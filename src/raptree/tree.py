"""The RAPTOR tree: data model, builder and on-disk format."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel

from raptree.clustering import ClusterConfig, two_stage_clusters
from raptree.summarize import ClusterView
from raptree.text import chunk, normalise, segment

if TYPE_CHECKING:
    from raptree.embed import Embedder, Matrix
    from raptree.reduce import Reducer
    from raptree.summarize import Summarizer
    from raptree.tokens import TokenCounter

FORMAT_VERSION = 1


class Segment(BaseModel):
    id: int
    text: str
    tokens: int
    origin: Literal["article", "summary"]
    # Character offsets into the normalised article; None for LLM-written summary sentences.
    start: int | None = None
    end: int | None = None


class Node(BaseModel):
    id: int
    layer: int
    text: str
    tokens: int
    segments: list[int]
    children: list[int]
    leaves: list[int]


@dataclass(frozen=True)
class BuildConfig:
    chunk_tokens: int = 100
    summary_tokens: int = 128
    max_layers: int = 5
    max_cluster_tokens: int = 3500
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_tokens": self.chunk_tokens,
            "summary_tokens": self.summary_tokens,
            "max_layers": self.max_layers,
            "max_cluster_tokens": self.max_cluster_tokens,
            "cluster": self.cluster.__dict__.copy(),
        }


@dataclass
class RaptorTree:
    doc_id: str
    title: str
    segments: list[Segment]
    nodes: list[Node]
    layers: list[list[int]]
    segment_embeddings: Matrix
    node_embeddings: Matrix
    meta: dict[str, Any]

    @property
    def leaves(self) -> list[Node]:
        return [self.nodes[i] for i in self.layers[0]]

    @property
    def depth(self) -> int:
        """Number of layers including the leaves."""
        return len(self.layers)

    def parents(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for node in self.nodes:
            for child in node.children:
                out[child].append(node.id)
        return out

    def save(self, path: Path) -> None:
        payload = {
            "format": FORMAT_VERSION,
            "doc_id": self.doc_id,
            "title": self.title,
            "segments": [s.model_dump() for s in self.segments],
            "nodes": [n.model_dump() for n in self.nodes],
            "layers": self.layers,
            "meta": self.meta,
        }
        blob = np.frombuffer(json.dumps(payload).encode("utf-8"), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez_compressed(
                fh,
                payload=blob,
                segment_embeddings=self.segment_embeddings,
                node_embeddings=self.node_embeddings,
            )

    @classmethod
    def load(cls, path: Path) -> RaptorTree:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(data["payload"].tobytes().decode("utf-8"))
            seg_emb = data["segment_embeddings"]
            node_emb = data["node_embeddings"]
        if payload.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported tree format {payload.get('format')}")
        return cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            segments=[Segment(**s) for s in payload["segments"]],
            nodes=[Node(**n) for n in payload["nodes"]],
            layers=payload["layers"],
            segment_embeddings=seg_emb,
            node_embeddings=node_emb,
            meta=payload["meta"],
        )


class TreeBuilder:
    def __init__(
        self,
        embedder: Embedder,
        counter: TokenCounter,
        reducer: Reducer,
        summarizer: Summarizer,
        config: BuildConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.counter = counter
        self.reducer = reducer
        self.summarizer = summarizer
        self.config = config or BuildConfig()

    def build(self, doc_id: str, title: str, text: str) -> RaptorTree:
        cfg = self.config
        timings = {"embed": 0.0, "cluster": 0.0, "summarize": 0.0}
        started = time.perf_counter()

        article = normalise(text)
        spans = segment(article, self.counter, cfg.chunk_tokens)
        if not spans:
            raise ValueError(f"{doc_id}: no text to index")
        segments = [
            Segment(id=i, text=s.text, tokens=s.tokens, origin="article", start=s.start, end=s.end)
            for i, s in enumerate(spans)
        ]
        nodes: list[Node] = []
        for group in chunk(spans, cfg.chunk_tokens):
            nodes.append(
                Node(
                    id=len(nodes),
                    layer=0,
                    text=" ".join(segments[i].text for i in group),
                    tokens=sum(segments[i].tokens for i in group),
                    segments=group,
                    children=[],
                    leaves=[len(nodes)],
                )
            )
        t = time.perf_counter()
        seg_emb = self.embedder.embed([s.text for s in segments])
        node_emb = self.embedder.embed([n.text for n in nodes])
        timings["embed"] += time.perf_counter() - t

        layers = [[n.id for n in nodes]]
        stop_at = cfg.cluster.reduction_dim + 1
        while len(layers) <= cfg.max_layers and len(layers[-1]) > stop_at:
            current = layers[-1]
            t = time.perf_counter()
            clusters = self._cluster(current, nodes, node_emb)
            timings["cluster"] += time.perf_counter() - t
            if len(clusters) >= len(current):
                break  # clustering no longer shrinks the layer
            new_layer: list[int] = []
            new_texts: list[str] = []
            for members in clusters:
                t = time.perf_counter()
                node, new_segments = self._summarize(members, nodes, node_emb, segments, seg_emb)
                timings["summarize"] += time.perf_counter() - t
                if new_segments:
                    t = time.perf_counter()
                    extra = self.embedder.embed([s.text for s in new_segments])
                    timings["embed"] += time.perf_counter() - t
                    segments.extend(new_segments)
                    seg_emb = np.vstack([seg_emb, extra])
                node.id = len(nodes)
                node.layer = len(layers)
                nodes.append(node)
                new_layer.append(node.id)
                new_texts.append(node.text)
            t = time.perf_counter()
            node_emb = np.vstack([node_emb, self.embedder.embed(new_texts)])
            timings["embed"] += time.perf_counter() - t
            layers.append(new_layer)

        meta = {
            "build_config": cfg.as_dict(),
            "embedder": self.embedder.name,
            "token_counter": self.counter.name,
            "reducer": self.reducer.name,
            "summarizer": self.summarizer.name,
            "build_seconds": time.perf_counter() - started,
            "stage_seconds": timings,
            "article_tokens": sum(s.tokens for s in segments if s.origin == "article"),
        }
        return RaptorTree(doc_id, title, segments, nodes, layers, seg_emb, node_emb, meta)

    def _cluster(self, ids: list[int], nodes: list[Node], node_emb: Matrix) -> list[list[int]]:
        """Two-stage clustering; clusters too long for a summariser prompt are re-clustered."""
        out: list[list[int]] = []
        for members in two_stage_clusters(node_emb[ids], self.reducer, self.config.cluster):
            member_ids = [ids[m] for m in members]
            total = sum(nodes[i].tokens for i in member_ids)
            if (
                total <= self.config.max_cluster_tokens
                or len(member_ids) == 1
                or len(member_ids) == len(ids)
            ):
                out.append(member_ids)
            else:
                out.extend(self._cluster(member_ids, nodes, node_emb))
        return out

    def _summarize(
        self,
        members: list[int],
        nodes: list[Node],
        node_emb: Matrix,
        segments: list[Segment],
        seg_emb: Matrix,
    ) -> tuple[Node, list[Segment]]:
        seg_ids = sorted({s for m in members for s in nodes[m].segments})
        view = ClusterView(
            child_texts=[nodes[m].text for m in members],
            child_embeddings=node_emb[members],
            segment_ids=seg_ids,
            segment_texts=[segments[s].text for s in seg_ids],
            segment_tokens=[segments[s].tokens for s in seg_ids],
            segment_embeddings=seg_emb[seg_ids],
        )
        summary = self.summarizer.summarize(view)
        new_segments: list[Segment] = []
        if summary.segment_ids is not None:
            node_segments = summary.segment_ids
        else:
            spans = segment(summary.text, self.counter, self.config.chunk_tokens)
            if not spans:
                raise ValueError(f"{self.summarizer.name} returned an empty summary")
            first = len(segments)
            new_segments = [
                Segment(id=first + i, text=s.text, tokens=s.tokens, origin="summary")
                for i, s in enumerate(spans)
            ]
            node_segments = [s.id for s in new_segments]
        lookup = {s.id: s for s in new_segments}
        seg_tokens = [
            lookup[i].tokens if i in lookup else segments[i].tokens for i in node_segments
        ]
        texts = [lookup[i].text if i in lookup else segments[i].text for i in node_segments]
        node = Node(
            id=-1,
            layer=-1,
            text=" ".join(texts),
            tokens=sum(seg_tokens),
            segments=node_segments,
            children=list(members),
            leaves=sorted({leaf for m in members for leaf in nodes[m].leaves}),
        )
        return node, new_segments
