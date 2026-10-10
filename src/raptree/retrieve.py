"""Retrieval under a token budget: truncation, BM25, dense, RAPTOR collapsed and traversal."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raptree.tree import RaptorTree

Method = Literal["truncation", "bm25", "dense", "raptor-collapsed", "raptor-traversal"]
METHODS: tuple[Method, ...] = (
    "truncation",
    "bm25",
    "dense",
    "raptor-collapsed",
    "raptor-traversal",
)
EMBEDDING_METHODS: frozenset[Method] = frozenset({"dense", "raptor-collapsed", "raptor-traversal"})


@dataclass(frozen=True)
class Retrieval:
    method: Method
    node_ids: list[int]  # in the order they are placed in the context
    tokens: int

    def text(self, tree: RaptorTree) -> str:
        return "\n\n".join(tree.nodes[i].text for i in self.node_ids)


def pack(ranked: Sequence[int], tokens: Sequence[int], budget: int) -> list[int]:
    """Greedy packing: walk the ranking, keep every node that still fits.

    A node that does not fit is skipped rather than ending the walk, so a long summary near
    the top of the ranking cannot leave most of the budget unused.
    """
    out: list[int] = []
    seen: set[int] = set()
    used = 0
    for i in ranked:
        if i in seen:
            continue
        if used + tokens[i] <= budget:
            out.append(i)
            seen.add(i)
            used += tokens[i]
    return out


_TERM = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has",
        "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "not", "of", "on", "or", "our", "she", "so", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when",
        "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your",
        "do", "does", "did", "been", "being", "how",
    }
)  # fmt: skip


def terms(text: str) -> list[str]:
    return [t for t in _TERM.findall(text.lower()) if t not in _STOP]


class Bm25:
    """Okapi BM25 with the non-negative Lucene IDF."""

    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf = [Counter(terms(d)) for d in documents]
        self._len = np.array([sum(tf.values()) for tf in self._tf], dtype=np.float64)
        self._avg = float(self._len.mean()) if len(documents) else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documents)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> NDArray[np.float64]:
        out = np.zeros(len(self._tf))
        if not self._tf:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self._len / max(self._avg, 1e-9))
        for term in set(terms(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            freq = np.array([tf.get(term, 0) for tf in self._tf], dtype=np.float64)
            out += idf * freq * (self.k1 + 1.0) / (freq + norm)
        return out


def _ranked(ids: Sequence[int], scores: NDArray) -> list[int]:
    # Stable sort on -score keeps document order among ties.
    order = np.argsort(-scores[list(ids)], kind="stable")
    return [ids[i] for i in order]


class Retriever:
    def __init__(self, tree: RaptorTree, budget: int, beam: int = 3) -> None:
        self.tree = tree
        self.budget = budget
        self.beam = beam
        self._tokens = [n.tokens for n in tree.nodes]
        self._leaf_ids = list(tree.layers[0])
        self._bm25: Bm25 | None = None

    def _finish(self, method: Method, ids: list[int]) -> Retrieval:
        return Retrieval(method, ids, sum(self._tokens[i] for i in ids))

    def truncation(self) -> Retrieval:
        ids: list[int] = []
        used = 0
        for i in self._leaf_ids:
            if used + self._tokens[i] > self.budget:
                break
            ids.append(i)
            used += self._tokens[i]
        return self._finish("truncation", ids)

    def bm25(self, query: str) -> Retrieval:
        if self._bm25 is None:
            self._bm25 = Bm25([self.tree.nodes[i].text for i in self._leaf_ids])
        scores = np.zeros(len(self.tree.nodes))
        scores[self._leaf_ids] = self._bm25.scores(query)
        return self._finish(
            "bm25", pack(_ranked(self._leaf_ids, scores), self._tokens, self.budget)
        )

    def dense(self, qvec: NDArray[np.float32]) -> Retrieval:
        sims = self.tree.node_embeddings @ qvec
        ranked = _ranked(self._leaf_ids, sims)
        return self._finish("dense", pack(ranked, self._tokens, self.budget))

    def collapsed(self, qvec: NDArray[np.float32]) -> Retrieval:
        """Rank every node of every layer together, then pack."""
        sims = self.tree.node_embeddings @ qvec
        ranked = _ranked(list(range(len(self.tree.nodes))), sims)
        return self._finish("raptor-collapsed", pack(ranked, self._tokens, self.budget))

    def traversal(self, qvec: NDArray[np.float32]) -> Retrieval:
        """Beam search from the top layer down, then fill with leaves under the final beam.

        At each layer the ``beam`` best children of the previous beam are kept. The context
        is the beam path (top layer first), followed by the remaining leaves under the last
        summary-layer beam in score order, packed to the budget.
        """
        sims = self.tree.node_embeddings @ qvec
        nodes = self.tree.nodes
        order: list[int] = []
        frontier = _ranked(self.tree.layers[-1], sims)
        while True:
            chosen = frontier[: self.beam]
            order.extend(chosen)
            children: list[int] = []
            seen = set(order)
            for c in chosen:
                for child in nodes[c].children:
                    if child not in seen:
                        seen.add(child)
                        children.append(child)
            if not children:
                order.extend(frontier[self.beam :])
                break
            frontier = _ranked(children, sims)
        return self._finish("raptor-traversal", pack(order, self._tokens, self.budget))

    def run(self, method: Method, query: str, qvec: NDArray[np.float32] | None) -> Retrieval:
        if method == "truncation":
            return self.truncation()
        if method == "bm25":
            return self.bm25(query)
        if qvec is None:
            raise ValueError(f"{method} needs a query embedding")
        if method == "dense":
            return self.dense(qvec)
        if method == "raptor-collapsed":
            return self.collapsed(qvec)
        return self.traversal(qvec)


def query_text(question: str, options: Sequence[str]) -> str:
    """Retrieval query: the question followed by all four options."""
    return " ".join([question, *options])
