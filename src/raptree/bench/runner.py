"""The QuALITY benchmark: build each article's tree, retrieve under the budget, read, score."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from raptree.profiles import Profile, make_builder
from raptree.quality import load_quality
from raptree.readers import (
    OnnxPairScorer,
    OptionEvidence,
    OverlapScorer,
    PairReader,
    hypothesis,
    select_evidence,
)
from raptree.retrieve import METHODS, Retriever, query_text
from raptree.sources import require_model
from raptree.stats import clustered_bootstrap, paired_permutation_test
from raptree.tree import BuildConfig

if TYPE_CHECKING:
    from pathlib import Path

    from raptree.embed import Embedder
    from raptree.reduce import Reducer

CONTROL = "no-context"
READERS = ("nli", "cross-encoder", "overlap")
SUBSETS = ("all", "hard", "easy")
Log = Callable[[str], None]


@dataclass(frozen=True)
class BenchConfig:
    budget: int = 2000
    beam: int = 3
    evidence_tokens: int = 192
    readers: tuple[str, ...] = READERS
    profile: Profile = "onnx"
    limit_articles: int | None = None
    limit_questions: int | None = None
    bootstrap_resamples: int = 2000
    seed: int = 0
    threads: int | None = None
    build: BuildConfig = field(default_factory=BuildConfig)

    @property
    def is_subset(self) -> bool:
        return self.limit_articles is not None or self.limit_questions is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget,
            "traversal_beam": self.beam,
            "evidence_tokens": self.evidence_tokens,
            "readers": list(self.readers),
            "profile": self.profile,
            "limit_articles": self.limit_articles,
            "limit_questions_per_article": self.limit_questions,
            "bootstrap_resamples": self.bootstrap_resamples,
            "seed": self.seed,
            "threads": self.threads,
            "build": self.build.as_dict(),
            "retrieval_query": "question + all four options",
            "summarizer": "extractive centroid+MMR",
        }


def make_reader(name: str, threads: int | None) -> PairReader:
    if name == "overlap":
        return PairReader(OverlapScorer(), "premise-first")
    if name == "nli":
        paths = require_model("nli")
        scorer = OnnxPairScorer(
            paths.weights,
            paths.tokenizer,
            kind="nli",
            name="xtremedistil-l6-h256-zeroshot (NLI)",
            threads=threads,
        )
        return PairReader(scorer, "premise-first")
    if name == "cross-encoder":
        paths = require_model("cross-encoder")
        scorer = OnnxPairScorer(
            paths.weights,
            paths.tokenizer,
            kind="relevance",
            name="ms-marco-MiniLM-L6-v2 (cross-encoder)",
            threads=threads,
        )
        return PairReader(scorer, "query-first")
    raise ValueError(f"unknown reader {name!r}; choose from {', '.join(READERS)}")


@dataclass
class Row:
    """One (question, method) pair."""

    qid: str
    article: str
    hard: bool
    gold: int
    method: str
    tokens: int
    distinct_tokens: int
    summary_tokens: int
    nodes: int
    latency_ms: float
    predictions: dict[str, int] = field(default_factory=dict)


def _warm_up(embedder: Embedder, reducer: Reducer) -> float:
    """Pay one-off costs (numba compilation inside UMAP, ONNX session warm-up) before timing."""
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    reducer.reduce(rng.normal(size=(40, embedder.dim)).astype(np.float32), 6, 10, 0)
    embedder.embed(["warm-up sentence"] * 4)
    return time.perf_counter() - started


def _summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _holm(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def run_benchmark(
    config: BenchConfig,
    dataset: Path,
    *,
    trees_dir: Path | None = None,
    log: Log = print,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run everything; returns the results document and one prediction record per question."""
    articles = load_quality(dataset)
    if config.limit_articles is not None:
        articles = articles[: config.limit_articles]
    builder = make_builder(config.profile, config.build, config.threads)
    embedder = builder.embedder
    warm_up_seconds = _warm_up(embedder, builder.reducer)
    log(f"warm-up {warm_up_seconds:.1f}s; {len(articles)} articles")

    tree_rows: list[dict[str, Any]] = []
    rows: list[Row] = []
    items: list[OptionEvidence] = []
    query_embed_seconds = 0.0
    n_queries = 0
    for index, article in enumerate(articles, 1):
        tree = builder.build(article.id, article.title, article.text)
        if trees_dir is not None:
            tree.save(trees_dir / f"{article.id}.npz")
        tree_rows.append(
            {
                "article": article.id,
                "build_seconds": tree.meta["build_seconds"],
                "stage_seconds": tree.meta["stage_seconds"],
                "article_tokens": tree.meta["article_tokens"],
                "layers": [len(layer) for layer in tree.layers],
                "nodes": len(tree.nodes),
            }
        )
        log(
            f"[{index}/{len(articles)}] {article.id} built in {tree.meta['build_seconds']:.1f}s, "
            f"layer sizes {[len(layer) for layer in tree.layers]}"
        )
        questions = article.questions
        if config.limit_questions is not None:
            questions = questions[: config.limit_questions]
        if not questions:
            continue
        queries = [query_text(q.question, q.options) for q in questions]
        started = time.perf_counter()
        qvecs = embedder.embed(queries)
        hvecs = embedder.embed([hypothesis(q.question, o) for q in questions for o in q.options])
        query_embed_seconds += time.perf_counter() - started
        n_queries += len(questions)
        offsets = np.cumsum([0, *[len(q.options) for q in questions]])
        retriever = Retriever(tree, config.budget, config.beam)
        seg_tokens = [s.tokens for s in tree.segments]
        for qi, q in enumerate(questions):
            option_vecs = hvecs[offsets[qi] : offsets[qi + 1]]
            for method in METHODS:
                started = time.perf_counter()
                found = retriever.run(method, queries[qi], qvecs[qi])
                latency_ms = (time.perf_counter() - started) * 1000.0
                seg_ids = [s for n in found.node_ids for s in tree.nodes[n].segments]
                evidence = []
                for vec in option_vecs:
                    chosen = select_evidence(
                        seg_ids, tree.segment_embeddings, seg_tokens, vec, config.evidence_tokens
                    )
                    evidence.append(" ".join(tree.segments[s].text for s in chosen))
                items.append(OptionEvidence(q.question, q.options, evidence))
                rows.append(
                    Row(
                        qid=q.id,
                        article=article.id,
                        hard=q.hard,
                        gold=q.gold,
                        method=method,
                        tokens=found.tokens,
                        distinct_tokens=sum(seg_tokens[s] for s in set(seg_ids)),
                        summary_tokens=sum(
                            tree.nodes[n].tokens for n in found.node_ids if tree.nodes[n].layer > 0
                        ),
                        nodes=len(found.node_ids),
                        latency_ms=latency_ms,
                    )
                )
            items.append(OptionEvidence(q.question, q.options, [""] * len(q.options)))
            rows.append(Row(q.id, article.id, q.hard, q.gold, CONTROL, 0, 0, 0, 0, 0.0))

    reader_stats: dict[str, dict[str, Any]] = {}
    for name in config.readers:
        reader = make_reader(name, config.threads)
        log(f"reader {name}: {len(items)} items")
        started = time.perf_counter()
        answers = reader.answer_many(
            items, progress=lambda done, total, n=name: log(f"  {n}: {done}/{total} pairs")
        )
        seconds = time.perf_counter() - started
        reader_stats[name] = {
            "model": reader.name,
            "pairs": sum(len(item.options) for item in items),
            "unique_pairs_scored": reader.pairs_scored,
            "seconds": seconds,
            "pairs_per_second": reader.pairs_scored / seconds if seconds > 0 else None,
        }
        for row, answer in zip(rows, answers, strict=True):
            row.predictions[name] = answer.choice

    results = {
        "config": config.as_dict(),
        "dataset": {
            "articles": len(articles),
            "questions": len({r.qid for r in rows}),
            "hard_questions": len({r.qid for r in rows if r.hard}),
        },
        "warm_up_seconds": warm_up_seconds,
        "trees": _tree_summary(tree_rows),
        "retrieval": _retrieval_summary(rows, query_embed_seconds, n_queries),
        "readers": reader_stats,
        "accuracy": _accuracy(rows, config),
        "paired_tests": _paired(rows, config),
        "not_run": [
            {
                "what": "RAPTOR with LLM-written summaries",
                "why": "needs an LLM endpoint; the extractive summariser was measured instead",
            },
            {
                "what": "LLM reader over the full 2,000-token context",
                "why": "needs an LLM endpoint; only the LLM-free readers were measured",
            },
        ],
    }
    return results, _prediction_records(rows)


def _tree_summary(tree_rows: list[dict[str, Any]]) -> dict[str, Any]:
    depth = Counter(len(r["layers"]) for r in tree_rows)
    return {
        "per_article": tree_rows,
        "build_seconds": _summary([r["build_seconds"] for r in tree_rows]),
        "total_build_seconds": float(sum(r["build_seconds"] for r in tree_rows)),
        "stage_seconds_total": {
            stage: float(sum(r["stage_seconds"][stage] for r in tree_rows))
            for stage in ("embed", "cluster", "summarize")
        },
        "nodes": _summary([r["nodes"] for r in tree_rows]),
        "leaves": _summary([r["layers"][0] for r in tree_rows]),
        "article_tokens": _summary([r["article_tokens"] for r in tree_rows]),
        "depth_histogram": {str(k): v for k, v in sorted(depth.items())},
    }


def _retrieval_summary(rows: list[Row], embed_seconds: float, n_queries: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "query_embedding_ms_per_question": 1000.0 * embed_seconds / max(n_queries, 1),
        "methods": {},
    }
    for method in METHODS:
        sel = [r for r in rows if r.method == method]
        if not sel:
            continue
        tokens = sum(r.tokens for r in sel)
        out["methods"][method] = {
            "mean_tokens": float(np.mean([r.tokens for r in sel])),
            "mean_distinct_tokens": float(np.mean([r.distinct_tokens for r in sel])),
            "mean_nodes": float(np.mean([r.nodes for r in sel])),
            "summary_token_share": sum(r.summary_tokens for r in sel) / tokens if tokens else 0.0,
            "latency_ms": _summary([r.latency_ms for r in sel]),
        }
    return out


def _subset(rows: list[Row], subset: str) -> list[Row]:
    if subset == "hard":
        return [r for r in rows if r.hard]
    if subset == "easy":
        return [r for r in rows if not r.hard]
    return rows


def _accuracy(rows: list[Row], config: BenchConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for reader in config.readers:
        out[reader] = {}
        for method in (*METHODS, CONTROL):
            out[reader][method] = {}
            for subset in SUBSETS:
                sel = [r for r in _subset(rows, subset) if r.method == method]
                if not sel:
                    continue
                interval = clustered_bootstrap(
                    [r.predictions[reader] == r.gold for r in sel],
                    [r.article for r in sel],
                    n_resamples=config.bootstrap_resamples,
                    seed=config.seed,
                )
                out[reader][method][subset] = {
                    "accuracy": interval.estimate,
                    "ci95": [interval.low, interval.high],
                    "n": interval.n,
                }
    return out


def _paired(rows: list[Row], config: BenchConfig) -> dict[str, Any]:
    by_method: dict[str, dict[str, Row]] = {}
    for r in rows:
        by_method.setdefault(r.method, {})[r.qid] = r
    out: dict[str, Any] = {}
    for reader in config.readers:
        tests = []
        for a, b in combinations((*METHODS, CONTROL), 2):
            qids = sorted(by_method[a])
            test = paired_permutation_test(
                [by_method[a][q].predictions[reader] == by_method[a][q].gold for q in qids],
                [by_method[b][q].predictions[reader] == by_method[b][q].gold for q in qids],
            )
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "difference": test.difference,
                    "p_value": test.p_value,
                    "a_only_correct": test.a_only,
                    "b_only_correct": test.b_only,
                    "n": test.n,
                }
            )
        for test, adjusted in zip(tests, _holm([t["p_value"] for t in tests]), strict=True):
            test["p_holm"] = adjusted
        out[reader] = tests
    return out


def _prediction_records(rows: list[Row]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for r in rows:
        rec = records.setdefault(
            r.qid,
            {"id": r.qid, "article": r.article, "hard": r.hard, "gold": r.gold, "methods": {}},
        )
        rec["methods"][r.method] = {"tokens": r.tokens, "predictions": r.predictions}
    return list(records.values())
