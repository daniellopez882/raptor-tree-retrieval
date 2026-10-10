"""Component sets. ``onnx`` is the measured configuration; ``offline`` needs no downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from raptree.embed import Embedder, HashingEmbedder, OnnxSentenceEmbedder
from raptree.reduce import PcaReducer, Reducer, UmapReducer
from raptree.sources import require_model
from raptree.summarize import ExtractiveSummarizer
from raptree.tokens import TokenCounter, WordCounter, WordPieceCounter
from raptree.tree import BuildConfig, TreeBuilder

Profile = Literal["onnx", "offline"]
PROFILES: dict[str, Profile] = {"onnx": "onnx", "offline": "offline"}


@dataclass(frozen=True)
class Components:
    embedder: Embedder
    counter: TokenCounter
    reducer: Reducer


def components(profile: Profile, threads: int | None = None) -> Components:
    if profile == "offline":
        return Components(HashingEmbedder(), WordCounter(), PcaReducer())
    paths = require_model("embedder")
    return Components(
        OnnxSentenceEmbedder(paths.weights, paths.tokenizer, threads=threads),
        WordPieceCounter(paths.tokenizer),
        UmapReducer(),
    )


def make_builder(
    profile: Profile, config: BuildConfig | None = None, threads: int | None = None
) -> TreeBuilder:
    """A tree builder with the extractive summariser, the configuration that is benchmarked."""
    cfg = config or BuildConfig()
    comps = components(profile, threads)
    return TreeBuilder(
        comps.embedder,
        comps.counter,
        comps.reducer,
        ExtractiveSummarizer(cfg.summary_tokens),
        cfg,
    )
