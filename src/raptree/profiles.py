"""Component sets. ``onnx`` is the measured configuration; ``offline`` needs no downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from raptree.embed import Embedder, HashingEmbedder, OnnxSentenceEmbedder
from raptree.llm import make_provider
from raptree.reduce import PcaReducer, Reducer, UmapReducer
from raptree.sources import require_model
from raptree.summarize import ExtractiveSummarizer, LlmSummarizer, Summarizer
from raptree.tokens import TokenCounter, WordCounter, WordPieceCounter
from raptree.tree import BuildConfig, TreeBuilder

Profile = Literal["onnx", "offline"]
PROFILES: dict[str, Profile] = {"onnx": "onnx", "offline": "offline"}

SummarizerKind = Literal["extractive", "llm"]
SUMMARIZERS: dict[str, SummarizerKind] = {"extractive": "extractive", "llm": "llm"}


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


def make_summarizer(kind: SummarizerKind, config: BuildConfig, llm: str = "openai") -> Summarizer:
    """``extractive`` is what the benchmark measures; ``llm`` goes through a chat provider."""
    if kind == "llm":
        return LlmSummarizer(make_provider(llm), max_tokens=2 * config.summary_tokens)
    return ExtractiveSummarizer(config.summary_tokens)


def make_builder(
    profile: Profile,
    config: BuildConfig | None = None,
    threads: int | None = None,
    *,
    summarizer: SummarizerKind = "extractive",
    llm: str = "openai",
) -> TreeBuilder:
    cfg = config or BuildConfig()
    comps = components(profile, threads)
    return TreeBuilder(
        comps.embedder,
        comps.counter,
        comps.reducer,
        make_summarizer(summarizer, cfg, llm),
        cfg,
    )
