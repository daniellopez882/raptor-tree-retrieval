"""Cluster summarisers: an extractive centroid + MMR summariser and an LLM summariser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from raptree.embed import Matrix
    from raptree.llm import ChatProvider


@dataclass(frozen=True)
class ClusterView:
    """What a summariser sees: the member nodes and every segment those members contain."""

    child_texts: list[str]
    child_embeddings: Matrix
    segment_ids: list[int]
    segment_texts: list[str]
    segment_tokens: list[int]
    segment_embeddings: Matrix


@dataclass(frozen=True)
class Summary:
    text: str
    # Segment ids reused verbatim (extractive), or None when the text is newly written.
    segment_ids: list[int] | None


class Summarizer(Protocol):
    name: str

    def summarize(self, view: ClusterView) -> Summary: ...


class ExtractiveSummarizer:
    """Maximal marginal relevance against the cluster centroid, under a token budget.

    Relevance is cosine similarity to the normalised mean of the member embeddings; the
    redundancy penalty is the highest similarity to an already chosen sentence. Chosen
    sentences are emitted in article order and copied verbatim, so every summary sentence
    has exact provenance.
    """

    def __init__(self, max_tokens: int = 128, diversity: float = 0.3) -> None:
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be in [0, 1]")
        self.max_tokens = max_tokens
        self.diversity = diversity
        self.name = f"extractive-mmr(max_tokens={max_tokens},diversity={diversity})"

    def summarize(self, view: ClusterView) -> Summary:
        centroid = view.child_embeddings.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        relevance = view.segment_embeddings @ centroid
        similarity = view.segment_embeddings @ view.segment_embeddings.T
        tokens = np.asarray(view.segment_tokens)
        chosen: list[int] = []
        redundancy = np.zeros(len(view.segment_ids))
        available = np.ones(len(view.segment_ids), dtype=bool)
        used = 0
        while True:
            fits = available & (tokens + used <= self.max_tokens)
            if not fits.any():
                break
            score = (1.0 - self.diversity) * relevance - self.diversity * redundancy
            score[~fits] = -np.inf
            pick = int(np.argmax(score))
            chosen.append(pick)
            used += int(tokens[pick])
            available[pick] = False
            redundancy = np.maximum(redundancy, similarity[pick])
        chosen.sort(key=lambda j: view.segment_ids[j])
        ids = [view.segment_ids[j] for j in chosen]
        return Summary(" ".join(view.segment_texts[j] for j in chosen), ids)


SUMMARY_INSTRUCTION = (
    "Summarise the passages below in one paragraph. Keep names, places, numbers and the order "
    "of events. Do not add anything that the passages do not say."
)


class LlmSummarizer:
    """Abstractive summaries through any chat provider (OpenAI-compatible or the fake)."""

    def __init__(self, provider: ChatProvider, max_tokens: int = 256) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.name = f"llm({provider.name},max_tokens={max_tokens})"

    def summarize(self, view: ClusterView) -> Summary:
        body = "\n\n".join(view.child_texts)
        text = self.provider.complete(
            [
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": body},
            ],
            max_tokens=self.max_tokens,
        )
        return Summary(" ".join(text.split()), None)
