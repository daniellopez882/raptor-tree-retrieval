from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from raptree.llm import FakeProvider
from raptree.readers import (
    LlmReader,
    OptionEvidence,
    OverlapScorer,
    PairReader,
    entailment_log_odds,
    hypothesis,
    parse_choice,
    select_evidence,
)


class LengthScorer:
    """Scores a pair by the length of its second element and counts calls."""

    name = "length"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.seen.extend(pairs)
        return np.array([len(b) for _, b in pairs], dtype=np.float64)


def test_evidence_is_budgeted_deduplicated_and_in_context_order() -> None:
    emb = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Similarities: segment 0 -> 1.0, 2 -> 0.8, 1 and 3 -> 0. Segment 2 appears twice.
    chosen = select_evidence([3, 2, 0, 2, 1], emb, [4, 3, 3, 3], target, budget=7)
    assert chosen == [2, 0]


def test_evidence_of_an_empty_context_is_empty() -> None:
    assert select_evidence([], np.zeros((0, 4), dtype=np.float32), [], np.ones(4), 10) == []


def test_pair_reader_scores_each_distinct_pair_once() -> None:
    scorer = LengthScorer()
    reader = PairReader(scorer, "premise-first")
    item = OptionEvidence("Q?", ["a", "bbb", "cc", "d"], ["same", "same", "same", "same"])
    answers = reader.answer_many([item, item])
    assert [a.choice for a in answers] == [1, 1]
    assert reader.pairs_scored == 4
    assert scorer.seen[1] == ("same", hypothesis("Q?", "bbb"))


def test_query_first_layout_puts_the_hypothesis_first() -> None:
    scorer = LengthScorer()
    reader = PairReader(scorer, "query-first")
    item = OptionEvidence("Q?", ["a", "b", "c", "d"], ["x", "yyyy", "zz", "w"])
    assert reader.answer_many([item])[0].choice == 1
    assert scorer.seen[0] == ("Q? a", "x")


def test_progress_is_reported_per_chunk() -> None:
    reader = PairReader(LengthScorer(), "premise-first")
    calls: list[tuple[int, int]] = []
    items = [OptionEvidence(f"Q{i}", ["a", "b", "c", "d"], ["e", "f", "g", "h"]) for i in range(3)]
    reader.answer_many(items, chunk_size=5, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(5, 12), (10, 12), (12, 12)]


def test_entailment_log_odds_for_two_and_three_labels() -> None:
    two = np.array([[2.0, 0.0]])
    assert entailment_log_odds(two, 0)[0] == pytest.approx(2.0)
    three = np.array([[1.0, 0.0, 0.0]])
    # log p(e) - log(p(n) + p(c)) = 1 - log(2)
    assert entailment_log_odds(three, 0)[0] == pytest.approx(1.0 - np.log(2.0))


def test_overlap_scorer_counts_shared_content_words() -> None:
    scores = OverlapScorer().score(
        [("the ferry left at dawn", "When did the ferry leave? dawn"), ("nothing here", "ferry")]
    )
    assert scores[0] == pytest.approx(2 / 3)  # ferry, dawn of {ferry, leave, dawn}
    assert scores[1] == 0.0


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("B", 1),
        ("(C) because of the fog", 2),
        ("The answer is D.", 3),
        ("I think option a", 0),
        ("none of them", -1),
    ],
)
def test_parse_choice(reply: str, expected: int) -> None:
    assert parse_choice(reply) == expected


def test_llm_reader_builds_a_lettered_prompt() -> None:
    provider = FakeProvider(lambda messages: "C")
    reader = LlmReader(provider)
    answer = reader.answer("Some passage.", "Which one?", ["w", "x", "y", "z"])
    assert answer.choice == 2
    assert answer.scores == [0.0, 0.0, 1.0, 0.0]
    prompt = provider.calls[0][1]["content"]
    assert "Passage:\nSome passage." in prompt
    assert "C. y" in prompt
    assert prompt.endswith("Answer:")
