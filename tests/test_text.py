from __future__ import annotations

from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from raptree.text import chunk, normalise, segment, split_sentences
from raptree.tokens import WordCounter


def sentences_of(text: str) -> list[str]:
    return [text[a:b] for a, b in split_sentences(text)]


def test_splits_on_terminal_punctuation_and_quotes() -> None:
    text = 'He stopped. "Is it far?" she asked. It was! Then they walked on.'
    assert sentences_of(text) == [
        "He stopped.",
        '"Is it far?" she asked.',
        "It was!",
        "Then they walked on.",
    ]


def test_does_not_split_after_abbreviations_or_initials() -> None:
    text = "Mr. Herne met Dr. Polglase at St. Ives. Robert F. Young wrote it."
    assert sentences_of(text) == [
        "Mr. Herne met Dr. Polglase at St. Ives.",
        "Robert F. Young wrote it.",
    ]


def test_lowercase_after_period_is_not_a_boundary() -> None:
    assert sentences_of("It cost 3.5 dollars. e.g. this one. Next.") == [
        "It cost 3.5 dollars. e.g. this one.",
        "Next.",
    ]


def test_normalise_collapses_line_breaks_inside_sentences() -> None:
    assert normalise("Every man's mind\n\n is a universe.\n") == "Every man's mind is a universe."


def test_long_sentences_are_cut_to_the_chunk_size() -> None:
    text = " ".join(f"w{i}" for i in range(250)) + "."
    spans = segment(text, WordCounter(), 100)
    assert [s.tokens for s in spans] == [100, 100, 50]
    assert " ".join(s.text for s in spans) == text


words = st.text(alphabet="abcdefgh", min_size=1, max_size=8)
sentence = st.lists(words, min_size=1, max_size=40).map(lambda ws: " ".join(ws).capitalize() + ".")


@settings(max_examples=60, deadline=None)
@given(st.lists(sentence, min_size=1, max_size=40), st.integers(5, 60))
def test_segments_tile_the_text_and_chunks_respect_the_limit(parts: list[str], limit: int) -> None:
    text = normalise(" ".join(parts))
    spans = segment(text, WordCounter(), limit)
    # Segments are ordered, disjoint, and together hold every word of the text.
    assert all(a.end <= b.start for a, b in pairwise(spans))
    assert " ".join(s.text for s in spans).split() == text.split()
    assert all(text[s.start : s.end] == s.text for s in spans)
    assert all(0 < s.tokens <= limit for s in spans)
    groups = chunk(spans, limit)
    assert [i for g in groups for i in g] == list(range(len(spans)))
    assert all(sum(spans[i].tokens for i in g) <= limit for g in groups)
