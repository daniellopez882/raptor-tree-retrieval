"""Sentence segmentation and sentence-aligned chunking.

Every piece of text that reaches a retriever or a reader is a *segment*: a sentence, or a
slice of a sentence that was longer than one chunk. Leaves are runs of consecutive segments,
and extractive summaries are subsets of segments, so any context can be traced back to
character offsets in the normalised article.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raptree.tokens import TokenCounter

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "prof", "capt", "lt", "col",
        "gen", "sgt", "rev", "gov", "sen", "rep", "no", "vol", "fig", "mt", "ft", "co", "inc",
        "ltd", "e.g", "i.e", "u.s", "u.n", "a.m", "p.m", "cf", "approx", "dept", "est",
    }
)  # fmt: skip

_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def normalise(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    QuALITY's HTML-stripped articles break lines in the middle of sentences, so line breaks
    carry no reliable paragraph signal and are dropped.
    """
    return " ".join(text.split())


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of the sentences in already-normalised text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        punct_at = match.start()
        word = _WORD_BEFORE.search(text, max(0, punct_at - 40), punct_at)
        if word is not None:
            token = word.group(1).lower().rstrip(".")
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
        end = match.end()
        piece_end = end
        while piece_end > start and text[piece_end - 1] == " ":
            piece_end -= 1
        if piece_end > start:
            spans.append((start, piece_end))
        start = end
    if start < len(text):
        tail_end = len(text.rstrip())
        if tail_end > start:
            spans.append((start, tail_end))
    return spans


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    text: str
    tokens: int


def segment(text: str, counter: TokenCounter, max_tokens: int) -> list[Span]:
    """Split normalised text into sentence segments of at most ``max_tokens`` tokens each.

    Sentences longer than ``max_tokens`` are cut at word boundaries so that no segment can
    overflow a leaf on its own.
    """
    sentence_spans = split_sentences(text)
    sentences = [text[a:b] for a, b in sentence_spans]
    counts = counter.count_many(sentences)
    out: list[Span] = []
    for (a, b), sentence, n in zip(sentence_spans, sentences, counts, strict=True):
        if n <= max_tokens:
            out.append(Span(a, b, sentence, n))
            continue
        out.extend(_split_long(text, a, b, counter, max_tokens))
    return out


def _split_long(text: str, a: int, b: int, counter: TokenCounter, max_tokens: int) -> list[Span]:
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text[a:b])]
    word_counts = counter.count_many([text[a + s : a + e] for s, e in words])
    pieces: list[Span] = []
    piece_start = 0
    used = 0
    for i, n in enumerate(word_counts):
        if used + n > max_tokens and i > piece_start:
            s, e = a + words[piece_start][0], a + words[i - 1][1]
            pieces.append(Span(s, e, text[s:e], used))
            piece_start, used = i, 0
        used += n
    s, e = a + words[piece_start][0], a + words[-1][1]
    pieces.append(Span(s, e, text[s:e], used))
    return pieces


def chunk(spans: list[Span], max_tokens: int) -> list[list[int]]:
    """Group consecutive segment indices into leaves of at most ``max_tokens`` tokens."""
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, span in enumerate(spans):
        if current and used + span.tokens > max_tokens:
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += span.tokens
    if current:
        groups.append(current)
    return groups
