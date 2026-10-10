"""Token counting used for chunk sizes, summary lengths and the retrieval budget."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def count_many(self, texts: Sequence[str]) -> list[int]: ...


class WordPieceCounter:
    """Counts BERT WordPiece tokens (no [CLS]/[SEP]) with the embedder's own tokenizer.

    WordPiece pre-tokenises on whitespace and punctuation, so the count of texts joined with
    whitespace equals the sum of their counts. Packing relies on that to stay under budget.
    """

    def __init__(self, tokenizer_json: Path) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_json))
        self._tok.no_truncation()
        self._tok.no_padding()
        self.name = "wordpiece:all-MiniLM-L6-v2"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._tok.encode_batch(list(texts), add_special_tokens=False)
        return [len(e.ids) for e in encodings]


class WordCounter:
    """Whitespace word count; the offline profile and the tests use it."""

    name = "words"

    def count(self, text: str) -> int:
        return len(text.split())

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]
