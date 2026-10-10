"""Sentence embedders. Every embedder returns L2-normalised float32 rows."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def _normalise_rows(x: NDArray[np.floating]) -> Matrix:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (x / norms).astype(np.float32)


class OnnxSentenceEmbedder:
    """all-MiniLM-L6-v2 through onnxruntime: mean pooling over real tokens, then L2 norm."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_json: Path,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tok = Tokenizer.from_file(str(tokenizer_json))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.no_padding()
        self._batch = batch_size
        self.name = "all-MiniLM-L6-v2-onnx-quint8"
        self.dim = 384

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        encodings = self._tok.encode_batch(list(texts))
        # Sorting by length keeps padding (and wasted compute) small inside each batch.
        order = sorted(range(len(texts)), key=lambda i: len(encodings[i].ids))
        for lo in range(0, len(order), self._batch):
            idx = order[lo : lo + self._batch]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = len(encodings[i].ids)
                ids[row, :n] = encodings[i].ids
                mask[row, :n] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]
            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            out[idx] = pooled
        return _normalise_rows(out)


_WORD = re.compile(r"[a-z0-9']+")


class HashingEmbedder:
    """Signed feature hashing of words and word bigrams with sublinear term frequency.

    It needs no model files, which is what the offline profile, the Docker smoke run and the
    unit tests need. CRC32 keeps the vectors identical across platforms and processes.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            feats = words + [f"{a} {b}" for a, b in pairwise(words)]
            counts: dict[int, float] = {}
            for feat in feats:
                h = zlib.crc32(feat.encode("utf-8"))
                slot = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0
                counts[slot] = counts.get(slot, 0.0) + sign
            for slot, value in counts.items():
                out[row, slot] = np.sign(value) * np.log1p(abs(value))
        return _normalise_rows(out)
