"""Multiple-choice readers that turn a retrieved context into an answer.

The two LLM-free readers share one recipe. For each option, the sentences of the context
most similar to "question + option" are kept up to a small evidence budget (the pair models
accept 512 tokens, the context holds 2,000), then a pair model scores (evidence, option):

* ``nli``: entailment log-odds from an NLI cross-encoder, evidence as premise and
  "question option" as hypothesis;
* ``cross-encoder``: the MS MARCO relevance logit of "question option" against the evidence.

The predicted option is the arg-max. An LLM reader with the full context is also provided;
it needs an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from raptree.retrieve import terms

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raptree.embed import Matrix
    from raptree.llm import ChatProvider


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]: ...


class OnnxPairScorer:
    """A sequence-pair classifier exported to ONNX.

    ``kind="nli"`` returns log P(entailment) - log P(other labels); ``kind="relevance"``
    returns the single relevance logit. The longer side of each pair is truncated.
    """

    def __init__(
        self,
        model_path: Path,
        tokenizer_json: Path,
        *,
        kind: Literal["nli", "relevance"],
        name: str,
        entailment_index: int = 0,
        max_length: int = 512,
        batch_size: int = 16,
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
        self._tok.enable_truncation(max_length=max_length, strategy="longest_first")
        self._tok.no_padding()
        self.kind = kind
        self.name = name
        self._entail = entailment_index
        self._batch = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]:
        out = np.zeros(len(pairs))
        if not pairs:
            return out
        encodings = self._tok.encode_batch(list(pairs))
        order = sorted(range(len(pairs)), key=lambda i: len(encodings[i].ids))
        for lo in range(0, len(order), self._batch):
            idx = order[lo : lo + self._batch]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for row, i in enumerate(idx):
                enc = encodings[i]
                n = len(enc.ids)
                ids[row, :n] = enc.ids
                mask[row, :n] = 1
                types[row, :n] = enc.type_ids
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = types
            logits = np.asarray(self._session.run(None, feeds)[0], dtype=np.float64)
            out[idx] = self._reduce(logits)
        return out

    def _reduce(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kind == "relevance":
            return logits[:, 0]
        return entailment_log_odds(logits, self._entail)


def entailment_log_odds(logits: NDArray[np.float64], entail: int) -> NDArray[np.float64]:
    """log P(entailment) - log P(any other label), from raw classifier logits."""
    others = np.delete(logits, entail, axis=1)
    top = others.max(axis=1, keepdims=True)
    log_other = np.log(np.exp(others - top).sum(axis=1)) + top[:, 0]
    return logits[:, entail] - log_other


def select_evidence(
    segment_ids: Sequence[int],
    segment_embeddings: Matrix,
    segment_tokens: Sequence[int],
    target: NDArray[np.float32],
    budget: int,
) -> list[int]:
    """Most similar context sentences to ``target`` within ``budget`` tokens, in context order.

    ``segment_ids`` index the tree's segment table; a sentence that appears twice in the
    context (a leaf and an extractive summary can share it) is counted once.
    """
    unique = list(dict.fromkeys(segment_ids))
    if not unique:
        return []
    sims = segment_embeddings[unique] @ target
    picked: list[int] = []
    used = 0
    for j in np.argsort(-sims, kind="stable"):
        cost = segment_tokens[unique[j]]
        if used + cost <= budget:
            picked.append(int(j))
            used += cost
    picked.sort()
    return [unique[j] for j in picked]


def hypothesis(question: str, option: str) -> str:
    return f"{question.strip()} {option.strip()}"


@dataclass(frozen=True)
class OptionEvidence:
    """One (question, method) item: the evidence text chosen for each answer option."""

    question: str
    options: list[str]
    evidence: list[str]


@dataclass(frozen=True)
class ReaderAnswer:
    scores: list[float]
    choice: int  # 0-based; -1 when the reader produced no usable answer


class PairReader:
    """Arg-max over per-option pair scores; identical pairs are scored once per batch."""

    def __init__(self, scorer: PairScorer, layout: Literal["premise-first", "query-first"]):
        self.scorer = scorer
        self.layout = layout
        self.name = scorer.name
        self.pairs_scored = 0

    def _pair(self, evidence: str, question: str, option: str) -> tuple[str, str]:
        h = hypothesis(question, option)
        return (evidence, h) if self.layout == "premise-first" else (h, evidence)

    def answer_many(
        self,
        items: Sequence[OptionEvidence],
        *,
        chunk_size: int = 4096,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[ReaderAnswer]:
        keys: dict[tuple[str, str], int] = {}
        slots: list[list[int]] = []
        for item in items:
            row = []
            for option, evidence in zip(item.options, item.evidence, strict=True):
                pair = self._pair(evidence, item.question, option)
                row.append(keys.setdefault(pair, len(keys)))
            slots.append(row)
        unique = list(keys)
        scores = np.zeros(len(unique))
        for lo in range(0, len(unique), chunk_size):
            scores[lo : lo + chunk_size] = self.scorer.score(unique[lo : lo + chunk_size])
            if progress is not None:
                progress(min(lo + chunk_size, len(unique)), len(unique))
        self.pairs_scored += len(unique)
        out = []
        for row in slots:
            values = [float(scores[k]) for k in row]
            out.append(ReaderAnswer(values, int(np.argmax(values))))
        return out


LETTERS = "ABCD"
_LEADING = re.compile(r"^\W*([ABCD])\b")
_NAMED = re.compile(r"\b(?:answer|option)\b\W*(?:is\W*)?\(?([ABCD])\b", re.IGNORECASE)
_ANY = re.compile(r"\b([ABCD])\b")


def parse_choice(reply: str) -> int:
    """Index of the option letter in a reply such as "B", "(C) ..." or "The answer is D"."""
    for pattern in (_LEADING, _NAMED, _ANY):
        match = pattern.search(reply)
        if match:
            return LETTERS.index(match.group(1).upper())
    return -1


READER_INSTRUCTION = (
    "Read the passage, then answer the multiple-choice question. Reply with the letter of the "
    "best option only."
)


class LlmReader:
    """Full-context multiple choice through a chat provider; parses the first A-D letter."""

    def __init__(self, provider: ChatProvider, max_tokens: int = 8) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.name = f"llm({provider.name})"

    def answer(self, context: str, question: str, options: Sequence[str]) -> ReaderAnswer:
        lines = [f"Passage:\n{context}", "", f"Question: {question}"]
        lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
        lines.append("Answer:")
        reply = self.provider.complete(
            [
                {"role": "system", "content": READER_INSTRUCTION},
                {"role": "user", "content": "\n".join(lines)},
            ],
            max_tokens=self.max_tokens,
        )
        choice = parse_choice(reply)
        scores = [1.0 if i == choice else 0.0 for i in range(len(options))]
        return ReaderAnswer(scores, choice)


class OverlapScorer:
    """Share of the hypothesis's content words found in the evidence (an MCTest-style baseline).

    Needs no model. It runs as a third, lexical reader next to the two neural ones and is
    what the offline profile and the tests use.
    """

    name = "lexical-overlap"

    def __init__(self, layout: Literal["premise-first", "query-first"] = "premise-first"):
        self.layout = layout

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]:
        out = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            evidence, hyp = (a, b) if self.layout == "premise-first" else (b, a)
            wanted = set(terms(hyp))
            if wanted:
                out[i] = len(wanted & set(terms(evidence))) / len(wanted)
        return out
