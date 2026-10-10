"""Checks against the real QuALITY file and ONNX models; run with ``pytest -m integration``.

Each test is skipped when ``raptree fetch`` has not been run on this machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from raptree.embed import OnnxSentenceEmbedder
from raptree.quality import Article, load_quality
from raptree.readers import OnnxPairScorer
from raptree.reduce import UmapReducer
from raptree.retrieve import METHODS, Retriever, query_text
from raptree.sources import QUALITY_DEV, ModelPaths, locate_model, quality_path, sha256_file
from raptree.text import normalise
from raptree.tokens import WordPieceCounter

pytestmark = pytest.mark.integration


def model_or_skip(key: str) -> ModelPaths:
    paths = locate_model(key)
    if paths is None:
        pytest.skip(f"model {key} not fetched; run `raptree fetch --only models`")
    return paths


@pytest.fixture(scope="module")
def articles() -> list[Article]:
    path = quality_path()
    if not path.exists():
        pytest.skip("QuALITY not fetched; run `raptree fetch --only quality`")
    assert sha256_file(path) == QUALITY_DEV.sha256
    return load_quality(path)


def test_quality_dev_counts(articles: list[Article]) -> None:
    assert len(articles) == 115
    questions = [q for a in articles for q in a.questions]
    assert len(questions) == 2086
    assert sum(q.hard for q in questions) == 1065
    assert all(len(q.options) == 4 and 0 <= q.gold < 4 for q in questions)


def test_minilm_embeddings_are_normalised_and_meaningful() -> None:
    paths = model_or_skip("embedder")
    emb = OnnxSentenceEmbedder(paths.weights, paths.tokenizer)
    vecs = emb.embed(
        ["The ship sank in the storm.", "A vessel went down in bad weather.", "I like jam."]
    )
    assert vecs.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2] + 0.2


def test_wordpiece_counts_add_up_over_whitespace(articles: list[Article]) -> None:
    paths = model_or_skip("embedder")
    counter = WordPieceCounter(paths.tokenizer)
    words = normalise(articles[0].text).split()[:400]
    parts = [" ".join(words[i : i + 37]) for i in range(0, len(words), 37)]
    assert counter.count(" ".join(parts)) == sum(counter.count_many(parts))


def test_nli_and_cross_encoder_prefer_the_supported_option() -> None:
    nli_paths, ce_paths = model_or_skip("nli"), model_or_skip("cross-encoder")
    premise = "Marta climbed the ninety-seven steps and lit the lamp before sunset."
    right = "How many steps did Marta climb? Ninety-seven"
    wrong = "How many steps did Marta climb? Twelve"
    nli = OnnxPairScorer(nli_paths.weights, nli_paths.tokenizer, kind="nli", name="nli")
    scores = nli.score([(premise, right), (premise, wrong)])
    assert scores[0] > scores[1]
    ce = OnnxPairScorer(ce_paths.weights, ce_paths.tokenizer, kind="relevance", name="ce")
    scores = ce.score([(right, premise), ("What is the capital of Peru?", premise)])
    assert scores[0] > scores[1]


def test_real_article_tree_with_umap(articles: list[Article]) -> None:
    from raptree.profiles import make_builder

    model_or_skip("embedder")
    article = articles[0]
    tree = make_builder("onnx").build(article.id, article.title, article.text)
    assert tree.meta["reducer"] == UmapReducer.name
    assert tree.depth >= 2
    covered = sorted({x for i in tree.layers[-1] for x in tree.nodes[i].leaves})
    assert covered == tree.layers[0]
    paths = model_or_skip("embedder")
    emb = OnnxSentenceEmbedder(paths.weights, paths.tokenizer)
    counter = WordPieceCounter(paths.tokenizer)
    q = article.questions[0]
    query = query_text(q.question, q.options)
    qvec = emb.embed([query])[0]
    for method in METHODS:
        found = Retriever(tree, 2000).run(method, query, qvec)
        assert counter.count(found.text(tree)) <= 2000
        assert found.tokens > 1500  # articles are longer than the budget, so it is nearly full
