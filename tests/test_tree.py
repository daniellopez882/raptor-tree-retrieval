from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from raptree.embed import HashingEmbedder
from raptree.llm import FakeProvider
from raptree.profiles import make_builder
from raptree.reduce import PcaReducer
from raptree.samples import sample_text
from raptree.summarize import ClusterView, ExtractiveSummarizer, LlmSummarizer
from raptree.text import normalise
from raptree.tokens import WordCounter
from raptree.tree import BuildConfig, RaptorTree, TreeBuilder

if TYPE_CHECKING:
    from pathlib import Path


def check_invariants(tree: RaptorTree) -> None:
    nodes = tree.nodes
    leaves = tree.layers[0]
    # Layers partition the nodes and carry their own layer number.
    assert sorted(i for layer in tree.layers for i in layer) == list(range(len(nodes)))
    for depth, layer in enumerate(tree.layers):
        assert all(nodes[i].layer == depth for i in layer)
    # Provenance: a node covers exactly the union of its children's leaves.
    for node in nodes:
        if node.children:
            assert node.leaves == sorted({x for c in node.children for x in nodes[c].leaves})
            assert all(nodes[c].layer == node.layer - 1 for c in node.children)
        else:
            assert node.leaves == [node.id]
    # Every leaf is reachable from the top layer by following children.
    seen: set[int] = set()
    stack = list(tree.layers[-1])
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack.extend(nodes[i].children)
    assert set(leaves) <= seen
    # Every node below the top has a parent; the top layer covers the whole article.
    parents = tree.parents()
    top = set(tree.layers[-1])
    assert all(parents[n.id] for n in nodes if n.id not in top)
    assert sorted({x for i in tree.layers[-1] for x in nodes[i].leaves}) == leaves
    # Token counts agree with the segment table.
    for node in nodes:
        assert node.tokens == sum(tree.segments[s].tokens for s in node.segments)
    assert tree.node_embeddings.shape[0] == len(nodes)
    assert tree.segment_embeddings.shape[0] == len(tree.segments)


def check_extractive(tree: RaptorTree, article: str, budget: int) -> None:
    for node in tree.nodes:
        if node.layer == 0:
            continue
        child_segments = {s for c in node.children for s in tree.nodes[c].segments}
        assert set(node.segments) <= child_segments
        assert node.segments == sorted(node.segments)
        assert node.tokens <= budget
        for s in node.segments:
            seg = tree.segments[s]
            assert seg.origin == "article"
            assert seg.start is not None
            assert article[seg.start : seg.end] == seg.text  # verbatim, with offsets


def test_sample_tree_satisfies_the_invariants(sample_tree: RaptorTree) -> None:
    check_invariants(sample_tree)
    check_extractive(sample_tree, normalise(sample_text()), 128)
    assert sample_tree.depth >= 2
    assert len(sample_tree.layers[-1]) <= 11


VOCABULARY = [
    "storm", "ferry", "harbour", "bread", "baker", "lamp", "keeper", "island", "letter",
    "brother", "schooner", "fog", "bell", "orchard", "apple", "cider", "well", "water",
    "tree", "barrel", "council", "vote", "money", "engine",
]  # fmt: skip
words = st.sampled_from(VOCABULARY)
sentence = st.lists(words, min_size=3, max_size=18).map(lambda ws: " ".join(ws).capitalize() + ".")


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(sentence, min_size=1, max_size=120), st.integers(20, 60))
def test_random_documents_build_valid_trees(parts: list[str], chunk_tokens: int) -> None:
    text = " ".join(parts)
    config = BuildConfig(chunk_tokens=chunk_tokens, summary_tokens=40)
    tree = make_builder("offline", config).build("doc", "random", text)
    check_invariants(tree)
    check_extractive(tree, normalise(text), 40)


def test_save_and_load_round_trip(sample_tree: RaptorTree, tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    sample_tree.save(path)
    again = RaptorTree.load(path)
    assert again.nodes == sample_tree.nodes
    assert again.segments == sample_tree.segments
    assert again.layers == sample_tree.layers
    assert again.meta == sample_tree.meta
    np.testing.assert_array_equal(again.node_embeddings, sample_tree.node_embeddings)


def test_mmr_respects_budget_and_prefers_central_sentences() -> None:
    emb = HashingEmbedder()
    texts = ["the storm hit the ferry", "the storm hit the harbour", "a recipe for bread"]
    view = ClusterView(
        child_texts=texts[:2],
        child_embeddings=emb.embed(texts[:2]),
        segment_ids=[10, 11, 12],
        segment_texts=texts,
        segment_tokens=[5, 5, 4],
        segment_embeddings=emb.embed(texts),
    )
    summary = ExtractiveSummarizer(max_tokens=5).summarize(view)
    assert summary.segment_ids in ([10], [11])
    full = ExtractiveSummarizer(max_tokens=100).summarize(view)
    assert full.segment_ids == [10, 11, 12]
    assert full.text == " ".join(texts)


def test_llm_summaries_become_new_segments() -> None:
    provider = FakeProvider(lambda messages: "A keeper lives on an island. She keeps the light.")
    builder = TreeBuilder(
        HashingEmbedder(), WordCounter(), PcaReducer(), LlmSummarizer(provider), BuildConfig()
    )
    tree = builder.build("s", "sample", sample_text())
    check_invariants(tree)
    summaries = [n for n in tree.nodes if n.layer > 0]
    assert summaries
    assert len(provider.calls) == len(summaries)
    for node in summaries:
        assert node.text == "A keeper lives on an island. She keeps the light."
        assert all(tree.segments[s].origin == "summary" for s in node.segments)
        assert all(tree.segments[s].start is None for s in node.segments)
    # The provider saw the member texts and the instruction.
    system, user = provider.calls[0]
    assert system["role"] == "system"
    assert "Do not add anything" in system["content"]
    assert user["content"].split("\n\n")[0] == tree.nodes[summaries[0].children[0]].text


def test_empty_llm_summary_is_an_error() -> None:
    builder = TreeBuilder(
        HashingEmbedder(), WordCounter(), PcaReducer(), LlmSummarizer(FakeProvider(lambda m: " "))
    )
    with pytest.raises(ValueError, match="empty summary"):
        builder.build("s", "sample", sample_text())


def test_oversized_clusters_are_split_again() -> None:
    def child_totals(tree: RaptorTree) -> list[int]:
        return [
            sum(tree.nodes[c].tokens for c in n.children) for n in tree.nodes if len(n.children) > 1
        ]

    loose = make_builder("offline").build("s", "sample", sample_text())
    tight = make_builder("offline", BuildConfig(max_cluster_tokens=250)).build(
        "s", "sample", sample_text()
    )
    check_invariants(tight)
    # A cluster may only stay over the limit when it is too small to be clustered again.
    small = tight.meta["build_config"]["cluster"]["reduction_dim"] + 1
    for node in tight.nodes:
        total = sum(tight.nodes[c].tokens for c in node.children)
        assert total <= 250 or len(node.children) <= small
    assert max(child_totals(tight), default=0) <= max(child_totals(loose))


def test_empty_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="no text"):
        make_builder("offline").build("e", "empty", "   \n ")
