from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from raptree.embed import HashingEmbedder
from raptree.retrieve import METHODS, Bm25, Retriever, pack, query_text, terms
from raptree.tokens import WordCounter

if TYPE_CHECKING:
    from raptree.tree import RaptorTree


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.integers(1, 300), min_size=1, max_size=60),
    st.integers(0, 2500),
    st.randoms(use_true_random=False),
)
def test_packing_never_exceeds_the_budget(tokens: list[int], budget: int, rnd) -> None:
    ranked = list(range(len(tokens)))
    rnd.shuffle(ranked)
    chosen = pack(ranked, tokens, budget)
    assert sum(tokens[i] for i in chosen) <= budget
    assert len(set(chosen)) == len(chosen)
    # Greedy: anything skipped did not fit at the moment it was considered.
    used = 0
    for i in ranked:
        if i in chosen:
            used += tokens[i]
        else:
            assert used + tokens[i] > budget


def test_packing_skips_a_large_node_and_keeps_going() -> None:
    assert pack([0, 1, 2], [50, 500, 40], budget=100) == [0, 2]


def test_bm25_prefers_documents_with_rare_query_terms() -> None:
    docs = ["the storm hit the ferry", "the baker sold bread", "storm storm warning at sea"]
    scores = Bm25(docs).scores("storm warning")
    assert scores.argmax() == 2
    assert scores[1] == 0.0


def test_terms_drop_stopwords_and_case() -> None:
    assert terms("The Ferry and THE storm") == ["ferry", "storm"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("budget", [0, 30, 150, 2000])
def test_every_method_stays_within_budget(
    sample_tree: RaptorTree, method: str, budget: int
) -> None:
    query = "Why did the Harbour Board keep a keeper on the island?"
    qvec = HashingEmbedder().embed([query])[0]
    found = Retriever(sample_tree, budget).run(method, query, qvec)
    assert found.tokens <= budget
    assert found.tokens == sum(sample_tree.nodes[i].tokens for i in found.node_ids)
    # The context string itself respects the budget under the tree's own counter.
    assert WordCounter().count(found.text(sample_tree)) == found.tokens


def test_truncation_is_a_prefix_of_the_leaves(sample_tree: RaptorTree) -> None:
    found = Retriever(sample_tree, 300).truncation()
    assert found.node_ids == sample_tree.layers[0][: len(found.node_ids)]
    nxt = sample_tree.nodes[sample_tree.layers[0][len(found.node_ids)]]
    assert found.tokens + nxt.tokens > 300


def test_dense_and_bm25_only_return_leaves(sample_tree: RaptorTree) -> None:
    query = "fog bell smugglers"
    qvec = HashingEmbedder().embed([query])[0]
    r = Retriever(sample_tree, 500)
    leaves = set(sample_tree.layers[0])
    assert set(r.dense(qvec).node_ids) <= leaves
    assert set(r.bm25(query).node_ids) <= leaves


def test_collapsed_ranks_all_layers_together(hand_tree: RaptorTree) -> None:
    qvec = hand_tree.node_embeddings[6]  # identical to the root's own embedding
    found = Retriever(hand_tree, 1000).collapsed(qvec)
    assert found.node_ids[0] == 6
    assert set(found.node_ids) == set(range(7))


def test_traversal_walks_down_then_fills_with_leaves(hand_tree: RaptorTree) -> None:
    qvec = HashingEmbedder().embed(["storm ferry captain Ines Ortega"])[0]
    found = Retriever(hand_tree, 1000, beam=1).traversal(qvec)
    layers = [hand_tree.nodes[i].layer for i in found.node_ids]
    assert layers == sorted(layers, reverse=True)  # path top-down, then leaves
    assert found.node_ids[0] == 6
    chosen_l1 = found.node_ids[1]
    assert hand_tree.nodes[chosen_l1].layer == 1
    # Leaves come only from under the chosen layer-1 node.
    assert set(found.node_ids[2:]) == set(hand_tree.nodes[chosen_l1].children)


def test_traversal_beam_limits_each_summary_layer(sample_tree: RaptorTree) -> None:
    qvec = HashingEmbedder().embed(["the keeper and the fog bell"])[0]
    found = Retriever(sample_tree, 5000, beam=2).traversal(qvec)
    upper = [i for i in found.node_ids if sample_tree.nodes[i].layer > 0]
    assert len(upper) <= 2 * (sample_tree.depth - 1)


def test_embedding_methods_need_a_query_vector(sample_tree: RaptorTree) -> None:
    with pytest.raises(ValueError, match="query embedding"):
        Retriever(sample_tree, 100).run("dense", "q", None)


def test_query_text_appends_the_options() -> None:
    assert query_text("Who?", ["Ann", "Bo"]) == "Who? Ann Bo"


def test_ties_keep_document_order(hand_tree: RaptorTree) -> None:
    qvec = np.zeros(hand_tree.node_embeddings.shape[1], dtype=np.float32)
    found = Retriever(hand_tree, 1000).dense(qvec)
    assert found.node_ids == [0, 1, 2, 3]
