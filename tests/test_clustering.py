from __future__ import annotations

import numpy as np
import pytest

from raptree.clustering import ClusterConfig, soft_assign, two_stage_clusters
from raptree.reduce import PcaReducer


def topic_embeddings(rng: np.random.Generator, topics: int, per: int, dim: int = 32) -> np.ndarray:
    centres = rng.normal(size=(topics, dim)) * 4.0
    X = np.vstack([c + rng.normal(size=(per, dim)) for c in centres])
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_small_inputs_form_a_single_cluster(rng: np.random.Generator) -> None:
    X = topic_embeddings(rng, 2, 5)
    assert two_stage_clusters(X, PcaReducer(), ClusterConfig()) == [list(range(10))]


def test_every_point_lands_in_at_least_one_cluster(rng: np.random.Generator) -> None:
    X = topic_embeddings(rng, 4, 15)
    clusters = two_stage_clusters(X, PcaReducer(), ClusterConfig(reduction_dim=4))
    covered = {i for c in clusters for i in c}
    assert covered == set(range(60))
    assert all(len(c) > 0 for c in clusters)


def test_separated_topics_are_not_mixed(rng: np.random.Generator) -> None:
    X = topic_embeddings(rng, 3, 20)
    clusters = two_stage_clusters(X, PcaReducer(), ClusterConfig(reduction_dim=4))
    for c in clusters:
        assert len({i // 20 for i in c}) == 1


@pytest.mark.parametrize("threshold", [0.05, 0.3])
def test_soft_assignment_keeps_points_without_a_confident_component(
    rng: np.random.Generator, threshold: float
) -> None:
    X = rng.normal(size=(40, 2))
    clusters = soft_assign(X, ClusterConfig(threshold=threshold, max_components=6))
    assert {i for c in clusters for i in c} == set(range(40))
