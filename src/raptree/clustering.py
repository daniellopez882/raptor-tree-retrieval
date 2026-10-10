"""RAPTOR's two-stage soft clustering: a global pass, then a local pass inside each cluster."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from raptree.gmm import CovarianceType, predict_proba, select_by_bic

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raptree.reduce import Reducer


@dataclass(frozen=True)
class ClusterConfig:
    reduction_dim: int = 10
    threshold: float = 0.1
    max_components: int = 50
    local_neighbors: int = 10
    covariance_type: CovarianceType = "full"
    seed: int = 224


def soft_assign(X: NDArray, config: ClusterConfig) -> list[list[int]]:
    """BIC-selected GMM; each point joins every component whose posterior exceeds the threshold.

    A point whose posteriors all fall below the threshold (possible once k > 1/threshold) is
    kept in its most likely component, so no node is ever dropped from the next layer.
    """
    selection = select_by_bic(
        X,
        config.max_components,
        covariance_type=config.covariance_type,
        seed=config.seed,
    )
    proba = predict_proba(X, selection.fit.params)
    clusters: list[list[int]] = [[] for _ in range(selection.n_components)]
    for i, row in enumerate(proba):
        hits = np.flatnonzero(row > config.threshold)
        if hits.size == 0:
            hits = np.array([int(np.argmax(row))])
        for c in hits:
            clusters[int(c)].append(i)
    return [c for c in clusters if c]


def two_stage_clusters(
    embeddings: NDArray[np.float32], reducer: Reducer, config: ClusterConfig
) -> list[list[int]]:
    """Cluster rows of ``embeddings``; returns lists of row indices (clusters may overlap)."""
    n = embeddings.shape[0]
    small = config.reduction_dim + 1
    if n <= small:
        return [list(range(n))]
    global_neighbors = int(math.sqrt(n - 1))
    reduced = reducer.reduce(embeddings, global_neighbors, config.reduction_dim, config.seed)
    out: list[list[int]] = []
    for members in soft_assign(reduced, config):
        if len(members) <= small:
            out.append(members)
            continue
        local = reducer.reduce(
            embeddings[members], config.local_neighbors, config.reduction_dim, config.seed
        )
        out.extend([members[j] for j in sub] for sub in soft_assign(local, config))
    return out
