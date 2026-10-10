"""Dimensionality reduction applied before clustering."""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Reducer(Protocol):
    name: str

    def reduce(self, X: NDArray[np.float32], n_neighbors: int, dim: int, seed: int) -> NDArray: ...


class UmapReducer:
    """UMAP with cosine distance, as in RAPTOR. umap-learn is imported lazily (numba start-up)."""

    name = "umap"

    def reduce(self, X: NDArray[np.float32], n_neighbors: int, dim: int, seed: int) -> NDArray:
        import umap

        n = X.shape[0]
        n_neighbors = max(2, min(n_neighbors, n - 1))
        # Spectral initialisation needs dim < n - 1; tiny inputs start from a random layout.
        init = "spectral" if dim < n - 2 else "random"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=dim,
                metric="cosine",
                random_state=seed,
                init=init,
                n_jobs=1,
            )
            return np.asarray(model.fit_transform(X), dtype=np.float64)


class PcaReducer:
    """Principal components through an SVD. Used by the offline profile and the unit tests."""

    name = "pca"

    def reduce(self, X: NDArray[np.float32], n_neighbors: int, dim: int, seed: int) -> NDArray:
        del n_neighbors, seed  # PCA is deterministic and has no neighbourhood graph
        centred = np.asarray(X, dtype=np.float64) - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[: min(dim, vt.shape[0])]
        # Fix the SVD sign ambiguity so results do not depend on the LAPACK build.
        signs = np.sign(components[np.arange(len(components)), np.argmax(np.abs(components), 1)])
        components = components * signs[:, None]
        return centred @ components.T
