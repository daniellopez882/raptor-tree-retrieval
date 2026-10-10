"""Gaussian mixtures fitted by expectation-maximisation, and BIC model selection.

Written against numpy only. The parameterisation mirrors scikit-learn's (Cholesky factors of
the precision matrices, ``reg_covar`` added to the covariance diagonal, a mean-log-likelihood
convergence test), so the test-suite can check this implementation against sklearn step for
step from the same starting point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CovarianceType = Literal["full", "diag"]
Array = NDArray[np.float64]

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class MixtureParams:
    weights: Array  # (k,)
    means: Array  # (k, d)
    covariances: Array  # (k, d, d) for "full", (k, d) for "diag"
    covariance_type: CovarianceType

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])


@dataclass(frozen=True)
class MixtureFit:
    params: MixtureParams
    # Mean per-sample log-likelihood of the starting parameters and after every M-step.
    log_likelihood: list[float]
    n_iter: int
    converged: bool

    @property
    def final_log_likelihood(self) -> float:
        return self.log_likelihood[-1]


def logsumexp(a: Array, axis: int) -> Array:
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):  # a row of -inf legitimately gives log(0) = -inf
        out = np.log(np.sum(np.exp(a - top), axis=axis, keepdims=True)) + top
    return np.squeeze(out, axis=axis)


def precision_cholesky(params: MixtureParams) -> Array:
    """Upper-triangular factors U with U Uᵀ = Σ⁻¹ ("full"), or 1/σ per dimension ("diag")."""
    if params.covariance_type == "diag":
        return 1.0 / np.sqrt(params.covariances)
    try:
        lower = np.linalg.cholesky(params.covariances)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "a covariance matrix is not positive definite; increase reg_covar"
        ) from exc
    d = params.means.shape[1]
    eye = np.broadcast_to(np.eye(d), lower.shape)
    return np.swapaxes(np.linalg.solve(lower, eye), 1, 2)


def log_component_density(X: Array, params: MixtureParams) -> Array:
    """log N(x_i | μ_k, Σ_k) for every sample i and component k, shape (n, k)."""
    d = X.shape[1]
    prec = precision_cholesky(params)
    if params.covariance_type == "diag":
        log_det = np.sum(np.log(prec), axis=1)
        sq = np.sum(((X[:, None, :] - params.means[None, :, :]) * prec[None, :, :]) ** 2, axis=2)
    else:
        log_det = np.sum(np.log(np.diagonal(prec, axis1=1, axis2=2)), axis=1)
        y = np.einsum("nd,kde->nke", X, prec) - np.einsum("kd,kde->ke", params.means, prec)[None]
        sq = np.sum(y * y, axis=2)
    return -0.5 * (d * _LOG_2PI + sq) + log_det[None, :]


def e_step(X: Array, params: MixtureParams) -> tuple[float, Array]:
    """Mean log-likelihood and log responsibilities, computed with log-sum-exp."""
    weighted = log_component_density(X, params) + np.log(params.weights)[None, :]
    norm = logsumexp(weighted, axis=1)
    return float(np.mean(norm)), weighted - norm[:, None]


def m_step(
    X: Array, resp: Array, covariance_type: CovarianceType, reg_covar: float
) -> MixtureParams:
    nk = resp.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
    means = (resp.T @ X) / nk[:, None]
    diff = X[None, :, :] - means[:, None, :]  # (k, n, d)
    if covariance_type == "diag":
        covariances = np.einsum("nk,knd->kd", resp, diff * diff) / nk[:, None] + reg_covar
    else:
        covariances = np.einsum("nk,knd,kne->kde", resp, diff, diff) / nk[:, None, None]
        d = X.shape[1]
        covariances[:, np.arange(d), np.arange(d)] += reg_covar
    weights = nk / nk.sum()
    return MixtureParams(weights, means, covariances, covariance_type)


def kmeans_labels(
    X: Array, k: int, rng: np.random.Generator, max_iter: int = 100
) -> NDArray[np.intp]:
    """k-means++ seeding followed by Lloyd iterations; used to initialise EM."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest.sum()
        pick = rng.integers(n) if total <= 0.0 else rng.choice(n, p=closest / total)
        centers[c] = X[pick]
        closest = np.minimum(closest, np.sum((X - centers[c]) ** 2, axis=1))
    labels = np.full(n, -1, dtype=np.intp)
    for _ in range(max_iter):
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = labels == c
            if members.any():
                centers[c] = X[members].mean(axis=0)
            else:
                # An empty cluster takes the point that is currently worst served.
                far = int(np.argmax(dist[np.arange(n), labels]))
                centers[c] = X[far]
                labels[far] = c
    return labels


def initial_params(
    X: Array, k: int, covariance_type: CovarianceType, reg_covar: float, seed: int
) -> MixtureParams:
    labels = kmeans_labels(X, k, np.random.default_rng(seed))
    resp = np.zeros((X.shape[0], k))
    resp[np.arange(X.shape[0]), labels] = 1.0
    return m_step(X, resp, covariance_type, reg_covar)


def fit_mixture(
    X: Array,
    n_components: int,
    *,
    covariance_type: CovarianceType = "full",
    tol: float = 1e-3,
    max_iter: int = 100,
    reg_covar: float = 1e-6,
    seed: int = 0,
    init: MixtureParams | None = None,
) -> MixtureFit:
    X = np.asarray(X, dtype=np.float64)
    if not 1 <= n_components <= X.shape[0]:
        raise ValueError(f"n_components must be in [1, {X.shape[0]}], got {n_components}")
    params = init or initial_params(X, n_components, covariance_type, reg_covar, seed)
    history: list[float] = []
    previous = -math.inf
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - n_iter is reported after the loop
        mean_ll, log_resp = e_step(X, params)
        history.append(mean_ll)
        params = m_step(X, np.exp(log_resp), covariance_type, reg_covar)
        if abs(mean_ll - previous) < tol:
            converged = True
            break
        previous = mean_ll
    history.append(e_step(X, params)[0])
    return MixtureFit(params, history, n_iter, converged)


def predict_proba(X: Array, params: MixtureParams) -> Array:
    return np.exp(e_step(np.asarray(X, dtype=np.float64), params)[1])


def n_free_parameters(n_components: int, dim: int, covariance_type: CovarianceType) -> int:
    cov = n_components * dim * (dim + 1) // 2 if covariance_type == "full" else n_components * dim
    return cov + n_components * dim + n_components - 1


def bic(X: Array, fit: MixtureFit) -> float:
    n, d = X.shape
    k = fit.params.n_components
    p = n_free_parameters(k, d, fit.params.covariance_type)
    return -2.0 * fit.final_log_likelihood * n + p * math.log(n)


@dataclass(frozen=True)
class BicSelection:
    n_components: int
    candidates: list[int]
    bics: list[float]
    fit: MixtureFit


def select_by_bic(
    X: Array,
    max_components: int,
    *,
    covariance_type: CovarianceType = "full",
    seed: int = 0,
    reg_covar: float = 1e-6,
) -> BicSelection:
    """Fit k = 1..min(max_components, n-1) and keep the fit with the lowest BIC."""
    X = np.asarray(X, dtype=np.float64)
    top = max(1, min(max_components, X.shape[0] - 1))
    candidates = list(range(1, top + 1))
    fits = [
        fit_mixture(X, k, covariance_type=covariance_type, seed=seed, reg_covar=reg_covar)
        for k in candidates
    ]
    scores = [bic(X, fit) for fit in fits]
    best = int(np.argmin(scores))
    return BicSelection(candidates[best], candidates, scores, fits[best])
