from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.mixture import GaussianMixture

from raptree.gmm import (
    MixtureParams,
    bic,
    e_step,
    fit_mixture,
    initial_params,
    kmeans_labels,
    logsumexp,
    n_free_parameters,
    predict_proba,
    select_by_bic,
)


def blobs(rng: np.random.Generator, k: int, dim: int, per: int, spread: float = 8.0) -> np.ndarray:
    centres = rng.normal(scale=spread, size=(k, dim))
    return np.vstack([c + rng.normal(size=(per, dim)) for c in centres])


def ring(rng: np.random.Generator, k: int, per: int, radius: float = 12.0) -> np.ndarray:
    """k unit-variance blobs evenly spaced on a circle, far enough apart to be unambiguous."""
    angles = 2 * np.pi * np.arange(k) / k
    centres = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([c + rng.normal(size=(per, 2)) for c in centres])


def test_logsumexp_matches_naive_and_survives_large_values() -> None:
    a = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, -np.inf]])
    out = logsumexp(a, axis=1)
    assert out[0] == pytest.approx(np.log(np.exp(1) + np.exp(2) + np.exp(3)))
    assert out[1] == pytest.approx(1000.0 + np.log(2.0))


def test_logsumexp_of_all_minus_inf_is_minus_inf() -> None:
    out = logsumexp(np.full((1, 3), -np.inf), axis=1)
    assert np.isneginf(out[0])


@pytest.mark.parametrize("cov", ["full", "diag"])
def test_em_matches_sklearn_from_the_same_start(rng: np.random.Generator, cov: str) -> None:
    X = blobs(rng, 3, 4, 60, spread=3.0)
    start = initial_params(X, 3, cov, 1e-6, seed=3)
    ours = fit_mixture(X, 3, covariance_type=cov, init=start, tol=1e-6, max_iter=500)

    precisions = np.linalg.inv(start.covariances) if cov == "full" else 1.0 / start.covariances
    ref = GaussianMixture(
        3,
        covariance_type=cov,
        weights_init=start.weights,
        means_init=start.means,
        precisions_init=precisions,
        tol=1e-6,
        max_iter=500,
        reg_covar=1e-6,
    ).fit(X)

    assert ours.converged == ref.converged_
    assert ours.n_iter == ref.n_iter_
    np.testing.assert_allclose(ours.params.weights, ref.weights_, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours.params.means, ref.means_, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours.params.covariances, ref.covariances_, rtol=1e-6, atol=1e-9)
    assert ours.final_log_likelihood == pytest.approx(ref.score(X), rel=1e-9)
    assert bic(X, ours) == pytest.approx(ref.bic(X), rel=1e-9)
    np.testing.assert_allclose(predict_proba(X, ours.params), ref.predict_proba(X), atol=1e-8)


@pytest.mark.parametrize("cov", ["full", "diag"])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_parameter_count_matches_sklearn(rng: np.random.Generator, cov: str, k: int) -> None:
    X = rng.normal(size=(40, 3))
    ref = GaussianMixture(k, covariance_type=cov, random_state=0).fit(X)
    assert n_free_parameters(k, 3, cov) == ref._n_parameters()


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    planted=st.integers(1, 4),
    extra_k=st.integers(0, 3),
    extra_dims=st.integers(0, 2),
    per=st.integers(15, 40),
    cov=st.sampled_from(["full", "diag"]),
)
def test_em_log_likelihood_never_decreases(
    seed: int, planted: int, extra_k: int, extra_dims: int, per: int, cov: str
) -> None:
    # Exact EM (reg_covar=0) is what the monotonicity theorem covers. The data are blobs with
    # unit spread so that no component can collapse onto a handful of points.
    rng = np.random.default_rng(seed)
    X = ring(rng, planted, per)
    X = np.hstack([X, rng.normal(size=(X.shape[0], extra_dims))])
    k = min(planted, max(1, planted - extra_k))
    fit = fit_mixture(X, k, covariance_type=cov, seed=seed, tol=1e-12, max_iter=80, reg_covar=0.0)
    steps = np.diff(fit.log_likelihood)
    assert (steps >= -1e-10).all(), steps.min()


@pytest.mark.parametrize("cov", ["full", "diag"])
@pytest.mark.parametrize("planted", [2, 3, 5])
def test_bic_recovers_the_planted_number_of_components(cov: str, planted: int) -> None:
    X = ring(np.random.default_rng(planted), planted, 50)
    selection = select_by_bic(X, 8, covariance_type=cov, seed=0)
    assert selection.n_components == planted
    assert selection.bics[planted - 1] == min(selection.bics)


def test_candidates_stop_below_the_sample_count(rng: np.random.Generator) -> None:
    X = rng.normal(size=(6, 2))
    selection = select_by_bic(X, 50, seed=0)
    assert selection.candidates == [1, 2, 3, 4, 5]


def test_responsibilities_are_a_distribution(rng: np.random.Generator) -> None:
    X = blobs(rng, 3, 3, 30)
    fit = fit_mixture(X, 3, seed=1)
    _, log_resp = e_step(X, fit.params)
    np.testing.assert_allclose(np.exp(log_resp).sum(axis=1), 1.0, atol=1e-12)
    assert fit.params.weights.sum() == pytest.approx(1.0)


def test_kmeans_labels_cover_every_cluster(rng: np.random.Generator) -> None:
    X = ring(rng, 4, 25)
    labels = kmeans_labels(X, 4, np.random.default_rng(0))
    assert set(labels.tolist()) == {0, 1, 2, 3}
    for c in range(4):
        assert len(set(labels[c * 25 : (c + 1) * 25].tolist())) == 1


def test_singular_covariance_without_regularisation_is_reported() -> None:
    X = np.zeros((5, 2))
    params = MixtureParams(np.ones(1), np.zeros((1, 2)), np.zeros((1, 2, 2)), "full")
    with pytest.raises(ValueError, match="positive definite"):
        e_step(X, params)


def test_component_count_is_validated(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="n_components"):
        fit_mixture(rng.normal(size=(3, 2)), 4)
