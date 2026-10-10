"""Uncertainty for accuracy: article-clustered bootstrap CIs and paired permutation tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int


def clustered_bootstrap(
    correct: Sequence[bool],
    clusters: Sequence[str],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile interval for accuracy, resampling whole articles with replacement.

    Questions about the same article share one retrieved document, so they are not
    independent; resampling articles keeps that correlation inside each replicate.
    """
    y = np.asarray(correct, dtype=np.float64)
    if y.size == 0:
        raise ValueError("no observations")
    labels, index = np.unique(np.asarray(clusters), return_inverse=True)
    hits = np.bincount(index, weights=y, minlength=len(labels))
    sizes = np.bincount(index, minlength=len(labels)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(n_resamples, len(labels)))
    stats = hits[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return Interval(float(y.mean()), float(low), float(high), int(y.size))


@dataclass(frozen=True)
class PairedTest:
    difference: float  # accuracy(a) - accuracy(b)
    p_value: float
    n: int
    a_only: int  # questions only a answers correctly
    b_only: int


def paired_permutation_test(a: Sequence[bool], b: Sequence[bool]) -> PairedTest:
    """Exact two-sided sign-flip permutation test on per-question differences.

    Under the null hypothesis the two methods are exchangeable on every question, so each
    per-question difference is equally likely to carry either sign. Concordant questions
    contribute zero whatever the sign, and over the m discordant ones the flipped sum is
    2*Binomial(m, 1/2) - m, so the full permutation distribution is a binomial and the
    p-value is computed exactly instead of by resampling.
    """
    x = np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)
    if x.size == 0:
        raise ValueError("no observations")
    a_only = int((x == 1).sum())
    b_only = int((x == -1).sum())
    m = a_only + b_only
    if m == 0:
        return PairedTest(0.0, 1.0, int(x.size), 0, 0)
    k = np.arange(m + 1)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m + 1)))])
    log_pmf = log_fact[m] - log_fact[k] - log_fact[m - k] - m * np.log(2.0)
    extreme = np.abs(2 * k - m) >= abs(a_only - b_only)
    p = float(np.exp(log_pmf[extreme]).sum())
    return PairedTest(float(x.mean()), min(1.0, p), int(x.size), a_only, b_only)
