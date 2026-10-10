from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binomtest

from raptree.stats import clustered_bootstrap, paired_permutation_test


def test_bootstrap_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random(400) < 0.4
    clusters = [f"a{i // 20}" for i in range(400)]
    ci = clustered_bootstrap(correct, clusters, seed=1)
    assert ci.low <= ci.estimate <= ci.high
    assert ci.estimate == pytest.approx(correct.mean())
    assert ci.n == 400
    assert clustered_bootstrap(correct, clusters, seed=1) == ci


def test_clustering_widens_the_interval_when_articles_differ() -> None:
    # Whole articles are either right or wrong: the effective sample is 20 articles, not 400.
    correct = np.repeat(np.arange(20) % 2 == 0, 20)
    by_article = clustered_bootstrap(correct, [str(i // 20) for i in range(400)], seed=0)
    by_question = clustered_bootstrap(correct, [str(i) for i in range(400)], seed=0)
    assert (by_article.high - by_article.low) > 2 * (by_question.high - by_question.low)


def test_known_small_case() -> None:
    a = [True] * 5 + [False] * 5
    b = [False] * 10
    t = paired_permutation_test(a, b)
    assert (t.a_only, t.b_only) == (5, 0)
    assert t.difference == pytest.approx(0.5)
    assert t.p_value == pytest.approx(2 / 32)


def test_identical_methods_have_p_one() -> None:
    t = paired_permutation_test([True, False, True], [True, False, True])
    assert t.p_value == 1.0
    assert t.difference == 0.0


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 300), st.integers(0, 300))
def test_permutation_p_value_equals_the_exact_sign_test(a_only: int, b_only: int) -> None:
    # scipy's exact binomial test is the parity reference for the sign-flip distribution.
    a = [True] * a_only + [False] * b_only + [True] * 7
    b = [False] * a_only + [True] * b_only + [True] * 7
    t = paired_permutation_test(a, b)
    if a_only + b_only == 0:
        assert t.p_value == 1.0
    else:
        ref = binomtest(a_only, a_only + b_only, 0.5).pvalue
        assert t.p_value == pytest.approx(ref, rel=1e-9, abs=1e-12)
    assert paired_permutation_test(b, a).p_value == pytest.approx(t.p_value)


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="no observations"):
        clustered_bootstrap([], [])
    with pytest.raises(ValueError, match="no observations"):
        paired_permutation_test([], [])
