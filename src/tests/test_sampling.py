import numpy as np
import pytest
import pandas as pd

from sampling import (
    sample_empirical_ecdf,
    sample_outcome,
    sample_poisson,
    sample_poisson_weekday_only,
)


# ECDF sampler tests

def test_empirical_ecdf_returns_value_from_samples():
    samples = pd.Series([5, 10, 10, 20, 30])
    rng = np.random.default_rng(123)

    draw = sample_empirical_ecdf(samples, rng=rng)

    # draw should be one of the observed values (after sorting)
    assert draw in samples.values


def test_empirical_ecdf_handles_nans():
    samples = pd.Series([np.nan, 5, 10, np.nan, 20])
    rng = np.random.default_rng(1)

    draws = [sample_empirical_ecdf(samples, rng=rng) for _ in range(50)]

    # should never return NaN
    assert not np.isnan(draws).any()
    # should only return values that exist in non-NaN set
    assert set(np.unique(draws)).issubset({5, 10, 20})
    assert all(d in [5, 10, 20] for d in draws)


def test_empirical_ecdf_raises_on_all_nan():
    samples = pd.Series([np.nan, np.nan])
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        sample_empirical_ecdf(samples, rng=rng)

def test_empirical_ecdf_returns_int():
    samples = pd.Series([1, 2, 3, 4, 5])
    rng = np.random.default_rng(42)

    draw = sample_empirical_ecdf(samples, rng=rng)

    assert isinstance(draw, int)

def test_empirical_ecdf_reproducible_with_same_seed():
    samples = pd.Series([3, 7, 9, 12, 20])

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    d1 = [sample_empirical_ecdf(samples, rng=rng1) for _ in range(25)]
    d2 = [sample_empirical_ecdf(samples, rng=rng2) for _ in range(25)]

    assert d1 == d2


def test_empirical_ecdf_can_use_fixed_u():
    samples = pd.Series([10, 20, 30, 40, 50])
    rng = np.random.default_rng(123)

    d1 = sample_empirical_ecdf(samples, rng=rng, u=0.5)
    d2 = sample_empirical_ecdf(samples, rng=rng, u=0.5)

    assert d1 == d2


def test_empirical_ecdf_multiple_draws_are_correct_length():
    samples = pd.Series([1, 2, 3, 4, 5])
    rng = np.random.default_rng(42)

    draws = [sample_empirical_ecdf(samples, rng=rng) for _ in range(100)]

    assert len(draws) == 100

def test_empirical_ecdf_fixed_u_hits_expected_value():
    samples = pd.Series([10, 20, 30, 40])
    rng = np.random.default_rng(1)

    draw = sample_empirical_ecdf(samples, rng=rng, u=0.0)

    assert draw == 10



# Outcome sampler tests

def test_sample_outcome_returns_allowed_outcomes():
    probs = {"biopsy": 0.4, "discharge": 0.6}
    rng = np.random.default_rng(7)

    draws = sample_outcome(probs, n=100, rng=rng)

    assert isinstance(draws, list)
    assert len(draws) == 100
    assert set(draws).issubset(set(probs.keys()))


def test_sample_outcome_reproducible_with_same_seed():
    probs = {"A": 0.2, "B": 0.8}

    rng1 = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)

    d1 = sample_outcome(probs, n=50, rng=rng1)
    d2 = sample_outcome(probs, n=50, rng=rng2)

    assert d1 == d2


def test_sample_outcome_normalises_if_not_sum_to_one():
    # sums to 10 not 1; should normalise if allow_unnormalised=True
    probs = {"A": 2.0, "B": 8.0}
    rng = np.random.default_rng(3)

    draw = sample_outcome(probs, n=1, rng=rng, allow_unnormalised=True)
    assert draw in probs


def test_sample_outcome_raises_if_not_sum_to_one_and_disallowed():
    probs = {"A": 2.0, "B": 8.0}
    rng = np.random.default_rng(3)

    with pytest.raises(ValueError):
        sample_outcome(probs, n=1, rng=rng, allow_unnormalised=False)


def test_sample_outcome_raises_on_empty_dict():
    with pytest.raises(ValueError):
        sample_outcome({}, n=1)


def test_sample_outcome_raises_on_negative_or_all_zero_weights():
    with pytest.raises(ValueError):
        sample_outcome({"A": -0.1, "B": 1.1}, n=1)

    with pytest.raises(ValueError):
        sample_outcome({"A": 0.0, "B": 0.0}, n=1)


# Poisson sampler tests

def test_sample_poisson_raises_on_negative_lambda():
    with pytest.raises(ValueError):
        sample_poisson(-1.0, n=1)


def test_sample_poisson_returns_nonnegative_ints():
    rng = np.random.default_rng(123)
    draws = sample_poisson(0.5, n=200, rng=rng)

    assert isinstance(draws, np.ndarray)
    assert draws.dtype.kind in {"i", "u"}  # integer / unsigned integer
    assert (draws >= 0).all()


def test_sample_poisson_reproducible_with_same_seed():
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    d1 = sample_poisson(1.2, n=50, rng=rng1)
    d2 = sample_poisson(1.2, n=50, rng=rng2)

    assert np.array_equal(d1, d2)


def test_sample_poisson_weekday_only_weekends_zero():
    rng = np.random.default_rng(1)

    # Saturday=5, Sunday=6
    assert sample_poisson_weekday_only(1.5, weekday=5, rng=rng) == 0
    assert sample_poisson_weekday_only(1.5, weekday=6, rng=rng) == 0


def test_sample_poisson_weekday_only_weekday_matches_base_poisson():
    # Use identical RNG seeds so the "weekday-only" draw should match the base Poisson draw
    lam = 2.0

    rng1 = np.random.default_rng(77)
    rng2 = np.random.default_rng(77)

    d_weekday_only = sample_poisson_weekday_only(lam, weekday=2, rng=rng1)  # Wednesday
    d_base = sample_poisson(lam, n=1, rng=rng2)

    assert d_weekday_only == d_base

def test_sample_outcome_matches_expected_probabilities_approximately():
    probs = {0: 0.2, 1: 0.5, 2: 0.3}
    rng = np.random.default_rng(123)

    draws = sample_outcome(probs, n=20_000, rng=rng)

    observed = {k: draws.count(k) / len(draws) for k in probs}

    assert observed[0] == pytest.approx(0.2, abs=0.02)
    assert observed[1] == pytest.approx(0.5, abs=0.02)
    assert observed[2] == pytest.approx(0.3, abs=0.02)
