import numpy as np
import pandas as pd
import pytest

from core.sampling import get_rng, sample_empirical_ecdf, sample_poisson_weekday_only


def test_get_rng_returns_supplied_generator():
    rng = np.random.default_rng(123)
    result = get_rng(rng)

    assert result is rng


def test_get_rng_creates_generator_when_none_supplied():
    result = get_rng()

    assert isinstance(result, np.random.Generator)


def test_sample_empirical_ecdf_returns_observed_value():
    rng = np.random.default_rng(123)
    samples = pd.Series([2, 4, 6, 8])

    value = sample_empirical_ecdf(samples, rng=rng)

    assert value in {2, 4, 6, 8}


def test_sample_empirical_ecdf_ignores_nans():
    rng = np.random.default_rng(123)
    samples = pd.Series([1, np.nan, 3, np.nan, 5])

    value = sample_empirical_ecdf(samples, rng=rng)

    assert value in {1, 3, 5}


def test_sample_empirical_ecdf_raises_on_empty_series():
    rng = np.random.default_rng(123)
    samples = pd.Series([], dtype=float)

    with pytest.raises(ValueError, match="Empty empirical sample array"):
        sample_empirical_ecdf(samples, rng=rng)


def test_sample_empirical_ecdf_raises_on_all_nan_series():
    rng = np.random.default_rng(123)
    samples = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError, match="Empty empirical sample array"):
        sample_empirical_ecdf(samples, rng=rng)


def test_sample_empirical_ecdf_raises_when_u_is_negative():
    rng = np.random.default_rng(123)
    samples = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="u must be in \\[0,1\\)"):
        sample_empirical_ecdf(samples, rng=rng, u=-0.1)


def test_sample_empirical_ecdf_raises_when_u_is_one():
    rng = np.random.default_rng(123)
    samples = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="u must be in \\[0,1\\)"):
        sample_empirical_ecdf(samples, rng=rng, u=1.0)


def test_sample_empirical_ecdf_with_u_zero_returns_minimum_value():
    rng = np.random.default_rng(123)
    samples = pd.Series([7, 2, 9, 4])

    value = sample_empirical_ecdf(samples, rng=rng, u=0.0)

    assert value == 2


def test_sample_empirical_ecdf_with_u_near_one_returns_maximum_value():
    rng = np.random.default_rng(123)
    samples = pd.Series([7, 2, 9, 4])

    value = sample_empirical_ecdf(samples, rng=rng, u=np.nextafter(1.0, 0.0))

    assert value == 9


def test_sample_empirical_ecdf_reproducible_with_fixed_seed():
    samples = pd.Series([1, 2, 3, 4, 5])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    draws1 = [sample_empirical_ecdf(samples, rng1) for _ in range(10)]
    draws2 = [sample_empirical_ecdf(samples, rng2) for _ in range(10)]

    assert draws1 == draws2


def test_sample_empirical_ecdf_maps_known_u_values_correctly():
    rng = np.random.default_rng(123)
    samples = pd.Series([10, 20, 30, 40])

    assert sample_empirical_ecdf(samples, rng=rng, u=0.00) == 10
    assert sample_empirical_ecdf(samples, rng=rng, u=0.24) == 10
    assert sample_empirical_ecdf(samples, rng=rng, u=0.25) == 10
    assert sample_empirical_ecdf(samples, rng=rng, u=0.26) == 20
    assert sample_empirical_ecdf(samples, rng=rng, u=0.50) == 20
    assert sample_empirical_ecdf(samples, rng=rng, u=0.51) == 30
    assert sample_empirical_ecdf(samples, rng=rng, u=0.75) == 30
    assert sample_empirical_ecdf(samples, rng=rng, u=0.76) == 40


def test_sample_poisson_weekday_only_returns_zero_on_saturday():
    rng = np.random.default_rng(123)

    value = sample_poisson_weekday_only(lam_per_workday=2.5, weekday=5, rng=rng)

    assert value == 0


def test_sample_poisson_weekday_only_returns_zero_on_sunday():
    rng = np.random.default_rng(123)

    value = sample_poisson_weekday_only(lam_per_workday=2.5, weekday=6, rng=rng)

    assert value == 0


def test_sample_poisson_weekday_only_raises_for_negative_lambda_on_weekday():
    rng = np.random.default_rng(123)

    with pytest.raises(ValueError, match="lam_per_workday must be >= 0"):
        sample_poisson_weekday_only(lam_per_workday=-1.0, weekday=2, rng=rng)


def test_sample_poisson_weekday_only_allows_zero_lambda():
    rng = np.random.default_rng(123)

    value = sample_poisson_weekday_only(lam_per_workday=0.0, weekday=2, rng=rng)

    assert value == 0


def test_sample_poisson_weekday_only_returns_non_negative_int_on_weekday():
    rng = np.random.default_rng(123)

    value = sample_poisson_weekday_only(lam_per_workday=2.5, weekday=2, rng=rng)

    assert isinstance(value, int)
    assert value >= 0


def test_sample_poisson_weekday_only_reproducible_with_fixed_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    draws1 = [sample_poisson_weekday_only(1.7, weekday=0, rng=rng1) for _ in range(20)]
    draws2 = [sample_poisson_weekday_only(1.7, weekday=0, rng=rng2) for _ in range(20)]

    assert draws1 == draws2