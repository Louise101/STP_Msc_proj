from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


Number = int | float


def get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return a supplied RNG or create a fresh one.

    Keeping RNG access behind one helper makes unit testing and reproducibility
    easier across the project.
    """
    return rng if rng is not None else np.random.default_rng()


def empirical_percentile(value: float, samples: pd.Series) -> float:
    """Return the empirical percentile of ``value`` within ``samples``.

    The return value is clipped away from exactly 0 and 1 so it can safely be
    passed through inverse-normal transforms in copula-based helpers.
    """
    x = np.sort(samples.dropna().to_numpy())
    if len(x) == 0:
        raise ValueError("Empty empirical sample array")

    rank = np.searchsorted(x, value, side="right")
    u = rank / len(x)
    eps = 1e-10
    return min(max(u, eps), 1.0 - eps)


def sample_empirical_ecdf(
    samples: pd.Series,
    rng: np.random.Generator,
    u: float | None = None,
) -> int:

    x = np.sort(samples.dropna().to_numpy())
    if len(x) == 0:
        raise ValueError("Empty empirical sample array")

    if u is None:
        u = rng.random()
    elif not (0.0 <= u < 1.0):
        raise ValueError("u must be in [0, 1)")

    n = len(x)
    k = int(np.ceil(u * n)) - 1
    k = max(0, min(k, n - 1))
    return int(x[k])


def sample_outcome(
    probs: dict[int | str, float],
    rng: Optional[np.random.Generator] = None,
) -> int | str:
    """Sample one categorical outcome from a probability dictionary."""
    rng = get_rng(rng)
    if not probs:
        raise ValueError("Cannot sample from an empty probability dictionary")

    outcomes = list(probs.keys())
    weights = np.asarray(list(probs.values()), dtype=float)
    if np.any(weights < 0) or np.all(weights == 0):
        raise ValueError("Outcome weights must be non-negative and not all zero")

    weights = weights / weights.sum()
    return rng.choice(outcomes, p=weights)


def sample_poisson_weekday_only(
    lam_per_workday: float,
    weekday: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample weekday referrals and force weekend referrals to zero."""
    rng = get_rng(rng)
    if weekday >= 5:
        return 0
    if lam_per_workday < 0:
        raise ValueError("lam_per_workday must be >= 0")
    return int(rng.poisson(lam_per_workday))


def sample_mri_to_report_correlated(
    ref_to_mri_wait: int,
    ref_to_mri_samples: pd.Series,
    mri_to_report_samples: pd.Series,
    rng: np.random.Generator,
    gaussian_corr: float,
) -> int:
    """Sample MRI->report with mild dependence on referral->MRI wait.

    This helper is optional and not used by the main combined engine, but it is
    retained as a reusable utility because it may still be useful in future
    sensitivity analyses.
    """
    if not (-0.999 < gaussian_corr < 0.999):
        raise ValueError("gaussian_corr must be between -0.999 and 0.999")

    u1 = empirical_percentile(ref_to_mri_wait, ref_to_mri_samples)
    z1 = norm.ppf(u1)

    eps = rng.standard_normal()
    z2 = gaussian_corr * z1 + np.sqrt(1.0 - gaussian_corr**2) * eps
    u2 = min(max(norm.cdf(z2), 0.0), 1.0 - 1e-12)

    return sample_empirical_ecdf(mri_to_report_samples, rng=rng, u=u2)
