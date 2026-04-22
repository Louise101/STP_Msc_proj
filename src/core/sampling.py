from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


Number = int | float

#from verify_batch_against_data import  empirical_percentile



"""Return a supplied RNG or create a fresh one.
    """
def get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()



"""
    Sample an integer value from empirical samples using inverse transform sampling.
    If u is provided, it must be in [0,1) and will be used instead of drawing a new uniform.
    """
def sample_empirical_ecdf(samples: pd.Series, rng: np.random.Generator, u: float | None = None) -> int:

    x = np.sort(samples.dropna().to_numpy())
    if len(x) == 0:
        raise ValueError("Empty empirical sample array")

    if u is None:
        u = rng.random()
    else:
        if not (0.0 <= u < 1.0):
            raise ValueError("u must be in [0,1)")

    # k = smallest index such that (k+1)/n >= u  (0-indexed)
    n = len(x)
    k = int(np.ceil(u * n)) - 1
    k = max(0, min(k, n - 1))
    return int(x[k])




"""Sample weekday referrals and force weekend referrals to zero."""
def sample_poisson_weekday_only(
    lam_per_workday: float,
    weekday: int,
    rng: Optional[np.random.Generator] = None,
) -> int: 
    rng = get_rng(rng)
    if weekday >= 5:
        return 0
    if lam_per_workday < 0:
        raise ValueError("lam_per_workday must be >= 0")
    return int(rng.poisson(lam_per_workday))