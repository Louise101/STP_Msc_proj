from __future__ import annotations
#from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


Number = Union[int, float]


# set rng and use the same seed - code is reproducable. If not, a new generator will be used and results will change each time. 
def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


#function to sample ecdf of waiting times 

def sample_empirical_ecdf_orig(
    samples: Sequence[Number], # input data
    n: int = 1, # number of samples to draw - default is 1 
    rng: Optional[np.random.Generator] = None, #optional rng for reprodcibility 
    nonnegative: bool = True, # reject negative waiting times  
    integer: bool = True, # return intiger day counts 
) -> Union[int, np.ndarray]: # integer returned if n=1, array if n>1
    """
    Sample from an empirical distribution using inverse transform sampling on the ECDF.

    Concept:
      Given observed durations {x_i}, define the empirical CDF:
        F_hat(x) = (1/N) * sum_{i=1..N} I(x_i <= x)
      Inverse transform sampling:
        U ~ Uniform(0,1)
        X = F_hat^{-1}(U)

    Implementation detail:
      Sorting observed samples x_(1) <= ... <= x_(N),
      we can sample an index k = ceil(U * N) and return x_(k).

    Args:
        samples: Observed durations (e.g., days between events).
        n: Number of samples to draw.
        rng: Optional numpy random Generator.
        nonnegative: If True, raise if any samples are negative.
        integer: If True, return integers (rounded) for durations.

    Returns:
        A single int if n=1, otherwise a numpy array of length n.
    """
    rng = _get_rng(rng) #gets rngf

    x = np.asarray(samples, dtype=float) #converts list into array 
    x = x[~np.isnan(x)] # removes NANs 
    if x.size == 0:
        raise ValueError("sample_empirical_ecdf: no valid (non-NaN) samples provided.") #stops if no values left after removing NAN

    if nonnegative and np.any(x < 0):
        raise ValueError("sample_empirical_ecdf: negative values found in samples.") # checks for negative values 

    x_sorted = np.sort(x) # sorts values into order for sampling
    N = x_sorted.size # counts number of values 

    # Draw U ~ Uniform(0,1) and map to indices 0..N-1 via ceil(U*N)-1
    U = rng.random(n) # generates random number from uniform distribution bewteen 0 and 1
    idx = np.ceil(U * N).astype(int) - 1 # multiplies U by N to get an index for the sampled value 
    idx = np.clip(idx, 0, N - 1) # handles case U=0 and idx would equal -1 which is invalid - so it make idx=0 in this case.  clip moves values outside of boundaries to boundary edges. 

    draws = x_sorted[idx] #picks sampled value from data

    if integer:
        draws = np.rint(draws).astype(int) #rounds to nearest integer if needed - shouldnt be needed as data already integer days

    if n == 1:
        return int(draws[0]) # restuns single value if n=1 and array if n>1 
    return draws


def sample_empirical_ecdf(samples: pd.Series, rng: np.random.Generator, u: float | None = None) -> int:
    """
    Sample an integer value from empirical samples using inverse transform sampling.
    If u is provided, it must be in [0,1) and will be used instead of drawing a new uniform.
    """
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

def correlated_u(u_patient: float, rng: np.random.Generator, alpha: float) -> float:
    """
    Blend patient-level quantile with stage-specific noise.
    alpha=1 -> independent
    alpha=0 -> fully correlated (uses u_patient only)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    u_stage = rng.random()
    u = (1 - alpha) * u_patient + alpha * u_stage
    # Keep strictly < 1.0 for indexing safety
    return min(u, np.nextafter(1.0, 0.0))


#function to sample outcomes 
def sample_outcome(
    probs: Dict[str, float], #probability data calculated in pdf_create.py
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
    allow_unnormalised: bool = True, # notmalises probabilites that don't sum to exactly 1
) -> Union[str, List[str]]:
    """
    Sample categorical outcomes from a probability dictionary.

    Args:
        probs: mapping outcome -> probability weight
        n: number of outcomes to sample
        rng: optional numpy Generator
        allow_unnormalised: if True, will normalise weights to sum to 1

    Returns:
        Single outcome string if n=1 else list of outcomes.
    """
    rng = _get_rng(rng)

    if not probs:
        raise ValueError("sample_outcome: empty probability dictionary.")

    outcomes = list(probs.keys()) # names of outcomes like 'discharge', 'biopsy'. 
    weights = np.array([probs[o] for o in outcomes], dtype=float) # corresponding probability 

    if np.any(weights < 0) or np.all(weights == 0):
        raise ValueError("sample_outcome: invalid weights (negative or all zero).") # rejects invalid probabilitys

    s = weights.sum()
    if not np.isclose(s, 1.0):
        if allow_unnormalised:
            weights = weights / s
        else:
            raise ValueError(f"sample_outcome: probabilities do not sum to 1 (sum={s}).") #sums weights and either normalises or rejects if sum doesnt equal 1

    draws = rng.choice(outcomes, size=n, p=weights) # draws n samples 

    if n == 1:
        return str(draws[0])
    return [str(d) for d in draws]


# poisson sampling for refferal rate frequency - get number of referrals in a day
def sample_poisson(
    lam: float, # expected mean count per interval 
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Union[int, np.ndarray]:
    """
    Sample counts from a Poisson(lam).

    Args:
        lam: rate parameter (expected count per interval)
        n: number of samples to draw
        rng: optional numpy Generator

    Returns:
        Single int if n=1, else array of ints.
    """
    rng = _get_rng(rng)
    if lam < 0:
        raise ValueError("sample_poisson: lam must be >= 0.") # ensures valid lam
    draws = rng.poisson(lam=lam, size=n) # draws n values 
    if n == 1:
        return int(draws[0])
    return draws

# adjust poisson sampling for weekdays only - no refferals on weekends 
def sample_poisson_weekday_only(
    lam_per_workday: float,
    weekday: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample Poisson referrals for a specific day, forcing weekends to 0.

    Args:
        lam_per_workday: mean referrals per workday
        weekday: 0=Mon ... 6=Sun (Python datetime.weekday convention)
        rng: optional numpy Generator

    Returns:
        referrals count for that day
    """
    if weekday >= 5:  # Sat/Sun #if sat or sun, refferals adjusted to 0
        return 0
    return sample_poisson(lam_per_workday, n=1, rng=rng) # if not, use above to do standard poisson distribution
