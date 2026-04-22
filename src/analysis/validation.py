from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from analysis.summaries import extract_full_pathway_lengths


def compare_wait_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict:
    """Compare simulated and real wait distributions using common summary metrics."""
    ks_stat, ks_p = ks_2samp(sim_series, real_series)
    return {
        "n_sim": len(sim_series),
        "n_real": len(real_series),
        "mean_sim": float(sim_series.mean()),
        "mean_real": float(real_series.mean()),
        "median_sim": float(sim_series.median()),
        "median_real": float(real_series.median()),
        "p90_sim": float(np.percentile(sim_series, 90)),
        "p90_real": float(np.percentile(real_series, 90)),
        "mean_diff": float(sim_series.mean() - real_series.mean()),
        "median_diff": float(sim_series.median() - real_series.median()),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def compare_pathway_distributions(sim_df: pd.DataFrame, real_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compare full pathway lengths between simulation and real data."""
    comp = compare_wait_distributions(sim_df["total_days"], real_df["total_days"])
    return pd.DataFrame([{"comparison": label, "level": "full_pathway", **comp}])


def run_basic_validation(
    baseline_result: dict,
    mixed_result: dict,
    real_pre_df: pd.DataFrame,
    real_pros_df: pd.DataFrame,
) -> pd.DataFrame:
    """Small reusable validation routine for baseline and observed-mix runs."""
    sim_baseline = extract_full_pathway_lengths(baseline_result, "ALL_BASELINE")
    sim_mix = extract_full_pathway_lengths(mixed_result, "OBS_MIX")
    baseline_comp = compare_pathway_distributions(sim_baseline, real_pre_df, "ALL_BASELINE vs BASELINE_REAL")
    mix_comp = compare_pathway_distributions(sim_mix, real_pros_df, "OBS_MIX vs OBS_MIX_REAL")
    return pd.concat([baseline_comp, mix_comp], ignore_index=True)
