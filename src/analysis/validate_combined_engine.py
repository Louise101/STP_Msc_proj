from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from engine.combined_engine import run_day_loop_combined_engine
from data_prep.empirical_inputs import load_real_stage_data

from analysis.validation import run_basic_validation, build_real_pathway_csvs, load_real_pathway_data
from analysis.summaries import flatten_wait_values, summarise_stage_activity, summarise_stage_weekly_arrivals, summarise_flow_counts, summarise_mri_resource, summarise_pathway_stats, summarise_mixed_pathway_type
from analysis.metrics import safe_pct_change, extract_stage_waits, extract_pathway_lengths
from analysis.plots import make_stage_pressure_plot, make_weekly_arrivals_plot, save_pathway_distribution_plot, save_obs_mix_pathway_split_plot

from engine.scenarios import build_combined_config, generate_daily_referrals


# ======================================================================================
# CONFIGURATION
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "refac_combined_stage_pressure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level settings for the multi-scenario analysis run."""

    start_date: date = date(2026, 1, 5)
    n_days: int = 365
    lam_per_workday: float = 1.7528735632183907
    p_prostad_obs: float = 0.5098039215686274
    seeds: tuple[int, ...] = tuple(range(1, 21))


CFG = AnalysisConfig()

# Named scenarios to run. This is intentionally a very small, central registry for this
# analysis script. The core scenario definitions should still live in config/scenarios.py.
SCENARIOS: dict[str, dict[str, Any]] = {
    "ALL_BASELINE": {"p_prostad": 0.0},
    "OBS_MIX": {"p_prostad": CFG.p_prostad_obs},
    "ALL_PROSTAD": {"p_prostad": 1.0},
}





FLOW_EVENTS = [
    "referral_received",
    "mri_performed",
    "mri_report_ready",
    "MDT_occured",
    "biopsy_done",
    "Path_report_recieved",
    "Path_report_outcome",
    "Treatment_options_MDT_occured",
    "Outpatient_appointment_occured",
]

FAST_FLOW_ELIGIBLE_STAGES = {"ref_to_mri", "mri_to_report", "report_to_biopmdt"}





# ======================================================================================
# SCENARIO EXECUTION
# ======================================================================================

def build_runtime_config(scenario_name: str, seed: int):
    return build_combined_config(
        scenario_name=scenario_name,
        start_date=CFG.start_date,
        n_days=CFG.n_days,
        seed=seed,
        lam_per_workday=CFG.lam_per_workday,
        overrides={"p_prostad": SCENARIOS[scenario_name]["p_prostad"]},
    )

def run_seed(seed: int) -> dict[str, dict[str, Any]]:
    """Run all scenarios for one seed using the same referral stream."""
    referral_schedule = generate_daily_referrals(
        start_date=CFG.start_date,
        n_days=CFG.n_days,
        lam_per_workday=CFG.lam_per_workday,
        seed=seed,
    )

    outputs: dict[str, dict[str, Any]] = {}
    for scenario_name in SCENARIOS:
        cfg = build_runtime_config(scenario_name, seed)
        outputs[scenario_name] = run_day_loop_combined_engine(
            cfg,
            daily_referrals_override=referral_schedule,
        )
    return outputs



# ======================================================================================
# AGGREGATION AND COMPARISON
# ======================================================================================

def aggregate_stage_wait_summary(stage_wait_df: pd.DataFrame) -> pd.DataFrame:
    """Average stage wait metrics across seeds and add percent-change columns."""
    agg = (
        stage_wait_df.groupby(["scenario", "stage"])
        .agg(
            mean_wait_days_mean=("wait_days", "mean"),
            mean_wait_days_std=("wait_days", "std"),
            median_wait_days_mean=("wait_days", "median"),
            p90_wait_days_mean=("wait_days", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )

    wide = agg.pivot(index="stage", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    wide["mix_wait_pct_change"] = wide.apply(
        lambda r: safe_pct_change(r.get("mean_wait_days_mean_OBS_MIX"), r.get("mean_wait_days_mean_ALL_BASELINE")),
        axis=1,
    )
    wide["allpros_wait_pct_change"] = wide.apply(
        lambda r: safe_pct_change(r.get("mean_wait_days_mean_ALL_PROSTAD"), r.get("mean_wait_days_mean_ALL_BASELINE")),
        axis=1,
    )
    return wide


def aggregate_stage_summary(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Average stage pressure metrics across seeds and add baseline-relative changes."""
    agg = (
        stage_df.groupby(["scenario", "stage"])
        .agg(
            total_arrivals_mean=("total_arrivals", "mean"),
            total_arrivals_std=("total_arrivals", "std"),
            mean_daily_arrivals_mean=("mean_daily_arrivals", "mean"),
            peak_daily_arrivals_mean=("peak_daily_arrivals", "mean"),
            mean_in_stage_mean=("mean_in_stage", "mean"),
            mean_in_stage_std=("mean_in_stage", "std"),
            peak_in_stage_mean=("peak_in_stage", "mean"),
            completion_ratio_mean=("completion_ratio", "mean"),
            completion_ratio_std=("completion_ratio", "std"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="stage", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    for prefix, scenario in (("mix", "OBS_MIX"), ("allpros", "ALL_PROSTAD")):
        wide[f"{prefix}_arrival_pct_change"] = wide.apply(
            lambda r: safe_pct_change(
                r.get(f"mean_daily_arrivals_mean_{scenario}"),
                r.get("mean_daily_arrivals_mean_ALL_BASELINE"),
            ),
            axis=1,
        )
        wide[f"{prefix}_peak_pct_change"] = wide.apply(
            lambda r: safe_pct_change(
                r.get(f"peak_daily_arrivals_mean_{scenario}"),
                r.get("peak_daily_arrivals_mean_ALL_BASELINE"),
            ),
            axis=1,
        )
        wide[f"{prefix}_in_stage_pct_change"] = wide.apply(
            lambda r: safe_pct_change(
                r.get(f"mean_in_stage_mean_{scenario}"),
                r.get("mean_in_stage_mean_ALL_BASELINE"),
            ),
            axis=1,
        )
        wide[f"{prefix}_completion_change"] = (
            wide.get(f"completion_ratio_mean_{scenario}") - wide.get("completion_ratio_mean_ALL_BASELINE")
        )

    return wide


def aggregate_weekly_stage_arrivals(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Average weekly arrivals across seeds for each stage and scenario."""
    return (
        weekly_df.groupby(["scenario", "stage", "week_start"])
        .agg(
            weekly_arrivals_mean=("weekly_arrivals", "mean"),
            weekly_arrivals_std=("weekly_arrivals", "std"),
        )
        .reset_index()
    )


def aggregate_flow_summary(flow_df: pd.DataFrame) -> pd.DataFrame:
    """Average milestone event counts across seeds and compare to baseline."""
    agg = (
        flow_df.groupby(["scenario", "event"])
        .agg(count_mean=("count", "mean"), count_std=("count", "std"))
        .reset_index()
    )
    wide = agg.pivot(index="event", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()
    wide["obs_mix_minus_baseline"] = wide.get("count_mean_OBS_MIX") - wide.get("count_mean_ALL_BASELINE")
    wide["allpros_minus_baseline"] = wide.get("count_mean_ALL_PROSTAD") - wide.get("count_mean_ALL_BASELINE")
    return wide


def aggregate_mri_resource_summary(resource_df: pd.DataFrame) -> pd.DataFrame:
    """Average MRI_PROSTAD resource metrics across seeds."""
    if resource_df.empty:
        return pd.DataFrame()
    return (
        resource_df.groupby(["scenario", "resource"])
        .agg(
            mean_queue_len_mean=("mean_queue_len", "mean"),
            mean_queue_len_std=("mean_queue_len", "std"),
            peak_queue_len_mean=("peak_queue_len", "mean"),
            peak_queue_len_std=("peak_queue_len", "std"),
            mean_wait_mean=("mean_wait", "mean"),
            mean_wait_std=("mean_wait", "std"),
            peak_wait_mean=("peak_wait", "mean"),
            peak_wait_std=("peak_wait", "std"),
            n_wait_observations_mean=("n_wait_observations", "mean"),
        )
        .reset_index()
    )




def compare_pathway_stats(pathway_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each scenario with the all-baseline reference."""
    baseline = pathway_summary[pathway_summary["scenario"] == "ALL_BASELINE"].iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in pathway_summary.iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "mean_days": row["mean_days"],
                "median_days": row["median_days"],
                "p90_days": row["p90_days"],
                "pct_within_62": row["pct_within_62"],
                "delta_mean_vs_baseline": row["mean_days"] - baseline["mean_days"],
                "delta_median_vs_baseline": row["median_days"] - baseline["median_days"],
                "delta_pct_within_62": row["pct_within_62"] - baseline["pct_within_62"],
            }
        )
    return pd.DataFrame(rows)




# ======================================================================================
# STAGE VALIDATION AGAINST REAL DATA
# ======================================================================================




def compare_stage_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict[str, float]:
    """Compare one simulated stage wait distribution to its real counterpart."""
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
        "ks_p": float(ks_p),
    }


def run_stage_validation(all_stage_waits: pd.DataFrame) -> pd.DataFrame:
    """Validate baseline and observed-mix stage waits against real stage-level data."""
    real_stage_pre = load_real_stage_data("pre", DATA_DIR)
    real_stage_pros = load_real_stage_data("pros", DATA_DIR)
    rows: list[dict[str, Any]] = []

    for stage in sorted(all_stage_waits["stage"].dropna().unique()):
        sim_baseline = all_stage_waits[
            (all_stage_waits["scenario"] == "ALL_BASELINE") & (all_stage_waits["stage"] == stage)
        ]["wait_days"]
        sim_mix = all_stage_waits[
            (all_stage_waits["scenario"] == "OBS_MIX") & (all_stage_waits["stage"] == stage)
        ]["wait_days"]

        real_baseline = real_stage_pre.get(stage)
        real_mix = real_stage_pros.get(stage)

        if real_baseline is not None and len(real_baseline) > 0 and len(sim_baseline) > 0:
            comparison = compare_stage_distributions(sim_baseline, real_baseline)
            comparison["stage"] = stage
            comparison["dataset"] = "ALL_BASELINE vs PRE"
            rows.append(comparison)

        if real_mix is not None and len(real_mix) > 0 and len(sim_mix) > 0:
            comparison = compare_stage_distributions(sim_mix, real_mix)
            comparison["stage"] = stage
            comparison["dataset"] = "OBS_MIX vs PROSTAD_PERIOD"
            rows.append(comparison)

    return pd.DataFrame(rows)


# ======================================================================================
# STAGE FLAGGING
# ======================================================================================

def classify_stage_from_changes(
    arrival_pct: float,
    peak_pct: float,
    in_stage_pct: float,
    completion_change: float,
    mean_in_stage_value: float,
    wait_pct_change: float | None = None,
    stage_name: str | None = None,
) -> str:
    """Apply simple rule-based labels to stage pressure changes.

    These are descriptive flags for interpretation, not hard statistical tests.
    """
    if (
        pd.notna(arrival_pct)
        and arrival_pct > 10
        and ((pd.notna(in_stage_pct) and in_stage_pct > 10) or (pd.notna(wait_pct_change) and wait_pct_change > 10))
        and (pd.isna(completion_change) or completion_change <= 0)
    ):
        return "ð´ EMERGING BOTTLENECK"

    if (
        stage_name in FAST_FLOW_ELIGIBLE_STAGES
        and pd.notna(arrival_pct)
        and arrival_pct > 10
        and pd.notna(wait_pct_change)
        and wait_pct_change < -10
        and (pd.isna(completion_change) or completion_change >= 0)
    ):
        return "ð FASTER FLOW"

    if pd.notna(arrival_pct) and arrival_pct > 5 and not (pd.notna(wait_pct_change) and wait_pct_change < -10):
        return "ð  PRESSURE â"

    if (pd.notna(wait_pct_change) and wait_pct_change < -10) or (pd.notna(in_stage_pct) and in_stage_pct < -10):
        return "ð¢ IMPROVED"

    if pd.notna(mean_in_stage_value) and mean_in_stage_value < 0.5:
        return "â¡ FAST TRACK"

    return "âª STABLE"


def add_stage_flags(stage_summary: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive stage flags for observed mix and all-PROSTAD comparisons."""
    out = stage_summary.copy()
    out["stage_flag_obs_mix"] = out.apply(
        lambda row: classify_stage_from_changes(
            row.get("mix_arrival_pct_change"),
            row.get("mix_peak_pct_change"),
            row.get("mix_in_stage_pct_change"),
            row.get("mix_completion_change"),
            row.get("mean_in_stage_mean_OBS_MIX"),
            wait_pct_change=row.get("mix_wait_pct_change"),
            stage_name=row.get("stage"),
        ),
        axis=1,
    )
    out["stage_flag_all_prostad"] = out.apply(
        lambda row: classify_stage_from_changes(
            row.get("allpros_arrival_pct_change"),
            row.get("allpros_peak_pct_change"),
            row.get("allpros_in_stage_pct_change"),
            row.get("allpros_completion_change"),
            row.get("mean_in_stage_mean_ALL_PROSTAD"),
            wait_pct_change=row.get("allpros_wait_pct_change"),
            stage_name=row.get("stage"),
        ),
        axis=1,
    )
    return out




# ======================================================================================
# MAIN ANALYSIS PIPELINE
# ======================================================================================

def main() -> None:
    """Run all scenarios across all seeds and save scenario comparisons plus real-data validation."""
    all_stage_activity: list[pd.DataFrame] = []
    all_weekly_arrivals: list[pd.DataFrame] = []
    all_flow_counts: list[pd.DataFrame] = []
    all_resource_rows: list[pd.DataFrame] = []
    all_pathway_rows: list[pd.DataFrame] = []
    all_obs_mix_pathway_rows: list[pd.DataFrame] = []
    all_stage_wait_rows: list[pd.DataFrame] = []
    validation_seed_results: dict[str, dict[str, Any]] = {}

    print(f"Running seeds: {list(CFG.seeds)}")

    for seed in CFG.seeds:
        print(f"Running seed {seed}...")
        seed_outputs = run_seed(seed)

        for scenario_name, result in seed_outputs.items():
            all_stage_activity.append(summarise_stage_activity(result, scenario_name, seed))
            all_weekly_arrivals.append(summarise_stage_weekly_arrivals(result, scenario_name, seed))
            all_flow_counts.append(summarise_flow_counts(result, scenario_name, seed))
            all_resource_rows.append(summarise_mri_resource(result, scenario_name, seed))
            all_pathway_rows.append(extract_pathway_lengths(result, scenario_name, seed))
            all_stage_wait_rows.append(extract_stage_waits(result, scenario_name, seed))

        all_obs_mix_pathway_rows.append(extract_pathway_lengths(seed_outputs["OBS_MIX"], "OBS_MIX", seed))

        if seed == CFG.seeds[0]:
            validation_seed_results = seed_outputs




    stage_df = pd.concat(all_stage_activity, ignore_index=True)
    weekly_df = pd.concat(all_weekly_arrivals, ignore_index=True)
    flow_df = pd.concat(all_flow_counts, ignore_index=True)
    resource_df = pd.concat(all_resource_rows, ignore_index=True)
    pathway_df = pd.concat(all_pathway_rows, ignore_index=True)
    obs_mix_pathway_df = pd.concat(all_obs_mix_pathway_rows, ignore_index=True)
    stage_wait_df = pd.concat(all_stage_wait_rows, ignore_index=True)


    # Save raw per-run outputs first for traceability.
    stage_df.to_csv(OUTPUT_DIR / "stage_activity_all_runs.csv", index=False)
    weekly_df.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_all_runs.csv", index=False)
    flow_df.to_csv(OUTPUT_DIR / "flow_counts_all_runs.csv", index=False)
    resource_df.to_csv(OUTPUT_DIR / "mri_resource_all_runs.csv", index=False)
    pathway_df.to_csv(OUTPUT_DIR / "pathway_lengths_all_runs.csv", index=False)
    obs_mix_pathway_df.to_csv(OUTPUT_DIR / "obs_mix_pathway_lengths_by_type_all_runs.csv", index=False)
    stage_wait_df.to_csv(OUTPUT_DIR / "stage_waits_all_runs.csv", index=False)

    # Build seed-averaged summaries.
    stage_wait_summary = aggregate_stage_wait_summary(stage_wait_df)
    stage_summary = aggregate_stage_summary(stage_df).merge(stage_wait_summary, on="stage", how="left")
    stage_summary = add_stage_flags(stage_summary)

    weekly_summary = aggregate_weekly_stage_arrivals(weekly_df)
    flow_summary = aggregate_flow_summary(flow_df)
    mri_resource_summary = aggregate_mri_resource_summary(resource_df)
    pathway_summary = summarise_pathway_stats(pathway_df)
    pathway_comparison = compare_pathway_stats(pathway_summary)
    obs_mix_pathway_summary = summarise_mixed_pathway_type(obs_mix_pathway_df)

    # Save summaries.
    stage_summary.to_csv(OUTPUT_DIR / "stage_pressure_summary.csv", index=False)
    stage_wait_summary.to_csv(OUTPUT_DIR / "stage_wait_summary.csv", index=False)
    weekly_summary.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_summary.csv", index=False)
    flow_summary.to_csv(OUTPUT_DIR / "flow_summary.csv", index=False)
    mri_resource_summary.to_csv(OUTPUT_DIR / "mri_resource_summary.csv", index=False)
    pathway_summary.to_csv(OUTPUT_DIR / "pathway_length_summary.csv", index=False)
    pathway_comparison.to_csv(OUTPUT_DIR / "pathway_length_comparison.csv", index=False)
    obs_mix_pathway_summary.to_csv(OUTPUT_DIR / "obs_mix_pathway_length_by_type_summary.csv", index=False)



    

    # Print the most useful interpretation tables.
    summary_cols = [
        "stage",
        "mean_in_stage_mean_ALL_BASELINE",
        "mean_in_stage_mean_OBS_MIX",
        "mean_in_stage_mean_ALL_PROSTAD",
        "mix_arrival_pct_change",
        "mix_peak_pct_change",
        "mix_in_stage_pct_change",
        "mix_completion_change",
        "mix_wait_pct_change",
        "stage_flag_obs_mix",
        "allpros_arrival_pct_change",
        "allpros_peak_pct_change",
        "allpros_in_stage_pct_change",
        "allpros_completion_change",
        "allpros_wait_pct_change",
        "stage_flag_all_prostad",
    ]

    print("\n=== STAGE PRESSURE SUMMARY ===")
    print(stage_summary[summary_cols].round(3).to_string(index=False))

    print("\n=== FLOW SUMMARY ===")
    print(flow_summary.round(3).to_string(index=False))

    print("\n=== MRI RESOURCE SUMMARY ===")
    print(mri_resource_summary.round(3).to_string(index=False))

    print("\n=== PATHWAY LENGTH SUMMARY (FULL PATHWAY ONLY) ===")
    print(pathway_summary.round(2).to_string(index=False))

    print("\n=== PATHWAY LENGTH COMPARISON (VS ALL_BASELINE) ===")
    print(pathway_comparison.round(2).to_string(index=False))

    print("\n=== OBS_MIX FULL-PATHWAY LENGTHS BY PATHWAY TYPE ===")
    print(obs_mix_pathway_summary.round(2).to_string(index=False))

    # Save plots.
    make_stage_pressure_plot(stage_summary)
    for stage_name in weekly_summary["stage"].dropna().unique():
        make_weekly_arrivals_plot(weekly_summary, stage_name)
    save_pathway_distribution_plot(pathway_df, "pathway_time_distributions.png")
    save_obs_mix_pathway_split_plot(obs_mix_pathway_df, "obs_mix_pathway_type_split.png")

    # Pathway-level validation against real data.
    build_real_pathway_csvs(
        pre_ref_file=str(DATA_DIR / "pre_ref_to_mri.csv"),
        pre_outpat_file=str(DATA_DIR / "pre_treatmdt_to_outpat.csv"),
        pros_ref_file=str(DATA_DIR / "pros_ref_to_mri.csv"),
        pros_outpat_file=str(DATA_DIR / "pros_treatmdt_to_outpat.csv"),
        out_pre_file=str(DATA_DIR / "pre_pathway.csv"),
        out_pros_file=str(DATA_DIR / "pros_pathway.csv"),
    )

    real_pre, real_pros = load_real_pathway_data(
        str(DATA_DIR / "pre_pathway.csv"),
        str(DATA_DIR / "pros_pathway.csv"),
    )

    validation_results = run_basic_validation(
        baseline_result=validation_seed_results["ALL_BASELINE"],
        mixed_result=validation_seed_results["OBS_MIX"],
        real_pre_df=real_pre,
        real_pros_df=real_pros,
        output_dir=str(OUTPUT_DIR),
    )
    validation_results.to_csv(OUTPUT_DIR / "pathway_validation_results.csv", index=False)

    # Stage-level validation against real data.
    stage_validation_df = run_stage_validation(stage_wait_df)
    stage_validation_df.to_csv(OUTPUT_DIR / "stage_validation_results.csv", index=False)

    print("\n=== STAGE VALIDATION AGAINST REAL DATA ===")
    if not stage_validation_df.empty:
        print(stage_validation_df.round(3).to_string(index=False))
    else:
        print("No stage validation rows were produced.")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()