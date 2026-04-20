from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from combined_des_engine import CombinedEngineConfig, run_day_loop_combined_engine
from combined_stage_engine import WAIT_MODE_MC, WAIT_MODE_DES


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "combined_stage_pressure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CALIBRATED INPUTS
# --------------------------------------------------
START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.7528735632183907
P_PROSTAD_OBS = 0.5098039215686274

SEEDS = list(range(1, 21))


# --------------------------------------------------
# SCENARIO BUILDER
# --------------------------------------------------
def build_combined_config(
    name: str,
    p_prostad: float,
    seed: int,
) -> CombinedEngineConfig:
    return CombinedEngineConfig(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        p_prostad=p_prostad,
        mri_capacity_by_weekday_prostad={1: 4},
        seed=seed,
        scenario_name=name,

        baseline_wait_time_mode={
            "ref_to_mri": WAIT_MODE_MC,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },

        prostad_wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },

        baseline_stage_timing_policy={
            "mri_to_report": "EMPIRICAL",
            "report_to_biopmdt": "EMPIRICAL",
        },

        prostad_stage_timing_policy={
            "mri_to_report": "FIXED",
            "report_to_biopmdt": "FIXED",
        },

        prostad_fixed_wait_days_by_stage={
            "mri_to_report": 1,
            "report_to_biopmdt": 0,
        },
    )


# --------------------------------------------------
# REFERRAL SCHEDULE
# --------------------------------------------------
def generate_daily_referrals(
    start_date: date,
    n_days: int,
    lam_per_workday: float,
    seed: int,
) -> Dict[date, int]:
    rng = np.random.default_rng(seed)
    referrals = {}
    current_date = start_date

    for _ in range(n_days):
        if current_date.weekday() < 5:
            referrals[current_date] = int(rng.poisson(lam_per_workday))
        else:
            referrals[current_date] = 0
        current_date += timedelta(days=1)

    return referrals


# --------------------------------------------------
# RUN 3 SCENARIOS WITH SAME REFERRAL STREAM
# --------------------------------------------------
def run_three_scenarios(seed: int):
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    # NOTE:
    # This assumes you add an optional daily_referrals_override
    # to run_day_loop_combined_engine(), just like you did before
    # in the separate model. If you have not added that yet,
    # I explain the tiny engine change below the script.
    baseline_cfg = build_combined_config("ALL_BASELINE", p_prostad=0.0, seed=seed)
    mixed_cfg = build_combined_config("OBS_MIX", p_prostad=P_PROSTAD_OBS, seed=seed)
    all_pros_cfg = build_combined_config("ALL_PROSTAD", p_prostad=1.0, seed=seed)

    baseline_res = run_day_loop_combined_engine(
        baseline_cfg,
        daily_referrals_override=referral_schedule,
    )
    mixed_res = run_day_loop_combined_engine(
        mixed_cfg,
        daily_referrals_override=referral_schedule,
    )
    all_pros_res = run_day_loop_combined_engine(
        all_pros_cfg,
        daily_referrals_override=referral_schedule,
    )

    return baseline_res, mixed_res, all_pros_res


# --------------------------------------------------
# STAGE SUMMARIES
# --------------------------------------------------
def summarise_stage_activity(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    for stage_name, metrics in result["stage_activity"].items():
        arrivals = metrics["daily_arrivals"]
        in_stage = metrics["daily_in_stage"]
        completed = metrics["daily_completed"]

        arrivals_vals = list(arrivals.values())
        in_stage_vals = list(in_stage.values())
        completed_vals = list(completed.values())

        total_arrivals = sum(arrivals_vals)
        total_completed = sum(completed_vals)
        completion_ratio = total_completed / total_arrivals if total_arrivals > 0 else np.nan

        rows.append({
            "scenario": scenario_name,
            "seed": seed,
            "stage": stage_name,
            "total_arrivals": total_arrivals,
            "mean_daily_arrivals": float(np.mean(arrivals_vals)) if arrivals_vals else 0.0,
            "peak_daily_arrivals": max(arrivals_vals) if arrivals_vals else 0,
            "mean_in_stage": float(np.mean(in_stage_vals)) if in_stage_vals else 0.0,
            "peak_in_stage": max(in_stage_vals) if in_stage_vals else 0,
            "total_completed": total_completed,
            "completion_ratio": completion_ratio,
        })

    return pd.DataFrame(rows)


def summarise_stage_weekly_arrivals(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    for stage_name, metrics in result["stage_activity"].items():
        arrivals = metrics["daily_arrivals"]

        if not arrivals:
            continue

        df = pd.DataFrame({
            "date": pd.to_datetime(list(arrivals.keys())),
            "n_arrivals": list(arrivals.values()),
        })

        df["week_index"] = ((df["date"] - df["date"].min()).dt.days // 7).astype(int)

        weekly = (
            df.groupby("week_index")["n_arrivals"]
            .sum()
            .reset_index()
        )

        for _, row in weekly.iterrows():
            rows.append({
                "scenario": scenario_name,
                "seed": seed,
                "stage": stage_name,
                "week_index": int(row["week_index"]),
                "weekly_arrivals": int(row["n_arrivals"]),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# FLOW COUNTS
# --------------------------------------------------
def count_event_occurrences(result: dict, event_name: str) -> int:
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return 0
    return int((event_log["event"] == event_name).sum())


def summarise_flow_counts(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    milestones = [
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

    rows = []
    for event_name in milestones:
        rows.append({
            "scenario": scenario_name,
            "seed": seed,
            "event": event_name,
            "count": count_event_occurrences(result, event_name),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# AGGREGATION
# --------------------------------------------------
def safe_pct_change(new, old):
    if old in (0, 0.0, None) or pd.isna(old):
        return np.nan
    return (new - old) / old * 100


def aggregate_stage_summary(stage_df: pd.DataFrame) -> pd.DataFrame:
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

    # Compare observed mix to baseline
    wide["mix_arrival_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("mean_daily_arrivals_mean_OBS_MIX"),
            r.get("mean_daily_arrivals_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["mix_peak_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("peak_daily_arrivals_mean_OBS_MIX"),
            r.get("peak_daily_arrivals_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["mix_in_stage_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("mean_in_stage_mean_OBS_MIX"),
            r.get("mean_in_stage_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["mix_completion_change"] = (
        wide.get("completion_ratio_mean_OBS_MIX") -
        wide.get("completion_ratio_mean_ALL_BASELINE")
    )

    # Compare all PROSTAD to baseline
    wide["allpros_arrival_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("mean_daily_arrivals_mean_ALL_PROSTAD"),
            r.get("mean_daily_arrivals_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["allpros_peak_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("peak_daily_arrivals_mean_ALL_PROSTAD"),
            r.get("peak_daily_arrivals_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["allpros_in_stage_pct_change"] = wide.apply(
        lambda r: safe_pct_change(
            r.get("mean_in_stage_mean_ALL_PROSTAD"),
            r.get("mean_in_stage_mean_ALL_BASELINE"),
        ),
        axis=1,
    )

    wide["allpros_completion_change"] = (
        wide.get("completion_ratio_mean_ALL_PROSTAD") -
        wide.get("completion_ratio_mean_ALL_BASELINE")
    )

    wide["stage_flag_obs_mix"] = wide.apply(classify_stage_obs_mix, axis=1)
    wide["stage_flag_all_prostad"] = wide.apply(classify_stage_all_prostad, axis=1)

    return wide


def classify_stage_from_changes(arrival_pct, peak_pct, in_stage_pct, completion_change, mean_in_stage_value):
    # Clear pressure / emerging bottleneck:
    # demand up, peaks up, in-stage burden up, and no improvement in clearance
    if (
        pd.notna(arrival_pct) and pd.notna(peak_pct) and pd.notna(in_stage_pct)
        and arrival_pct > 10
        and peak_pct > 15
        and in_stage_pct > 10
        and (pd.isna(completion_change) or completion_change <= 0)
    ):
        return "🔴 EMERGING BOTTLENECK"

    # Faster flow:
    # arrivals rise because patients move through faster, but burden in stage falls
    if (
        pd.notna(arrival_pct) and arrival_pct > 10
        and pd.notna(in_stage_pct) and in_stage_pct < -20
        and (pd.isna(completion_change) or completion_change >= 0)
    ):
        return "🚀 FASTER FLOW"

    # Pressure increasing:
    # demand or in-stage burden increases, but not enough to call a bottleneck
    if (
        (pd.notna(arrival_pct) and arrival_pct > 5)
        or (pd.notna(in_stage_pct) and in_stage_pct > 5)
    ):
        return "🟠 PRESSURE ↑"

    # Improved:
    if pd.notna(in_stage_pct) and in_stage_pct < -10:
        return "🟢 IMPROVED"

    # Near-instant stage
    if pd.notna(mean_in_stage_value) and mean_in_stage_value < 0.5:
        return "⚡ FAST TRACK"

    return "⚪ STABLE"


def classify_stage_obs_mix(row):
    return classify_stage_from_changes(
        row.get("mix_arrival_pct_change"),
        row.get("mix_peak_pct_change"),
        row.get("mix_in_stage_pct_change"),
        row.get("mix_completion_change"),
        row.get("mean_in_stage_mean_OBS_MIX"),
    )


def classify_stage_all_prostad(row):
    return classify_stage_from_changes(
        row.get("allpros_arrival_pct_change"),
        row.get("allpros_peak_pct_change"),
        row.get("allpros_in_stage_pct_change"),
        row.get("allpros_completion_change"),
        row.get("mean_in_stage_mean_ALL_PROSTAD"),
    )


def aggregate_weekly_stage_arrivals(weekly_df: pd.DataFrame) -> pd.DataFrame:
    return (
        weekly_df.groupby(["scenario", "stage", "week_index"])
        .agg(
            weekly_arrivals_mean=("weekly_arrivals", "mean"),
            weekly_arrivals_std=("weekly_arrivals", "std"),
        )
        .reset_index()
    )


def aggregate_flow_summary(flow_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        flow_df.groupby(["scenario", "event"])
        .agg(
            count_mean=("count", "mean"),
            count_std=("count", "std"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="event", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    wide["obs_mix_minus_baseline"] = (
        wide.get("count_mean_OBS_MIX") - wide.get("count_mean_ALL_BASELINE")
    )
    wide["allpros_minus_baseline"] = (
        wide.get("count_mean_ALL_PROSTAD") - wide.get("count_mean_ALL_BASELINE")
    )

    return wide


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
def make_stage_pressure_plot(stage_summary: pd.DataFrame):
    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    plot_df = stage_summary.copy()
    plot_df["stage_label"] = plot_df["stage"].map(stage_labels).fillna(plot_df["stage"])

    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["mean_in_stage_mean_ALL_BASELINE"], width, label="All baseline")
    plt.bar(x, plot_df["mean_in_stage_mean_OBS_MIX"], width, label="Observed mix")
    plt.bar(x + width, plot_df["mean_in_stage_mean_ALL_PROSTAD"], width, label="All PROSTAD")

    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Stage pressure comparison across pathway mix scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_pressure_mean_in_stage.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_weekly_arrivals_plot(weekly_summary: pd.DataFrame, stage_name: str):
    df = weekly_summary[weekly_summary["stage"] == stage_name].copy()
    if df.empty:
        return

    plt.figure(figsize=(12, 6))

    for scenario in ["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"]:
        sub = df[df["scenario"] == scenario].sort_values("week_index")
        if sub.empty:
            continue
        plt.plot(sub["week_index"], sub["weekly_arrivals_mean"], label=scenario)

    plt.xlabel("Week index")
    plt.ylabel("Mean weekly arrivals")
    plt.title(f"Weekly arrivals by stage: {stage_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weekly_arrivals_{stage_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def flatten_wait_values(daily_waits: dict) -> list[float]:
    flat_wait_vals = []

    for v in (daily_waits or {}).values():
        if v is None:
            continue
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            flat_wait_vals.extend([x for x in v if pd.notna(x)])
        else:
            if pd.notna(v):
                flat_wait_vals.append(v)

    return flat_wait_vals


def summarise_mri_resource(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    for resource_name, metrics in result.get("resources", {}).items():
        # only summarise MRI_PROSTAD resource here
        if resource_name != "MRI_PROSTAD":
            continue

        daily_queue = metrics.get("daily_queue_len", {}) or {}
        daily_waits = metrics.get("daily_waits", {}) or {}

        queue_vals = list(daily_queue.values())
        flat_wait_vals = flatten_wait_values(daily_waits)

        rows.append({
            "scenario": scenario_name,
            "seed": seed,
            "resource": resource_name,
            "mean_queue_len": float(np.mean(queue_vals)) if queue_vals else 0.0,
            "peak_queue_len": max(queue_vals) if queue_vals else 0,
            "mean_wait": float(np.mean(flat_wait_vals)) if flat_wait_vals else 0.0,
            "peak_wait": max(flat_wait_vals) if flat_wait_vals else 0,
            "n_wait_observations": len(flat_wait_vals),
        })

    return pd.DataFrame(rows)

def aggregate_mri_resource_summary(resource_df: pd.DataFrame) -> pd.DataFrame:
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

def extract_pathway_lengths(result: dict, scenario_name: str) -> pd.DataFrame:
    rows = []

    for events, total_days in result["patient_results"]:
        if not events:
            continue

        patient_id = events[0].get("patient_id")

        # check if full pathway completed
        event_names = {e["event"] for e in events}

        if "Outpatient_appointment_occured" not in event_names:
            continue  # ❌ exclude partial pathways

        rows.append({
            "scenario": scenario_name,
            "patient_id": patient_id,
            "total_days": total_days,
        })

    return pd.DataFrame(rows)

def extract_pathway_lengths_with_type(result: dict, scenario_name: str) -> pd.DataFrame:
    rows = []

    for patient in result["completed_patients_objects"]:
        events = patient.events
        total_days = (patient.current_date - patient.start_date).days

        # only full pathway
        event_names = {e["event"] for e in events}
        if "Outpatient_appointment_occured" not in event_names:
            continue

        rows.append({
            "scenario": scenario_name,
            "patient_id": patient.patient_id,
            "pathway_type": patient.data.get("pathway_type"),
            "total_days": total_days,
        })

    return pd.DataFrame(rows)

def summarise_pathway_stats(df: pd.DataFrame) -> pd.DataFrame:
    def pct_within_target(x, target=62):
        return (x <= target).mean() * 100

    summary = (
        df.groupby("scenario")["total_days"]
        .agg(
            n="count",
            mean_days="mean",
            median_days="median",
            std_days="std",
            min_days="min",
            max_days="max",
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            pct_within_62=lambda x: pct_within_target(x, 62),
        )
        .reset_index()
    )

    return summary

def compare_pathway_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    base = summary_df[summary_df["scenario"] == "ALL_BASELINE"].iloc[0]

    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            "scenario": row["scenario"],
            "mean_days": row["mean_days"],
            "median_days": row["median_days"],
            "pct_within_62": row["pct_within_62"],
            "delta_mean_vs_baseline": row["mean_days"] - base["mean_days"],
            "delta_pct_within_62": row["pct_within_62"] - base["pct_within_62"],
        })

    return pd.DataFrame(rows)


  

def plot_pathway_distributions(df):
    plt.figure(figsize=(10,6))

    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario]
        subset["total_days"].plot(kind="kde", label=scenario)

    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlabel("Total pathway days")
    plt.title("Pathway time distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def summarise_mixed_pathway_type(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("pathway_type")["total_days"]
        .agg(
            n="count",
            mean_days="mean",
            median_days="median",
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
            pct_within_62=lambda x: (x <= 62).mean() * 100,
        )
        .reset_index()
    )


def plot_mixed_pathway_split(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    for ptype in ["BASELINE", "PROSTAD"]:
        subset = df[df["pathway_type"] == ptype]
        if not subset.empty:
            subset["total_days"].plot(kind="kde", label=ptype)

    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlim(left=0)
    plt.xlabel("Total pathway days")
    plt.ylabel("Density")
    plt.title("OBS_MIX full-pathway times by pathway type")
    plt.legend()
    plt.tight_layout()
    plt.show()



# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    all_stage_rows = []
    all_weekly_rows = []
    all_flow_rows = []
    all_resource_rows = []
    pathway_dfs = []
    mixed_pathway_type_dfs = []

    print(f"Running seeds: {SEEDS}")

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        baseline_res, mixed_res, allpros_res = run_three_scenarios(seed)

        # Full-pathway completed patients only, across scenarios
        pathway_dfs.append(extract_pathway_lengths(baseline_res, "ALL_BASELINE"))
        pathway_dfs.append(extract_pathway_lengths(mixed_res, "OBS_MIX"))
        pathway_dfs.append(extract_pathway_lengths(allpros_res, "ALL_PROSTAD"))

        # Mixed scenario only: keep pathway type
        mixed_pathway_type_dfs.append(
            extract_pathway_lengths_with_type(mixed_res, "OBS_MIX")
        )

        # Stage summaries
        all_stage_rows.append(summarise_stage_activity(baseline_res, "ALL_BASELINE", seed))
        all_stage_rows.append(summarise_stage_activity(mixed_res, "OBS_MIX", seed))
        all_stage_rows.append(summarise_stage_activity(allpros_res, "ALL_PROSTAD", seed))

        # Weekly arrivals
        all_weekly_rows.append(summarise_stage_weekly_arrivals(baseline_res, "ALL_BASELINE", seed))
        all_weekly_rows.append(summarise_stage_weekly_arrivals(mixed_res, "OBS_MIX", seed))
        all_weekly_rows.append(summarise_stage_weekly_arrivals(allpros_res, "ALL_PROSTAD", seed))

        # Flow counts
        all_flow_rows.append(summarise_flow_counts(baseline_res, "ALL_BASELINE", seed))
        all_flow_rows.append(summarise_flow_counts(mixed_res, "OBS_MIX", seed))
        all_flow_rows.append(summarise_flow_counts(allpros_res, "ALL_PROSTAD", seed))

        # MRI resource summary
        all_resource_rows.append(summarise_mri_resource(baseline_res, "ALL_BASELINE", seed))
        all_resource_rows.append(summarise_mri_resource(mixed_res, "OBS_MIX", seed))
        all_resource_rows.append(summarise_mri_resource(allpros_res, "ALL_PROSTAD", seed))

    # Combine all runs
    stage_df = pd.concat(all_stage_rows, ignore_index=True)
    weekly_df = pd.concat(all_weekly_rows, ignore_index=True)
    flow_df = pd.concat(all_flow_rows, ignore_index=True)
    resource_df = pd.concat(all_resource_rows, ignore_index=True)
    pathway_df = pd.concat(pathway_dfs, ignore_index=True)
    mixed_pathway_type_df = pd.concat(mixed_pathway_type_dfs, ignore_index=True)

    # Save raw outputs
    stage_df.to_csv(OUTPUT_DIR / "stage_activity_all_runs.csv", index=False)
    weekly_df.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_all_runs.csv", index=False)
    flow_df.to_csv(OUTPUT_DIR / "flow_counts_all_runs.csv", index=False)
    resource_df.to_csv(OUTPUT_DIR / "mri_resource_all_runs.csv", index=False)
    pathway_df.to_csv(OUTPUT_DIR / "pathway_lengths_all_runs.csv", index=False)
    mixed_pathway_type_df.to_csv(OUTPUT_DIR / "obs_mix_pathway_lengths_by_type_all_runs.csv", index=False)

    # Aggregate summaries
    stage_summary = aggregate_stage_summary(stage_df)
    weekly_summary = aggregate_weekly_stage_arrivals(weekly_df)
    flow_summary = aggregate_flow_summary(flow_df)
    mri_resource_summary = aggregate_mri_resource_summary(resource_df)

    pathway_summary = summarise_pathway_stats(pathway_df)
    pathway_comparison = compare_pathway_stats(pathway_summary)

    mixed_pathway_type_summary = summarise_mixed_pathway_type(mixed_pathway_type_df)

    # Save summaries
    stage_summary.to_csv(OUTPUT_DIR / "stage_pressure_summary.csv", index=False)
    weekly_summary.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_summary.csv", index=False)
    flow_summary.to_csv(OUTPUT_DIR / "flow_summary.csv", index=False)
    mri_resource_summary.to_csv(OUTPUT_DIR / "mri_resource_summary.csv", index=False)
    pathway_summary.to_csv(OUTPUT_DIR / "pathway_length_summary.csv", index=False)
    pathway_comparison.to_csv(OUTPUT_DIR / "pathway_length_comparison.csv", index=False)
    mixed_pathway_type_summary.to_csv(OUTPUT_DIR / "obs_mix_pathway_length_by_type_summary.csv", index=False)

    summary_cols = [
        "stage",
        "mean_in_stage_mean_ALL_BASELINE",
        "mean_in_stage_mean_OBS_MIX",
        "mean_in_stage_mean_ALL_PROSTAD",
        "mix_arrival_pct_change",
        "mix_peak_pct_change",
        "mix_in_stage_pct_change",
        "mix_completion_change",
        "stage_flag_obs_mix",
        "allpros_arrival_pct_change",
        "allpros_peak_pct_change",
        "allpros_in_stage_pct_change",
        "allpros_completion_change",
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

    print("\n=== PATHWAY LENGTH COMPARISON (FULL PATHWAY ONLY) ===")
    print(pathway_comparison.round(2).to_string(index=False))

    print("\n=== OBS_MIX PATHWAY LENGTH SUMMARY BY PATHWAY TYPE ===")
    print(mixed_pathway_type_summary.round(2).to_string(index=False))

    # Plots
    make_stage_pressure_plot(stage_summary)

    for stage_name in weekly_summary["stage"].unique():
        make_weekly_arrivals_plot(weekly_summary, stage_name)

    plot_pathway_distributions(pathway_df)
    plot_mixed_pathway_split(mixed_pathway_type_df)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")

  

if __name__ == "__main__":
    main()