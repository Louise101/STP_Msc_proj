from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from validate_scenarios import run_scenario


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "multiseed_stage_activity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEEDS = list(range(1, 21))
START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 0.586

def generate_daily_referrals(start_date, n_days, lam_per_workday, seed):
    rng = np.random.default_rng(seed)
    referrals = {}
    current_date = start_date

    for _ in range(n_days):
        weekday = current_date.weekday()
        if weekday < 5:
            referrals[current_date] = int(rng.poisson(lam_per_workday))
        else:
            referrals[current_date] = 0
        current_date += timedelta(days=1)

    return referrals


def safe_div(numerator, denominator):
    if denominator in (0, 0.0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def safe_pct_change(new, old):
    if old in (0, 0.0, None) or pd.isna(old):
        return np.nan
    return (new - old) / old * 100


def summarise_stage_demand_weekly(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    stage_activity = result.get("stage_activity", {})

    for stage_name, metrics in stage_activity.items():
        arrivals = metrics.get("daily_arrivals", {}) or {}
        in_stage = metrics.get("daily_in_stage", {}) or {}
        completed = metrics.get("daily_completed", {}) or {}

        # -----------------------------
        # Weekly arrivals
        # -----------------------------
        if arrivals:
            arrivals_df = pd.DataFrame({
                "date": pd.to_datetime(list(arrivals.keys())),
                "arrivals": list(arrivals.values()),
            })
            arrivals_df["week"] = arrivals_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
            weekly_arrivals = arrivals_df.groupby("week")["arrivals"].sum()
        else:
            weekly_arrivals = pd.Series(dtype=float)

        total_arrivals = sum(arrivals.values())
        total_completed = sum(completed.values())

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "stage": stage_name,

                # weekly arrival metrics
                "mean_weekly_arrivals": float(weekly_arrivals.mean()) if not weekly_arrivals.empty else 0.0,
                "peak_weekly_arrivals": float(weekly_arrivals.max()) if not weekly_arrivals.empty else 0.0,
                "std_weekly_arrivals": float(weekly_arrivals.std(ddof=1)) if len(weekly_arrivals) > 1 else 0.0,
                "n_weeks": int(len(weekly_arrivals)),

                # keep old metrics too
                "total_arrivals": total_arrivals,
                "mean_daily_arrivals": float(np.mean(list(arrivals.values()))) if arrivals else 0.0,
                "peak_daily_arrivals": max(arrivals.values()) if arrivals else 0,
                "mean_in_stage": float(np.mean(list(in_stage.values()))) if in_stage else 0.0,
                "peak_in_stage": max(in_stage.values()) if in_stage else 0,
                "total_completed": total_completed,
                "completion_ratio": safe_div(total_completed, total_arrivals),
            }
        )

    return pd.DataFrame(rows)


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
        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "event": event_name,
                "count": count_event_occurrences(result, event_name),
            }
        )

    return pd.DataFrame(rows)


def summarise_branching(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    biopsy_done = int((event_log["event"] == "biopsy_done").sum())
    pathrep_done = int((event_log["event"] == "Path_report_recieved").sum())
    treat_mdt_done = int((event_log["event"] == "Treatment_options_MDT_occured").sum())
    outpat_done = int((event_log["event"] == "Outpatient_appointment_occured").sum())

    rows = [
        {
            "scenario": scenario_name,
            "seed": seed,
            "transition": "biopsy_done -> path_report",
            "from_n": biopsy_done,
            "to_n": pathrep_done,
            "ratio": safe_div(pathrep_done, biopsy_done),
        },
        {
            "scenario": scenario_name,
            "seed": seed,
            "transition": "path_report -> treat_mdt",
            "from_n": pathrep_done,
            "to_n": treat_mdt_done,
            "ratio": safe_div(treat_mdt_done, pathrep_done),
        },
        {
            "scenario": scenario_name,
            "seed": seed,
            "transition": "treat_mdt -> outpatient",
            "from_n": treat_mdt_done,
            "to_n": outpat_done,
            "ratio": safe_div(outpat_done, treat_mdt_done),
        },
    ]
    return pd.DataFrame(rows)


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


def summarise_resource_pressure(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    for resource_name, metrics in result.get("resources", {}).items():
        daily_queue = metrics.get("daily_queue_len", {}) or {}
        daily_waits = metrics.get("daily_waits", {}) or {}

        queue_vals = list(daily_queue.values())
        flat_wait_vals = flatten_wait_values(daily_waits)

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "resource": resource_name,
                "mean_queue_len": float(np.mean(queue_vals)) if queue_vals else 0.0,
                "peak_queue_len": max(queue_vals) if queue_vals else 0,
                "mean_wait": float(np.mean(flat_wait_vals)) if flat_wait_vals else 0.0,
                "peak_wait": max(flat_wait_vals) if flat_wait_vals else 0,
                "n_wait_observations": len(flat_wait_vals),
            }
        )

    return pd.DataFrame(rows)


def summarise_late_pipeline_patients(
    result: dict,
    scenario_name: str,
    seed: int,
    last_n_days: int = 30,
) -> pd.DataFrame:
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    df = event_log.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=last_n_days)

    late_df = df[df["date"] >= cutoff]

    milestones = [
        "mri_performed",
        "biopsy_done",
        "Path_report_recieved",
        "Treatment_options_MDT_occured",
        "Outpatient_appointment_occured",
    ]

    rows = []
    for event_name in milestones:
        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "window_start": cutoff.date(),
                "window_end": max_date.date(),
                "event": event_name,
                "count_in_last_window": int((late_df["event"] == event_name).sum()),
            }
        )

    return pd.DataFrame(rows)


def classify_stage(row):
    weekly_arrival_change = row["mean_weekly_arrivals_diff_mean"]
    peak_weekly_change = row["peak_weekly_arrivals_diff_mean"]
    variability_change = row["std_weekly_arrivals_diff_mean"]
    in_stage_change = row["mean_in_stage_diff_mean"]
    completion_ratio = row["completion_ratio_mean_PROSTAD"]

    if (
        pd.notna(completion_ratio)
        and peak_weekly_change >= 2
        and in_stage_change >= 0.5
        and completion_ratio < 0.95
    ):
        return "🔴 BOTTLENECK"

    if weekly_arrival_change > 0.25 and in_stage_change > 0.1:
        return "🟠 PRESSURE ↑"

    if in_stage_change < -0.5:
        return "🟢 IMPROVED"

    if row["mean_in_stage_mean_PROSTAD"] < 0.5:
        return "⚡ FAST TRACK"

    return "⚪ STABLE"

def run_both_scenarios(seed: int):
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    mc_res = run_scenario(
        "ALL_MC_BASELINE",
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
        daily_referrals_override=referral_schedule,
    )

    prostad_res = run_scenario(
        "PROSTAD",
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
        daily_referrals_override=referral_schedule,
    )

    return mc_res, prostad_res


def aggregate_stage_metrics(all_stage_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        all_stage_df.groupby(["scenario", "stage"])
        .agg(
            total_arrivals_mean=("total_arrivals", "mean"),
            total_arrivals_std=("total_arrivals", "std"),
            mean_in_stage_mean=("mean_in_stage", "mean"),
            mean_in_stage_std=("mean_in_stage", "std"),
            peak_in_stage_mean=("peak_in_stage", "mean"),
            peak_in_stage_std=("peak_in_stage", "std"),
            completion_ratio_mean=("completion_ratio", "mean"),
            completion_ratio_std=("completion_ratio", "std"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="stage", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    wide["arrival_diff_mean"] = (
        wide["total_arrivals_mean_PROSTAD"] - wide["total_arrivals_mean_ALL_MC_BASELINE"]
    )
    wide["arrival_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["total_arrivals_mean_PROSTAD"], r["total_arrivals_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["mean_in_stage_diff_mean"] = (
        wide["mean_in_stage_mean_PROSTAD"] - wide["mean_in_stage_mean_ALL_MC_BASELINE"]
    )
    wide["mean_in_stage_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["mean_in_stage_mean_PROSTAD"], r["mean_in_stage_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["completion_ratio_diff_mean"] = (
        wide["completion_ratio_mean_PROSTAD"] - wide["completion_ratio_mean_ALL_MC_BASELINE"]
    )
    wide["completion_ratio_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["completion_ratio_mean_PROSTAD"], r["completion_ratio_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["stage_flag"] = wide.apply(classify_stage, axis=1)

    return wide
def aggregate_stage_metrics(all_stage_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        all_stage_df.groupby(["scenario", "stage"])
        .agg(
            # weekly arrivals
            mean_weekly_arrivals_mean=("mean_weekly_arrivals", "mean"),
            mean_weekly_arrivals_std=("mean_weekly_arrivals", "std"),
            peak_weekly_arrivals_mean=("peak_weekly_arrivals", "mean"),
            peak_weekly_arrivals_std=("peak_weekly_arrivals", "std"),
            std_weekly_arrivals_mean=("std_weekly_arrivals", "mean"),
            std_weekly_arrivals_std=("std_weekly_arrivals", "std"),

            # old metrics kept
            total_arrivals_mean=("total_arrivals", "mean"),
            total_arrivals_std=("total_arrivals", "std"),
            mean_in_stage_mean=("mean_in_stage", "mean"),
            mean_in_stage_std=("mean_in_stage", "std"),
            peak_in_stage_mean=("peak_in_stage", "mean"),
            peak_in_stage_std=("peak_in_stage", "std"),
            completion_ratio_mean=("completion_ratio", "mean"),
            completion_ratio_std=("completion_ratio", "std"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="stage", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    # weekly arrival differences
    wide["mean_weekly_arrivals_diff_mean"] = (
        wide["mean_weekly_arrivals_mean_PROSTAD"] - wide["mean_weekly_arrivals_mean_ALL_MC_BASELINE"]
    )
    wide["mean_weekly_arrivals_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(
            r["mean_weekly_arrivals_mean_PROSTAD"],
            r["mean_weekly_arrivals_mean_ALL_MC_BASELINE"],
        ),
        axis=1,
    )

    wide["peak_weekly_arrivals_diff_mean"] = (
        wide["peak_weekly_arrivals_mean_PROSTAD"] - wide["peak_weekly_arrivals_mean_ALL_MC_BASELINE"]
    )

    wide["std_weekly_arrivals_diff_mean"] = (
        wide["std_weekly_arrivals_mean_PROSTAD"] - wide["std_weekly_arrivals_mean_ALL_MC_BASELINE"]
    )

    # keep old comparisons too
    wide["arrival_diff_mean"] = (
        wide["total_arrivals_mean_PROSTAD"] - wide["total_arrivals_mean_ALL_MC_BASELINE"]
    )
    wide["arrival_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["total_arrivals_mean_PROSTAD"], r["total_arrivals_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["mean_in_stage_diff_mean"] = (
        wide["mean_in_stage_mean_PROSTAD"] - wide["mean_in_stage_mean_ALL_MC_BASELINE"]
    )
    wide["mean_in_stage_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["mean_in_stage_mean_PROSTAD"], r["mean_in_stage_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["completion_ratio_diff_mean"] = (
        wide["completion_ratio_mean_PROSTAD"] - wide["completion_ratio_mean_ALL_MC_BASELINE"]
    )
    wide["completion_ratio_diff_pct"] = wide.apply(
        lambda r: safe_pct_change(r["completion_ratio_mean_PROSTAD"], r["completion_ratio_mean_ALL_MC_BASELINE"]),
        axis=1,
    )

    wide["stage_flag"] = wide.apply(classify_stage, axis=1)

    return wide

def aggregate_flow_counts(flow_df: pd.DataFrame) -> pd.DataFrame:
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
    wide["difference_mean"] = wide["count_mean_PROSTAD"] - wide["count_mean_ALL_MC_BASELINE"]
    return wide


def aggregate_branching(branch_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        branch_df.groupby(["scenario", "transition"])
        .agg(
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="transition", columns="scenario")
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()
    wide["difference_mean"] = wide["ratio_mean_PROSTAD"] - wide["ratio_mean_ALL_MC_BASELINE"]
    return wide


def aggregate_resource_pressure(resource_df: pd.DataFrame) -> pd.DataFrame:
    return (
        resource_df.groupby(["scenario", "resource"])
        .agg(
            mean_queue_len_mean=("mean_queue_len", "mean"),
            mean_queue_len_std=("mean_queue_len", "std"),
            peak_queue_len_mean=("peak_queue_len", "mean"),
            peak_queue_len_std=("peak_queue_len", "std"),
            mean_wait_mean=("mean_wait", "mean"),
            mean_wait_std=("mean_wait", "std"),
        )
        .reset_index()
    )


def aggregate_late_pipeline(late_df: pd.DataFrame) -> pd.DataFrame:
    return (
        late_df.groupby(["scenario", "event"])
        .agg(
            count_in_last_window_mean=("count_in_last_window", "mean"),
            count_in_last_window_std=("count_in_last_window", "std"),
        )
        .reset_index()
    )


def make_stage_plots(stage_summary: pd.DataFrame):
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
    width = 0.38

    # --------------------------------------------------
    # Plot 1: mean number in stage
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        plot_df["mean_in_stage_mean_ALL_MC_BASELINE"],
        width,
        yerr=plot_df["mean_in_stage_std_ALL_MC_BASELINE"],
        capsize=4,
        label="ALL_MC_BASELINE",
    )
    plt.bar(
        x + width / 2,
        plot_df["mean_in_stage_mean_PROSTAD"],
        width,
        yerr=plot_df["mean_in_stage_std_PROSTAD"],
        capsize=4,
        label="PROSTAD",
    )
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Bottleneck shift across seeds: mean number in stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_bottleneck_shift_mean_in_stage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # Plot 2: change in mean number in stage
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["stage_label"], plot_df["mean_in_stage_diff_mean"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("PROSTAD - MC mean in stage")
    plt.title("Mean change in stage occupancy across seeds")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_bottleneck_shift_delta_in_stage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # Plot 3: mean weekly arrivals by stage
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        plot_df["mean_weekly_arrivals_mean_ALL_MC_BASELINE"],
        width,
        yerr=plot_df["mean_weekly_arrivals_std_ALL_MC_BASELINE"],
        capsize=4,
        label="ALL_MC_BASELINE",
    )
    plt.bar(
        x + width / 2,
        plot_df["mean_weekly_arrivals_mean_PROSTAD"],
        width,
        yerr=plot_df["mean_weekly_arrivals_std_PROSTAD"],
        capsize=4,
        label="PROSTAD",
    )
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean weekly arrivals")
    plt.title("Weekly demand by stage across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_mean_weekly_arrivals_by_stage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # Plot 4: peak weekly arrivals by stage
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        plot_df["peak_weekly_arrivals_mean_ALL_MC_BASELINE"],
        width,
        yerr=plot_df["peak_weekly_arrivals_std_ALL_MC_BASELINE"],
        capsize=4,
        label="ALL_MC_BASELINE",
    )
    plt.bar(
        x + width / 2,
        plot_df["peak_weekly_arrivals_mean_PROSTAD"],
        width,
        yerr=plot_df["peak_weekly_arrivals_std_PROSTAD"],
        capsize=4,
        label="PROSTAD",
    )
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Peak weekly arrivals")
    plt.title("Peak weekly demand by stage across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_peak_weekly_arrivals_by_stage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # Plot 5: change in mean weekly arrivals
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["stage_label"], plot_df["mean_weekly_arrivals_diff_mean"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("PROSTAD - MC mean weekly arrivals")
    plt.title("Change in weekly demand across seeds")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_delta_mean_weekly_arrivals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # Plot 6: change in peak weekly arrivals
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["stage_label"], plot_df["peak_weekly_arrivals_diff_mean"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("PROSTAD - MC peak weekly arrivals")
    plt.title("Change in peak weekly demand across seeds")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multiseed_delta_peak_weekly_arrivals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_weekly_arrivals_by_stage(weekly_summary_df: pd.DataFrame):
    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    stages = weekly_summary_df["stage"].dropna().unique()

    for stage_name in stages:
        stage_df = weekly_summary_df[weekly_summary_df["stage"] == stage_name].copy()

        plt.figure(figsize=(12, 6))

        for scenario in ["ALL_MC_BASELINE", "PROSTAD"]:
            s = stage_df[stage_df["scenario"] == scenario].sort_values("week_start")

            if s.empty:
                continue

            x = pd.to_datetime(s["week_start"])
            y = s["mean_weekly_arrivals"].to_numpy()
            sd = s["std_weekly_arrivals"].fillna(0).to_numpy()

            plt.plot(x, y, label=scenario)
            plt.fill_between(x, y - sd, y + sd, alpha=0.2)

        plt.xlabel("Week")
        plt.ylabel("Weekly arrivals")
        plt.title(f"Weekly arrivals over time: {stage_labels.get(stage_name, stage_name)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"weekly_arrivals_timeseries_{stage_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()   

def plot_weekly_arrivals_by_stage_with_seeds(weekly_arrivals_df: pd.DataFrame):
    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    stages = weekly_arrivals_df["stage"].dropna().unique()

    for stage_name in stages:
        stage_df = weekly_arrivals_df[weekly_arrivals_df["stage"] == stage_name].copy()

        plt.figure(figsize=(12, 6))

        for scenario in ["ALL_MC_BASELINE", "PROSTAD"]:
            s_all = stage_df[stage_df["scenario"] == scenario]

            for seed in sorted(s_all["seed"].unique()):
                s = s_all[s_all["seed"] == seed].sort_values("week_start")
                plt.plot(
                    pd.to_datetime(s["week_start"]),
                    s["weekly_arrivals"],
                    alpha=0.15,
                    linewidth=1,
                )

            mean_s = (
                s_all.groupby("week_start", as_index=False)["weekly_arrivals"]
                .mean()
                .sort_values("week_start")
            )

            plt.plot(
                pd.to_datetime(mean_s["week_start"]),
                mean_s["weekly_arrivals"],
                linewidth=2.5,
                label=scenario,
            )

        plt.xlabel("Week")
        plt.ylabel("Weekly arrivals")
        plt.title(f"Weekly arrivals by seed: {stage_labels.get(stage_name, stage_name)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"weekly_arrivals_timeseries_with_seeds_{stage_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close() 
def plot_weekly_arrivals_by_seed(raw_df: pd.DataFrame):
    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    stages = raw_df["stage"].unique()
    seeds = sorted(raw_df["seed"].unique())

    for stage in stages:
        stage_df = raw_df[raw_df["stage"] == stage]

        plt.figure(figsize=(12, 6))

        for seed in seeds:
            mc = stage_df[
                (stage_df["scenario"] == "ALL_MC_BASELINE") &
                (stage_df["seed"] == seed)
            ].sort_values("week_start")

            pros = stage_df[
                (stage_df["scenario"] == "PROSTAD") &
                (stage_df["seed"] == seed)
            ].sort_values("week_start")

            if mc.empty or pros.empty:
                continue

            x_mc = pd.to_datetime(mc["week_start"])
            y_mc = mc["weekly_arrivals"]

            x_pr = pd.to_datetime(pros["week_start"])
            y_pr = pros["weekly_arrivals"]

            # SAME colour per seed
            colour = plt.cm.tab20(seed % 20)

            plt.plot(x_mc, y_mc, linestyle="--", color=colour, alpha=0.7)
            plt.plot(x_pr, y_pr, linestyle="-", color=colour, alpha=0.9)

        plt.title(f"Weekly arrivals by seed: {stage_labels.get(stage, stage)}")
        plt.xlabel("Week")
        plt.ylabel("Patients entering stage")

        # custom legend
        plt.plot([], [], linestyle="--", color="black", label="MC baseline")
        plt.plot([], [], linestyle="-", color="black", label="PROSTAD")
        plt.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"weekly_arrivals_by_seed_{stage}.png", dpi=300)
        plt.close()
def plot_weekly_arrivals_stage_by_seed(raw_df: pd.DataFrame):
    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    stages = sorted(raw_df["stage"].dropna().unique())
    seeds = sorted(raw_df["seed"].dropna().unique())

    for stage in stages:
        for seed in seeds:
            df = raw_df[
                (raw_df["stage"] == stage) &
                (raw_df["seed"] == seed)
            ].copy()

            if df.empty:
                continue

            plt.figure(figsize=(10, 5))

            for scenario in ["ALL_MC_BASELINE", "PROSTAD"]:
                s = df[df["scenario"] == scenario].sort_values("week_start")
                if s.empty:
                    continue

                plt.plot(
                    pd.to_datetime(s["week_start"]),
                    s["weekly_arrivals"],
                    label=scenario,
                )

            plt.title(f"{stage_labels.get(stage, stage)} — Seed {seed}")
            plt.xlabel("Week")
            plt.ylabel("Weekly arrivals")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / f"weekly_arrivals_{stage}_seed_{seed}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

def stage_arrivals_to_weekly_df(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    stage_activity = result.get("stage_activity", {})

    for stage_name, metrics in stage_activity.items():
        arrivals = metrics.get("daily_arrivals", {}) or {}

        if not arrivals:
            continue

        df = pd.DataFrame({
            "date": pd.to_datetime(list(arrivals.keys())),
            "arrivals": list(arrivals.values()),
        })

        df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

        weekly = (
            df.groupby("week_start", as_index=False)["arrivals"]
            .sum()
            .rename(columns={"arrivals": "weekly_arrivals"})
        )

        weekly["stage"] = stage_name
        weekly["scenario"] = scenario_name
        weekly["seed"] = seed

        rows.append(weekly)

    if not rows:
        return pd.DataFrame(columns=["week_start", "weekly_arrivals", "stage", "scenario", "seed"])

    return pd.concat(rows, ignore_index=True)

def aggregate_weekly_arrivals_across_seeds(weekly_arrivals_df: pd.DataFrame) -> pd.DataFrame:
    return (
        weekly_arrivals_df
        .groupby(["scenario", "stage", "week_start"])
        .agg(
            mean_weekly_arrivals=("weekly_arrivals", "mean"),
            std_weekly_arrivals=("weekly_arrivals", "std"),
            min_weekly_arrivals=("weekly_arrivals", "min"),
            max_weekly_arrivals=("weekly_arrivals", "max"),
        )
        .reset_index()
    )

#debug
def debug_patient_inputs(result, patient_ids=(1, 2, 3, 4, 5)):
    rows = []
    for p in result.get("all_patients_objects", []):
        if p.patient_id in patient_ids:
            rows.append({
                "patient_id": p.patient_id,
                "wait_ref_to_mri": p.data.get("wait_ref_to_mri"),
                "ref_to_mri_pre_delay": p.data.get("ref_to_mri_pre_delay"),
                "ref_to_mri_queue_wait": p.data.get("ref_to_mri_queue_wait"),
                "wait_biopmdt_to_biopsy": p.data.get("wait_biopmdt_to_biopsy"),
                "biopmdt_outcome": p.data.get("biopmdt_outcome"),
                "pathrep_outcome": p.data.get("pathrep_outcome"),
            })
    return pd.DataFrame(rows).sort_values("patient_id")

def compare_referral_streams(mc_res, pros_res):
    mc_daily = mc_res["daily_referrals"]
    pros_daily = pros_res["daily_referrals"]

    same_dates = set(mc_daily.keys()) == set(pros_daily.keys())
    same_counts = all(mc_daily[d] == pros_daily[d] for d in mc_daily)

    print("\n=== Referral stream comparison ===")
    print("Same referral dates:", same_dates)
    print("Same referral counts by day:", same_counts)
    print("Total referrals MC:", sum(mc_daily.values()))
    print("Total referrals PROSTAD:", sum(pros_daily.values()))

def per_seed_flow_delta(flow_df: pd.DataFrame, event_name: str) -> pd.DataFrame:
    sub = flow_df[flow_df["event"] == event_name].copy()
    pivot = sub.pivot(index="seed", columns="scenario", values="count").reset_index()
    pivot["delta"] = pivot["PROSTAD"] - pivot["ALL_MC_BASELINE"]
    return pivot

def debug_patients_with_downstream_progress(result, n=10):
    rows = []

    for p in result.get("all_patients_objects", []):
        if (
            p.data.get("biopmdt_outcome") is not None
            or p.data.get("pathrep_outcome") is not None
            or p.data.get("wait_biopmdt_to_biopsy") is not None
            or any(
                e.get("event") in {
                    "biopsy_done",
                    "Path_report_recieved",
                    "Treatment_options_MDT_occured",
                    "Outpatient_appointment_occured",
                }
                for e in getattr(p, "events", [])
            )
        ):
            rows.append({
                "patient_id": p.patient_id,
                "wait_ref_to_mri": p.data.get("wait_ref_to_mri"),
                "ref_to_mri_pre_delay": p.data.get("ref_to_mri_pre_delay"),
                "ref_to_mri_queue_wait": p.data.get("ref_to_mri_queue_wait"),
                "wait_biopmdt_to_biopsy": p.data.get("wait_biopmdt_to_biopsy"),
                "biopmdt_outcome": p.data.get("biopmdt_outcome"),
                "pathrep_outcome": p.data.get("pathrep_outcome"),
                "n_events": len(getattr(p, "events", [])),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "patient_id",
            "wait_ref_to_mri",
            "ref_to_mri_pre_delay",
            "ref_to_mri_queue_wait",
            "wait_biopmdt_to_biopsy",
            "biopmdt_outcome",
            "pathrep_outcome",
            "n_events",
        ])

    df = pd.DataFrame(rows).sort_values("patient_id")
    return df.head(n)

def debug_patient_from_events(result, patient_id: int) -> pd.DataFrame:
    for p in result.get("all_patients_objects", []):
        if p.patient_id == patient_id:
            rows = []
            for e in p.events:
                rows.append({
                    "patient_id": p.patient_id,
                    "event": e.get("event"),
                    "date": e.get("date"),
                    "wait_days": e.get("wait_days"),
                    "outcome": e.get("outcome"),
                })
            return pd.DataFrame(rows)

    return pd.DataFrame(columns=["patient_id", "event", "date", "wait_days", "outcome"])

def main():
    all_stage_rows = []
    all_flow_rows = []
    all_branch_rows = []
    all_resource_rows = []
    all_late_rows = []
    all_weekly_arrival_rows = []

    # --------------------------------------------------
    # ONE-SEED VALIDATION CHECK
    # --------------------------------------------------
    mc_check, pros_check = run_both_scenarios(seed=1)

    compare_referral_streams(mc_check, pros_check)

    mc_debug = debug_patient_inputs(mc_check)
    pros_debug = debug_patient_inputs(pros_check)

    print("\n=== Patient-level debug: MC ===")
    print(mc_debug.to_string(index=False))

    print("\n=== Patient-level debug: PROSTAD ===")
    print(pros_debug.to_string(index=False))

    print(f"Running {len(SEEDS)} seeds: {SEEDS}")

    print("\n=== Patients with downstream progress: MC ===")
    print(debug_patients_with_downstream_progress(mc_check, n=10).to_string(index=False))

    print("\n=== Patients with downstream progress: PROSTAD ===")
    print(debug_patients_with_downstream_progress(pros_check, n=10).to_string(index=False))

   # p = mc_check["all_patients_objects"][1]
   # print("\n=== Example MC patient data ===")
    #print(p.data)
    #print("\n=== Example MC patient events ===")
    #for e in p.events:
     #   print(e)

    #p = pros_check["all_patients_objects"][1]
    #print("\n=== Example PROSTAD patient data ===")
    #print(p.data)
    #print("\n=== Example PROSTAD patient events ===")
    #for e in p.events:
     #   print(e)

    print("\n=== Example MC patient events (patient 2) ===")
    print(debug_patient_from_events(mc_check, patient_id=2).to_string(index=False))

    print("\n=== Example PROSTAD patient events (patient 2) ===")
    print(debug_patient_from_events(pros_check, patient_id=2).to_string(index=False))

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        mc_res, prostad_res = run_both_scenarios(seed)

        all_stage_rows.append(summarise_stage_demand_weekly(mc_res, "ALL_MC_BASELINE", seed))
        all_stage_rows.append(summarise_stage_demand_weekly(prostad_res, "PROSTAD", seed))

        all_flow_rows.append(summarise_flow_counts(mc_res, "ALL_MC_BASELINE", seed))
        all_flow_rows.append(summarise_flow_counts(prostad_res, "PROSTAD", seed))

        all_branch_rows.append(summarise_branching(mc_res, "ALL_MC_BASELINE", seed))
        all_branch_rows.append(summarise_branching(prostad_res, "PROSTAD", seed))

        all_resource_rows.append(summarise_resource_pressure(mc_res, "ALL_MC_BASELINE", seed))
        all_resource_rows.append(summarise_resource_pressure(prostad_res, "PROSTAD", seed))

        all_late_rows.append(summarise_late_pipeline_patients(mc_res, "ALL_MC_BASELINE", seed, last_n_days=30))
        all_late_rows.append(summarise_late_pipeline_patients(prostad_res, "PROSTAD", seed, last_n_days=30))

        all_weekly_arrival_rows.append(stage_arrivals_to_weekly_df(mc_res, "ALL_MC_BASELINE", seed))
        all_weekly_arrival_rows.append(stage_arrivals_to_weekly_df(prostad_res, "PROSTAD", seed))


    weekly_arrivals_df = pd.concat(all_weekly_arrival_rows, ignore_index=True)
    weekly_arrivals_df.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_all_runs.csv", index=False)
    stage_df = pd.concat(all_stage_rows, ignore_index=True)
    flow_df = pd.concat(all_flow_rows, ignore_index=True)
    branch_df = pd.concat(all_branch_rows, ignore_index=True)
    resource_df = pd.concat(all_resource_rows, ignore_index=True)
    late_df = pd.concat(all_late_rows, ignore_index=True)

    stage_df.to_csv(OUTPUT_DIR / "multiseed_stage_demand_all_runs.csv", index=False)
    flow_df.to_csv(OUTPUT_DIR / "multiseed_flow_counts_all_runs.csv", index=False)
    branch_df.to_csv(OUTPUT_DIR / "multiseed_branching_all_runs.csv", index=False)
    resource_df.to_csv(OUTPUT_DIR / "multiseed_resource_pressure_all_runs.csv", index=False)
    late_df.to_csv(OUTPUT_DIR / "multiseed_late_pipeline_all_runs.csv", index=False)

    stage_summary = aggregate_stage_metrics(stage_df)
    flow_summary = aggregate_flow_counts(flow_df)
    branch_summary = aggregate_branching(branch_df)
    resource_summary = aggregate_resource_pressure(resource_df)
    late_summary = aggregate_late_pipeline(late_df)

    stage_summary.to_csv(OUTPUT_DIR / "multiseed_stage_summary.csv", index=False)
    flow_summary.to_csv(OUTPUT_DIR / "multiseed_flow_summary.csv", index=False)
    branch_summary.to_csv(OUTPUT_DIR / "multiseed_branch_summary.csv", index=False)
    resource_summary.to_csv(OUTPUT_DIR / "multiseed_resource_summary.csv", index=False)
    late_summary.to_csv(OUTPUT_DIR / "multiseed_late_pipeline_summary.csv", index=False)

    weekly_arrivals_summary = aggregate_weekly_arrivals_across_seeds(weekly_arrivals_df)
    weekly_arrivals_summary.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_summary.csv", index=False)

    # --------------------------------------------------
    # DELTA VARIABILITY CHECKS
    # --------------------------------------------------
    for event_name in ["biopsy_done", "Path_report_recieved", "Treatment_options_MDT_occured"]:
        delta_df = per_seed_flow_delta(flow_df, event_name)

        print(f"\n=== Per-seed delta summary: {event_name} ===")
        print(delta_df["delta"].describe())

        delta_df.to_csv(OUTPUT_DIR / f"per_seed_delta_{event_name}.csv", index=False)

    summary_cols = [
        "stage",

        "mean_weekly_arrivals_mean_ALL_MC_BASELINE",
        "mean_weekly_arrivals_mean_PROSTAD",
        "mean_weekly_arrivals_diff_mean",
        "mean_weekly_arrivals_diff_pct",

        "peak_weekly_arrivals_mean_ALL_MC_BASELINE",
        "peak_weekly_arrivals_mean_PROSTAD",
        "peak_weekly_arrivals_diff_mean",

        "std_weekly_arrivals_mean_ALL_MC_BASELINE",
        "std_weekly_arrivals_mean_PROSTAD",
        "std_weekly_arrivals_diff_mean",

        "mean_in_stage_mean_ALL_MC_BASELINE",
        "mean_in_stage_mean_PROSTAD",
        "mean_in_stage_diff_mean",

        "completion_ratio_mean_ALL_MC_BASELINE",
        "completion_ratio_mean_PROSTAD",
        "completion_ratio_diff_mean",

        "stage_flag",
    ]

    print("\n=== MULTI-SEED STAGE SUMMARY (MEAN OVER SEEDS) ===")
    print(stage_summary[summary_cols].round(3).to_string(index=False))

    print("\n=== MULTI-SEED FLOW SUMMARY ===")
    print(flow_summary.round(3).to_string(index=False))

    print("\n=== MULTI-SEED BRANCH SUMMARY ===")
    print(branch_summary.round(3).to_string(index=False))

    print("\n=== MULTI-SEED RESOURCE SUMMARY ===")
    print(resource_summary.round(3).to_string(index=False))

    print("\n=== MULTI-SEED LATE PIPELINE SUMMARY ===")
    print(late_summary.round(3).to_string(index=False))

    make_stage_plots(stage_summary)

    plot_weekly_arrivals_by_seed(weekly_arrivals_df)
    plot_weekly_arrivals_stage_by_seed(weekly_arrivals_df)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()