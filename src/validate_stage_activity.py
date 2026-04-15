from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from validate_scenarios import run_scenario


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator not in (0, 0.0, None) else np.nan


def safe_pct_change(new, old):
    if old in (0, 0.0, None) or pd.isna(old):
        return np.nan
    return (new - old) / old * 100


def summarise_stage_demand(result):
    rows = []

    stage_activity = result.get("stage_activity", {})

    for stage_name, metrics in stage_activity.items():
        arrivals = metrics.get("daily_arrivals", {}) or {}
        in_stage = metrics.get("daily_in_stage", {}) or {}
        completed = metrics.get("daily_completed", {}) or {}

        total_arrivals = sum(arrivals.values())
        total_completed = sum(completed.values())

        rows.append(
            {
                "stage": stage_name,
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


def classify_stage(row):
    arrival_change = row["arrival_change"]
    in_stage_change = row["in_stage_change"]
    completion_ratio = row["prostad_completion_ratio"]

    # Strong bottleneck
    if (
        arrival_change >= 5 and
        in_stage_change >= 1 and
        pd.notna(completion_ratio) and
        completion_ratio < 0.95
    ):
        return "🔴 BOTTLENECK"

    # Pressure increasing
    if (
        arrival_change > 0 and
        in_stage_change > 0.25
    ):
        return "🟠 PRESSURE ↑"

    # Improved flow
    if in_stage_change < -0.5:
        return "🟢 IMPROVED"

    # Fast / near-instant stage
    if row["prostad_mean_in_stage"] < 0.5:
        return "⚡ FAST TRACK"

    return "⚪ STABLE"


def stage_activity_to_daily_df(result, stage_name: str) -> pd.DataFrame:
    metrics = result["stage_activity"].get(stage_name, {})

    arrivals = metrics.get("daily_arrivals", {}) or {}
    in_stage = metrics.get("daily_in_stage", {}) or {}
    completed = metrics.get("daily_completed", {}) or {}

    all_dates = sorted(set(arrivals.keys()) | set(in_stage.keys()) | set(completed.keys()))

    rows = []
    for d in all_dates:
        rows.append(
            {
                "date": pd.to_datetime(d),
                "daily_arrivals": arrivals.get(d, 0),
                "daily_in_stage": in_stage.get(d, 0),
                "daily_completed": completed.get(d, 0),
            }
        )

    return pd.DataFrame(rows).sort_values("date")
def count_event_occurrences(result, event_name: str) -> int:
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return 0
    return int((event_log["event"] == event_name).sum())


def summarise_flow_counts(result, scenario_name: str) -> pd.DataFrame:
    """
    Count how many patients hit each key milestone in the event log.
    """
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
                "event": event_name,
                "count": count_event_occurrences(result, event_name),
            }
        )

    return pd.DataFrame(rows)


def summarise_branching(result, scenario_name: str) -> pd.DataFrame:
    """
    Summarise implied branch proportions from the event log.
    """
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    biopsy_done = (event_log["event"] == "biopsy_done").sum()
    pathrep_done = (event_log["event"] == "Path_report_recieved").sum()
    treat_mdt_done = (event_log["event"] == "Treatment_options_MDT_occured").sum()
    outpat_done = (event_log["event"] == "Outpatient_appointment_occured").sum()

    rows = [
        {
            "scenario": scenario_name,
            "transition": "biopsy_done -> path_report",
            "from_n": int(biopsy_done),
            "to_n": int(pathrep_done),
            "ratio": safe_div(pathrep_done, biopsy_done),
        },
        {
            "scenario": scenario_name,
            "transition": "path_report -> treat_mdt",
            "from_n": int(pathrep_done),
            "to_n": int(treat_mdt_done),
            "ratio": safe_div(treat_mdt_done, pathrep_done),
        },
        {
            "scenario": scenario_name,
            "transition": "treat_mdt -> outpatient",
            "from_n": int(treat_mdt_done),
            "to_n": int(outpat_done),
            "ratio": safe_div(outpat_done, treat_mdt_done),
        },
    ]
    return pd.DataFrame(rows)


def summarise_resource_pressure(result, scenario_name: str) -> pd.DataFrame:
    """
    Summarise DES resource queue behaviour.
    Handles daily_waits whether stored as scalars or lists per day.
    """
    rows = []

    for resource_name, metrics in result.get("resources", {}).items():
        daily_queue = metrics.get("daily_queue_len", {}) or {}
        daily_waits = metrics.get("daily_waits", {}) or {}

        queue_vals = list(daily_queue.values())

        # Flatten daily_waits safely
        flat_wait_vals = []
        for v in daily_waits.values():
            if v is None:
                continue
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                flat_wait_vals.extend([x for x in v if pd.notna(x)])
            else:
                if pd.notna(v):
                    flat_wait_vals.append(v)

        rows.append(
            {
                "scenario": scenario_name,
                "resource": resource_name,
                "mean_queue_len": float(np.mean(queue_vals)) if queue_vals else 0.0,
                "peak_queue_len": max(queue_vals) if queue_vals else 0,
                "mean_wait": float(np.mean(flat_wait_vals)) if flat_wait_vals else 0.0,
                "peak_wait": max(flat_wait_vals) if flat_wait_vals else 0,
                "n_wait_observations": len(flat_wait_vals),
            }
        )

    return pd.DataFrame(rows)


def summarise_late_pipeline_patients(result, scenario_name: str, last_n_days: int = 30) -> pd.DataFrame:
    """
    Check how many late MRI/biopsy/pathology events happen near the end of the run.
    If many do, some downstream pressure may simply not have had time to appear yet.
    """
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
                "window_start": cutoff.date(),
                "window_end": max_date.date(),
                "event": event_name,
                "count_in_last_window": int((late_df["event"] == event_name).sum()),
            }
        )

    return pd.DataFrame(rows)

def main():
    mc_res = run_scenario(
        "ALL_MC_BASELINE",
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,
    )

    prostad_res = run_scenario(
        name="PROSTAD",
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,
    )

        # --------------------------------------------------
    # FLOW / BRANCHING / RESOURCE CHECKS
    # --------------------------------------------------
    mc_flow = summarise_flow_counts(mc_res, "ALL_MC_BASELINE")
    prostad_flow = summarise_flow_counts(prostad_res, "PROSTAD")
    flow_summary = pd.concat([mc_flow, prostad_flow], ignore_index=True)
    print("\n=== Flow counts by milestone ===")
    print(flow_summary.to_string(index=False))
    flow_summary.to_csv(OUTPUT_DIR / "flow_counts_by_milestone.csv", index=False)

    flow_pivot = flow_summary.pivot(index="event", columns="scenario", values="count").reset_index()
    flow_pivot["difference"] = flow_pivot["PROSTAD"] - flow_pivot["ALL_MC_BASELINE"]

    print("\n=== Flow count comparison ===")
    print(flow_pivot.to_string(index=False))
    flow_pivot.to_csv(OUTPUT_DIR / "flow_count_comparison.csv", index=False)

   

    mc_branch = summarise_branching(mc_res, "ALL_MC_BASELINE")
    prostad_branch = summarise_branching(prostad_res, "PROSTAD")
    branch_summary = pd.concat([mc_branch, prostad_branch], ignore_index=True)
    print("\n=== Branching / continuation summary ===")
    print(branch_summary.to_string(index=False))
    branch_summary.to_csv(OUTPUT_DIR / "branching_summary.csv", index=False)

    branch_pivot = branch_summary.pivot(index="transition", columns="scenario", values="ratio").reset_index()
    branch_pivot["difference"] = branch_pivot["PROSTAD"] - branch_pivot["ALL_MC_BASELINE"]

    print("\n=== Branch continuation comparison ===")
    print(branch_pivot.to_string(index=False))
    branch_pivot.to_csv(OUTPUT_DIR / "branch_ratio_comparison.csv", index=False)

    mc_resource_pressure = summarise_resource_pressure(mc_res, "ALL_MC_BASELINE")
    prostad_resource_pressure = summarise_resource_pressure(prostad_res, "PROSTAD")
    resource_pressure_summary = pd.concat(
        [mc_resource_pressure, prostad_resource_pressure],
        ignore_index=True,
    )
    print("\n=== Resource pressure summary ===")
    print(resource_pressure_summary.to_string(index=False))
    resource_pressure_summary.to_csv(OUTPUT_DIR / "resource_pressure_summary.csv", index=False)

    mc_late = summarise_late_pipeline_patients(mc_res, "ALL_MC_BASELINE", last_n_days=30)
    prostad_late = summarise_late_pipeline_patients(prostad_res, "PROSTAD", last_n_days=30)
    late_pipeline_summary = pd.concat([mc_late, prostad_late], ignore_index=True)
    print("\n=== Late pipeline events (last 30 days) ===")
    print(late_pipeline_summary.to_string(index=False))
    late_pipeline_summary.to_csv(OUTPUT_DIR / "late_pipeline_summary.csv", index=False)

    mc_demand = summarise_stage_demand(mc_res).rename(
        columns=lambda c: f"mc_{c}" if c != "stage" else c
    )
    prostad_demand = summarise_stage_demand(prostad_res).rename(
        columns=lambda c: f"prostad_{c}" if c != "stage" else c
    )

    comparison = mc_demand.merge(prostad_demand, on="stage", how="outer")

    comparison["arrival_change"] = (
        comparison["prostad_total_arrivals"] - comparison["mc_total_arrivals"]
    )
    comparison["in_stage_change"] = (
        comparison["prostad_mean_in_stage"] - comparison["mc_mean_in_stage"]
    )
    comparison["peak_in_stage_change"] = (
        comparison["prostad_peak_in_stage"] - comparison["mc_peak_in_stage"]
    )
    comparison["completion_change"] = (
        comparison["prostad_completion_ratio"] - comparison["mc_completion_ratio"]
    )

    comparison["arrival_pct_change"] = comparison.apply(
        lambda row: safe_pct_change(row["prostad_total_arrivals"], row["mc_total_arrivals"]),
        axis=1,
    )
    comparison["in_stage_pct_change"] = comparison.apply(
        lambda row: safe_pct_change(row["prostad_mean_in_stage"], row["mc_mean_in_stage"]),
        axis=1,
    )
    comparison["completion_pct_change"] = comparison.apply(
        lambda row: safe_pct_change(row["prostad_completion_ratio"], row["mc_completion_ratio"]),
        axis=1,
    )

    comparison["stage_flag"] = comparison.apply(classify_stage, axis=1)

    summary_cols = [
        "stage",
        "mc_total_arrivals",
        "prostad_total_arrivals",
        "arrival_change",
        "arrival_pct_change",
        "mc_mean_in_stage",
        "prostad_mean_in_stage",
        "in_stage_change",
        "in_stage_pct_change",
        "mc_completion_ratio",
        "prostad_completion_ratio",
        "completion_change",
        "completion_pct_change",
        "stage_flag",
    ]

    summary_table = comparison[summary_cols].copy().round(3)

    print("\n=== MC vs PROSTAD Demand & Bottleneck Summary ===")
    print(summary_table.to_string(index=False))

    summary_table.to_csv(OUTPUT_DIR / "bottleneck_summary_mc_vs_prostad.csv", index=False)

    # -----------------------------------------
    # Plot: mean number in stage (MC vs PROSTAD)
    # -----------------------------------------
    plot_df = comparison.copy()

    stage_labels = {
        "ref_to_mri": "Referral → MRI",
        "mri_to_report": "MRI → Report",
        "report_to_biopmdt": "Report → Biopsy MDT",
        "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
        "biopsy_to_pathrep": "Biopsy → Path Report",
        "pathrep_to_treatmdt": "Path Report → Treat MDT",
        "treatmdt_to_outpat": "Treat MDT → Outpatient",
    }

    plot_df["stage_label"] = plot_df["stage"].map(stage_labels).fillna(plot_df["stage"])

    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df["mc_mean_in_stage"], width, label="ALL_MC_BASELINE")
    plt.bar(x + width / 2, plot_df["prostad_mean_in_stage"], width, label="PROSTAD")
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Bottleneck shift: mean number in stage by scenario")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottleneck_shift_mean_in_stage.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # -----------------------------------------
    # Plot: change in mean number in stage
    # -----------------------------------------
    delta_df = comparison.copy()
    delta_df["stage_label"] = delta_df["stage"].map(stage_labels).fillna(delta_df["stage"])

    plt.figure(figsize=(12, 6))
    plt.bar(delta_df["stage_label"], delta_df["in_stage_change"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("PROSTAD - MC mean in stage")
    plt.title("Change in stage occupancy under PROSTAD")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottleneck_shift_delta_in_stage.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # -----------------------------------------
    # Plot: daily in-stage occupancy over time
    # -----------------------------------------
    stage_name = "biopmdt_to_biopsy"

    mc_stage_df = stage_activity_to_daily_df(mc_res, stage_name)
    prostad_stage_df = stage_activity_to_daily_df(prostad_res, stage_name)

    plt.figure(figsize=(12, 6))
    plt.plot(mc_stage_df["date"], mc_stage_df["daily_in_stage"], label="ALL_MC_BASELINE")
    plt.plot(prostad_stage_df["date"], prostad_stage_df["daily_in_stage"], label="PROSTAD")
    plt.ylabel("Patients in stage")
    plt.xlabel("Date")
    plt.title(f"Daily occupancy over time: {stage_labels.get(stage_name, stage_name)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "biopsy_stage_occupancy_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


    


if __name__ == "__main__":
    main()