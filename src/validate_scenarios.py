from __future__ import annotations

from datetime import date
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from des_engine import run_day_loop_with_stage_engine
from scenarios import build_scenario_config


# =========================================================
# Helpers
# =========================================================

def patient_results_to_dataframe(patient_results) -> pd.DataFrame:
    rows = []

    for events, total_days in patient_results:
        row = {"total_days": total_days}

        event_dates = {}
        event_outcomes = {}

        for e in events:
            event_name = e.get("event")
            event_date = e.get("date")
            outcome = e.get("outcome", None)
            wait_days = e.get("wait_days", None)

            if event_name is not None and event_date is not None:
                event_dates[event_name] = event_date

            if outcome is not None:
                event_outcomes[event_name] = outcome

            if wait_days is not None:
                row[f"wait_at_{event_name}"] = wait_days

        row.update({f"date_{k}": v for k, v in event_dates.items()})
        row.update({f"outcome_{k}": v for k, v in event_outcomes.items()})

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_results(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    patient_results = result["patient_results"]
    total_completed = result["summary_stats"]["total_patients_completed"]

    total_days_list = [x[1] for x in patient_results]
    mean_total_days = float(np.mean(total_days_list)) if total_days_list else None
    median_total_days = float(np.median(total_days_list)) if total_days_list else None

    mri_waits = []
    biopsy_waits = []

    for waits in result["resources"]["MRI"]["daily_waits"].values():
        mri_waits.extend(waits)

    for waits in result["resources"]["Biopsy"]["daily_waits"].values():
        biopsy_waits.extend(waits)

    return {
        "scenario": name,
        "completed": total_completed,
        "mean_total_days": mean_total_days,
        "median_total_days": median_total_days,
        "final_mri_queue": result["summary_stats"]["final_queue_length_by_resource"]["MRI"],
        "final_biopsy_queue": result["summary_stats"]["final_queue_length_by_resource"]["Biopsy"],
        "mean_mri_des_wait": float(np.mean(mri_waits)) if mri_waits else None,
        "mean_biopsy_des_wait": float(np.mean(biopsy_waits)) if biopsy_waits else None,
    }


def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(summary_rows)
    print("\n=== Scenario summary ===")
    print(df.to_string(index=False))


def extract_stage_waits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"date_referral_recieved", "date_mri_performed"}.issubset(out.columns):
        out["obs_ref_to_mri"] = (
            pd.to_datetime(out["date_mri_performed"]) -
            pd.to_datetime(out["date_referral_recieved"])
        ).dt.days

    if {"date_mri_performed", "date_mri_report_ready"}.issubset(out.columns):
        out["obs_mri_to_report"] = (
            pd.to_datetime(out["date_mri_report_ready"]) -
            pd.to_datetime(out["date_mri_performed"])
        ).dt.days

    if {"date_mri_report_ready", "date_MDT_occured"}.issubset(out.columns):
        out["obs_report_to_biopmdt"] = (
            pd.to_datetime(out["date_MDT_occured"]) -
            pd.to_datetime(out["date_mri_report_ready"])
        ).dt.days

    return out


# =========================================================
# Validation helpers
# =========================================================

def run_scenario(name: str, start_date: date, n_days: int, lam_per_workday: float, seed: int = 1234):
    cfg = build_scenario_config(
        name=name,
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
    )
    cfg.seed = seed
    return run_day_loop_with_stage_engine(cfg)


# =========================================================
# Validation checks
# =========================================================

def validate_main_scenarios() -> None:
    print("\n=== ALL_MC vs PROSTAD comparison ===")

    scenario_names = [
        "ALL_MC_BASELINE",
        "PROSTAD",
    ]

    summaries = []
    for name in scenario_names:
        result = run_scenario(
            name=name,
            start_date=date(2026, 1, 5),
            n_days=365,
            lam_per_workday=0.586,
            seed=123,
        )
        summaries.append(summarise_results(name, result))

    print_summary_table(summaries)

    print("\nInterpretation guide:")
    print("- PROSTAD should reduce pathway time compared with ALL_MC baseline if improved rules/capacity are represented")
    print("- ALL_MC represents the empirical historical baseline")


def validate_fixed_rule_scenario() -> None:
    print("\n=== PROSTAD fixed-rule validation ===")

    result = run_scenario(
        name="PROSTAD",
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,
    )

    df = patient_results_to_dataframe(result["patient_results"])
    df = extract_stage_waits(df)

    if "obs_mri_to_report" in df.columns:
        print("Observed MRI->report waits:")
        print(df["obs_mri_to_report"].dropna().describe())

    if "obs_report_to_biopmdt" in df.columns:
        print("Observed report->biopsy MDT waits:")
        print(df["obs_report_to_biopmdt"].dropna().describe())

    print("\nExpected under PROSTAD fixed rules:")
    print("- mri_to_report should be 1 day")
    print("- report_to_biopmdt should be 0 days")


def validate_event_ordering() -> None:
    print("\n=== PROSTAD event ordering validation ===")

    result = run_scenario(
        name="PROSTAD",
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,
    )

    bad_order_count = 0

    for events, total_days in result["patient_results"]:
        dates = [e["date"] for e in events if "date" in e]
        if any(dates[i] > dates[i + 1] for i in range(len(dates) - 1)):
            bad_order_count += 1

    print(f"Patients with non-monotonic event dates: {bad_order_count}")
    print("Expected: 0")


# =========================================================
# Main
# =========================================================

def main():
    print("Running simplified ALL_MC vs PROSTAD validation checks...")

    validate_main_scenarios()
    validate_fixed_rule_scenario()
    validate_event_ordering()

    print("\nDone.")


if __name__ == "__main__":
    main()