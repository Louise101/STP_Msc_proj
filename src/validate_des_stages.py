from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from des_engine import run_day_loop_with_stage_engine
from scenarios import build_scenario_config


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# Real data loaders
# =========================================================

def load_real_stage_waits() -> dict[str, pd.Series]:
    """
    Load observed pre-PROSTAD stage waits from CSVs.
    """
    stage_files = {
        "ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI"),
        "mri_to_report": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported"),
        "report_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT"),
        "biopmdt_to_biopsy": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy"),
        "biopsy_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report"),
        "pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)"),
        "treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt"),
    }

    out = {}

    for stage_name, (filename, start_col, end_col) in stage_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Warning: missing real-data file for {stage_name}: {path}")
            continue

        df = pd.read_csv(path)
        df[start_col] = pd.to_datetime(df[start_col], dayfirst=True, errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], dayfirst=True, errors="coerce")

        df = df.dropna(subset=[start_col, end_col]).copy()
        waits = (df[end_col] - df[start_col]).dt.days.dropna().astype(int)

        out[stage_name] = waits

    return out


# =========================================================
# Simulation helpers
# =========================================================

def run_des_scenario(
    scenario_name: str = "DES_PRE_PROSTAD",
    start_date: date = date(2026, 1, 5),
    n_days: int = 365,
    lam_per_workday: float = 2.0,
    seed: int = 1234,
) -> Dict[str, Any]:
    cfg = build_scenario_config(
        name=scenario_name,
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
    )
    cfg.seed = seed
    return run_day_loop_with_stage_engine(cfg)


def patient_results_to_event_df(patient_results) -> pd.DataFrame:
    rows = []

    for pid, (events, total_days) in enumerate(patient_results, start=1):
        row = {
            "patient_id": pid,
            "total_days": total_days,
        }

        for e in events:
            event_name = e.get("event")
            event_date = e.get("date")
            wait_days = e.get("wait_days", None)
            outcome = e.get("outcome", None)

            if event_name is not None:
                row[f"date_{event_name}"] = event_date
            if wait_days is not None:
                row[f"wait_at_{event_name}"] = wait_days
            if outcome is not None:
                row[f"outcome_{event_name}"] = outcome

        rows.append(row)

    return pd.DataFrame(rows)


def derive_sim_stage_waits(sim_events_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Derive stage waits from event dates.
    """
    df = sim_events_df.copy()

    date_cols = [c for c in df.columns if c.startswith("date_")]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    waits = {}

    def diff(col_end, col_start):
        if col_end in df.columns and col_start in df.columns:
            return (df[col_end] - df[col_start]).dt.days.dropna().astype(int)
        return pd.Series(dtype=int)

    waits["ref_to_mri"] = diff("date_mri_performed", "date_referral_recieved")
    waits["mri_to_report"] = diff("date_mri_report_ready", "date_mri_performed")
    waits["report_to_biopmdt"] = diff("date_MDT_occured", "date_mri_report_ready")
    waits["biopmdt_to_biopsy"] = diff("date_biopsy_done", "date_MDT_occured")
    waits["biopsy_to_pathrep"] = diff("date_Path_report_recieved", "date_biopsy_done")
    waits["pathrep_to_treatmdt"] = diff("date_Treatment_options_MDT_occured", "date_Path_report_recieved")
    waits["treatmdt_to_outpat"] = diff("date_Outpatient_appointment_occured", "date_Treatment_options_MDT_occured")

    return waits


# =========================================================
# Comparison summaries
# =========================================================

def summarise_stage(obs: pd.Series, sim: pd.Series, stage_name: str) -> dict:
    ks_stat, ks_p = ks_2samp(obs, sim)

    return {
        "stage": stage_name,
        "obs_n": int(obs.shape[0]),
        "sim_n": int(sim.shape[0]),
        "obs_mean": float(obs.mean()),
        "sim_mean": float(sim.mean()),
        "obs_median": float(obs.median()),
        "sim_median": float(sim.median()),
        "obs_std": float(obs.std(ddof=1)) if obs.shape[0] > 1 else 0.0,
        "sim_std": float(sim.std(ddof=1)) if sim.shape[0] > 1 else 0.0,
        "obs_min": int(obs.min()),
        "sim_min": int(sim.min()),
        "obs_max": int(obs.max()),
        "sim_max": int(sim.max()),
        "mean_diff": float(sim.mean() - obs.mean()),
        "median_diff": float(sim.median() - obs.median()),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def compare_stage_waits(
    real_waits: dict[str, pd.Series],
    sim_waits: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []

    for stage_name, obs in real_waits.items():
        sim = sim_waits.get(stage_name, pd.Series(dtype=int))

        if len(obs) == 0 or len(sim) == 0:
            rows.append({
                "stage": stage_name,
                "obs_n": int(len(obs)),
                "sim_n": int(len(sim)),
                "obs_mean": float(obs.mean()) if len(obs) else np.nan,
                "sim_mean": float(sim.mean()) if len(sim) else np.nan,
                "obs_median": float(obs.median()) if len(obs) else np.nan,
                "sim_median": float(sim.median()) if len(sim) else np.nan,
                "obs_std": float(obs.std(ddof=1)) if len(obs) > 1 else np.nan,
                "sim_std": float(sim.std(ddof=1)) if len(sim) > 1 else np.nan,
                "obs_min": int(obs.min()) if len(obs) else np.nan,
                "sim_min": int(sim.min()) if len(sim) else np.nan,
                "obs_max": int(obs.max()) if len(obs) else np.nan,
                "sim_max": int(sim.max()) if len(sim) else np.nan,
                "mean_diff": np.nan,
                "median_diff": np.nan,
                "ks_stat": np.nan,
                "ks_p": np.nan,
            })
            continue

        rows.append(summarise_stage(obs, sim, stage_name))

    return pd.DataFrame(rows)


def add_interpretation_flags(df: pd.DataFrame, mean_tol: float = 2.0, median_tol: float = 2.0) -> pd.DataFrame:
    out = df.copy()

    def classify(row):
        if pd.isna(row["sim_n"]) or row["sim_n"] == 0:
            return "NO_SIM_DATA"
        if abs(row["mean_diff"]) <= mean_tol and abs(row["median_diff"]) <= median_tol:
            return "CLOSE_MATCH"
        if row["mean_diff"] > mean_tol:
            return "SIM_TOO_SLOW"
        if row["mean_diff"] < -mean_tol:
            return "SIM_TOO_FAST"
        return "CHECK"

    out["match_flag"] = out.apply(classify, axis=1)
    return out


# =========================================================
# Optional DES queue summary
# =========================================================

def summarise_des_resources(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    for resource_name, resource_info in result["resources"].items():
        daily_waits = resource_info.get("daily_waits", {})
        waits = [w for day_waits in daily_waits.values() for w in day_waits]

        rows.append({
            "resource": resource_name,
            "days_with_activity": sum(1 for v in resource_info.get("daily_started", {}).values() if v > 0),
            "total_started": int(sum(resource_info.get("daily_started", {}).values())),
            "final_queue_len": np.nan,  # final queue already in summary_stats
            "mean_des_wait": float(np.mean(waits)) if waits else np.nan,
            "median_des_wait": float(np.median(waits)) if waits else np.nan,
            "max_des_wait": int(np.max(waits)) if waits else np.nan,
        })

    df = pd.DataFrame(rows)

    final_q = result["summary_stats"]["final_queue_length_by_resource"]
    if "resource" in df.columns:
        df["final_queue_len"] = df["resource"].map(final_q)

    return df


# =========================================================
# Main
# =========================================================

def main():
    scenario_name = "DES_PRE_PROSTAD"
    n_days = 365
    lam_per_workday = 0.586
    seed = 1234

    print(f"Running stage-by-stage DES validation for scenario: {scenario_name}")

    real_waits = load_real_stage_waits()
    result = run_des_scenario(
        scenario_name=scenario_name,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        seed=seed,
    )

    sim_events_df = patient_results_to_event_df(result["patient_results"])
    sim_waits = derive_sim_stage_waits(sim_events_df)

    comparison = compare_stage_waits(real_waits, sim_waits)
    comparison = add_interpretation_flags(comparison, mean_tol=2.0, median_tol=2.0)

    print("\n=== DES vs real stage-by-stage comparison ===")
    print(comparison.to_string(index=False))

    resource_summary = summarise_des_resources(result)
    print("\n=== DES resource summary ===")
    print(resource_summary.to_string(index=False))

    print("\n=== Overall summary stats ===")
    print(result["summary_stats"])

    comparison_path = OUTPUT_DIR / "des_stage_by_stage_validation.csv"
    resource_path = OUTPUT_DIR / "des_resource_summary.csv"

    comparison.to_csv(comparison_path, index=False)
    resource_summary.to_csv(resource_path, index=False)

    print("\nSaved:")
    print(comparison_path)
    print(resource_path)


if __name__ == "__main__":
    main()