from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from des_engine import run_day_loop_with_stage_engine
from scenarios import build_scenario_config


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Real data loaders
# --------------------------------------------------
def load_real_stage_waits() -> dict[str, pd.Series]:
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
        df = pd.read_csv(DATA_DIR / filename)

        df[start_col] = pd.to_datetime(df[start_col], dayfirst=True, errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], dayfirst=True, errors="coerce")

        df = df.dropna(subset=[start_col, end_col]).copy()
        waits = (df[end_col] - df[start_col]).dt.days
        out[stage_name] = waits.dropna().astype(int)

    return out


# --------------------------------------------------
# Simulation helpers
# --------------------------------------------------
def run_all_mc_baseline(n_days: int = 365, lam_per_workday: float = 2.0, seed: int = 1234):
    cfg = build_scenario_config(
        name="ALL_MC_BASELINE",
        start_date=date(2026, 1, 5),
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
            outcome = e.get("outcome", None)

            if event_name is not None:
                row[f"date_{event_name}"] = event_date
            if outcome is not None:
                row[f"outcome_{event_name}"] = outcome

        rows.append(row)

    return pd.DataFrame(rows)


def derive_sim_stage_waits(sim_events_df: pd.DataFrame) -> dict[str, pd.Series]:
    df = sim_events_df.copy()

    date_cols = [c for c in df.columns if c.startswith("date_")]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    waits = {}

    def diff(col_end, col_start):
        return (df[col_end] - df[col_start]).dt.days.dropna().astype(int)

    if {"date_mri_performed", "date_referral_recieved"}.issubset(df.columns):
        waits["ref_to_mri"] = diff("date_mri_performed", "date_referral_recieved")

    if {"date_mri_report_ready", "date_mri_performed"}.issubset(df.columns):
        waits["mri_to_report"] = diff("date_mri_report_ready", "date_mri_performed")

    if {"date_MDT_occured", "date_mri_report_ready"}.issubset(df.columns):
        waits["report_to_biopmdt"] = diff("date_MDT_occured", "date_mri_report_ready")

    if {"date_biopsy_done", "date_MDT_occured"}.issubset(df.columns):
        waits["biopmdt_to_biopsy"] = diff("date_biopsy_done", "date_MDT_occured")

    if {"date_Path_report_recieved", "date_biopsy_done"}.issubset(df.columns):
        waits["biopsy_to_pathrep"] = diff("date_Path_report_recieved", "date_biopsy_done")

    if {"date_Treatment_options_MDT_occured", "date_Path_report_recieved"}.issubset(df.columns):
        waits["pathrep_to_treatmdt"] = diff("date_Treatment_options_MDT_occured", "date_Path_report_recieved")

    if {"date_Outpatient_appointment_occured", "date_Treatment_options_MDT_occured"}.issubset(df.columns):
        waits["treatmdt_to_outpat"] = diff("date_Outpatient_appointment_occured", "date_Treatment_options_MDT_occured")

    return waits


# --------------------------------------------------
# Comparison summaries
# --------------------------------------------------
def summarise_series(name: str, obs: pd.Series, sim: pd.Series) -> dict:
    ks_stat, ks_p = ks_2samp(obs, sim)

    return {
        "stage": name,
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


def compare_stage_waits(real_waits: dict[str, pd.Series], sim_waits: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []

    for stage_name in real_waits:
        if stage_name in sim_waits and len(sim_waits[stage_name]) > 0:
            rows.append(summarise_series(stage_name, real_waits[stage_name], sim_waits[stage_name]))

    return pd.DataFrame(rows)


# --------------------------------------------------
# Total pathway subgroup comparison
# --------------------------------------------------
def classify_sim_patients(sim_events_df: pd.DataFrame) -> pd.DataFrame:
    df = sim_events_df.copy()

    has_biopsy = df["date_biopsy_done"].notna() if "date_biopsy_done" in df.columns else False
    has_path = df["date_Path_report_recieved"].notna() if "date_Path_report_recieved" in df.columns else False
    has_treat = df["date_Treatment_options_MDT_occured"].notna() if "date_Treatment_options_MDT_occured" in df.columns else False
    has_outpat = df["date_Outpatient_appointment_occured"].notna() if "date_Outpatient_appointment_occured" in df.columns else False

    subgroup = np.where(
        has_outpat,
        "full_pathway_to_outpatient",
        np.where(
            has_biopsy & has_path & (~has_outpat),
            "biopsy_pathology_only",
            np.where(~has_biopsy, "ended_before_biopsy", "other")
        ),
    )

    df["pathway_subgroup"] = subgroup
    return df


def summarise_total_pathway_by_group(sim_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sim_df.groupby("pathway_subgroup")["total_days"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "count": "sim_n",
            "mean": "sim_mean",
            "median": "sim_median",
            "std": "sim_std",
            "min": "sim_min",
            "max": "sim_max",
        })
    )
    return grouped


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Running ALL_MC baseline against real pre-PROSTAD data...")

    real_waits = load_real_stage_waits()

    result = run_all_mc_baseline(n_days=365, lam_per_workday=2.0, seed=1234)
    sim_events_df = patient_results_to_event_df(result["patient_results"])
    sim_waits = derive_sim_stage_waits(sim_events_df)

    stage_comparison = compare_stage_waits(real_waits, sim_waits)
    print("\n=== Stage wait comparison: real vs ALL_MC baseline ===")
    print(stage_comparison.to_string(index=False))

    stage_comparison.to_csv(OUTPUT_DIR / "all_mc_stage_wait_validation.csv", index=False)

    sim_classified = classify_sim_patients(sim_events_df)
    sim_total_summary = summarise_total_pathway_by_group(sim_classified)

    print("\n=== Simulated total pathway summary by subgroup (ALL_MC baseline) ===")
    print(sim_total_summary.to_string(index=False))

    sim_total_summary.to_csv(OUTPUT_DIR / "all_mc_total_pathway_summary.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "all_mc_stage_wait_validation.csv")
    print(OUTPUT_DIR / "all_mc_total_pathway_summary.csv")


if __name__ == "__main__":
    main()