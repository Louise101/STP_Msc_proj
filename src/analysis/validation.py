from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from analysis.summaries import extract_full_pathway_lengths

def build_real_pathway_csvs(
    pre_ref_file: str,
    pre_outpat_file: str,
    pros_ref_file: str,
    pros_outpat_file: str,
    out_pre_file: str,
    out_pros_file: str,
):
    # PRE
    pre_ref = pd.read_csv(pre_ref_file).copy()
    pre_out = pd.read_csv(pre_outpat_file).copy()

    pre_ref = pre_ref.rename(columns={
        "Subject number": "patient_id",
        "Date of referral to pathway": "referral_date",
    })
    pre_out = pre_out.rename(columns={
        "Subject number": "patient_id",
        "Date of outpat appt": "outpatient_date",
    })

    pre_ref["patient_id"] = pre_ref["patient_id"].astype(str)
    pre_out["patient_id"] = pre_out["patient_id"].astype(str)

    pre_ref["referral_date"] = pd.to_datetime(pre_ref["referral_date"], dayfirst=True, errors="coerce")
    pre_out["outpatient_date"] = pd.to_datetime(pre_out["outpatient_date"], dayfirst=True, errors="coerce")

    pre = pre_ref[["patient_id", "referral_date"]].merge(
        pre_out[["patient_id", "outpatient_date"]],
        on="patient_id",
        how="inner",
    )
    pre["total_days"] = (pre["outpatient_date"] - pre["referral_date"]).dt.days
    pre = pre[(pre["total_days"].notna()) & (pre["total_days"] >= 0)].copy()
    pre.to_csv(out_pre_file, index=False)

    # PROSTAD
    pros_ref = pd.read_csv(pros_ref_file).copy()
    pros_out = pd.read_csv(pros_outpat_file).copy()

    pros_ref = pros_ref.rename(columns={
        "Subject number": "patient_id",
        "Date of referral to pathway": "referral_date",
    })
    pros_out = pros_out.rename(columns={
        "Subject number": "patient_id",
        "Date of OPD appt": "outpatient_date",
    })

    pros_ref["patient_id"] = pros_ref["patient_id"].astype(str)
    pros_out["patient_id"] = pros_out["patient_id"].astype(str)

    pros_ref["referral_date"] = pd.to_datetime(pros_ref["referral_date"], format="%m/%d/%y", errors="coerce")
    pros_out["outpatient_date"] = pd.to_datetime(pros_out["outpatient_date"], format="%m/%d/%y", errors="coerce")

    pros = pros_ref[["patient_id", "referral_date"]].merge(
        pros_out[["patient_id", "outpatient_date"]],
        on="patient_id",
        how="inner",
    )
    pros["total_days"] = (pros["outpatient_date"] - pros["referral_date"]).dt.days
    pros = pros[(pros["total_days"].notna()) & (pros["total_days"] >= 0)].copy()
    pros.to_csv(out_pros_file, index=False)


# --------------------------------------------------
# LOAD REAL DATA
# --------------------------------------------------
def load_real_pathway_data(pre_path: str, pros_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = pd.read_csv(pre_path).copy()
    pros = pd.read_csv(pros_path).copy()

    # Adjust these if your actual CSV column names differ
    pre["referral_date"] = pd.to_datetime(pre["referral_date"], dayfirst=True, errors="coerce")
    pre["outpatient_date"] = pd.to_datetime(pre["outpatient_date"], dayfirst=True, errors="coerce")

    pros["referral_date"] = pd.to_datetime(pros["referral_date"], dayfirst=True, errors="coerce")
    pros["outpatient_date"] = pd.to_datetime(pros["outpatient_date"], dayfirst=True, errors="coerce")

    for df in [pre, pros]:
        df["total_days"] = (df["outpatient_date"] - df["referral_date"]).dt.days
        df.dropna(subset=["total_days"], inplace=True)
        df = df[df["total_days"] >= 0]

    pre = pre[(pre["total_days"].notna()) & (pre["total_days"] >= 0)].copy()
    pros = pros[(pros["total_days"].notna()) & (pros["total_days"] >= 0)].copy()

    return pre, pros


# --------------------------------------------------
# SIM DATA EXTRACTION
# --------------------------------------------------
def extract_sim_pathway_lengths(result: dict, scenario_name: str) -> pd.DataFrame:
    """
    Extract full-pathway completed patients only.
    """
    rows = []

    for patient in result["completed_patients_objects"]:
        events = patient.events
        if not events:
            continue

        event_names = {e["event"] for e in events}
        if "Outpatient_appointment_occured" not in event_names:
            continue

        total_days = (patient.current_date - patient.start_date).days
        if total_days < 0:
            continue

        rows.append({
            "scenario": scenario_name,
            "patient_id": patient.patient_id,
            "total_days": total_days,
        })

    return pd.DataFrame(rows)



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
