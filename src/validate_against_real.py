from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

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


# --------------------------------------------------
# SUMMARY STATS
# --------------------------------------------------
def summarise_distribution(df: pd.DataFrame, label: str) -> dict:
    x = df["total_days"].dropna()

    return {
        "label": label,
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "pct_within_62": float((x <= 62).mean() * 100),
    }


def compare_sim_vs_real(sim_df: pd.DataFrame, real_df: pd.DataFrame, label: str) -> pd.DataFrame:
    sim_stats = summarise_distribution(sim_df, f"{label}_SIM")
    real_stats = summarise_distribution(real_df, f"{label}_REAL")

    ks_stat, ks_p = ks_2samp(sim_df["total_days"], real_df["total_days"])

    return pd.DataFrame([{
        "scenario": label,
        "n_sim": sim_stats["n"],
        "n_real": real_stats["n"],
        "mean_sim": sim_stats["mean"],
        "mean_real": real_stats["mean"],
        "median_sim": sim_stats["median"],
        "median_real": real_stats["median"],
        "p90_sim": sim_stats["p90"],
        "p90_real": real_stats["p90"],
        "pct62_sim": sim_stats["pct_within_62"],
        "pct62_real": real_stats["pct_within_62"],
        "mean_diff": sim_stats["mean"] - real_stats["mean"],
        "median_diff": sim_stats["median"] - real_stats["median"],
        "pct62_diff": sim_stats["pct_within_62"] - real_stats["pct_within_62"],
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }])


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
def plot_distribution_comparison(sim_df, real_df, title, save_path=None):
    plt.figure(figsize=(10, 6))

    sim_df["total_days"].plot(kind="kde", label="Simulated")
    real_df["total_days"].plot(kind="kde", label="Real")

    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlim(left=0)
    plt.xlabel("Total pathway days")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_boxplot_comparison(sim_df, real_df, title, save_path=None):
    plt.figure(figsize=(6, 6))

    data = [sim_df["total_days"], real_df["total_days"]]
    plt.boxplot(data, labels=["Sim", "Real"])

    plt.ylabel("Total pathway days")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# --------------------------------------------------
# MASTER VALIDATION FUNCTION
# --------------------------------------------------
def run_validation(
    baseline_result,
    mixed_result,
    real_pre_df,
    real_pros_df,
    output_dir=None,
):
    sim_baseline = extract_sim_pathway_lengths(baseline_result, "BASELINE")
    sim_mix = extract_sim_pathway_lengths(mixed_result, "OBS_MIX")

    baseline_comp = compare_sim_vs_real(sim_baseline, real_pre_df, "BASELINE")
    mix_comp = compare_sim_vs_real(sim_mix, real_pros_df, "OBS_MIX")

    results = pd.concat([baseline_comp, mix_comp], ignore_index=True)

    print("\n=== VALIDATION RESULTS ===")
    print(results.round(2).to_string(index=False))

    plot_distribution_comparison(
        sim_baseline,
        real_pre_df,
        "Baseline: Sim vs Real (Pre-PROSTAD, full pathway only)",
        save_path=None if not output_dir else f"{output_dir}/baseline_kde.png",
    )

    plot_distribution_comparison(
        sim_mix,
        real_pros_df,
        "Observed Mix: Sim vs Real (PROSTAD period, full pathway only)",
        save_path=None if not output_dir else f"{output_dir}/mix_kde.png",
    )

    plot_boxplot_comparison(
        sim_baseline,
        real_pre_df,
        "Baseline: Boxplot comparison",
        save_path=None if not output_dir else f"{output_dir}/baseline_boxplot.png",
    )

    plot_boxplot_comparison(
        sim_mix,
        real_pros_df,
        "Observed Mix: Boxplot comparison",
        save_path=None if not output_dir else f"{output_dir}/mix_boxplot.png",
    )

    return results