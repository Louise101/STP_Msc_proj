import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from scipy.stats import ks_2samp, chisquare

from PDF_create import build_pdfs, build_branching

# Configuration
# set where to save results

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_RESULTS = PROJECT_ROOT / "batch_results.csv"
BATCH_EVENTS = PROJECT_ROOT / "batch_events.csv"
BATCH_SIMWAITS = PROJECT_ROOT / "sim_waits.csv"

OUTPUT_DIR = PROJECT_ROOT / "verification_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Map event names (in batch_events.csv) to PDF keys (from build_pdfs())
STAGE_MAP = {
    "mri_performed": "pre_referral_to_mri",
    "mri_report_ready": "pre_mri_to_mrireport",
    "MDT_occured": "pre_mrirep_to_biopsymdt",
    "biopsy_done": "pre_biopmdt_to_biop",
    "Path_report_recieved": "pre_biop_to_pathrep",
    "Treatment_options_MDT_occured": "pre_pathrep_to_treatmdt",
    "Outpatient_appointment_occured": "pre_treatmdt_to_outpat",
}
# set OBSERVED_TOTALS to the path and define the column name.
OBSERVED_TOTALS = PROJECT_ROOT / "data" / "Pre_observed_total_times.csv"
OBSERVED_TOTAL_COL = "total_days"

def ecdf(x: np.ndarray):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y

def empirical_percentile(value: float, samples: pd.Series) -> float:
    """
    Return the empirical percentile F(value) based on the observed samples.
    Output is clipped to [eps, 1-eps] for numerical stability.
    """
    x, _ = ecdf(samples.to_numpy())

    if len(x) == 0:
        raise ValueError("Empty empirical sample array")

    # proportion of observed values <= value
    rank = np.searchsorted(x, value, side="right")
    u = rank / len(x)

    # Avoid exactly 0 or 1 because inverse normal transform needs open interval
    eps = 1e-10
    u = min(max(u, eps), 1.0 - eps)
    return u



def save_hist_overlay(sim, obs, title, filename, bins=30):
    plt.figure()
    #plt.hist(obs, bins=bins, alpha=0.5, label="Observed (empirical)")
    #plt.hist(sim, bins=bins, alpha=0.5, label="Simulated")
    plt.hist(sim, bins=30, density=True, alpha=0.5, label="Simulated")
    plt.hist(obs, bins=30, density=True, alpha=0.5, label="Observed")
    plt.xlabel("Days")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def save_ecdf_overlay(sim, obs, title, filename):
    sx, sy = ecdf(sim)
    ox, oy = ecdf(obs)

    plt.figure()
    if len(ox):
        plt.step(ox, oy, where="post", label="Observed (empirical)")
    if len(sx):
        plt.step(sx, sy, where="post", label="Simulated")
    plt.xlabel("Days")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def summary_stats(arr: np.ndarray):
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }



# Main verification

def main():
    # Load simulated outputs
    results_df = pd.read_csv(BATCH_RESULTS)
    events_df = pd.read_csv(BATCH_EVENTS)
    simwaits_df = pd.read_csv(BATCH_SIMWAITS)


    # Ensure wait_days numeric where present
    if "wait_days" in events_df.columns:
        events_df["wait_days"] = pd.to_numeric(events_df["wait_days"], errors="coerce")

    # Load empirical inputs
    pdfs = build_pdfs()
    branching = build_branching()

  
    #  Stage wait-time distribution checks (Sim vs empirical PDF vectors)
  
    stage_rows = []

    for event_name, pdf_key in STAGE_MAP.items():
        if pdf_key not in pdfs:
            print(f"[WARN] PDF key '{pdf_key}' not found for stage '{event_name}'. Skipping.")
            continue

        # Observed: the empirical vector used to create the PDF
        obs = pd.to_numeric(pdfs[pdf_key], errors="coerce").dropna().values

        # Simulated: wait_days recorded for this event
        sim = events_df.loc[events_df["event"] == event_name, "wait_days"].dropna().values

        # Stats + KS test
        obs_stats = summary_stats(obs)
        sim_stats = summary_stats(sim)

        ks_stat, ks_p = (np.nan, np.nan)
        if len(obs) > 0 and len(sim) > 0:
            ks = ks_2samp(sim, obs)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)

        stage_rows.append({
            "stage_event": event_name,
            "pdf_key": pdf_key,
            "obs_n": obs_stats.get("n", 0),
            "sim_n": sim_stats.get("n", 0),
            "obs_mean": obs_stats.get("mean", np.nan),
            "sim_mean": sim_stats.get("mean", np.nan),
            "obs_median": obs_stats.get("p50", np.nan),
            "sim_median": sim_stats.get("p50", np.nan),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
        })

        # Plots
        save_hist_overlay(
            sim, obs,
            title=f"Stage wait time: {event_name} (Sim vs Observed PDF)",
            filename=OUTPUT_DIR / f"hist_{event_name}.png",
        )
        save_ecdf_overlay(
            sim, obs,
            title=f"ECDF: {event_name} (Sim vs Observed PDF)",
            filename=OUTPUT_DIR / f"ecdf_{event_name}.png",
        )

    stage_summary = pd.DataFrame(stage_rows).sort_values("stage_event")
    stage_summary.to_csv(OUTPUT_DIR / "stage_waittime_verification.csv", index=False)

    #total pathway time verification

    overall_rows = []
    sim_total = pd.to_numeric(results_df["total_days"], errors="coerce").dropna().values
    sim_total_stats = summary_stats(sim_total)

    overall_rows.append({"metric": "sim_total_days", **sim_total_stats})

    
    obs_total_df = pd.read_csv(OBSERVED_TOTALS)
    obs_total = pd.to_numeric(obs_total_df[OBSERVED_TOTAL_COL], errors="coerce").dropna().values
    obs_total_stats = summary_stats(obs_total)
    overall_rows.append({"metric": "obs_total_days", **obs_total_stats})

        # KS test for overall totals
    ks = ks_2samp(sim_total, obs_total)
    overall_rows.append({
        "metric": "ks_total_days",
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    })

    save_hist_overlay(
        sim_total, obs_total,
        title="Overall total pathway time (Sim vs Observed)",
        filename=OUTPUT_DIR / "hist_total_days.png",
    )
    save_ecdf_overlay(
        sim_total, obs_total,
        title="Overall ECDF total pathway time (Sim vs Observed)",
        filename=OUTPUT_DIR / "ecdf_total_days.png",)
        
    

    overall_summary = pd.DataFrame(overall_rows)
    overall_summary.to_csv(OUTPUT_DIR / "overall_verification.csv", index=False)

    #comparison of obs and sim overall pathways 


    obs_mean = np.mean(obs_total)
    sim_mean = np.mean(sim_total)

    obs_median = np.median(obs_total)
    sim_median = np.median(sim_total)

# Absolute differences
    mean_diff = sim_mean - obs_mean
    median_diff = sim_median - obs_median

# Relative differences (%)
    mean_pct_diff = 100 * mean_diff / obs_mean
    median_pct_diff = 100 * median_diff / obs_median

# KS test
    ks_stat, ks_p = ks_2samp(obs_total, sim_total)

    print("Observed mean:", obs_mean)
    print("Simulated mean:", sim_mean)
    print("Mean difference:", mean_diff, f"({mean_pct_diff:.2f}%)")

    print("Observed median:", obs_median)
    print("Simulated median:", sim_median)
    print("Median difference:", median_diff, f"({median_pct_diff:.2f}%)")

    print("KS statistic:", ks_stat)
    print("KS p-value:", ks_p)


    #  Outcome distribution checks (Sim vs observed branching probabilities)

    outcome_rows = []

    # Biopsy MDT outcome: from results_df (always present in your current pathway)
    sim_biop = results_df["biopmdt_outcome"].dropna()
    sim_biop_counts = sim_biop.value_counts().sort_index()
    sim_biop_props = (sim_biop_counts / sim_biop_counts.sum()).to_dict()

    obs_biop_probs = branching.get("biopmdt_outcome", {})
    # Ensure matching keys
    all_keys = sorted(set(sim_biop_counts.index.astype(int)).union(set(obs_biop_probs.keys())))
    sim_vec = np.array([sim_biop_props.get(k, 0.0) for k in all_keys])
    obs_vec = np.array([obs_biop_probs.get(k, 0.0) for k in all_keys])

    # Chi-square requires counts, so compare observed counts to expected proportions
    # (Expected counts = obs_probs * N_sim)
    expected = obs_vec * sim_biop_counts.sum()
    # Avoid zero expected counts
    mask = expected > 0
    chi_stat, chi_p = chisquare(f_obs=sim_biop_counts.reindex(all_keys, fill_value=0).values[mask],
                               f_exp=expected[mask])

    outcome_rows.append({
        "outcome_set": "biopmdt_outcome",
        "keys": str(all_keys),
        "chi2_stat": float(chi_stat),
        "chi2_pvalue": float(chi_p),
        "sim_props": str(sim_biop_props),
        "obs_probs": str(obs_biop_probs),
    })

    # Path report outcome: only among those who reached pathology (non-null)
    sim_path = results_df["pathrep_outcome"].dropna()
    if len(sim_path) > 0:
        sim_path_counts = sim_path.value_counts().sort_index()
        sim_path_props = (sim_path_counts / sim_path_counts.sum()).to_dict()

        obs_path_probs = branching.get("pathrep_outcome", {})
        all_keys = sorted(set(sim_path_counts.index.astype(int)).union(set(obs_path_probs.keys())))
        sim_vec = np.array([sim_path_props.get(k, 0.0) for k in all_keys])
        obs_vec = np.array([obs_path_probs.get(k, 0.0) for k in all_keys])

        expected = obs_vec * sim_path_counts.sum()
        mask = expected > 0
        chi_stat, chi_p = chisquare(f_obs=sim_path_counts.reindex(all_keys, fill_value=0).values[mask],
                                   f_exp=expected[mask])

        outcome_rows.append({
            "outcome_set": "pathrep_outcome",
            "keys": str(all_keys),
            "chi2_stat": float(chi_stat),
            "chi2_pvalue": float(chi_p),
            "sim_props": str(sim_path_props),
            "obs_probs": str(obs_path_probs),
        })
    else:
        outcome_rows.append({
            "outcome_set": "pathrep_outcome",
            "note": "No simulated patients reached pathology stage in this run."
        })

    outcome_summary = pd.DataFrame(outcome_rows)
    outcome_summary.to_csv(OUTPUT_DIR / "outcome_verification.csv", index=False)

    print(f"\nVerification outputs saved to: {OUTPUT_DIR}")
    print("Created:")
    print("- stage_waittime_verification.csv")
    print("- overall_verification.csv")
    print("- outcome_verification.csv")
    print("- hist_*.png and ecdf_*.png per stage (plus totals)")

    print("\nSIMULATED CORRELATION CHECK")

    subset = simwaits_df[["wait_ref_to_mri", "wait_mri_to_report"]].dropna()

    print("\nSIMULATED FIRST-TWO-STAGE CORRELATION CHECK")

    print("\nSpearman:")
    print(subset.corr(method="spearman"))

    print("\nKendall:")
    print(subset.corr(method="kendall"))



if __name__ == "__main__":
    main()

