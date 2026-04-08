from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Convert raw patient_results into event log
# --------------------------------------------------
def patient_results_to_event_log(patient_results) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert engine patient_results into:
    1) long event log dataframe
    2) raw patient container dataframe

    Observed structure:
      each item in patient_results is like [event_list, total_value]
    where event_list is a list of dicts.
    """
    raw_df = pd.DataFrame(patient_results)

    if raw_df.shape[1] < 1:
        raise ValueError("patient_results is empty or malformed")

    event_rows = []
    for i, row in raw_df.iterrows():
        events = row.iloc[0]

        total_val = row.iloc[1] if raw_df.shape[1] > 1 else np.nan

        if not isinstance(events, list):
            continue

        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev2 = ev.copy()
            ev2["raw_total_value"] = total_val
            event_rows.append(ev2)

    events_df = pd.DataFrame(event_rows)

    if events_df.empty:
        raise ValueError("No event rows could be extracted from patient_results")

    if "date" in events_df.columns:
        events_df["date"] = pd.to_datetime(events_df["date"])

    return events_df, raw_df


# --------------------------------------------------
# Build patient-level summary from event log
# --------------------------------------------------
def build_patient_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long event log into patient-level summary table.
    """
    df = events_df.copy()

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found. Columns are: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])

    # referral
    referral = (
        df[df["event"] == "referral_recieved"]
        .groupby("patient_id")["date"]
        .min()
        .rename("referral_date")
    )

    # MRI
    mri = (
        df[df["event"] == "mri_performed"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mri_date")
    )

    # MRI report
    mri_report = (
        df[df["event"] == "mri_report_ready"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mri_report_date")
    )

    # MDT
    mdt = (
        df[df["event"] == "MDT_occured"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mdt_date")
    )

    # biopsy
    biopsy = (
        df[df["event"] == "biopsy_done"]
        .groupby("patient_id")["date"]
        .min()
        .rename("biopsy_date")
    )

    # pathology outcome date
    pathrep = (
        df[df["event"] == "Path_report_outcome"]
        .groupby("patient_id")["date"]
        .min()
        .rename("pathrep_date")
    )

    # pathway end / exit
    pathway_end = (
        df[df["event"].isin(["pathway_end", "pathway_exit"])]
        .groupby("patient_id")["date"]
        .min()
        .rename("pathway_end_date")
    )

    patient = pd.concat(
        [referral, mri, mri_report, mdt, biopsy, pathrep, pathway_end],
        axis=1
    ).reset_index()

    # stage waits from event log
    stage_waits = (
        df.dropna(subset=["wait_days"])
        .pivot_table(
            index="patient_id",
            columns="event",
            values="wait_days",
            aggfunc="first"
        )
        .rename(columns={
            "mri_performed": "wait_ref_to_mri",
            "mri_report_ready": "wait_mri_to_report",
            "MDT_occured": "wait_report_to_biopmdt",
            "biopsy_done": "wait_biopmdt_to_biopsy",
        })
        .reset_index()
    )

    patient = patient.merge(stage_waits, on="patient_id", how="left")

    # pathology outcome
    path_outcome = (
        df[df["event"] == "Path_report_outcome"]
        .groupby("patient_id")["outcome"]
        .first()
        .rename("pathology_outcome")
        .reset_index()
    )

    patient = patient.merge(path_outcome, on="patient_id", how="left")

    # total times
    patient["total_days_ref_to_pathrep"] = (
        patient["pathrep_date"] - patient["referral_date"]
    ).dt.days

    patient["total_days_ref_to_outpat"] = (
        patient["pathway_end_date"] - patient["referral_date"]
    ).dt.days

    patient["met_62d_to_pathrep"] = patient["total_days_ref_to_pathrep"] <= 62

    return patient


# --------------------------------------------------
# Stage-level summaries from event log
# --------------------------------------------------
def extract_stage_wait_means(events_df: pd.DataFrame) -> dict:
    df = events_df.copy()

    stage_map = {
        "mri_performed": "wait_ref_to_mri",
        "mri_report_ready": "wait_mri_to_report",
        "MDT_occured": "wait_report_to_biopmdt",
        "biopsy_done": "wait_biopmdt_to_biopsy",
    }

    out = {}
    for event_name, metric_name in stage_map.items():
        vals = df.loc[df["event"] == event_name, "wait_days"].dropna()
        out[metric_name] = vals.mean() if len(vals) > 0 else np.nan

    return out


# --------------------------------------------------
# Run one simulation replicate
# --------------------------------------------------
def run_one_rep(seed: int, n_days: int = 365, lam_per_workday: float = 2.0) -> dict:
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        mri_capacity_by_weekday={2: 4},
        biopsy_capacity_by_weekday={3: 1, 4: 1},
        seed=seed,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_repot": WAIT_MODE_DES,   # keep engine spelling if that is what des_engine expects
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
            "biopsy_to_pathrep": WAIT_MODE_MC,
        }
    )

    result = run_day_loop_with_stage_engine(cfg)

    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result)}")

    if "patient_results" not in result:
        raise ValueError(f"'patient_results' not found. Keys are: {list(result.keys())}")

    events_df, raw_df = patient_results_to_event_log(result["patient_results"])
    patient_df = build_patient_summary(events_df)
    stage_means = extract_stage_wait_means(events_df)

    res = {
        "seed": seed,
        "n_patients": patient_df["patient_id"].nunique(),
        "mean_total_ref_to_outpat": patient_df["total_days_ref_to_outpat"].dropna().mean(),
        "median_total_ref_to_outpat": patient_df["total_days_ref_to_outpat"].dropna().median(),
        "sd_total_ref_to_outpat": patient_df["total_days_ref_to_outpat"].dropna().std(),
        "mean_total_ref_to_pathrep": patient_df["total_days_ref_to_pathrep"].dropna().mean(),
        "prop_met_62d_to_pathrep": patient_df["met_62d_to_pathrep"].dropna().mean(),
    }

    res.update(stage_means)
    return res


# --------------------------------------------------
# Summarise across replications
# --------------------------------------------------
def summarise_variability(results_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in results_df.columns if c != "seed"]

    rows = []
    for col in numeric_cols:
        vals = results_df[col].dropna()
        if len(vals) == 0:
            continue

        mean_val = vals.mean()
        sd_val = vals.std(ddof=1)
        cv_val = sd_val / mean_val if mean_val != 0 else np.nan

        rows.append({
            "metric": col,
            "n_reps": len(vals),
            "mean_across_reps": mean_val,
            "sd_across_reps": sd_val,
            "cv_across_reps": cv_val,
            "min": vals.min(),
            "q2.5": vals.quantile(0.025),
            "median": vals.median(),
            "q97.5": vals.quantile(0.975),
            "max": vals.max(),
            "range": vals.max() - vals.min(),
        })

    return pd.DataFrame(rows).sort_values("cv_across_reps", ascending=False)


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_metric_by_seed(results_df: pd.DataFrame, metric: str, outpath: Path):
    vals = results_df[["seed", metric]].dropna()
    if vals.empty:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(vals["seed"], vals[metric], marker="o")
    plt.xlabel("Seed")
    plt.ylabel(metric)
    plt.title(f"Sensitivity of {metric} to random seed")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    seeds = range(1001, 1051)

    all_results = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        rep_result = run_one_rep(seed=seed, n_days=365, lam_per_workday=2.0)
        all_results.append(rep_result)

    results_df = pd.DataFrame(all_results)
    summary_df = summarise_variability(results_df)

    results_path = OUTPUT_DIR / "seed_sensitivity_results.csv"
    summary_path = OUTPUT_DIR / "seed_sensitivity_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(summary_path)

    print("\nTop unstable metrics:")
    print(summary_df.head(10).to_string(index=False))

    for metric in [
        "mean_total_ref_to_outpat",
        "mean_total_ref_to_pathrep",
        "prop_met_62d_to_pathrep",
        "wait_ref_to_mri",
        "wait_biopmdt_to_biopsy",
    ]:
        if metric in results_df.columns:
            plot_metric_by_seed(results_df, metric, OUTPUT_DIR / f"{metric}_by_seed.png")


if __name__ == "__main__":
    main()