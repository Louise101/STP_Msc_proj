import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from single_walk_mdt_day import trace_one_patient_mdtday
from PDF_create import build_pdfs, build_branching, build_pdfs2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "batch_results.csv"
EVENTS_FILE = OUTPUT_DIR / "batch_events.csv"
WAITS_FILE = OUTPUT_DIR / "sim_waits.csv"


def _get_event(log, event_name: str):
    return next((e for e in log if e.get("event") == event_name), None)


def make_patient_wait_table(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long event log into wide patient-level wait table.
    """
    if sim_df is None or sim_df.empty:
        return pd.DataFrame(columns=["patient_id"])

    event_to_wait_col = {
        "mri_performed": "wait_ref_to_mri",
        "mri_report_ready": "wait_mri_to_report",
        "MDT_occured": "wait_report_to_biopmdt",
        "biopsy_done": "wait_biopmdt_to_biopsy",
        "Path_report_recieved": "wait_biopsy_to_pathreport",
        "Treatment_options_MDT_occured": "wait_pathrep_to_treatmdt",
        "Outpatient_appointment_occured": "wait_treatmdt_to_outpat",
    }

    waits = sim_df[sim_df["event"].isin(event_to_wait_col.keys())].copy()
    waits["wait_name"] = waits["event"].map(event_to_wait_col)
    waits = waits[["patient_id", "wait_name", "wait_days"]].copy()

    wide_df = waits.pivot_table(
        index="patient_id",
        columns="wait_name",
        values="wait_days",
        aggfunc="first"
    ).reset_index()

    wide_df.columns.name = None
    return wide_df


def validate_events_df(events_df: pd.DataFrame) -> None:
    """
    Basic sanity checks on event log.
    """
    if events_df is None or events_df.empty:
        raise ValueError("events_df is empty")

    required_cols = {"patient_id", "event", "date"}
    missing = required_cols - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df missing required columns: {sorted(missing)}")

    n_missing_dates = events_df["date"].isna().sum()
    print(f"\nMissing dates in event log: {n_missing_dates}")

    # Check a few obvious ordering problems
    bad_rows = events_df.loc[events_df["date"].isna()]
    if len(bad_rows) > 0:
        print("\nExample rows with missing dates:")
        print(bad_rows.head(10).to_string(index=False))

    print("\nUnique event names:")
    print(sorted(events_df["event"].dropna().unique()))

    print("\nEvent counts:")
    print(events_df["event"].value_counts())


def simulate_n_patients(
    n: int,
    start_date: dt.date,
    seed: int = 42,
    keep_event_log: bool = True,
    exclude_np053_ref_to_mri: bool = False,
):
    master_rng = np.random.default_rng(seed)

    results = []
    all_events = []

    pdfs = build_pdfs()
   # pdfs = build_pdfs2(exclude_np053_ref_to_mri=exclude_np053_ref_to_mri)
    branching = build_branching()

    for i in range(n):
        patient_id = f"VP{i:05d}"

        # Stable independent RNG per patient
        patient_seed = int(master_rng.integers(0, 2**32 - 1))
        patient_rng = np.random.default_rng(patient_seed)

        log, total_days = trace_one_patient_mdtday(
            start_date=start_date,
            rng=patient_rng,
            pdfs=pdfs,
            branching=branching,
            patient_id=patient_id,
        )

        referral_event = _get_event(log, "referral_received")
        end_event = log[-1] if log else None
        mdt_decision = _get_event(log, "mdt_decision")
        path_outcome = _get_event(log, "Path_report_outcome")

        had_biopsy = any(e.get("event") == "biopsy_done" for e in log)
        had_treat_mdt = any(e.get("event") == "Treatment_options_MDT_occured" for e in log)
        had_outpat = any(e.get("event") == "Outpatient_appointment_occured" for e in log)

        results.append({
            "patient_id": patient_id,
            "patient_seed": patient_seed,
            "start_date": referral_event.get("date") if referral_event else start_date,
            "end_date": end_event.get("date") if end_event else None,
            "end_event": end_event.get("event") if end_event else None,
            "total_days": total_days,
            "biopmdt_outcome": int(mdt_decision["outcome"]) if mdt_decision and mdt_decision.get("outcome") is not None else None,
            "pathrep_outcome": int(path_outcome["outcome"]) if path_outcome and path_outcome.get("outcome") is not None else None,
            "had_biopsy": had_biopsy,
            "had_treat_mdt": had_treat_mdt,
            "had_outpat": had_outpat,
        })

        if keep_event_log:
            for ev in log:
                ev2 = dict(ev)
                ev2["patient_id"] = patient_id
                ev2["patient_seed"] = patient_seed
                all_events.append(ev2)

    results_df = pd.DataFrame(results)

    if keep_event_log:
        events_df = pd.DataFrame(all_events)

        # Force dates to datetime here before saving
        events_df["date"] = pd.to_datetime(events_df["date"], errors="coerce")

        # Sort for readability/debugging
        events_df = events_df.sort_values(["patient_id", "date", "event"]).reset_index(drop=True)

        sim_waits = make_patient_wait_table(events_df)
    else:
        events_df = None
        sim_waits = None

    return results_df, events_df, sim_waits


def plot_total_days(results_df: pd.DataFrame, title: str):
    plt.figure()
    plt.hist(results_df["total_days"].dropna(), bins=30)
    plt.xlabel("Total pathway time (days)")
    plt.ylabel("Number of patients")
    plt.title(title)
    plt.show()


def print_basic_summaries(results_df: pd.DataFrame):
    print("\nTotal_days summary:")
    print(results_df["total_days"].describe())

    print("\nEnd event counts:")
    print(results_df["end_event"].value_counts(dropna=False))

    if "biopmdt_outcome" in results_df.columns:
        print("\nBiopsy MDT outcome proportions:")
        print(results_df["biopmdt_outcome"].value_counts(normalize=True, dropna=False))

    if "pathrep_outcome" in results_df.columns:
        reached_path = results_df["pathrep_outcome"].notna()
        print("\nPath report outcome proportions (among those who reached pathology):")
        print(results_df.loc[reached_path, "pathrep_outcome"].value_counts(normalize=True))


if __name__ == "__main__":
    # Main model
    results_df, events_df, sim_waits = simulate_n_patients(
        n=10000,
        start_date=dt.date(2026, 1, 5),
        seed=42,
        keep_event_log=True,
    )

   
    print(results_df.head())
    print_basic_summaries(results_df)
    plot_total_days(results_df, title="Simulated total pathway time (n=10000, seed=42)")

    results_df.to_csv(RESULTS_FILE, index=False)

    if events_df is not None:
        validate_events_df(events_df)
        events_df.to_csv(EVENTS_FILE, index=False)

    if sim_waits is not None:
        sim_waits.to_csv(WAITS_FILE, index=False)

    print("\nSaved:")
    print(RESULTS_FILE)
    print(EVENTS_FILE)
    print(WAITS_FILE)

 