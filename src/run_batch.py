import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from single_patient_walk import trace_one_patient
from PDF_create import build_pdfs, build_branching


def _get_event(log, event_name: str):
    return next((e for e in log if e.get("event") == event_name), None)

# run single patient pathway n times and output results
def simulate_n_patients(
    n: int,
    start_date: dt.date,
    seed: int = 42,
    keep_event_log: bool = True,
):

    master_rng = np.random.default_rng(seed)

    results = []
    all_events = []

    pdfs = build_pdfs()
    branching = build_branching()

    for i in range(n):
        patient_id = f"VP{i:05d}"

        # independent RNG stream per patient (stable + avoids accidental coupling)
        patient_seed = int(master_rng.integers(0, 2**32 - 1))
        patient_rng = np.random.default_rng(patient_seed)

      
        #log, total_days = trace_one_patient(start_date, patient_rng, patient_id=patient_id)
        log, total_days = trace_one_patient(start_date, patient_rng, pdfs, branching, patient_id=patient_id)

        # Pull key events/outcomes using your exact event names
        referral_event = _get_event(log, "referral_received")
        end_event = log[-1] if log else None

        mdt_decision = _get_event(log, "mdt_decision")
        path_outcome = _get_event(log, "Path_report_outcome")

        # Convenience flags
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

            # Branch outcomes (may be None if patient exited before that stage)
            "biopmdt_outcome": int(mdt_decision["outcome"]) if mdt_decision and mdt_decision.get("outcome") is not None else None,
            "pathrep_outcome": int(path_outcome["outcome"]) if path_outcome and path_outcome.get("outcome") is not None else None,

            # Stage completion flags
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
    events_df = pd.DataFrame(all_events) if keep_event_log else None
    return results_df, events_df


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
        print("\nBiopsy MDT outcome proportions (incl NaN if exited before MDT):")
        print(results_df["biopmdt_outcome"].value_counts(normalize=True, dropna=False))

    if "pathrep_outcome" in results_df.columns:
        # Often better to look only among those who *reached* pathology outcome
        reached_path = results_df["pathrep_outcome"].notna()
        print("\nPath report outcome proportions (among those who reached pathology):")
        print(results_df.loc[reached_path, "pathrep_outcome"].value_counts(normalize=True))


if __name__ == "__main__":
    results_df, events_df = simulate_n_patients(
        n=10000,
        start_date=dt.date(2026, 1, 5),
        seed=42,
        keep_event_log=True,
    )

    print(results_df.head())
    print_basic_summaries(results_df)
    plot_total_days(results_df, title="Simulated total pathway time (n=1000, seed=42)")

    # Optional saves
    results_df.to_csv("batch_results.csv", index=False)
    if events_df is not None:
        events_df.to_csv("batch_events.csv", index=False)
