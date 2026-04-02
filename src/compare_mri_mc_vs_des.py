from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_MC, WAIT_MODE_DES
from single_walk_mdt_day import trace_one_patient_mdtday

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_wait(events, start_event, end_event):
    start_date = None
    end_date = None

    for e in events:
        if e["event"] == start_event:
            start_date = e["date"]
        elif e["event"] == end_event:
            end_date = e["date"]

    if start_date is not None and end_date is not None:
        return (end_date - start_date).days
    return None


def summarise_patient_results(patient_results):
    biopsy_waits = []
    total_pathway_waits = []

    for events, total_days in patient_results:
        w = extract_wait(events, "MDT_occured", "biopsy_done")
        if w is not None:
            biopsy_waits.append(w)

        if total_days is not None:
            total_pathway_waits.append(total_days)

    def summary(arr):
        if len(arr) == 0:
            return {"n": 0, "mean": None, "median": None, "p90": None}
        arr = np.asarray(arr)
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "biopsy_wait": summary(biopsy_waits),
        "total_pathway": summary(total_pathway_waits),
    }


def run_scenario(name, wait_time_mode):
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=365,
        lam_per_workday=0.586,
        mri_capacity_by_weekday={2: 4},      # PROSTAD MRI list
        biopsy_capacity_by_weekday={3: 1, 4: 1},
        biopsy_ready_delay_days=0,
        seed=42,
        wait_time_mode=wait_time_mode,
        initial_biopsy_queue_n=1,
        initial_biopsy_pending_n=1,
        biopsy_capacity_dropout_prob_by_weekday={3: 0.15, 4: 0.2},
        biopsy_backlog_capacity_tiers=[
            (6, {3: 2, 4: 1}),
            (10, {3: 2, 4: 2}),
        ],
    )

    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    summary = summarise_patient_results(results["patient_results"])

    row = {
        "scenario": name,
        "patients_completed": results["summary_stats"]["total_patients_completed"],
        "mri_final_backlog": results["summary_stats"]["final_queue_length_by_resource"]["MRI"],
        "biopsy_final_backlog": results["summary_stats"]["final_backlog_by_resource"]["Biopsy"],
        "mri_mean_queue_wait": results["summary_stats"]["mean_queue_wait_days_by_resource"]["MRI"],
        "biopsy_mean_queue_wait": results["summary_stats"]["mean_queue_wait_days_by_resource"]["Biopsy"],
        "biopsy_wait_n": summary["biopsy_wait"]["n"],
        "biopsy_wait_mean": summary["biopsy_wait"]["mean"],
        "biopsy_wait_median": summary["biopsy_wait"]["median"],
        "biopsy_wait_p90": summary["biopsy_wait"]["p90"],
        "pathway_n": summary["total_pathway"]["n"],
        "pathway_mean": summary["total_pathway"]["mean"],
        "pathway_median": summary["total_pathway"]["median"],
        "pathway_p90": summary["total_pathway"]["p90"],
    }

    return results, row


def main():
    scenarios = {
        "baseline_mc_mri": {
            "ref_to_mri": WAIT_MODE_MC,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
        "mri_des_only": {
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
        "full_frontend_des": {
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    }

    rows = []
    all_results = {}

    for name, mode in scenarios.items():
        results, row = run_scenario(name, mode)
        rows.append(row)
        all_results[name] = results

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "mri_mc_vs_des_comparison.csv", index=False)
    print("\nSaved:", OUTPUT_DIR / "mri_mc_vs_des_comparison.csv")

    baseline = df.loc[df["scenario"] == "baseline_mc_mri"].iloc[0]

    delta_rows = []
    for _, row in df.iterrows():
        delta_rows.append({
            "scenario": row["scenario"],
            "delta_patients_completed": row["patients_completed"] - baseline["patients_completed"],
            "delta_mri_final_backlog": row["mri_final_backlog"] - baseline["mri_final_backlog"],
            "delta_biopsy_final_backlog": row["biopsy_final_backlog"] - baseline["biopsy_final_backlog"],
            "delta_biopsy_wait_mean": (
                None if pd.isna(row["biopsy_wait_mean"]) else row["biopsy_wait_mean"] - baseline["biopsy_wait_mean"]
            ),
            "delta_pathway_mean": (
                None if pd.isna(row["pathway_mean"]) else row["pathway_mean"] - baseline["pathway_mean"]
            ),
        })

    delta_df = pd.DataFrame(delta_rows)
    print("\nChanges vs baseline:\n")
    print(delta_df.to_string(index=False))
    delta_df.to_csv(OUTPUT_DIR / "mri_mc_vs_des_deltas.csv", index=False)


if __name__ == "__main__":
    main()