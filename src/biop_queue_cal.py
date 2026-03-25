from datetime import date
import numpy as np
import pandas as pd

from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES
from single_walk_mdt_day import trace_one_patient_mdtday


# -----------------------------
# Helpers: real / simulated biopsy waits
# -----------------------------

def load_real_biopsy_waits(csv_path: str) -> np.ndarray:
    real_df = pd.read_csv(csv_path)

    real_df["Date of Prostate MRI MDT"] = pd.to_datetime(
        real_df["Date of Prostate MRI MDT"], dayfirst=True
    )
    real_df["Date of Biopsy"] = pd.to_datetime(
        real_df["Date of Biopsy"], dayfirst=True
    )

    real_df = real_df.dropna(subset=["Date of Prostate MRI MDT", "Date of Biopsy"])

    real_waits = (
        real_df["Date of Biopsy"] - real_df["Date of Prostate MRI MDT"]
    ).dt.days.to_numpy()

    return real_waits


def extract_simulated_biopsy_waits(patient_results) -> np.ndarray:
    """
    Extract biopsy waits from trace_one_patient_mdtday outputs.

    Each patient result is currently a tuple:
        (events_list, total_pathway_days)

    We extract wait_days from the 'biopsy_done' event.
    """
    waits = []

    for result in patient_results:
        if not isinstance(result, tuple) or len(result) < 1:
            continue

        events = result[0]
        if not isinstance(events, list):
            continue

        for event in events:
            if event.get("event") == "biopsy_done":
                biopsy_wait = event.get("wait_days")
                if biopsy_wait is not None:
                    waits.append(biopsy_wait)
                break

    return np.asarray(waits)


def summarise_biopsy_waits(patient_results):
    waits = extract_simulated_biopsy_waits(patient_results)

    if len(waits) == 0:
        return {
            "n": 0,
            "mean_total_wait": None,
            "median_total_wait": None,
            "p90_total_wait": None,
        }

    return {
        "n": len(waits),
        "mean_total_wait": float(np.mean(waits)),
        "median_total_wait": float(np.median(waits)),
        "p90_total_wait": float(np.percentile(waits, 90)),
    }


def make_biopsy_residual_samples(real_waits, sim_waits) -> np.ndarray:
    """
    Build empirical residual samples by quantile-matching real waits
    against simulated waits and clipping negatives to zero.
    """
    real_waits = np.sort(np.asarray(real_waits))
    sim_waits = np.sort(np.asarray(sim_waits))

    if len(real_waits) == 0 or len(sim_waits) == 0:
        raise ValueError("real_waits and sim_waits must both be non-empty")

    n = min(len(real_waits), len(sim_waits))
    q = np.linspace(0, 1, n, endpoint=False)

    real_q = np.quantile(real_waits, q)
    sim_q = np.quantile(sim_waits, q)

    residuals = real_q - sim_q
    residuals = np.clip(residuals, 0, None)

    return residuals


def summarise_wait_array(waits: np.ndarray, label: str) -> pd.DataFrame:
    waits = np.asarray(waits)

    if len(waits) == 0:
        return pd.DataFrame([{
            "label": label,
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }])

    return pd.DataFrame([{
        "label": label,
        "n": len(waits),
        "mean": float(np.mean(waits)),
        "median": float(np.median(waits)),
        "p90": float(np.percentile(waits, 90)),
        "min": float(np.min(waits)),
        "max": float(np.max(waits)),
    }])


# -----------------------------
# Main scenario runner
# -----------------------------
def run_biopsy_scenarios(return_results: bool = False):
    schedules = [
        {"name": "thu_3", "capacity": {3: 3}},
        {"name": "thu2_fri1", "capacity": {3: 2, 4: 1}},
        {"name": "thu1_fri2", "capacity": {3: 1, 4: 2}},
        {"name": "thu2_fri2", "capacity": {3: 2, 4: 2}},
    ]

    readiness_delays = [0, 1, 2]
    rows = []

    for sched in schedules:
        for delay in readiness_delays:
            print(f"\nRunning: {sched['name']} | delay={delay}")

            cfg = EngineConfig(
                start_date=date(2024, 1, 1),
                n_days=365,
                lam_per_workday=0.586,
                mri_capacity_by_weekday={2: 4},
                biopsy_capacity_by_weekday=sched["capacity"],
                biopsy_ready_delay_days=delay,
                seed=42,
                wait_time_mode={
                    "ref_to_mri": WAIT_MODE_DES,
                    "mri_to_report": WAIT_MODE_DES,
                    "report_to_biopmdt": WAIT_MODE_DES,
                    "biopmdt_to_biopsy": WAIT_MODE_DES,
                },
            )

            results = run_day_loop_with_stage_engine(
                cfg,
                trace_one_patient_mdtday,
            )

            biopsy_stats = summarise_biopsy_waits(results["patient_results"])
            print(biopsy_stats)

            row = {
                "schedule": sched["name"],
                "capacity": str(sched["capacity"]),
                "delay_days": delay,
                "completed": results["summary_stats"]["total_patients_completed"],
                "final_queue": results["summary_stats"]["final_queue_length_by_resource"]["Biopsy"],
                "mean_queue_wait": results["summary_stats"]["mean_queue_wait_days_by_resource"]["Biopsy"],
                "median_queue_wait": results["summary_stats"]["median_queue_wait_days_by_resource"]["Biopsy"],
                **biopsy_stats,
            }

            if return_results:
                row["patient_results"] = results["patient_results"]

            rows.append(row)

    return pd.DataFrame(rows)


def pick_scenario(results_df: pd.DataFrame, schedule: str, delay_days: int) -> pd.Series:
    match = results_df[
        (results_df["schedule"] == schedule) &
        (results_df["delay_days"] == delay_days)
    ]

    if len(match) != 1:
        raise ValueError(
            f"Expected exactly one matching scenario for schedule={schedule}, delay_days={delay_days}, found {len(match)}"
        )

    return match.iloc[0]


def build_comparison_table(results_df, observed_waits=None):
    rows = []

    for _, row in results_df.iterrows():
        rows.append({
            "scenario": f"{row['schedule']} | delay={row['delay_days']}",
            "type": "simulated",
            "mean_wait": row["mean_total_wait"],
            "median_wait": row["median_total_wait"],
            "p90_wait": row["p90_total_wait"],
            "final_queue": row["final_queue"],
        })

    if observed_waits is not None:
        observed_waits = np.array(observed_waits)

        rows.insert(0, {
            "scenario": "Observed",
            "type": "observed",
            "mean_wait": float(np.mean(observed_waits)),
            "median_wait": float(np.median(observed_waits)),
            "p90_wait": float(np.percentile(observed_waits, 90)),
            "final_queue": None,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Run + print
# -----------------------------
if __name__ == "__main__":
    # Keep full patient_results so we can extract scenario-level waits later
    df = run_biopsy_scenarios(return_results=True)

    print("\n\n===== RESULTS =====\n")
    print(
        df.drop(columns=["patient_results"], errors="ignore").to_string(index=False)
    )

    real_biopsy_waits = load_real_biopsy_waits("data/pre_biopmdt_to_biop.csv")

    comparison_df = build_comparison_table(
        df.drop(columns=["patient_results"], errors="ignore"),
        observed_waits=real_biopsy_waits
    )

    print("\n===== COMPARISON TABLE =====\n")
    print(comparison_df.to_string(index=False))

    # Choose a candidate scenario for residual calculation
    chosen = pick_scenario(df, schedule="thu2_fri1", delay_days=1)

    sim_biopsy_waits = extract_simulated_biopsy_waits(chosen["patient_results"])
    biopsy_residual_samples = make_biopsy_residual_samples(
        real_biopsy_waits,
        sim_biopsy_waits,
    )

    residual_summary_df = summarise_wait_array(
        biopsy_residual_samples,
        label="Biopsy residual samples",
    )

    print("\n===== RESIDUAL SUMMARY =====\n")
    print(residual_summary_df.to_string(index=False))

    # Save outputs
    df.drop(columns=["patient_results"], errors="ignore").to_csv(
        "biopsy_scenario_results.csv", index=False
    )
    comparison_df.to_csv("biopsy_comparison_table.csv", index=False)
    pd.DataFrame({"biopsy_residual_samples": biopsy_residual_samples}).to_csv(
        "biopsy_residual_samples.csv", index=False
    )
    residual_summary_df.to_csv("biopsy_residual_summary.csv", index=False)

    print("\nSaved to biopsy_scenario_results.csv")
    print("Saved to biopsy_comparison_table.csv")
    print("Saved to biopsy_residual_samples.csv")
    print("Saved to biopsy_residual_summary.csv")

    