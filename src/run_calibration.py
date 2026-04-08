from datetime import date
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from des_engine import (
    EngineConfig,
    run_day_loop_with_stage_engine,
    WAIT_MODE_MC,
    WAIT_MODE_DES
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def patient_results_to_event_log(patient_results):
    raw_df = pd.DataFrame(patient_results)

    if raw_df.shape[1] < 1:
        raise ValueError("patient_results is empty or malformed")

    event_rows = []
    for _, row in raw_df.iterrows():
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


def build_patient_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found. Columns are: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])

    referral = (
        df[df["event"] == "referral_recieved"]
        .groupby("patient_id")["date"]
        .min()
        .rename("referral_date")
    )

    mri = (
        df[df["event"] == "mri_performed"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mri_date")
    )

    mri_report = (
        df[df["event"] == "mri_report_ready"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mri_report_date")
    )

    mdt = (
        df[df["event"] == "MDT_occured"]
        .groupby("patient_id")["date"]
        .min()
        .rename("mdt_date")
    )

    biopsy = (
        df[df["event"] == "biopsy_done"]
        .groupby("patient_id")["date"]
        .min()
        .rename("biopsy_date")
    )

    pathrep = (
        df[df["event"] == "Path_report_outcome"]
        .groupby("patient_id")["date"]
        .min()
        .rename("pathrep_date")
    )

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

    patient["total_days_ref_to_pathrep"] = (
        patient["pathrep_date"] - patient["referral_date"]
    ).dt.days

    patient["total_days_ref_to_outpat"] = (
        patient["pathway_end_date"] - patient["referral_date"]
    ).dt.days

    patient["met_62d_to_pathrep"] = patient["total_days_ref_to_pathrep"] <= 62

    return patient


def compute_sim_targets(patient_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    targets = {}

    def mean_wait(event_name):
        vals = events_df.loc[events_df["event"] == event_name, "wait_days"].dropna()
        return vals.mean() if len(vals) else np.nan

    targets["mean_wait_ref_to_mri"] = mean_wait("mri_performed")
    targets["mean_wait_mri_to_report"] = mean_wait("mri_report_ready")
    targets["mean_wait_report_to_biopmdt"] = mean_wait("MDT_occured")
    targets["mean_wait_biopmdt_to_biopsy"] = mean_wait("biopsy_done")
    targets["mean_wait_biopsy_to_pathrep"] = mean_wait("Path_report_outcome")

    targets["mean_total_ref_to_pathrep"] = patient_df["total_days_ref_to_pathrep"].mean()
    targets["median_total_ref_to_outpat"] = patient_df["total_days_ref_to_outpat"].median()

    mdt = events_df[events_df["event"] == "mdt_decision"]
    targets["prop_biopsy_after_mdt"] = (mdt["outcome"] == 1).mean()
    targets["prop_exit_after_mdt"] = (mdt["outcome"] == 0).mean()

    path = events_df[events_df["event"] == "Path_report_outcome"]
    targets["prop_path_positive"] = (path["outcome"] == 1).mean()

    targets["prop_met_62d_to_pathrep"] = patient_df["met_62d_to_pathrep"].mean()

    return targets


def compute_calibration_score(sim_targets: dict, target_df: pd.DataFrame) -> float:
    score = 0.0

    for _, row in target_df.iterrows():
        metric = row["metric"]
        obs = row["observed_value"]
        weight = row["weight"]

        sim = sim_targets.get(metric, np.nan)

        if pd.isna(sim) or obs == 0:
            continue

        error = abs(sim - obs) / obs
        score += weight * error

    return score


def compute_biopsy_overload_metrics(result: dict, cfg: EngineConfig) -> dict:
    arrivals_dict = result["stage_activity"]["Biopsy"]["daily_arrivals"]

    arrivals_df = pd.DataFrame(
        [{"date": d, "biopsy_ready_arrivals": n} for d, n in arrivals_dict.items()]
    )

    if arrivals_df.empty:
        return {
            "mean_weekly_biopsy_ready": np.nan,
            "max_weekly_biopsy_ready": np.nan,
            "weekly_biopsy_capacity": np.nan,
            "mean_biopsy_overload_ratio": np.nan,
            "max_biopsy_overload_ratio": np.nan,
            "n_overloaded_weeks": np.nan,
            "prop_overloaded_weeks": np.nan,
        }

    arrivals_df["date"] = pd.to_datetime(arrivals_df["date"])
    arrivals_df["week_start"] = arrivals_df["date"] - pd.to_timedelta(
        arrivals_df["date"].dt.weekday, unit="D"
    )

    weekly = (
        arrivals_df.groupby("week_start", as_index=False)["biopsy_ready_arrivals"]
        .sum()
        .rename(columns={"biopsy_ready_arrivals": "weekly_biopsy_ready"})
    )

    weekly_capacity = sum((cfg.biopsy_capacity_by_weekday or {}).values())

    if weekly_capacity > 0:
        weekly["overload_ratio"] = weekly["weekly_biopsy_ready"] / weekly_capacity
        weekly["is_overloaded"] = weekly["weekly_biopsy_ready"] > weekly_capacity
    else:
        weekly["overload_ratio"] = np.nan
        weekly["is_overloaded"] = np.nan

    return {
        "mean_weekly_biopsy_ready": weekly["weekly_biopsy_ready"].mean(),
        "max_weekly_biopsy_ready": weekly["weekly_biopsy_ready"].max(),
        "weekly_biopsy_capacity": weekly_capacity,
        "mean_biopsy_overload_ratio": weekly["overload_ratio"].mean(),
        "max_biopsy_overload_ratio": weekly["overload_ratio"].max(),
        "n_overloaded_weeks": weekly["is_overloaded"].sum(),
        "prop_overloaded_weeks": weekly["is_overloaded"].mean(),
    }


def run_calibration_rep(seed: int, cfg: EngineConfig) -> dict:
    cfg_rep = replace(cfg, seed=seed)

    result = run_day_loop_with_stage_engine(cfg_rep)

    events_df, _ = patient_results_to_event_log(result["patient_results"])
    patient_df = build_patient_summary(events_df)

    sim_targets = compute_sim_targets(patient_df, events_df)
    overload_metrics = compute_biopsy_overload_metrics(result, cfg_rep)
    sim_targets.update(overload_metrics)

    sim_targets["seed"] = seed
    return sim_targets


def evaluate_parameter_set(
    cfg: EngineConfig,
    target_df: pd.DataFrame,
    n_reps: int = 30,
    seed_start: int = 1000,
) -> tuple[float, dict, pd.DataFrame]:
    all_targets = []

    for i in range(n_reps):
        seed = seed_start + i
        sim_targets = run_calibration_rep(seed, cfg)
        all_targets.append(sim_targets)

    rep_df = pd.DataFrame(all_targets)
    mean_targets = rep_df.drop(columns=["seed"], errors="ignore").mean(numeric_only=True).to_dict()

    score = compute_calibration_score(mean_targets, target_df)
    return score, mean_targets, rep_df


scenarios = {
    "baseline": {
        "lam": 2.0,
        "mri_capacity": {2: 4},
    },
    "faster_mri": {
        "lam": 2.0,
        "mri_capacity": {1: 4, 2: 4},
    },
    "faster_mri_more_demand": {
        "lam": 2.5,
        "mri_capacity": {1: 4, 2: 4},
    },
}


def main():
    target_path = DATA_DIR / "calibration_targets_pre.csv"
    target_df = pd.read_csv(target_path)

    print("\nLoaded calibration targets:")
    print(target_df.to_string(index=False))

    scenario_results = []
    all_replication_rows = []

    for scenario_name, scenario_params in scenarios.items():
        print(f"\n====================")
        print(f"Running scenario: {scenario_name}")
        print(f"====================")

        for biopsy_cap in [2, 3]:
            cfg = EngineConfig(
                start_date=date(2024, 1, 1),
                n_days=365,
                lam_per_workday=scenario_params["lam"],
                mri_capacity_by_weekday=scenario_params["mri_capacity"],
                biopsy_capacity_by_weekday={3: biopsy_cap, 4: biopsy_cap},
                seed=1234,
                wait_time_mode={
                    "ref_to_mri": WAIT_MODE_DES,
                    "mri_to_repot": WAIT_MODE_MC,
                    "report_to_biopmdt": WAIT_MODE_MC,
                    "biopmdt_to_biopsy": WAIT_MODE_MC,
                    "biopsy_to_pathrep": WAIT_MODE_MC,
                },
            )

            score, targets, rep_df = evaluate_parameter_set(cfg, target_df)

            print(f"\nScenario={scenario_name}, cap={biopsy_cap}")
            print(f"Score: {score:.4f}")
            print(f"  mean_total_ref_to_pathrep: {targets.get('mean_total_ref_to_pathrep', np.nan):.3f}")
            print(f"  prop_met_62d_to_pathrep: {targets.get('prop_met_62d_to_pathrep', np.nan):.3f}")
            print(f"  mean_biopsy_overload_ratio: {targets.get('mean_biopsy_overload_ratio', np.nan):.3f}")
            print(f"  max_biopsy_overload_ratio: {targets.get('max_biopsy_overload_ratio', np.nan):.3f}")
            print(f"  n_overloaded_weeks: {targets.get('n_overloaded_weeks', np.nan)}")

            rep_df["scenario"] = scenario_name
            rep_df["biopsy_cap"] = biopsy_cap
            all_replication_rows.append(rep_df)

            scenario_results.append({
                "scenario": scenario_name,
                "biopsy_cap": biopsy_cap,
                "score": score,
                **targets,
            })

    scenario_df = pd.DataFrame(scenario_results).sort_values(["scenario", "score"]).reset_index(drop=True)
    rep_results_df = pd.concat(all_replication_rows, ignore_index=True)

    scenario_path = OUTPUT_DIR / "scenario_comparison.csv"
    reps_path = OUTPUT_DIR / "scenario_replications.csv"

    scenario_df.to_csv(scenario_path, index=False)
    rep_results_df.to_csv(reps_path, index=False)

    print("\n=== Scenario comparison ===")
    print(
        scenario_df[
            [
                "scenario",
                "biopsy_cap",
                "score",
                "mean_total_ref_to_pathrep",
                "prop_met_62d_to_pathrep",
                "mean_biopsy_overload_ratio",
                "max_biopsy_overload_ratio",
                "n_overloaded_weeks",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved scenario summary to: {scenario_path}")
    print(f"Saved scenario replications to: {reps_path}")


if __name__ == "__main__":
    main()