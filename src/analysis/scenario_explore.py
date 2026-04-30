from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "capacity_scenario_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748

SEEDS = range(1000, 1030)

SCENARIO_NAME = "OBS_MIX_DES_BIOPSY"

OUTPATIENT_EVENT = "Outpatient_appointment_occured"

MRI_CAPACITIES = [2, 3, 4, 5, 6, 7, 8]
BIOPSY_CAPACITIES = [1, 2, 3, 4, 5, 6]


def make_referral_schedule(seed: int) -> dict:
    return generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )


def apply_capacity_overrides(cfg, mri_capacity: int, biopsy_capacity: int):
    cfg.mri_capacity_by_weekday_prostad = {
        0: 0,
        1: mri_capacity,
        2: 0,
        3: 0,
        4: 0,
    }

    cfg.biopsy_capacity_by_weekday = {
        0: biopsy_capacity,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    return cfg


def get_first_event_date(patient, event_name: str):
    dates = [
        event.get("date")
        for event in patient.events
        if event.get("event") == event_name and event.get("date") is not None
    ]
    return min(dates) if dates else None


def extract_patient_metrics(
    result: dict,
    seed: int,
    mri_capacity: int,
    biopsy_capacity: int,
) -> pd.DataFrame:
    rows = []

    for patient in result["all_patients_objects"]:
        referral_date = get_first_event_date(patient, "referral_received")
        outpatient_date = get_first_event_date(patient, OUTPATIENT_EVENT)

        biopsy_decision_date = get_first_event_date(patient, "MDT_occured")
        biopsy_date = get_first_event_date(patient, "biopsy_done")

        mri_date = get_first_event_date(patient, "mri_performed")

        time_to_outpatient = np.nan
        biopsy_wait = np.nan
        ref_to_mri_wait = np.nan

        if referral_date is not None and outpatient_date is not None:
            value = (outpatient_date - referral_date).days
            if value >= 0:
                time_to_outpatient = value

        if biopsy_decision_date is not None and biopsy_date is not None:
            value = (biopsy_date - biopsy_decision_date).days
            if value >= 0:
                biopsy_wait = value

        if referral_date is not None and mri_date is not None:
            value = (mri_date - referral_date).days
            if value >= 0:
                ref_to_mri_wait = value

        rows.append(
            {
                "seed": seed,
                "mri_capacity": mri_capacity,
                "biopsy_capacity": biopsy_capacity,
                "patient_id": patient.patient_id,
                "pathway_type": patient.data.get("pathway_type"),
                "time_to_outpatient": time_to_outpatient,
                "clinic_mdt_to_biopsy_wait": biopsy_wait,
                "ref_to_mri_wait": ref_to_mri_wait,
            }
        )

    return pd.DataFrame(rows)


def run_one_capacity_seed(seed: int, mri_capacity: int, biopsy_capacity: int) -> pd.DataFrame:
    referral_schedule = make_referral_schedule(seed)

    cfg = build_combined_config(
        SCENARIO_NAME,
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    cfg = apply_capacity_overrides(
        cfg=cfg,
        mri_capacity=mri_capacity,
        biopsy_capacity=biopsy_capacity,
    )

    result = run_day_loop_combined_engine(
        cfg,
        daily_referrals_override=referral_schedule,
    )

    return extract_patient_metrics(
        result=result,
        seed=seed,
        mri_capacity=mri_capacity,
        biopsy_capacity=biopsy_capacity,
    )


def summarise_capacity_results_by_pathway(
    patient_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
) -> dict[str, pd.DataFrame]:

    outputs = {}

    groups = {
        "ALL": patient_df,
        "PROSTAD": patient_df[patient_df["pathway_type"] == "PROSTAD"],
        "BASELINE": patient_df[patient_df["pathway_type"] == "BASELINE"],
    }

    for group_name, df_subset in groups.items():
        df_subset = df_subset[df_subset[metric_col].notna()].copy()

        if df_subset.empty:
            continue

        seed_summary = (
            df_subset
            .groupby(["seed", "mri_capacity", "biopsy_capacity"], as_index=False)
            .agg(
                n_patients=(metric_col, "count"),
                metric_mean=(metric_col, "mean"),
                metric_median=(metric_col, "median"),
                metric_p90=(metric_col, lambda x: np.percentile(x, 90)),
            )
        )

        across_seed_summary = (
            seed_summary
            .groupby(["mri_capacity", "biopsy_capacity"], as_index=False)
            .agg(
                n_runs=("seed", "nunique"),
                n_patients_mean=("n_patients", "mean"),
                metric_mean_mean=("metric_mean", "mean"),
                metric_mean_sd=("metric_mean", "std"),
                metric_median_mean=("metric_median", "mean"),
                metric_p90_mean=("metric_p90", "mean"),
            )
        )

        across_seed_summary["metric"] = metric_label
        across_seed_summary["group"] = group_name

        outputs[group_name] = across_seed_summary

    return outputs


def plot_heatmap(
    summary_df: pd.DataFrame,
    value_col: str,
    colourbar_label: str,
    title: str,
    filename: str,
) -> None:
    heatmap_df = summary_df.pivot(
        index="biopsy_capacity",
        columns="mri_capacity",
        values=value_col,
    )

    plt.figure(figsize=(10, 6))

    im = plt.imshow(
        heatmap_df.values,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(im, label=colourbar_label)

    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)),
        labels=heatmap_df.columns,
    )

    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index,
    )

    plt.xlabel("MRI capacity")
    plt.ylabel("Biopsy capacity")
    plt.title(title)

    for y in range(len(heatmap_df.index)):
        for x in range(len(heatmap_df.columns)):
            value = heatmap_df.values[y, x]
            if pd.notna(value):
                plt.text(x, y, f"{value:.1f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_and_plot_summary_set(
    summary_dict: dict[str, pd.DataFrame],
    metric_name: str,
    heatmap_label: str,
    filename_prefix: str,
) -> None:
    group_labels = {
        "ALL": "All patients",
        "PROSTAD": "PROSTAD patients",
        "BASELINE": "Standard pathway patients",
    }

    file_labels = {
        "ALL": "all_patients",
        "PROSTAD": "prostad_only",
        "BASELINE": "baseline_only",
    }

    for group_name, summary_df in summary_dict.items():
        group_label = group_labels[group_name]
        file_label = file_labels[group_name]

        summary_df.to_csv(
            OUTPUT_DIR / f"{filename_prefix}_summary_{file_label}.csv",
            index=False,
        )

        plot_heatmap(
            summary_df=summary_df,
            value_col="metric_mean_mean",
            colourbar_label=heatmap_label,
            title=f"{metric_name}\n{group_label} ({len(SEEDS)} seeds)",
            filename=f"{filename_prefix}_heatmap_{file_label}.png",
        )


def main() -> None:
    all_patient_results = []

    for mri_capacity in MRI_CAPACITIES:
        for biopsy_capacity in BIOPSY_CAPACITIES:
            print(
                f"\nRunning MRI capacity={mri_capacity}, "
                f"biopsy capacity={biopsy_capacity}"
            )

            for seed in SEEDS:
                print(f"  Seed {seed}")

                df = run_one_capacity_seed(
                    seed=seed,
                    mri_capacity=mri_capacity,
                    biopsy_capacity=biopsy_capacity,
                )

                all_patient_results.append(df)

    patient_df = pd.concat(all_patient_results, ignore_index=True)

    patient_df.to_csv(
        OUTPUT_DIR / "capacity_scenario_patient_level_metrics.csv",
        index=False,
    )

    outpatient_summary_dict = summarise_capacity_results_by_pathway(
        patient_df=patient_df,
        metric_col="time_to_outpatient",
        metric_label="time_to_outpatient",
    )

    biopsy_wait_summary_dict = summarise_capacity_results_by_pathway(
        patient_df=patient_df,
        metric_col="clinic_mdt_to_biopsy_wait",
        metric_label="clinic_mdt_to_biopsy_wait",
    )

    ref_to_mri_summary_dict = summarise_capacity_results_by_pathway(
        patient_df=patient_df,
        metric_col="ref_to_mri_wait",
        metric_label="ref_to_mri_wait",
    )

    save_and_plot_summary_set(
        summary_dict=outpatient_summary_dict,
        metric_name="Mean time to outpatient appointment",
        heatmap_label="Mean time to outpatient appointment (days)",
        filename_prefix="time_to_outpatient",
    )

    save_and_plot_summary_set(
        summary_dict=biopsy_wait_summary_dict,
        metric_name="Mean clinic/MDT to biopsy wait",
        heatmap_label="Mean clinic/MDT to biopsy wait (days)",
        filename_prefix="clinic_mdt_to_biopsy_wait",
    )

    save_and_plot_summary_set(
        summary_dict=ref_to_mri_summary_dict,
        metric_name="Mean referral to MRI wait",
        heatmap_label="Mean referral to MRI wait (days)",
        filename_prefix="ref_to_mri_wait",
    )

    print("\n=== TIME TO OUTPATIENT APPOINTMENT SUMMARY ===")
    for group_name, summary_df in outpatient_summary_dict.items():
        print(f"\n--- {group_name} ---")
        print(
            summary_df[
                [
                    "mri_capacity",
                    "biopsy_capacity",
                    "n_runs",
                    "n_patients_mean",
                    "metric_mean_mean",
                    "metric_mean_sd",
                    "metric_median_mean",
                    "metric_p90_mean",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

    print("\n=== CLINIC/MDT TO BIOPSY WAIT SUMMARY ===")
    for group_name, summary_df in biopsy_wait_summary_dict.items():
        print(f"\n--- {group_name} ---")
        print(
            summary_df[
                [
                    "mri_capacity",
                    "biopsy_capacity",
                    "n_runs",
                    "n_patients_mean",
                    "metric_mean_mean",
                    "metric_mean_sd",
                    "metric_median_mean",
                    "metric_p90_mean",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

    print("\n=== REFERRAL TO MRI WAIT SUMMARY ===")
    for group_name, summary_df in ref_to_mri_summary_dict.items():
        print(f"\n--- {group_name} ---")
        print(
            summary_df[
                [
                    "mri_capacity",
                    "biopsy_capacity",
                    "n_runs",
                    "n_patients_mean",
                    "metric_mean_mean",
                    "metric_mean_sd",
                    "metric_median_mean",
                    "metric_p90_mean",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()