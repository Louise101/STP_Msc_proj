from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "capacity_scenario_exploration_all_prostad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748

SEEDS = range(1000, 1030)

SCENARIO_NAME = "ALL_PROSTAD"

OUTPATIENT_EVENT = "Outpatient_appointment_occured"

#MRI_CAPACITIES = [2, 3, 4, 5, 6, 7, 8]
#BIOPSY_CAPACITIES = [1, 2, 3, 4, 5]

MRI_CAPACITIES = [4, 5, 6]
BIOPSY_CAPACITIES = [2, 3,4]


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
        0: 0,
        1: 0,
        2:0, 
        3: 0,
        4: biopsy_capacity,
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
        clinic_mdt_to_biopsy_wait = np.nan
        ref_to_mri_wait = np.nan

        if referral_date is not None and outpatient_date is not None:
            value = (outpatient_date - referral_date).days
            if value >= 0:
                time_to_outpatient = value

        if biopsy_decision_date is not None and biopsy_date is not None:
            value = (biopsy_date - biopsy_decision_date).days
            if value >= 0:
                clinic_mdt_to_biopsy_wait = value

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
                "clinic_mdt_to_biopsy_wait": clinic_mdt_to_biopsy_wait,
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


def summarise_capacity_results(
    patient_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
) -> pd.DataFrame:
    df = patient_df[patient_df[metric_col].notna()].copy()

    seed_summary = (
        df.groupby(["seed", "mri_capacity", "biopsy_capacity"], as_index=False)
        .agg(
            n_patients=(metric_col, "count"),
            metric_mean=(metric_col, "mean"),
            metric_median=(metric_col, "median"),
            metric_p90=(metric_col, lambda x: np.percentile(x, 90)),
        )
    )

    across_seed_summary = (
        seed_summary.groupby(["mri_capacity", "biopsy_capacity"], as_index=False)
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
    across_seed_summary["scenario"] = SCENARIO_NAME

    return across_seed_summary


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

    fig, ax = plt.subplots(figsize=(10, 6))

    custom_cmap = LinearSegmentedColormap.from_list(
    "poster_teal_blue",
    [
        "#E6F4F3",  # very light teal (background match)
        "#A8D5D2",  # light teal
        "#5FB3B3",  # mid teal
        "#2F7F8F",  # teal-blue
        "#1F4E79",  # dark blue
    ],
)

    im = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        origin="lower",
        cmap=custom_cmap
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colourbar_label, fontsize=17, labelpad=12)
    cbar.ax.tick_params(labelsize=24)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, fontsize=24)

    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=24)

    ax.set_xlabel("MRI capacity (per week)", fontsize=24, labelpad=10)
    ax.set_ylabel("Biopsy capacity (per week)", fontsize=24, labelpad=10)



    values = heatmap_df.values
    threshold = np.nanmax(values) * 0.55

    for y in range(len(heatmap_df.index)):
        for x in range(len(heatmap_df.columns)):
            value = values[y, x]
            if pd.notna(value):
                text_colour = "white" if value >= threshold else "black"
                ax.text(
                    x,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=24,
                    fontweight="bold",
                    color=text_colour,
                )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, format="png", bbox_inches="tight")
    plt.close(fig)

def plot_capacity_response_curve(
    summary_df: pd.DataFrame,
    value_col: str,
    y_label: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for biopsy_capacity in sorted(summary_df["biopsy_capacity"].unique()):
        sub = (
            summary_df[summary_df["biopsy_capacity"] == biopsy_capacity]
            .sort_values("mri_capacity")
        )

        ax.plot(
            sub["mri_capacity"],
            sub[value_col],
            marker="o",
            linewidth=2,
            label=f"Biopsy capacity = {biopsy_capacity}/week",
        )

    ax.set_xlabel("MRI capacity per week")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def save_and_plot_summary(
    summary_df: pd.DataFrame,
    metric_name: str,
    heatmap_label: str,
    filename_prefix: str,
) -> None:
    summary_df.to_csv(
        OUTPUT_DIR / f"{filename_prefix}_summary_all_prostad.csv",
        index=False,
    )

    plot_heatmap(
        summary_df=summary_df,
        value_col="metric_median_mean",
        colourbar_label=heatmap_label,
        title=f"{metric_name}\nAll PROSTAD patients ({len(SEEDS)} seeds)",
        filename=f"{filename_prefix}_heatmap_all_prostad_fri.png",
    )

    plot_capacity_response_curve(
        summary_df=summary_df,
        value_col="metric_median_mean",
        y_label=heatmap_label,
        filename=f"{filename_prefix}_capacity_response_curve_all_prostad_bopcap1.png",
    )


def print_summary(summary_df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
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
        OUTPUT_DIR / "capacity_scenario_patient_level_metrics_all_prostad.csv",
        index=False,
    )

    outpatient_summary = summarise_capacity_results(
        patient_df=patient_df,
        metric_col="time_to_outpatient",
        metric_label="time_to_outpatient",
    )

    biopsy_wait_summary = summarise_capacity_results(
        patient_df=patient_df,
        metric_col="clinic_mdt_to_biopsy_wait",
        metric_label="clinic_mdt_to_biopsy_wait",
    )

    ref_to_mri_summary = summarise_capacity_results(
        patient_df=patient_df,
        metric_col="ref_to_mri_wait",
        metric_label="ref_to_mri_wait",
    )

    save_and_plot_summary(
        summary_df=outpatient_summary,
        metric_name="Median time to outpatient appointment",
        heatmap_label="Median days (referral to patient informed)",
        filename_prefix="time_to_outpatient",
    )

    save_and_plot_summary(
        summary_df=biopsy_wait_summary,
        metric_name="Mean clinic/MDT to biopsy wait",
        heatmap_label="Mean clinic/MDT to biopsy wait (days)",
        filename_prefix="clinic_mdt_to_biopsy_wait",
    )

    save_and_plot_summary(
        summary_df=ref_to_mri_summary,
        metric_name="Mean referral to MRI wait",
        heatmap_label="Mean referral to MRI wait (days)",
        filename_prefix="ref_to_mri_wait",
    )

    print_summary(outpatient_summary, "TIME TO OUTPATIENT APPOINTMENT SUMMARY")
    print_summary(biopsy_wait_summary, "CLINIC/MDT TO BIOPSY WAIT SUMMARY")
    print_summary(ref_to_mri_summary, "REFERRAL TO MRI WAIT SUMMARY")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()