from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from engine.pathway_definitions import STAGE_EVENT_PAIRS


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stage_occupancy_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.7528735632183907
SEED = 1234

SCENARIOS = ["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"]

STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT/clinic → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Pathology",
    "pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "treatmdt_to_outpat": "Treatment MDT → Outpatient",
}


def make_referral_schedule() -> dict:
    return generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )


def run_scenarios() -> dict[str, dict]:
    referral_schedule = make_referral_schedule()
    outputs = {}

    for scenario in SCENARIOS:
        cfg = build_combined_config(
            scenario,
            start_date=START_DATE,
            n_days=N_DAYS,
            lam_per_workday=LAM_PER_WORKDAY,
            seed=SEED,
        )

        outputs[scenario] = run_day_loop_combined_engine(
            cfg,
            daily_referrals_override=referral_schedule,
        )

    return outputs


def run_obs_mix() -> dict:
    referral_schedule = make_referral_schedule()

    cfg = build_combined_config(
        "OBS_MIX",
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

    return run_day_loop_combined_engine(
        cfg,
        daily_referrals_override=referral_schedule,
    )


# ------------------------------------------------------------------
# PART 1: ALL SCENARIOS OCCUPANCY
# ------------------------------------------------------------------

def extract_daily_stage_occupancy(outputs: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for scenario, result in outputs.items():
        for stage, metrics in result["stage_activity"].items():
            for day, count in metrics["daily_in_stage"].items():
                rows.append(
                    {
                        "scenario": scenario,
                        "stage": stage,
                        "date": pd.Timestamp(day),
                        "n_in_stage": int(count),
                    }
                )

    return pd.DataFrame(rows)


def add_rolling_mean_by_scenario(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.sort_values(["scenario", "stage", "date"]).copy()

    df["n_in_stage_rolling"] = (
        df.groupby(["scenario", "stage"])["n_in_stage"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def plot_stage_occupancy_all_scenarios(df: pd.DataFrame, stage: str) -> None:
    plot_df = df[df["stage"] == stage].copy()

    plt.figure(figsize=(12, 5))

    for scenario in SCENARIOS:
        subset = plot_df[plot_df["scenario"] == scenario]
        plt.plot(
            subset["date"],
            subset["n_in_stage_rolling"],
            label=scenario,
        )

    plt.title(f"Patients currently waiting: {STAGE_LABELS.get(stage, stage)}")
    plt.xlabel("Date")
    plt.ylabel("Number in stage, 14-day rolling mean")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"occupancy_all_scenarios_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def summarise_occupancy_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scenario", "stage"])
        .agg(
            mean_in_stage=("n_in_stage", "mean"),
            median_in_stage=("n_in_stage", "median"),
            peak_in_stage=("n_in_stage", "max"),
            mean_rolling_in_stage=("n_in_stage_rolling", "mean"),
            peak_rolling_in_stage=("n_in_stage_rolling", "max"),
        )
        .reset_index()
        .sort_values(["stage", "scenario"])
    )


# ------------------------------------------------------------------
# PART 2: OBS_MIX OCCUPANCY BY PATHWAY TYPE
# ------------------------------------------------------------------

def get_event_date(patient, event_name: str):
    dates = [
        event.get("date")
        for event in patient.events
        if event.get("event") == event_name and event.get("date") is not None
    ]
    return min(dates) if dates else None


def get_patient_pathway_type(patient) -> str:
    pathway_type = getattr(patient, "pathway_type", None)

    if pathway_type is None:
        pathway_type = patient.data.get("pathway_type")

    if pathway_type is None:
        pathway_type = patient.data.get("pathway")

    return pathway_type if pathway_type is not None else "UNKNOWN"


def extract_occupancy_by_pathway_type(obs_mix_result: dict) -> pd.DataFrame:
    columns = ["date", "pathway_type", "stage", "n_in_stage"]
    rows = []
    all_dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    patients = obs_mix_result.get("all_patients_objects", [])
    if not patients:
        patients = obs_mix_result.get("completed_patients_objects", [])

    for patient in patients:
        pathway_type = get_patient_pathway_type(patient)

        for (start_event, end_event), stage in STAGE_EVENT_PAIRS.items():
            start = get_event_date(patient, start_event)
            end = get_event_date(patient, end_event)

            # Patient never reached this stage.
            if start is None:
                continue

            # If there is no end event, only count them if they are genuinely
            # still active in this stage. Do NOT count branched-out patients
            # to the end of the simulation.
            if end is None:
                if getattr(patient, "is_complete", False):
                    continue

                if getattr(patient, "current_stage", None) != stage:
                    continue

                end = pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS)
            else:
                end = pd.Timestamp(end)

            start = pd.Timestamp(start)

            # Skip zero-length stages, e.g. fixed 0-day report -> MDT.
            if end <= start:
                continue

            in_stage_dates = all_dates[(all_dates >= start) & (all_dates < end)]

            for current_date in in_stage_dates:
                rows.append(
                    {
                        "date": current_date,
                        "pathway_type": pathway_type,
                        "stage": stage,
                        "n_in_stage": 1,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)

    df = (
        df.groupby(["date", "pathway_type", "stage"], as_index=False)["n_in_stage"]
        .sum()
    )

    stage_names = list(STAGE_EVENT_PAIRS.values())
    pathway_types = sorted(df["pathway_type"].dropna().unique())

    full_index = pd.MultiIndex.from_product(
        [all_dates, pathway_types, stage_names],
        names=["date", "pathway_type", "stage"],
    )

    df = (
        df.set_index(["date", "pathway_type", "stage"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return df[columns]

def add_rolling_mean_by_pathway(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.sort_values(["pathway_type", "stage", "date"]).copy()

    df["n_in_stage_rolling"] = (
        df.groupby(["pathway_type", "stage"])["n_in_stage"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def plot_mixed_pathway_occupancy(df: pd.DataFrame, stage: str) -> None:
    plot_df = df[df["stage"] == stage].copy()

    plt.figure(figsize=(12, 5))

    colours = {
        "BASELINE": "tab:blue",
        "PROSTAD": "tab:orange",
    }

    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = plot_df[plot_df["pathway_type"] == pathway_type]
        if subset.empty:
            continue

        colour = colours.get(pathway_type, None)
        mean_value = subset["n_in_stage"].mean()

        # Rolling line (solid)
        plt.plot(
            subset["date"],
            subset["n_in_stage_rolling"],
            color=colour,
            linestyle="-",
            linewidth=2,
            label=f"{pathway_type} (rolling)",
        )

        # Mean line (dashed)
        plt.axhline(
            y=mean_value,
            color=colour,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"{pathway_type} mean = {mean_value:.1f}",
        )

    plt.title(f"OBS_MIX: Patients currently waiting — {STAGE_LABELS.get(stage, stage)}")
    plt.xlabel("Date")
    plt.ylabel("Number in stage")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"obs_mix_occupancy_by_pathway_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def summarise_mixed_pathway_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pathway_type", "stage"])
        .agg(
            mean_in_stage=("n_in_stage", "mean"),
            median_in_stage=("n_in_stage", "median"),
            peak_in_stage=("n_in_stage", "max"),
            mean_rolling_in_stage=("n_in_stage_rolling", "mean"),
            peak_rolling_in_stage=("n_in_stage_rolling", "max"),
        )
        .reset_index()
        .sort_values(["stage", "pathway_type"])
    )


def main() -> None:
    # --------------------------------------------------
    # 1. All scenarios comparison
    # --------------------------------------------------
    outputs = run_scenarios()

    all_scenario_df = extract_daily_stage_occupancy(outputs)
    all_scenario_df = add_rolling_mean_by_scenario(all_scenario_df, window=14)

    all_scenario_df.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_all_scenarios.csv",
        index=False,
    )

    all_scenario_summary = summarise_occupancy_all_scenarios(all_scenario_df)
    all_scenario_summary.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_summary_all_scenarios.csv",
        index=False,
    )

    for stage in sorted(all_scenario_df["stage"].unique()):
        plot_stage_occupancy_all_scenarios(all_scenario_df, stage)

    print("\n=== ALL SCENARIOS STAGE OCCUPANCY SUMMARY ===")
    print(all_scenario_summary.round(3).to_string(index=False))

    # --------------------------------------------------
    # 2. OBS_MIX split by pathway type
    # --------------------------------------------------
    obs_mix_result = outputs["OBS_MIX"]

    pathway_df = extract_occupancy_by_pathway_type(obs_mix_result)
    pathway_df = add_rolling_mean_by_pathway(pathway_df, window=14)

    pathway_df.to_csv(
        OUTPUT_DIR / "obs_mix_daily_occupancy_by_pathway_type.csv",
        index=False,
    )

    pathway_summary = summarise_mixed_pathway_occupancy(pathway_df)
    pathway_summary.to_csv(
        OUTPUT_DIR / "obs_mix_occupancy_by_pathway_type_summary.csv",
        index=False,
    )

    for stage in STAGE_EVENT_PAIRS.values():
        plot_mixed_pathway_occupancy(pathway_df, stage)

    print("\n=== OBS_MIX OCCUPANCY BY PATHWAY TYPE ===")
    print(pathway_summary.round(3).to_string(index=False))

    print(f"\nSaved plots and CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()








