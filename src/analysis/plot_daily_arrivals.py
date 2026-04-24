from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stage_daily_arrivals"
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


def run_scenarios() -> dict[str, dict]:
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

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


def extract_daily_stage_arrivals(outputs: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for scenario, result in outputs.items():
        stage_activity = result["stage_activity"]

        for stage, metrics in stage_activity.items():
            daily_arrivals = metrics["daily_arrivals"]

            for day, count in daily_arrivals.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "stage": stage,
                        "date": pd.Timestamp(day),
                        "daily_arrivals": int(count),
                    }
                )

    df = pd.DataFrame(rows)

    # Make sure missing dates are filled with 0 for every scenario/stage.
    all_dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    all_stages = sorted(df["stage"].unique())

    full_index = pd.MultiIndex.from_product(
        [SCENARIOS, all_stages, all_dates],
        names=["scenario", "stage", "date"],
    )

    df = (
        df.set_index(["scenario", "stage", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return df


def add_rolling_average(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.sort_values(["scenario", "stage", "date"]).copy()

    df["rolling_arrivals"] = (
        df.groupby(["scenario", "stage"])["daily_arrivals"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )

    return df


def plot_stage_arrivals(df: pd.DataFrame, stage: str) -> None:
    plot_df = df[df["stage"] == stage].copy()

    plt.figure(figsize=(12, 5))

    for scenario in SCENARIOS:
        subset = plot_df[plot_df["scenario"] == scenario]
        plt.plot(
            subset["date"],
            subset["rolling_arrivals"],
            label=scenario,
        )

    title = STAGE_LABELS.get(stage, stage)
    plt.title(f"Daily arrivals to stage: {title}")
    plt.xlabel("Date")
    plt.ylabel("Daily arrivals, 14-day rolling mean")
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"daily_arrivals_{stage}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stage_arrivals_raw_and_smoothed(df: pd.DataFrame, stage: str) -> None:
    plot_df = df[df["stage"] == stage].copy()

    plt.figure(figsize=(12, 5))

    for scenario in SCENARIOS:
        subset = plot_df[plot_df["scenario"] == scenario]

        plt.plot(
            subset["date"],
            subset["rolling_arrivals"],
            label=f"{scenario} rolling mean",
        )

    title = STAGE_LABELS.get(stage, stage)
    plt.title(f"Daily arrivals to stage: {title}")
    plt.xlabel("Date")
    plt.ylabel("Daily arrivals, 14-day rolling mean")
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"daily_arrivals_smoothed_{stage}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def summarise_stage_arrival_peaks(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scenario", "stage"])
        .agg(
            total_arrivals=("daily_arrivals", "sum"),
            mean_daily_arrivals=("daily_arrivals", "mean"),
            peak_daily_arrivals=("daily_arrivals", "max"),
            mean_14d_rolling_arrivals=("rolling_arrivals", "mean"),
            peak_14d_rolling_arrivals=("rolling_arrivals", "max"),
        )
        .reset_index()
        .sort_values(["stage", "scenario"])
    )


def main() -> None:
    outputs = run_scenarios()

    daily_df = extract_daily_stage_arrivals(outputs)
    daily_df = add_rolling_average(daily_df, window=14)

    daily_df.to_csv(OUTPUT_DIR / "daily_stage_arrivals_all_scenarios.csv", index=False)

    summary_df = summarise_stage_arrival_peaks(daily_df)
    summary_df.to_csv(OUTPUT_DIR / "daily_stage_arrival_summary.csv", index=False)

    for stage in sorted(daily_df["stage"].unique()):
        plot_stage_arrivals(daily_df, stage)

    print("\n=== DAILY STAGE ARRIVAL SUMMARY ===")
    print(summary_df.round(3).to_string(index=False))

    print(f"\nSaved plots and CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()