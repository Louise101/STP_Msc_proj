from __future__ import annotations

"""Small plotting helpers for common project outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.pathway_definitions import STAGE_CONFIG
from data_prep.empirical_inputs import STAGE_LABELS
from engine.scenarios import SCENARIO_LIBRARY

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "combined_stage_pressure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_stage_pressure_plot(stage_summary: pd.DataFrame, output_path: Path) -> None:
    """Save a mean-in-stage comparison plot across scenarios."""
    plot_df = stage_summary.copy()
    plot_df["stage_label"] = plot_df["stage"].map(STAGE_CONFIG).fillna(plot_df["stage"])

    x = np.arange(len(plot_df))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["mean_in_stage_ALL_BASELINE"], width, label="All baseline")
    plt.bar(x, plot_df["mean_in_stage_OBS_MIX"], width, label="Observed mix")
    plt.bar(x + width, plot_df["mean_in_stage_ALL_PROSTAD"], width, label="All PROSTAD")
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Stage pressure comparison")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def make_stage_pressure_plot(stage_summary: pd.DataFrame) -> None:
    """Plot average mean occupancy for each stage across the three scenarios."""
    plot_df = stage_summary.copy()
    plot_df["stage_label"] = plot_df["stage"].map(STAGE_LABELS).fillna(plot_df["stage"])

    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["mean_in_stage_mean_ALL_BASELINE"], width, label="All baseline")
    plt.bar(x, plot_df["mean_in_stage_mean_OBS_MIX"], width, label="Observed mix")
    plt.bar(x + width, plot_df["mean_in_stage_mean_ALL_PROSTAD"], width, label="All PROSTAD")
    plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Stage pressure comparison across pathway mix scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_pressure_mean_in_stage.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_weekly_arrivals_plot(weekly_summary: pd.DataFrame, stage_name: str) -> None:
    """Plot weekly arrivals over time for one stage."""
    stage_df = weekly_summary[weekly_summary["stage"] == stage_name].copy()
    if stage_df.empty:
        return

    plt.figure(figsize=(12, 6))
    for scenario_name in SCENARIO_LIBRARY:
        scenario_df = stage_df[stage_df["scenario"] == scenario_name].sort_values("week_index")
        if scenario_df.empty:
            continue
        plt.plot(scenario_df["week_index"], scenario_df["weekly_arrivals_mean"], label=scenario_name)

    plt.xlabel("Week index")
    plt.ylabel("Mean weekly arrivals")
    plt.title(f"Weekly arrivals by stage: {STAGE_LABELS.get(stage_name, stage_name)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weekly_arrivals_{stage_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_pathway_distribution_plot(pathway_df: pd.DataFrame, filename: str) -> None:
    """Save full-pathway KDE curves for each scenario."""
    plt.figure(figsize=(10, 6))
    for scenario_name in pathway_df["scenario"].dropna().unique():
        subset = pathway_df[pathway_df["scenario"] == scenario_name]["total_days"]
        if len(subset) > 1:
            subset.plot(kind="kde", label=scenario_name)
    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlim(left=0)
    plt.xlabel("Total pathway days")
    plt.ylabel("Density")
    plt.title("Full-pathway time distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_obs_mix_pathway_split_plot(obs_mix_df: pd.DataFrame, filename: str) -> None:
    """Save KDE curves for baseline-pathway vs PROSTAD-pathway patients inside OBS_MIX."""
    plt.figure(figsize=(10, 6))
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = obs_mix_df[obs_mix_df["pathway_type"] == pathway_type]["total_days"]
        if len(subset) > 1:
            subset.plot(kind="kde", label=pathway_type)
    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlim(left=0)
    plt.xlabel("Total pathway days")
    plt.ylabel("Density")
    plt.title("OBS_MIX full-pathway times by pathway type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

