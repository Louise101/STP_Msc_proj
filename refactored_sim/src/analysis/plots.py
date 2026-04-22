from __future__ import annotations

"""Small plotting helpers for common project outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.pathway_definitions import STAGE_LABELS


def save_stage_pressure_plot(stage_summary: pd.DataFrame, output_path: Path) -> None:
    """Save a mean-in-stage comparison plot across scenarios."""
    plot_df = stage_summary.copy()
    plot_df["stage_label"] = plot_df["stage"].map(STAGE_LABELS).fillna(plot_df["stage"])

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
