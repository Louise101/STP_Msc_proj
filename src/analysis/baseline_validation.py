from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable
from scipy.stats import ks_2samp, shapiro

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config
from analysis.validation import build_real_pathway_csvs, load_real_pathway_data
from data_prep.empirical_inputs import load_real_stage_waits


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline_validation_ecdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 0.586 # pre lamda from combined_ref_value_calc.py
SEEDS = range(1000, 1030)

FULL_PATHWAY_EVENT = "Outpatient_appointment_occured"

STAGE_ORDER = [
    "ref_to_mri",
    "mri_to_report",
    "report_to_biopmdt",
    "biopmdt_to_biopsy",
    "biopsy_to_pathrep",
    "pathrep_to_treatmdt",
    "treatmdt_to_outpat",
]

STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Pathology",
    "pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "treatmdt_to_outpat": "Treatment MDT → Outpatient",
}


def build_baseline_results() -> list[dict]:
    """Run the ALL_BASELINE scenario across multiple seeds."""
    results = []

    for seed in SEEDS:
        print(f"Running seed {seed}")

        cfg = build_combined_config(
            "ALL_BASELINE",
            start_date=START_DATE,
            n_days=N_DAYS,
            lam_per_workday=LAM_PER_WORKDAY,
            seed=seed,
        )

        results.append(run_day_loop_combined_engine(cfg))

    return results


def extract_stage_waits(result: dict, scenario_name: str) -> pd.DataFrame:
    """Extract patient-level stage waits from event dates.

    Uses explicit event lookup rather than adjacent event pairs so that
    branch events like 'mdt_decision' do not break the stage extraction.
    """
    rows: list[dict] = []

    stage_pairs = [
        ("referral_received", "mri_performed", "ref_to_mri"),
        ("mri_performed", "mri_report_ready", "mri_to_report"),
        ("mri_report_ready", "MDT_occured", "report_to_biopmdt"),
        ("MDT_occured", "biopsy_done", "biopmdt_to_biopsy"),
        ("biopsy_done", "Path_report_recieved", "biopsy_to_pathrep"),
        ("Path_report_recieved", "Treatment_options_MDT_occured", "pathrep_to_treatmdt"),
        ("Treatment_options_MDT_occured", "Outpatient_appointment_occured", "treatmdt_to_outpat"),
    ]

    for patient in result["all_patients_objects"]:
        event_dates: dict[str, date] = {}
        for event in patient.events:
            event_dates[event["event"]] = event["date"]

        for start_event, end_event, stage_name in stage_pairs:
            if start_event in event_dates and end_event in event_dates:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "patient_id": patient.patient_id,
                        "stage": stage_name,
                        "wait_days": (event_dates[end_event] - event_dates[start_event]).days,
                    }
                )

    return pd.DataFrame(rows)


def extract_full_pathway_lengths(result: dict, scenario_name: str) -> pd.DataFrame:
    """Extract full-pathway times for patients who reached outpatient."""
    rows: list[dict] = []

    for patient in result["completed_patients_objects"]:
        event_names = {e["event"] for e in patient.events}
        if FULL_PATHWAY_EVENT not in event_names:
            continue

        rows.append(
            {
                "scenario": scenario_name,
                "patient_id": patient.patient_id,
                "total_days": (patient.current_date - patient.start_date).days,
            }
        )

    return pd.DataFrame(rows)


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinates for an ECDF."""
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y


def compare_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict:
    """Calculate summary statistics and KS test results."""
    ks_stat, ks_p = ks_2samp(sim_series, real_series)

    return {
        "n_sim": len(sim_series),
        "n_real": len(real_series),
        "mean_sim": float(sim_series.mean()),
        "mean_real": float(real_series.mean()),
        "median_sim": float(sim_series.median()),
        "median_real": float(real_series.median()),
        "p90_sim": float(np.percentile(sim_series, 90)),
        "p90_real": float(np.percentile(real_series, 90)),
        "mean_diff": float(sim_series.mean() - real_series.mean()),
        "median_diff": float(sim_series.median() - real_series.median()),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def plot_ecdf(sim_series: pd.Series, real_series: pd.Series, title: str, out_path: Path) -> None:
    """Save one ECDF comparison plot with validation statistics."""
    sx, sy = ecdf(sim_series)
    rx, ry = ecdf(real_series)

    stats = compare_distributions(sim_series, real_series)

    stats_text = (
        f"n real = {stats['n_real']}, n sim = {stats['n_sim']}\n"
        f"Mean diff = {stats['mean_diff']:.1f} days\n"
        f"Median diff = {stats['median_diff']:.1f} days\n"
        f"KS stat = {stats['ks_stat']:.3f}\n"
        f"KS p = {stats['ks_pvalue']:.3f}"
    )

    plt.figure(figsize=(8, 5))

    if len(rx):
        plt.step(rx, ry, where="post", label="Real baseline")

    if len(sx):
        plt.step(sx, sy, where="post", label="Simulated baseline")

    plt.xlabel("Days")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend(loc="lower right")

    plt.text(
        0.03,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_boxplot(sim_series: pd.Series, real_series: pd.Series, title: str, out_path: Path) -> None:
    """Save one boxplot comparison plot."""
    sim_values = sim_series.dropna().to_numpy()
    real_values = real_series.dropna().to_numpy()

    stats = compare_distributions(sim_series, real_series)

    stats_text = (
        f"n real = {stats['n_real']}, n sim = {stats['n_sim']}\n"
        f"Mean diff = {stats['mean_diff']:.1f} days\n"
        f"Median diff = {stats['median_diff']:.1f} days\n"
        f"KS stat = {stats['ks_stat']:.3f}\n"
        f"KS p = {stats['ks_pvalue']:.3f}"
    )

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [real_values, sim_values],
        labels=["Real baseline", "Simulated baseline"],
        showfliers=True,
    )

    plt.ylabel("Days")
    plt.title(
              f"{title}\nKS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3f}),"
              f"MeanDiff.={stats['mean_diff']: .1f} days")
    

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def summarise_real_wait_distributions(real_stage_waits: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive summary table for real baseline stage wait distributions."""
    rows: list[dict] = []

    for stage in STAGE_ORDER:
        values = (
            real_stage_waits.loc[real_stage_waits["stage"] == stage, "wait_days"]
            .dropna()
            .astype(float)
        )

        if values.empty:
            continue

        # Shapiro-Wilk is suitable here because each stage has n < 5000.
        shapiro_stat, shapiro_p = shapiro(values)

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        rows.append(
            {
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage),
                "n": len(values),
                "mean_days": values.mean(),
                "median_days": values.median(),
                "iqr_days": q3 - q1,
                "q1_days": q1,
                "q3_days": q3,
                "min_days": values.min(),
                "max_days": values.max(),
                "skewness": values.skew(),
                "shapiro_w": shapiro_stat,
                "shapiro_p": shapiro_p,
            }
        )

    return pd.DataFrame(rows)

def summarise_stage_validation(sim_series: pd.Series, real_series: pd.Series) -> dict:
    """Return compact validation statistics for one stage or full pathway."""
    sim = sim_series.dropna().astype(float)
    real = real_series.dropna().astype(float)

    ks_stat, ks_p = ks_2samp(sim, real)

    return {
        "n_real": len(real),
        "n_sim": len(sim),
        "median_real": real.median(),
        "median_sim": sim.median(),
        "iqr_real": real.quantile(0.75) - real.quantile(0.25),
        "iqr_sim": sim.quantile(0.75) - sim.quantile(0.25),
        "p90_real": np.percentile(real, 90),
        "p90_sim": np.percentile(sim, 90),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }

def plot_combined_stage_ecdfs(
    sim_stage_waits: pd.DataFrame,
    real_stage_waits: pd.DataFrame,
    sim_pathway: pd.DataFrame,
    real_pathway: pd.DataFrame,
    out_path: Path,
) -> None:
    """Create one multi-panel ECDF figure for stage waits plus total pathway time."""

    plot_items = []

    for stage in STAGE_ORDER:
        plot_items.append(
            {
                "label": STAGE_LABELS.get(stage, stage),
                "sim_values": sim_stage_waits.loc[
                    sim_stage_waits["stage"] == stage, "wait_days"
                ],
                "real_values": real_stage_waits.loc[
                    real_stage_waits["stage"] == stage, "wait_days"
                ],
            }
        )

    plot_items.append(
        {
            "label": "Total pathway time",
            "sim_values": sim_pathway["total_days"],
            "real_values": real_pathway["total_days"],
        }
    )

    n_items = len(plot_items)
    n_cols = 2
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 15),
        sharey=True,
    )

    axes = axes.flatten()

    for ax, item in zip(axes, plot_items):
        sim_series = item["sim_values"].dropna().astype(float)
        real_series = item["real_values"].dropna().astype(float)

        if sim_series.empty or real_series.empty:
            ax.axis("off")
            continue

        rx, ry = ecdf(real_series)
        sx, sy = ecdf(sim_series)

        ax.step(rx, ry, where="post", label="Observed", linewidth=2.5)
        ax.step(sx, sy, where="post", label="Simulated", linewidth=2.5)

        stats = summarise_stage_validation(sim_series, real_series)

        ax.set_title(item["label"], fontsize=18)
        ax.set_xlabel("Wait time (days)", fontsize=15)
        ax.set_ylabel("ECDF", fontsize=15)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=15)

        ax.text(
            0.03,
            0.97,
            f"KS={stats['ks_stat']:.2f}\np={stats['ks_pvalue']:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    for ax in axes[n_items:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=18,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Run baseline simulation
    # ------------------------------------------------------------------
    results = build_baseline_results()

# Pool patient-level outputs across all seeds
    sim_stage_waits = pd.concat(
        [
            extract_stage_waits(result, f"ALL_BASELINE_seed_{i}")
            for i, result in enumerate(results)
        ],
        ignore_index=True,
    )

    sim_pathway = pd.concat(
        [
            extract_full_pathway_lengths(result, f"ALL_BASELINE_seed_{i}")
            for i, result in enumerate(results)
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 2. Load real baseline pathway data
    # ------------------------------------------------------------------
    build_real_pathway_csvs(
        pre_ref_file=str(DATA_DIR / "pre_ref_to_mri.csv"),
        pre_outpat_file=str(DATA_DIR / "pre_treatmdt_to_outpat.csv"),
        pros_ref_file=str(DATA_DIR / "pros_ref_to_mri.csv"),
        pros_outpat_file=str(DATA_DIR / "pros_treatmdt_to_outpat.csv"),
        out_pre_file=str(DATA_DIR / "pre_pathway.csv"),
        out_pros_file=str(DATA_DIR / "pros_pathway.csv"),
    )

    real_pre_path, _ = load_real_pathway_data(
        str(DATA_DIR / "pre_pathway.csv"),
        str(DATA_DIR / "pros_pathway.csv"),
    )

    real_stage_waits = load_real_stage_waits(DATA_DIR).copy()
    real_stage_waits = real_stage_waits[real_stage_waits["scenario"] == "pre"].copy()

        # ------------------------------------------------------------------
    # 2b. Real baseline descriptive wait distribution summary
    # ------------------------------------------------------------------
    real_wait_distribution_summary = summarise_real_wait_distributions(real_stage_waits)

    real_wait_distribution_summary.to_csv(
        OUTPUT_DIR / "baseline_real_wait_distribution_summary.csv",
        index=False,
    )

    print("\n=== REAL BASELINE WAIT DISTRIBUTION SUMMARY ===")
    print(
        real_wait_distribution_summary[
            [
                "label",
                "n",
                "median_days",
                "q1_days",
                "q3_days",
                "iqr_days",
                "skewness",
                "shapiro_p",
            ]
        ]
        .round(
            {
                "median_days": 1,
                "q1_days": 1,
                "q3_days": 1,
                "iqr_days": 1,
                "skewness": 2,
                "shapiro_p": 4,
            }
        )
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 3. Combined stage-level ECDF figure + compact summary table
    #    Including total pathway time
    # ------------------------------------------------------------------
    stage_rows: list[dict] = []

    for stage in STAGE_ORDER:
        sim_series = sim_stage_waits.loc[
            sim_stage_waits["stage"] == stage, "wait_days"
        ]

        real_series = real_stage_waits.loc[
            real_stage_waits["stage"] == stage, "wait_days"
        ]

        if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
            continue

        stage_rows.append(
            {
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage),
                **summarise_stage_validation(sim_series, real_series),
            }
        )

    # Total pathway time row
    if len(sim_pathway["total_days"].dropna()) > 0 and len(real_pre_path["total_days"].dropna()) > 0:
        stage_rows.append(
            {
                "stage": "total_pathway_time",
                "label": "Total pathway time",
                **summarise_stage_validation(
                    sim_pathway["total_days"],
                    real_pre_path["total_days"],
                ),
            }
        )

    stage_summary_df = pd.DataFrame(stage_rows)

    stage_summary_df.to_csv(
        OUTPUT_DIR / "standard_pathway_stage_and_total_validation_summary.csv",
        index=False,
    )

    plot_combined_stage_ecdfs(
        sim_stage_waits=sim_stage_waits,
        real_stage_waits=real_stage_waits,
        sim_pathway=sim_pathway,
        real_pathway=real_pre_path,
        out_path=OUTPUT_DIR / "combined_stage_and_total_ecdfs_standard_pathway.png",
    )

    plot_combined_stage_ecdfs(
        sim_stage_waits=sim_stage_waits,
        real_stage_waits=real_stage_waits,
        sim_pathway=sim_pathway,
        real_pathway=real_pre_path,
        out_path=OUTPUT_DIR / "combined_stage_and_total_ecdfs_standard_pathway.svg",
    )

    print("\n=== STANDARD PATHWAY STAGE + TOTAL VALIDATION SUMMARY ===")
    print(
        stage_summary_df[
            [
                "label",
                "n_real",
                "n_sim",
                "median_real",
                "median_sim",
                "iqr_real",
                "iqr_sim",
                "p90_real",
                "p90_sim",
                "ks_stat",
                "ks_pvalue",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )



if __name__ == "__main__":
    main()