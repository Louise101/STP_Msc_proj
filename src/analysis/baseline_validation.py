from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

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
SEED = 1

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


def build_baseline_result() -> dict:
    """Run the ALL_BASELINE scenario once for validation plots."""
    cfg = build_combined_config(
        "ALL_BASELINE",
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )
    return run_day_loop_combined_engine(cfg)


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

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Run baseline simulation
    # ------------------------------------------------------------------
    result = build_baseline_result()

    sim_stage_waits = extract_stage_waits(result, "ALL_BASELINE")
    sim_pathway = extract_full_pathway_lengths(result, "ALL_BASELINE")

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
    # 3. Stage-level ECDF plots + summary table
    # ------------------------------------------------------------------
    stage_rows: list[dict] = []

    for stage in STAGE_ORDER:
        sim_series = sim_stage_waits.loc[sim_stage_waits["stage"] == stage, "wait_days"]
        real_series = real_stage_waits.loc[real_stage_waits["stage"] == stage, "wait_days"]

        if len(sim_series) == 0 or len(real_series) == 0:
            continue

        stage_rows.append(
            {
                "level": "stage",
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage),
                **compare_distributions(sim_series, real_series),
            }
        )

        plot_ecdf(
            sim_series=sim_series,
            real_series=real_series,
            title=f"Baseline validation ECDF: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"ecdf_{stage}.png",
        )

        plot_boxplot(
            sim_series=sim_series,
            real_series=real_series,
            title=f"Baseline validation boxplot: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"boxplot_{stage}.png",
        )

    stage_summary_df = pd.DataFrame(stage_rows)
    stage_summary_df.to_csv(OUTPUT_DIR / "baseline_stage_validation_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Full-pathway ECDF + summary row
    # ------------------------------------------------------------------
    pathway_summary = compare_distributions(sim_pathway["total_days"], real_pre_path["total_days"])
    pathway_summary_df = pd.DataFrame(
        [
            {
                "level": "full_pathway",
                "stage": "full_pathway",
                "label": "Full pathway",
                **pathway_summary,
            }
        ]
    )
    pathway_summary_df.to_csv(OUTPUT_DIR / "baseline_full_pathway_validation_summary.csv", index=False)

    plot_ecdf(
        sim_series=sim_pathway["total_days"],
        real_series=real_pre_path["total_days"],
        title="Baseline validation ECDF: Full pathway time",
        out_path=OUTPUT_DIR / "ecdf_full_pathway.png",
    )

    plot_boxplot(
        sim_series=sim_pathway["total_days"],
        real_series=real_pre_path["total_days"],
        title="Baseline validation boxplot: Full pathway time",
        out_path=OUTPUT_DIR / "boxplot_full_pathway.png",
    )

    # ------------------------------------------------------------------
    # 5. Combined summary table
    # ------------------------------------------------------------------
    validation_summary = pd.concat([stage_summary_df, pathway_summary_df], ignore_index=True)
    validation_summary.to_csv(OUTPUT_DIR / "baseline_validation_summary_all.csv", index=False)

    print("\n=== BASELINE VALIDATION SUMMARY ===")
    print(
        validation_summary[
            [
                "label",
                "n_sim",
                "n_real",
                "mean_sim",
                "mean_real",
                "median_sim",
                "median_real",
                "p90_sim",
                "p90_real",
                "mean_diff",
                "median_diff",
                "ks_stat",
                "ks_pvalue",
            ]
        ].round(3).to_string(index=False)
    )
    
    
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()