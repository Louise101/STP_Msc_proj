from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from analysis.validation import build_real_pathway_csvs, load_real_pathway_data
from data_prep.empirical_inputs import load_real_stage_waits


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "mixed_standard_validation_ecdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748
SEED = 1234

SCENARIO_NAME = "OBS_MIX"
SIM_PATHWAY_TYPE = "BASELINE"
SIM_PATHWAY_LABEL = "Standard"

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


def build_mixed_result() -> dict:
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

    cfg = build_combined_config(
        SCENARIO_NAME,
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

    return run_day_loop_combined_engine(
        cfg,
        daily_referrals_override=referral_schedule,
    )


def get_patient_pathway_type(patient) -> str:
    pathway_type = getattr(patient, "pathway_type", None)

    if pathway_type is None:
        pathway_type = patient.data.get("pathway_type")

    if pathway_type is None:
        pathway_type = patient.data.get("pathway")

    return pathway_type if pathway_type is not None else "UNKNOWN"


def extract_standard_stage_waits(result: dict) -> pd.DataFrame:
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
        if get_patient_pathway_type(patient) != SIM_PATHWAY_TYPE:
            continue

        event_dates: dict[str, date] = {}

        for event in patient.events:
            event_dates[event["event"]] = event["date"]

        for start_event, end_event, stage_name in stage_pairs:
            if start_event in event_dates and end_event in event_dates:
                wait_days = (event_dates[end_event] - event_dates[start_event]).days

                if wait_days >= 0:
                    rows.append(
                        {
                            "scenario": SCENARIO_NAME,
                            "pathway_type": SIM_PATHWAY_TYPE,
                            "patient_id": patient.patient_id,
                            "stage": stage_name,
                            "wait_days": wait_days,
                        }
                    )

    return pd.DataFrame(rows)


def extract_standard_full_pathway_lengths(result: dict) -> pd.DataFrame:
    rows: list[dict] = []

    for patient in result["completed_patients_objects"]:
        if get_patient_pathway_type(patient) != SIM_PATHWAY_TYPE:
            continue

        event_names = {e["event"] for e in patient.events}

        if FULL_PATHWAY_EVENT not in event_names:
            continue

        rows.append(
            {
                "scenario": SCENARIO_NAME,
                "pathway_type": SIM_PATHWAY_TYPE,
                "patient_id": patient.patient_id,
                "total_days": (patient.current_date - patient.start_date).days,
            }
        )

    return pd.DataFrame(rows)


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    x = np.sort(x)

    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])

    return x, y


def compare_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict:
    sim_series = pd.to_numeric(sim_series, errors="coerce").dropna()
    real_series = pd.to_numeric(real_series, errors="coerce").dropna()

    if len(sim_series) == 0 or len(real_series) == 0:
        return {
            "n_sim": len(sim_series),
            "n_real": len(real_series),
            "mean_sim": np.nan,
            "mean_real": np.nan,
            "median_sim": np.nan,
            "median_real": np.nan,
            "p90_sim": np.nan,
            "p90_real": np.nan,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
        }

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


def plot_ecdf(
    sim_series: pd.Series,
    real_series: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    sim_series = pd.to_numeric(sim_series, errors="coerce").dropna()
    real_series = pd.to_numeric(real_series, errors="coerce").dropna()

    if len(sim_series) == 0 or len(real_series) == 0:
        print(f"Skipped ECDF: {title}")
        return

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

    plt.step(rx, ry, where="post", label="Real standard")
    plt.step(sx, sy, where="post", label="Simulated standard in mixed model")

    plt.xlabel("Days")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend(loc="lower right")

    plt.gcf().text(
        0.75,
        0.5,
        stats_text,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.subplots_adjust(right=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(
    sim_series: pd.Series,
    real_series: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    sim_values = pd.to_numeric(sim_series, errors="coerce").dropna().to_numpy()
    real_values = pd.to_numeric(real_series, errors="coerce").dropna().to_numpy()

    if len(sim_values) == 0 or len(real_values) == 0:
        print(f"Skipped boxplot: {title}")
        return

    stats = compare_distributions(pd.Series(sim_values), pd.Series(real_values))

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [real_values, sim_values],
        labels=["Real standard", "Simulated standard"],
        showfliers=True,
        flierprops=dict(markersize=3),
    )

    plt.ylabel("Days")
    plt.title(
        f"{title}\n"
        f"KS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3f}, "
        f"Mean diff={stats['mean_diff']:.1f} days"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Run mixed simulation and extract standard-pathway patients
    # ------------------------------------------------------------------
    result = build_mixed_result()

    sim_stage_waits = extract_standard_stage_waits(result)
    sim_pathway = extract_standard_full_pathway_lengths(result)

    # ------------------------------------------------------------------
    # 2. Load observed pre-PROSTAD standard pathway data
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
    # 3. Stage-level ECDFs, boxplots and summary table
    # ------------------------------------------------------------------
    stage_rows: list[dict] = []

    for stage in STAGE_ORDER:
        sim_series = sim_stage_waits.loc[
            sim_stage_waits["stage"] == stage,
            "wait_days",
        ]

        real_series = real_stage_waits.loc[
            real_stage_waits["stage"] == stage,
            "wait_days",
        ]

        if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
            print(f"Skipping {stage}: no matching observed/simulated data")
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
            title=f"Mixed simulation standard-pathway validation ECDF: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"ecdf_mixed_standard_{stage}.png",
        )

        plot_boxplot(
            sim_series=sim_series,
            real_series=real_series,
            title=f"Mixed simulation standard-pathway validation boxplot: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"boxplot_mixed_standard_{stage}.png",
        )

    stage_summary_df = pd.DataFrame(stage_rows)
    stage_summary_df.to_csv(
        OUTPUT_DIR / "mixed_standard_stage_validation_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Full-pathway ECDF, boxplot and summary row
    # ------------------------------------------------------------------
    pathway_summary = compare_distributions(
        sim_pathway["total_days"],
        real_pre_path["total_days"],
    )

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

    pathway_summary_df.to_csv(
        OUTPUT_DIR / "mixed_standard_full_pathway_validation_summary.csv",
        index=False,
    )

    plot_ecdf(
        sim_series=sim_pathway["total_days"],
        real_series=real_pre_path["total_days"],
        title="Mixed simulation standard-pathway validation ECDF: Full pathway time",
        out_path=OUTPUT_DIR / "ecdf_mixed_standard_full_pathway.png",
    )

    plot_boxplot(
        sim_series=sim_pathway["total_days"],
        real_series=real_pre_path["total_days"],
        title="Mixed simulation standard-pathway validation boxplot: Full pathway time",
        out_path=OUTPUT_DIR / "boxplot_mixed_standard_full_pathway.png",
    )

    # ------------------------------------------------------------------
    # 5. Combined summary table
    # ------------------------------------------------------------------
    validation_summary = pd.concat(
        [stage_summary_df, pathway_summary_df],
        ignore_index=True,
    )

    validation_summary.to_csv(
        OUTPUT_DIR / "mixed_standard_validation_summary_all.csv",
        index=False,
    )

    print("\n=== MIXED SIMULATION STANDARD-PATHWAY VALIDATION SUMMARY ===")
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