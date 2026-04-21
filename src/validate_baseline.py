from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from combined_des_engine import CombinedEngineConfig, run_day_loop_combined_engine
from combined_stage_engine import WAIT_MODE_MC, WAIT_MODE_DES
from validate_against_real import build_real_pathway_csvs, load_real_pathway_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_config(seed: int) -> CombinedEngineConfig:
    return CombinedEngineConfig(
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=1.7528735632183907,
        p_prostad=0.0,
        mri_capacity_by_weekday_prostad={1: 4},
        seed=seed,
        scenario_name="ALL_BASELINE",
        baseline_wait_time_mode={
            "ref_to_mri": WAIT_MODE_MC,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        prostad_wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        baseline_stage_timing_policy={
            "mri_to_report": "EMPIRICAL",
            "report_to_biopmdt": "EMPIRICAL",
        },
        prostad_stage_timing_policy={
            "mri_to_report": "FIXED",
            "report_to_biopmdt": "FIXED",
        },
        prostad_fixed_wait_days_by_stage={
            "mri_to_report": 1,
            "report_to_biopmdt": 0,
        },
    )

def extract_full_pathway(result: dict) -> pd.Series:
    vals = []
    for patient in result["completed_patients_objects"]:
        names = {e["event"] for e in patient.events}
        if "Outpatient_appointment_occured" in names:
            vals.append((patient.current_date - patient.start_date).days)
    return pd.Series(vals, dtype=float)

def main():
    res = run_day_loop_combined_engine(build_config(seed=123))

    build_real_pathway_csvs(
        pre_ref_file=str(DATA_DIR / "pre_ref_to_mri.csv"),
        pre_outpat_file=str(DATA_DIR / "pre_treatmdt_to_outpat.csv"),
        pros_ref_file=str(DATA_DIR / "pros_ref_to_mri.csv"),
        pros_outpat_file=str(DATA_DIR / "pros_treatmdt_to_outpat.csv"),
        out_pre_file=str(DATA_DIR / "pre_pathway.csv"),
        out_pros_file=str(DATA_DIR / "pros_pathway.csv"),
    )

    real_pre, _ = load_real_pathway_data(
        str(DATA_DIR / "pre_pathway.csv"),
        str(DATA_DIR / "pros_pathway.csv"),
    )

    sim = extract_full_pathway(res)
    real = real_pre["total_days"]

    ks_stat, ks_p = ks_2samp(sim, real)

    summary = pd.DataFrame([{
        "n_sim": len(sim),
        "n_real": len(real),
        "mean_sim": sim.mean(),
        "mean_real": real.mean(),
        "median_sim": sim.median(),
        "median_real": real.median(),
        "p90_sim": np.percentile(sim, 90),
        "p90_real": np.percentile(real, 90),
        "pct62_sim": (sim <= 62).mean() * 100,
        "pct62_real": (real <= 62).mean() * 100,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
    }])

    print(summary.round(3).to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "baseline_validation_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    sim.plot(kind="kde", label="Simulated baseline")
    real.plot(kind="kde", label="Real baseline")
    plt.axvline(62, linestyle="--", label="62-day target")
    plt.xlabel("Total pathway days")
    plt.title("Baseline validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_validation_kde.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()