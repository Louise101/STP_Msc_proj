from __future__ import annotations

from datetime import date
from combined_des_engine import CombinedEngineConfig
from stage_engine2 import WAIT_MODE_MC, WAIT_MODE_DES


def build_combined_config(name: str, start_date: date, n_days: int, seed: int = 1234) -> CombinedEngineConfig:
    # replace with values derived from your combined real data
    lam_per_workday = 1.10
    p_prostad = 0.48

    return CombinedEngineConfig(
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        p_prostad=p_prostad,
        mri_capacity_by_weekday_prostad={1: 4},  # example only
        seed=seed,

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

        scenario_name=name,
    )