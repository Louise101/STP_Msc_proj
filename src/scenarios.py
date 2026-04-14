from datetime import date
from des_engine import EngineConfig
from stage_engine2 import WAIT_MODE_MC, WAIT_MODE_DES

ALL_STAGES = [
    "ref_to_mri",
    "mri_to_report",
    "report_to_biopmdt",
    "biopmdt_to_biopsy",
    "biopsy_to_pathrep",
    "pathrep_to_treatmdt",
    "treatmdt_to_outpat",
]


def build_scenario_config(name: str, start_date: date, n_days: int, lam_per_workday: float) -> EngineConfig:
    if name == "ALL_MC_BASELINE":
        return EngineConfig(
            start_date=start_date,
            n_days=n_days,
            lam_per_workday=lam_per_workday,
            mri_capacity_by_weekday={},
            biopsy_capacity_by_weekday={},
            wait_time_mode={stage: WAIT_MODE_MC for stage in ALL_STAGES},
            stage_timing_policy={stage: "EMPIRICAL" for stage in ALL_STAGES},
            fixed_wait_days_by_stage={},
            scenario_name=name,
        )

    if name == "HYBRID_BASELINE":
        return EngineConfig(
            start_date=start_date,
            n_days=n_days,
            lam_per_workday=lam_per_workday,
            mri_capacity_by_weekday={0: 0, 1: 4, 2: 0, 3: 0, 4: 0},
            biopsy_capacity_by_weekday={0: 0, 1: 0, 2: 0, 3: 2, 4: 1},
            wait_time_mode={
                "ref_to_mri": WAIT_MODE_DES,
                "mri_to_report": WAIT_MODE_MC,
                "report_to_biopmdt": WAIT_MODE_MC,
                "biopmdt_to_biopsy": WAIT_MODE_DES,
                "biopsy_to_pathrep": WAIT_MODE_MC,
                "pathrep_to_treatmdt": WAIT_MODE_MC,
                "treatmdt_to_outpat": WAIT_MODE_MC,
            },
            stage_timing_policy={
                "mri_to_report": "EMPIRICAL",
                "report_to_biopmdt": "EMPIRICAL",
                "biopsy_to_pathrep": "EMPIRICAL",
                "pathrep_to_treatmdt": "EMPIRICAL",
                "treatmdt_to_outpat": "EMPIRICAL",
            },
            fixed_wait_days_by_stage={},
            scenario_name=name,
        )

    if name == "PROSTAD":
        return EngineConfig(
            start_date=start_date,
            n_days=n_days,
            lam_per_workday=lam_per_workday,
            mri_capacity_by_weekday={0: 0, 1: 4, 2: 0, 3: 0, 4: 0},
            biopsy_capacity_by_weekday={0: 0, 1: 0, 2: 0, 3: 2, 4: 1},
            wait_time_mode={
                "ref_to_mri": WAIT_MODE_DES,
                "mri_to_report": WAIT_MODE_MC,
                "report_to_biopmdt": WAIT_MODE_MC,
                "biopmdt_to_biopsy": WAIT_MODE_MC,
                "biopsy_to_pathrep": WAIT_MODE_MC,
                "pathrep_to_treatmdt": WAIT_MODE_MC,
                "treatmdt_to_outpat": WAIT_MODE_MC,
            },
            stage_timing_policy={
                "mri_to_report": "FIXED",
                "report_to_biopmdt": "FIXED",
                "biopmdt_to_biopsy" : "EMPIRICAL",
                "biopsy_to_pathrep": "EMPIRICAL",
                "pathrep_to_treatmdt": "EMPIRICAL",
                "treatmdt_to_outpat": "EMPIRICAL",
            },
            fixed_wait_days_by_stage={
                "mri_to_report": 1,
                "report_to_biopmdt": 0,
                #"biopsy_to_pathrep": 8
            },
            scenario_name=name,
        )
    

    # can tweak any way 
    if name == "DES_PRE_PROSTAD":
        return EngineConfig(
            start_date=start_date,
            n_days=n_days,
            lam_per_workday=lam_per_workday,

            mri_capacity_by_weekday={ 2:2},
            biopsy_capacity_by_weekday={3: 2, 4: 1},
            mri_report_capacity_by_weekday={0: 3, 1: 3, 2: 3, 3: 3, 4: 3},
            biopsy_mdt_capacity_by_weekday={2: 50},
            pathology_capacity_by_weekday={0: 4, 1: 4, 2: 4, 3: 4, 4: 4},
            treatment_mdt_capacity_by_weekday={4: 3},
            outpatient_capacity_by_weekday={0: 3, 1: 3, 2: 3, 3: 3, 4: 3},

            wait_time_mode={
                "ref_to_mri": "DES",
                "mri_to_report": "DES",
                "report_to_biopmdt": "DES",
                "biopmdt_to_biopsy": "DES",
                "biopsy_to_pathrep": "DES",
                "pathrep_to_treatmdt": "DES",
                "treatmdt_to_outpat": "DES",
            },

            stage_timing_policy={},
            fixed_wait_days_by_stage={},
            scenario_name=name,
        )

    raise ValueError(f"Unknown scenario name: {name}")