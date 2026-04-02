from datetime import date

from des_engine import EngineConfig, WAIT_MODE_MC, WAIT_MODE_DES


FINAL_BIOPSY_DES_CFG = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586,
    mri_capacity_by_weekday={2: 4},
    biopsy_capacity_by_weekday={3: 1, 4: 1},
    biopsy_ready_delay_days=0,
    seed=42,
    wait_time_mode={
        "ref_to_mri": WAIT_MODE_MC,
        "mri_to_report": WAIT_MODE_MC,
        "report_to_biopmdt": WAIT_MODE_MC,
        "biopmdt_to_biopsy": WAIT_MODE_DES,
    },
    initial_biopsy_queue_n=1,
    initial_biopsy_pending_n=2,
    biopsy_capacity_dropout_prob_by_weekday={3: 0.15, 4: 0.2},
    biopsy_backlog_capacity_tiers=[
        (5, {3: 2, 4: 1}),
        (10, {3: 2, 4: 2}),
    ],
)