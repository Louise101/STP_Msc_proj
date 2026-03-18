from datetime import date
from des_engine import EngineConfig, run_day_loop_with_mri_queue, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday

cfg = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586, #calculated in poisson_calc.py
    mri_capacity_by_weekday= {1: 6}, #6 slots on day 1 (Tuesday)
    seed=42,
    wait_time_mode={
        "ref_to_mri": WAIT_MODE_DES,   # use for DES queue
       # "ref_to_mri": WAIT_MODE_MC, # use for MC sampling
    }
)

results = run_day_loop_with_mri_queue(cfg, trace_one_patient_mdtday)
print(results["summary_stats"])

