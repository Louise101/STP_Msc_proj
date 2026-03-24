from datetime import date
from des_engine import EngineConfig, run_day_loop_with_mri_queue, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday

cfg = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586, #calculated in poisson_calc.py
    mri_capacity_by_weekday= {2: 4}, #4 slots on day 2 (Wednesday) - first number  = day. second number = number of slots
    biopsy_capacity_by_weekday={3: 2, 4: 1},  # Thu + Fri
    seed=42,
    wait_time_mode={
        "ref_to_mri": WAIT_MODE_DES,   # use for DES queue
       # "ref_to_mri": WAIT_MODE_MC, # use for MC sampling
        "mri_to_repot": WAIT_MODE_DES,
       #"mri_to_repot": WAIT_MODE_MC,
        "report_to_biopmdt": WAIT_MODE_DES,
       #"report_to_biopmdt": WAIT_MODE_MC,
       "biopmdt_to_biopsy" : WAIT_MODE_DES,
      # "biopmdt_to_biopsy" : WAIT_MODE_MC,


    }
)

results = run_day_loop_with_mri_queue(cfg, trace_one_patient_mdtday)
print(results["summary_stats"])

