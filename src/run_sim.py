from datetime import date
from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday
from utils import extract_mdt_to_biopsy_waits, summarise_waits

cfg = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586, #calculated in poisson_calc.py
    mri_capacity_by_weekday= {2: 4}, #4 slots on day 2 (Wednesday) - first number  = day. second number = number of slots
    biopsy_capacity_by_weekday={3: 2, 4: 1},  # Thu + Fri
    seed=42,
    wait_time_mode={
        #"ref_to_mri": WAIT_MODE_DES,   # use for DES queue
        "ref_to_mri": WAIT_MODE_MC, # use for MC sampling
        #"mri_to_repot": WAIT_MODE_DES,
       "mri_to_repot": WAIT_MODE_MC,
       #"report_to_biopmdt": WAIT_MODE_DES,
       "report_to_biopmdt": WAIT_MODE_MC,
       "biopmdt_to_biopsy" : WAIT_MODE_DES,
      # "biopmdt_to_biopsy" : WAIT_MODE_MC,


    }
)

results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
print(results["summary_stats"])
print("Biopsy starts:",
      sum(results["resources"]["Biopsy"]["daily_started"].values()))
print("Any biopsy waits recorded:",
      any(len(v) > 0 for v in results["resources"]["Biopsy"]["daily_waits"].values()))


mc_waits = extract_mdt_to_biopsy_waits(results["patient_results"])
print(summarise_waits(mc_waits, label="MC upstream + DES biopsy").to_string(index=False))

#des_waits = extract_mdt_to_biopsy_waits(results["patient_results"])
#print(summarise_waits(des_waits, label="All DES").to_string(index=False))


