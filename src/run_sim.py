from datetime import date
from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday
from utils import extract_mdt_to_biopsy_waits, summarise_waits
import random

random.seed(10)

cfg = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    #lam_per_workday=0.586, #calculated in poisson_calc.py
    lam_per_workday= 0.586, #* random.uniform(0.8,1.2), # 20% variation
    mri_capacity_by_weekday= {2: 4}, #4 slots on day 2 (Wednesday) - first number  = day. second number = number of slots
    biopsy_capacity_by_weekday={3: 1, 4:1},  # Thu + Fri
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

results["daily_referrals"]
results["resources"]["Biopsy"]["daily_started"]
results["resources"]["Biopsy"]["daily_queue_len"]

import pandas as pd

# Daily data
daily_ref = pd.Series(results["daily_referrals"])
daily_completed = pd.Series(results["resources"]["Biopsy"]["daily_started"])
daily_queue = pd.Series(results["resources"]["Biopsy"]["daily_queue_len"])

# Convert index to datetime
daily_ref.index = pd.to_datetime(daily_ref.index)
daily_completed.index = pd.to_datetime(daily_completed.index)
daily_queue.index = pd.to_datetime(daily_queue.index)

# Weekly aggregation (Monday start to match your real data)
weekly_sim = pd.DataFrame({
    "demand": daily_ref.resample("W-MON").sum(),
    "completed": daily_completed.resample("W-MON").sum(),
    "mean_queue": daily_queue.resample("W-MON").mean(),
    "end_queue": daily_queue.resample("W-MON").last(),
}).fillna(0)

print(weekly_sim.head())

weekly_sim["prev_waiting"] = weekly_sim["demand"].shift(1)

print(weekly_sim[[ "prev_waiting", "completed"]].corr())

weekly_sim["prev_waiting"] = weekly_sim["demand"].shift(1)


# Previous week queue (this is key)
weekly_sim["prev_queue"] = weekly_sim["end_queue"].shift(1)

print("\n=== Correlations ===")
print("Completed vs demand:",
      weekly_sim["completed"].corr(weekly_sim["demand"]))

print("Completed vs previous queue:",
      weekly_sim["completed"].corr(weekly_sim["prev_queue"]))


