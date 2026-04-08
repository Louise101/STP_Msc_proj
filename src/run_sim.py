from datetime import date
from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday
from utils import extract_mdt_to_biopsy_waits, summarise_waits
import random
from scenarios import build_scenario_config

random.seed(10)

cfg = build_scenario_config (
    name = "ALL_MC_BASELINE",
    start_date=date(2024, 1, 1),
    n_days = 365,
    lam_per_workday= 0.586,


)

#cfg = EngineConfig(
 #   start_date=date(2024, 1, 1),
  #  n_days=365,
    #lam_per_workday=0.586, #calculated in poisson_calc.py
   # lam_per_workday= 0.586, #* random.uniform(0.8,1.2), # 20% variation
    #mri_capacity_by_weekday= {2: 4}, #4 slots on day 2 (Wednesday) - first number  = day. second number = number of slots
   # biopsy_capacity_by_weekday={3: 1, 4:1},  # Thu + Fri
    #seed=42,
   # wait_time_mode={
    #    "ref_to_mri": "DES",
     #   "mri_to_report": "MC",
      #  "report_to_biopmdt": "MC",
       # "biopmdt_to_biopsy": "DES",
   # },
    #stage_rule_mode={
     #   "mri_to_report": "FIXED",
      #  "report_to_biopmdt": "FIXED",
   # },
    #fixed_wait_days_by_stage={
     #   "mri_to_report": 1,
      #  "report_to_biopmdt": 0,
   # },
#)



results = run_day_loop_with_stage_engine(cfg)
print(results["summary_stats"])
print("Biopsy starts:",
      sum(results["resources"]["Biopsy"]["daily_started"].values()))
print("Any biopsy waits recorded:",
      any(len(v) > 0 for v in results["resources"]["Biopsy"]["daily_waits"].values()))


#mc_waits = extract_mdt_to_biopsy_waits(results["patient_results"])
#print(summarise_waits(mc_waits, label="MC upstream + DES biopsy").to_string(index=False))

#des_waits = extract_mdt_to_biopsy_waits(results["patient_results"])
#print(summarise_waits(des_waits, label="All DES").to_string(index=False))

#import pandas as pd

#daily_biopsy_arrivals = pd.Series(results["stage_activity"]["Biopsy"]["daily_arrivals"])
#daily_biopsy_completed = pd.Series(results["resources"]["Biopsy"]["daily_started"])
#daily_biopsy_backlog = pd.Series(results["stage_activity"]["Biopsy"]["daily_backlog"])

#daily_biopsy_arrivals.index = pd.to_datetime(daily_biopsy_arrivals.index)
#daily_biopsy_completed.index = pd.to_datetime(daily_biopsy_completed.index)
#daily_biopsy_backlog.index = pd.to_datetime(daily_biopsy_backlog.index)

#weekly_sim = pd.DataFrame({
 #   "demand": daily_biopsy_arrivals.resample("W-MON").sum(),
  #  "biopsy_completed": daily_biopsy_completed.resample("W-MON").sum(),
   # "mean_backlog": daily_biopsy_backlog.resample("W-MON").mean(),
    #"end_backlog": daily_biopsy_backlog.resample("W-MON").last(),
#}).fillna(0)

#weekly_sim["prev_backlog"] = weekly_sim["end_backlog"].shift(1)

#print(weekly_sim.head())

#print("\n=== Correlations ===")
#print(weekly_sim[["demand", "biopsy_completed", "prev_backlog"]].corr())

#print("Biopsy completed vs demand:",
 #     weekly_sim["biopsy_completed"].corr(weekly_sim["demand"]))

#print("Biopsy completed vs previous backlog:",
 #     weekly_sim["biopsy_completed"].corr(weekly_sim["prev_backlog"]))