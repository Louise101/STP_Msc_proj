from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES,WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday
from datetime import date

# 1. Run simulation

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

sim_output= run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)


# 2. Build weekly data
import pandas as pd

def build_weekly_from_sim(sim_output):
    # Extract biopsy events
    records = []

    for events, _ in sim_output["patient_results"]:
        for e in events:
            if e["event"] == "biopsy_performed":
                records.append(e["date"])

    df = pd.DataFrame({"biopsy_date": pd.to_datetime(records)})

    df["week"] = df["biopsy_date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly = df.groupby("week").size().rename("completed").reset_index()

    return weekly


weekly_sim = build_weekly_from_sim(sim_output)

print(weekly_sim["completed"].describe())

# 3. Test 1: peaks
print(weekly_sim["completed"].describe())

# 4. Test 2: queue dynamics
# build waiting
# plot

# 5. Test 3: distribution comparison
# KS test

# 6. Test 4: correlations
# backlog vs completed

# 7. Save outputs (VERY IMPORTANT)
weekly_sim.to_csv("outputs/weekly_sim.csv", index=False)