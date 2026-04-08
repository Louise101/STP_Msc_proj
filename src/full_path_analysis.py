import pandas as pd
import numpy as np

def get_event_date(events, name):
    for e in events:
        if e["event"] == name:
            return e["date"]
    return None


def summarise_stage_waits(patient_results):
    rows = []

    for events, total_days in patient_results:
        referral = get_event_date(events, "referral_recieved")
        mri = get_event_date(events, "mri_performed")
        report = get_event_date(events, "mri_report_ready")
        biopmdt = get_event_date(events, "MDT_occured")
        biopsy = get_event_date(events, "biopsy_done")
        pathrep = get_event_date(events, "Path_report_recieved")
        treatmdt = get_event_date(events, "Treatment_options_MDT_occured")
        outpat = get_event_date(events, "Outpatient_appointment_occured")

        row = {
            "total_pathway": total_days,
            "wait_ref_to_mri": (mri - referral).days if referral and mri else None,
            "wait_mri_to_report": (report - mri).days if mri and report else None,
            "wait_report_to_biopmdt": (biopmdt - report).days if report and biopmdt else None,
            "wait_biopmdt_to_biopsy": (biopsy - biopmdt).days if biopmdt and biopsy else None,
            "wait_biopsy_to_pathrep": (pathrep - biopsy).days if biopsy and pathrep else None,
            "wait_pathrep_to_treatmdt": (treatmdt - pathrep).days if pathrep and treatmdt else None,
            "wait_treatmdt_to_outpat": (outpat - treatmdt).days if treatmdt and outpat else None,
        }

        rows.append(row)

    return pd.DataFrame(rows)

from datetime import date
from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday
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
       "biopsy_to_pathrep" : WAIT_MODE_MC,}
)

results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
stage_df = summarise_stage_waits(results["patient_results"])

summary = stage_df.agg(["count", "mean", "median", "std", "min", "max"]).T
summary["p90"] = stage_df.quantile(0.9)
print(summary)

def summarise_series(s):
    s = s.dropna()
    return {
        "n": len(s),
        "mean": float(s.mean()) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "p90": float(s.quantile(0.9)) if len(s) else None,
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
    }

for col in stage_df.columns:
    print(col, summarise_series(stage_df[col]))