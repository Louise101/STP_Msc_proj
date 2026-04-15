from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from patient_state import PatientState
from queue_resource import QueuePatient, QueueResource
from sampling import sample_poisson_weekday_only
from PDF_create import build_pdfs, build_branching
from stage_engine2 import (
    StageContext,
    WAIT_MODE_MC,
    WAIT_MODE_DES,
    STAGE_CONFIG,
    initialize_pending_mc,
    initialize_pending_des_arrivals,
    release_due_des_arrivals_for_day,
    enter_wait_stage,
    process_des_resource_for_day,
    process_all_mc_due_today_until_stable,
    create_new_patient,
    initialize_stage_activity,
    snapshot_stage_occupancy,
)
from event_log_utils import patient_results_to_event_log




@dataclass
class EngineConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    mri_capacity_by_weekday: Dict[int, int]
    biopsy_capacity_by_weekday: Dict[int, int]
    daily_referrals_override: Dict[date, int] | None = None

    mri_report_capacity_by_weekday: Dict[int, int] | None = None
    biopsy_mdt_capacity_by_weekday: Dict[int, int] | None = None
    pathology_capacity_by_weekday: Dict[int, int] | None = None
    treatment_mdt_capacity_by_weekday: Dict[int, int] | None = None
    outpatient_capacity_by_weekday: Dict[int, int] | None = None

    biopsy_ready_delay_days: int = 0
    seed: int = 1234
    wait_time_mode: Dict[str, str] | None = None
    stage_rule_mode: Dict[str, str] | None = None
    fixed_wait_days_by_stage:Dict[str, str] | None = None

    scenario_name : str | None = None


    stage_timing_policy: Dict[str, str] | None = None

    initial_biopsy_queue_n: int = 1
    initial_biopsy_pending_n: int = 2

    biopsy_capacity_dropout_prob_by_weekday: Dict[int, float] = field(
        default_factory=lambda: {3: 0.15, 4: 0.2}
    )

    biopsy_backlog_capacity_tiers: List[Tuple[int, Dict[int, int]]] = field(
        default_factory=lambda: [
            (5, {3: 2, 4: 1}),
            (10, {3: 2, 4: 2}),
        ]
    )





def run_day_loop_with_stage_engine(
    cfg: EngineConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    rng = rng or np.random.default_rng(cfg.seed)
    wait_time_mode = cfg.wait_time_mode or {}

    pdfs = build_pdfs()
    branching = build_branching()

    mri_resource = QueueResource("MRI", cfg.mri_capacity_by_weekday)
    biopsy_resource = QueueResource("Biopsy", cfg.biopsy_capacity_by_weekday or {})
    mri_report_resource = QueueResource("MRI_REPORT", cfg.mri_report_capacity_by_weekday or {})
    biopsy_mdt_resource = QueueResource("BIOPSY_MDT", cfg.biopsy_mdt_capacity_by_weekday or {})
    pathology_resource = QueueResource("PATHOLOGY", cfg.pathology_capacity_by_weekday or {})
    treatment_mdt_resource = QueueResource("TREATMENT_MDT", cfg.treatment_mdt_capacity_by_weekday or {})
    outpatient_resource = QueueResource("OUTPATIENT", cfg.outpatient_capacity_by_weekday or {})

    pending_mc = initialize_pending_mc()
    pending_des_arrivals = initialize_pending_des_arrivals()
    stage_activity = initialize_stage_activity()

    ctx = StageContext(
        rng=rng,
        pdfs=pdfs,
        branching=branching,
        wait_time_mode=wait_time_mode,
        pending_mc=pending_mc,
        base_seed=cfg.seed,
        resources={
            "MRI": mri_resource,
            "Biopsy": biopsy_resource,
            "MRI_REPORT":mri_report_resource,
            "BIOPSY_MDT":biopsy_mdt_resource,
            "PATHOLOGY": pathology_resource,
            "TREATMENT_MDT":treatment_mdt_resource, 
            "OUTPATIENT":outpatient_resource 


        },
        pending_des_arrivals=pending_des_arrivals,
        stage_timing_policy = cfg.stage_timing_policy or {},
        fixed_wait_days_by_stage=cfg.fixed_wait_days_by_stage or {},
        scenario_name= None,
        stage_activity=stage_activity,
    )

    completed_patients = []
    all_patients = []
    daily_referrals = {}
    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        # 1) Add new referrals first
        if cfg.daily_referrals_override is not None:
            n_new = int(cfg.daily_referrals_override.get(current_date, 0))
        else:
            n_new = sample_poisson_weekday_only(
                lam_per_workday=cfg.lam_per_workday,
                weekday=current_date.weekday(),
                rng=rng,
            )

        daily_referrals[current_date] = n_new

        for _ in range(n_new):
            patient = create_new_patient(next_pid, current_date)
            all_patients.append(patient)
            next_pid += 1
            enter_wait_stage(patient, "ref_to_mri", ctx)

        # 2) Release DES arrivals scheduled for today into their queues
        release_due_des_arrivals_for_day(current_date, ctx)

        # 3) Process DES resources for today
        
        des_resources_to_process = set()
        for stage_name, stage_cfg in STAGE_CONFIG.items():
            if wait_time_mode.get(stage_name, WAIT_MODE_MC) == WAIT_MODE_DES:
                resource_name = stage_cfg["resource"]
                if resource_name is not None:
                    des_resources_to_process.add(resource_name)

        for resource_name in des_resources_to_process:
            process_des_resource_for_day(resource_name, current_date, ctx, completed_patients)

        # 3) Repeatedly process all MC work due today until stable
        process_all_mc_due_today_until_stable(current_date, ctx, completed_patients)

        # 4) snapshot occupancy at end of day
        snapshot_stage_occupancy(current_date, ctx)

        # 4) Advance day
        current_date += timedelta(days=1)

    patient_results = []
    for p in completed_patients:
        total_days = (p.current_date - p.start_date).days
        patient_results.append((p.events, total_days))

    all_patient_results = []
    for p in all_patients:
        total_days = (p.current_date - p.start_date).days
        all_patient_results.append((p.events, total_days))

    

    summary_stats = {
        "total_days": cfg.n_days,
        "total_patients_completed": len(patient_results),
        "lambda_target": cfg.lam_per_workday,
        "capacity_by_resource": {
            "MRI": cfg.mri_capacity_by_weekday,
            "Biopsy": cfg.biopsy_capacity_by_weekday or {},
            "MRI_REPORT":cfg.mri_report_capacity_by_weekday or {},
            "BIOPSY_MDT":cfg.biopsy_mdt_capacity_by_weekday or {},
            "PATHOLOGY": cfg.pathology_capacity_by_weekday or{},
            "TREATMENT_MDT":cfg.treatment_mdt_capacity_by_weekday or {}, 
            "OUTPATIENT":cfg.outpatient_capacity_by_weekday or {}

        },
        "final_queue_length_by_resource": {
            "MRI": mri_resource.queue_length(),
            "Biopsy": biopsy_resource.queue_length(),
            "MRI_REPORT":mri_report_resource.queue_length(),
            "BIOPSY_MDT":biopsy_mdt_resource.queue_length(),
            "PATHOLOGY": pathology_resource.queue_length(),
            "TREATMENT_MDT":treatment_mdt_resource.queue_length(), 
            "OUTPATIENT":outpatient_resource.queue_length() 

        },
    }

    event_log_df = patient_results_to_event_log(
        all_patient_results,
        source_engine="STAGE_ENGINE",
        start_date=cfg.start_date,
    )

   # print("\nDebug check: first 10 completed patients with MRI wait components")
   # shown = 0
    #for p in completed_patients:
     #   if "wait_ref_to_mri" in p.data:
      #      print(
       #         p.patient_id,
        #        p.data.get("ref_to_mri_pre_delay"),
         #       p.data.get("ref_to_mri_queue_wait"),
          #      p.data.get("wait_ref_to_mri"),
           # )
            #shown += 1
            #if shown >= 10:
             #   break   

    return {
        "patient_results": patient_results,
        "all_patient_results" : all_patient_results,
        "event_log" : event_log_df,
        "daily_referrals": daily_referrals,
        "completed_patients_objects": completed_patients,
        "all_patients_objects": all_patients,
        "resources": {
            "MRI": {
                "daily_started": mri_resource.daily_started,
                "daily_queue_len": mri_resource.daily_queue_len,
                "daily_waits": mri_resource.daily_waits,
            },
            "Biopsy": {
                "daily_started": biopsy_resource.daily_started,
                "daily_queue_len": biopsy_resource.daily_queue_len,
                "daily_waits": biopsy_resource.daily_waits,
            },
            "MRI_REPORT":{
                "daily_started":mri_report_resource.daily_started,
                "daily_queue_len": mri_report_resource.daily_queue_len,
                "daily_waits":mri_report_resource.daily_waits,
            },    
            "BIOPSY_MDT":{
                "daily_started":biopsy_mdt_resource.daily_started,
                "daily_queue_len": biopsy_mdt_resource.daily_queue_len,
                "daily_waits":biopsy_mdt_resource.daily_waits,
            },
            "PATHOLOGY":{
                "daily_started":pathology_resource.daily_started,
                "daily_queue_len": pathology_resource.daily_queue_len,
                "daily_waits":pathology_resource.daily_waits,
            },
            
            "TREATMENT_MDT":{
                "daily_started":treatment_mdt_resource.daily_started,
                "daily_queue_len": treatment_mdt_resource.daily_queue_len,
                "daily_waits":treatment_mdt_resource.daily_waits,
            },
            "OUTPATIENT":{
                "daily_started":outpatient_resource .daily_started,
                "daily_queue_len": outpatient_resource .daily_queue_len,
                "daily_waits":outpatient_resource .daily_waits,
            },
           

        },
        "stage_activity" : ctx.stage_activity,
        "summary_stats": summary_stats,
        
    }

