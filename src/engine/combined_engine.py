from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Any, Dict, List

import numpy as np

from core.event_log import patient_results_to_event_log
from core.patient import PatientState
from core.queueing import QueueResource
from core.sampling import sample_poisson_weekday_only
from data_prep.empirical_inputs import build_branching, build_pdfs
from engine.pathway_definitions import STAGE_CONFIG, WAIT_MODE_DES, WAIT_MODE_MC



from engine.stage_logic import (
    StageContext,
    initialize_pending_des_arrivals,
    initialize_pending_mc,
    initialize_stage_activity,
    make_patient_rng,
    release_due_des_arrivals_for_day,
    sample_mdt_decision,
    sample_pathology_outcome,
   # sample_ref_to_mri_pre_delay,
    sample_wait_for_stage,
    snapshot_stage_occupancy,
    get_rule_based_wait, 
    DelayQueueItem,
)


#configuration for the combined pathway engine.

@dataclass
class CombinedEngineConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    p_prostad: float
    mri_capacity_by_weekday_prostad: Dict[int, int]
    biopsy_capacity_by_weekday: Dict[int, int] | None = None

    seed: int = 1234

    baseline_wait_time_mode: Dict[str, str] | None = None
    prostad_wait_time_mode: Dict[str, str] | None = None

    baseline_stage_timing_policy: Dict[str, str] | None = None
    prostad_stage_timing_policy: Dict[str, str] | None = None

    baseline_fixed_wait_days_by_stage: Dict[str, int] | None = None
    prostad_fixed_wait_days_by_stage: Dict[str, int] | None = None

    scenario_name: str | None = None


#Create a new patient at pathway referral
def create_new_combined_patient(patient_id: int, current_date: date) -> PatientState:
    patient = PatientState(
        patient_id=patient_id,
        start_date=current_date,
        current_date=current_date,
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_received", current_date)
    return patient


#Assign a patient to BASELINE or PROSTAD
def sample_patient_pathway(patient: PatientState, base_seed: int, p_prostad: float) -> str:
    rng = make_patient_rng(base_seed, patient.patient_id, "route_pathway")
    u = rng.random()
    patient.data["u_pathway"] = u
    return "PROSTAD" if u < p_prostad else "BASELINE"


#assign the rule set for the patient
def get_patient_stage_rules(patient: PatientState, cfg: CombinedEngineConfig):
    if patient.pathway_type == "PROSTAD":
        return (
            cfg.prostad_wait_time_mode or {},
            cfg.prostad_stage_timing_policy or {},
            cfg.prostad_fixed_wait_days_by_stage or {},
        )

    return (
        cfg.baseline_wait_time_mode or {},
        cfg.baseline_stage_timing_policy or {},
        cfg.baseline_fixed_wait_days_by_stage or {},
    )



#Route a patient into the chosen stage using their pathway-specific rules
def enter_stage_for_patient(patient: PatientState, stage_name: str, ctx: StageContext, cfg: CombinedEngineConfig) -> None:
    wait_mode_map, timing_policy_map, fixed_wait_map = get_patient_stage_rules(patient, cfg)
    mode = wait_mode_map.get(stage_name, WAIT_MODE_MC)
    stage_cfg = STAGE_CONFIG[stage_name]

    patient.current_stage = stage_name
    entry_date = patient.current_date
    ctx.stage_activity[stage_name]["daily_arrivals"][entry_date] = (
        ctx.stage_activity[stage_name]["daily_arrivals"].get(entry_date, 0) + 1
    )

    if stage_name == "ref_to_mri" and mode == WAIT_MODE_DES:
        #pre_delay = sample_ref_to_mri_pre_delay(patient, ctx)
        #patient.data["ref_to_mri_pre_delay"] = pre_delay
        patient.data["ref_to_mri_queue_entry_date"] = entry_date

    if mode == WAIT_MODE_MC:
        timing_policy = timing_policy_map.get(stage_name, "EMPIRICAL")
        sampled_wait = (
            get_rule_based_wait(stage_name)
            if timing_policy == "FIXED"
            else sample_wait_for_stage(stage_name, patient, ctx)
        )
        ready_date = entry_date + timedelta(days=sampled_wait)
        ctx.pending_mc[stage_name].setdefault(ready_date, []).append(
            DelayQueueItem(
                patient=patient,
                entry_date=entry_date,
                ready_date=ready_date,
                sampled_wait=sampled_wait,
                stage_name=stage_name,
            )
        )
        return

    if mode == WAIT_MODE_DES:
        resource_name = stage_cfg["resource"]
        if resource_name is None:
            raise ValueError(f"Stage {stage_name} is DES but has no resource")
        from core.queueing import QueuePatient
        queue_patient = QueuePatient(
            patient_id=patient.patient_id,
            referral_date=entry_date,
            payload={
                "patient": patient,
                "stage_name": stage_name,
                "entry_date": entry_date,
            },
        )
        ctx.pending_des_arrivals[resource_name].setdefault(entry_date, []).append(queue_patient)
        return

    raise ValueError(f"Unknown wait mode '{mode}' for stage '{stage_name}'")



#Complete one stage and route the patient onward
def complete_wait_stage_combined(
    patient: PatientState,
    stage_name: str,
    completion_date: date,
    wait_days: int,
    ctx: StageContext,
    cfg: CombinedEngineConfig,
) -> None:
    patient.current_date = completion_date

    ctx.stage_activity[stage_name]["daily_completed"][completion_date] = (
        ctx.stage_activity[stage_name]["daily_completed"].get(completion_date, 0) + 1
    )

    if stage_name == "ref_to_mri":
        patient.data["wait_ref_to_mri"] = wait_days
        patient.add_event("mri_performed", completion_date, wait_days=wait_days)
        enter_stage_for_patient(patient, "mri_to_report", ctx, cfg)
        return

    if stage_name == "mri_to_report":
        patient.add_event("mri_report_ready", completion_date, wait_days=wait_days)
        enter_stage_for_patient(patient, "report_to_biopmdt", ctx, cfg)
        return

    if stage_name == "report_to_biopmdt":
        patient.add_event("MDT_occured", completion_date, wait_days=wait_days)

        mdt_outcome = sample_mdt_decision(patient, ctx)
        patient.add_event("mdt_decision", completion_date, outcome=mdt_outcome)

        if mdt_outcome == 1:
            enter_stage_for_patient(patient, "biopmdt_to_biopsy", ctx, cfg)
        else:
            patient.is_complete = True
            patient.exit_reason = "no_biopsy_after_mdt"
        return

    if stage_name == "biopmdt_to_biopsy":
        patient.add_event("biopsy_done", completion_date, wait_days=wait_days)
        enter_stage_for_patient(patient, "biopsy_to_pathrep", ctx, cfg)
        return

    if stage_name == "biopsy_to_pathrep":
        patient.add_event("Path_report_recieved", completion_date, wait_days=wait_days)

        path_outcome = sample_pathology_outcome(patient, ctx)
        patient.add_event("Path_report_outcome", completion_date, outcome=path_outcome)

        if path_outcome == 1:
            enter_stage_for_patient(patient, "pathrep_to_treatmdt", ctx, cfg)
        else:
            patient.is_complete = True
            patient.exit_reason = "no_cancer_on_pathology"
        return

    if stage_name == "pathrep_to_treatmdt":
        patient.add_event("Treatment_options_MDT_occured", completion_date, wait_days=wait_days)
        enter_stage_for_patient(patient, "treatmdt_to_outpat", ctx, cfg)
        return

    if stage_name == "treatmdt_to_outpat":
        patient.add_event("Outpatient_appointment_occured", completion_date, wait_days=wait_days)
        patient.is_complete = True
        patient.exit_reason = "full_pathway_complete"
        return

    raise ValueError(f"Unknown stage_name '{stage_name}'")


#Release all MC waits due today
def process_all_mc_due_today_until_stable_combined(
    current_date: date,
    ctx: StageContext,
    cfg: CombinedEngineConfig,
    completed_patients: List[PatientState],
) -> int:
    total_released = 0

    while True:
        released_this_pass = 0

        for stage_name in STAGE_CONFIG:
            ready_items = ctx.pending_mc[stage_name].pop(current_date, [])
            if not ready_items:
                continue

            released_this_pass += len(ready_items)

            for item in ready_items:
                patient = item.patient

                complete_wait_stage_combined(
                    patient=patient,
                    stage_name=stage_name,
                    completion_date=current_date,
                    wait_days=item.sampled_wait,
                    ctx=ctx,
                    cfg=cfg,
                )

                if patient.is_complete:
                    completed_patients.append(patient)

        total_released += released_this_pass

        if released_this_pass == 0:
            break

    return total_released

#Process one DES resource and route any started patients onward
def process_des_resource_for_day_combined(
    resource_name: str,
    current_date: date,
    ctx: StageContext,
    cfg: CombinedEngineConfig,
    completed_patients: List[PatientState],
) -> None:
    resource = ctx.resources[resource_name]
    started_today = resource.process_day(current_date)

    for service_event in started_today:
        queue_patient = service_event.patient
        patient = queue_patient.payload["patient"]
        stage_name = queue_patient.payload["stage_name"]

        queue_wait_days = int(service_event.wait_days)
        total_wait_days = queue_wait_days

        if stage_name == "ref_to_mri":
            #pre_delay = int(patient.data.get("ref_to_mri_pre_delay", 0))
            total_wait_days = queue_wait_days #+ pre_delay
            patient.data["ref_to_mri_queue_wait"] = queue_wait_days
            patient.data["wait_ref_to_mri"] = total_wait_days

        complete_wait_stage_combined(
            patient=patient,
            stage_name=stage_name,
            completion_date=current_date,
            wait_days=total_wait_days,
            ctx=ctx,
            cfg=cfg,
        )

        if patient.is_complete:
            completed_patients.append(patient)



#Run the combined engine for one full simulation
def run_day_loop_combined_engine(
    cfg: CombinedEngineConfig,
    *,
    rng: Optional[np.random.Generator] = None,
    daily_referrals_override: Dict[date, int] | None = None,
) -> Dict[str, Any]:
    rng = rng or np.random.default_rng(cfg.seed)

    pdfs = build_pdfs()
    branching = build_branching()

    mri_resource_prostad = QueueResource("MRI_PROSTAD", cfg.mri_capacity_by_weekday_prostad)

    biopsy_capacity = cfg.biopsy_capacity_by_weekday or {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    }
    biopsy_resource = QueueResource("BIOPSY", biopsy_capacity)


    ctx = StageContext(
        rng=rng,
        pdfs=pdfs,
        branching=branching,
        wait_time_mode={},
        pending_mc=initialize_pending_mc(),
        resources={
            "MRI_PROSTAD": mri_resource_prostad,
            "BIOPSY": biopsy_resource,
            },
        base_seed=cfg.seed,
        stage_timing_policy={},
        fixed_wait_days_by_stage={},
        scenario_name=cfg.scenario_name,
        stage_activity=initialize_stage_activity(),
        pending_des_arrivals=initialize_pending_des_arrivals(),
    )

    completed_patients: list[PatientState] = []
    all_patients: list[PatientState] = []
    daily_referrals: dict[date, int] = {}

    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        if daily_referrals_override is not None:
            n_new = int(daily_referrals_override.get(current_date, 0))
        else:
            n_new = sample_poisson_weekday_only(
            lam_per_workday=cfg.lam_per_workday,
            weekday=current_date.weekday(),
            rng=rng,
        )
        daily_referrals[current_date] = n_new

        for _ in range(n_new):
            patient = create_new_combined_patient(next_pid, current_date)
            patient.pathway_type = sample_patient_pathway(patient, cfg.seed, cfg.p_prostad)
            patient.data["pathway_type"] = patient.pathway_type

            all_patients.append(patient)
            next_pid += 1

            enter_stage_for_patient(patient, "ref_to_mri", ctx, cfg)

        release_due_des_arrivals_for_day(current_date, ctx)

        process_des_resource_for_day_combined(
            "MRI_PROSTAD",
        current_date,
            ctx,
            cfg,
            completed_patients,
        )

        process_des_resource_for_day_combined(
            "BIOPSY",
            current_date,
            ctx,
            cfg,
            completed_patients,
        )

        process_all_mc_due_today_until_stable_combined(
            current_date,
            ctx,
            cfg,
            completed_patients,
        )

        release_due_des_arrivals_for_day(current_date, ctx)

        process_des_resource_for_day_combined(
            "BIOPSY",
            current_date,
            ctx,
            cfg,
            completed_patients,
        )

        process_all_mc_due_today_until_stable_combined(
            current_date,
            ctx,
            cfg,
            completed_patients,
        )

        snapshot_stage_occupancy(current_date, ctx)

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
        "p_prostad": cfg.p_prostad,
        "capacity_by_resource": {
            "MRI_PROSTAD": cfg.mri_capacity_by_weekday_prostad,
            "BIOPSY": biopsy_capacity,
        },
        "final_queue_length_by_resource": {
            "MRI_PROSTAD": mri_resource_prostad.queue_length(),
            "BIOPSY": biopsy_resource.queue_length(),
        },
    }

    event_log_df = patient_results_to_event_log(
        all_patient_results,
        source_engine="COMBINED_ENGINE",
        start_date=cfg.start_date,
    )

    return {
        "patient_results": patient_results,
        "all_patient_results": all_patient_results,
        "event_log": event_log_df,
        "daily_referrals": daily_referrals,
        "completed_patients_objects": completed_patients,
        "all_patients_objects": all_patients,
        "resources": {
            "MRI_PROSTAD": {
                "daily_started": mri_resource_prostad.daily_started,
                "daily_queue_len": mri_resource_prostad.daily_queue_len,
                "daily_waits": mri_resource_prostad.daily_waits,
            },
            "BIOPSY": {
                "daily_started": biopsy_resource.daily_started,
                "daily_queue_len": biopsy_resource.daily_queue_len,
                "daily_waits": biopsy_resource.daily_waits,
            },
        },
        "stage_activity": ctx.stage_activity,
        "summary_stats": summary_stats,
    }