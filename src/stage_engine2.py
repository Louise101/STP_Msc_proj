from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from patient_state import PatientState
from queue_resource import QueuePatient
from sampling import sample_empirical_ecdf

WAIT_MODE_MC = "MC"
WAIT_MODE_DES = "DES"


# --------------------------------------------------
# Waiting stage definitions
# --------------------------------------------------
STAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "ref_to_mri": {
        "resource": "MRI",
        "pdf_key": "pre_referral_to_mri",
        "completion_event": "mri_performed",
        "timing_type": "DES_OR_MC",
    },
    "mri_to_report": {
        "resource": "MRI_REPORT",
        "pdf_key": "pre_mri_to_mrireport",
        "completion_event": "mri_report_ready",
        "timing_type": "RULE",
        "rule_name": "next_day",

    },
    "report_to_biopmdt": {
        "resource": "BIOPSY_MDT",
        "pdf_key": "pre_mrirep_to_biopsymdt",
        "completion_event": "MDT_occured",
        "timing_type": "RULE",
        "rule_name": "same_day_or_next_day",
    },
    "biopmdt_to_biopsy": {
        "resource": "Biopsy",
        "pdf_key": "pre_biopmdt_to_biop",
        "completion_event": "biopsy_done",
    },
    "biopsy_to_pathrep": {
        "resource": "PATHOLOGY",
        "pdf_key": "pre_biop_to_pathrep",
        "completion_event": "Path_report_recieved",
    },
    "pathrep_to_treatmdt": {
        "resource": "TREATMENT_MDT",
        "pdf_key": "pre_pathrep_to_treatmdt",
        "completion_event": "Treatment_options_MDT_occured",
    },
    "treatmdt_to_outpat": {
        "resource": "OUTPATIENT",
        "pdf_key": "pre_treatmdt_to_outpat",
        "completion_event": "Outpatient_appointment_occured",
    },
}


# --------------------------------------------------
# MC delay queue item
# --------------------------------------------------
@dataclass
class DelayQueueItem:
    patient: PatientState
    entry_date: date
    ready_date: date
    sampled_wait: int
    stage_name: str


# --------------------------------------------------
# Context
# --------------------------------------------------
@dataclass
class StageContext:
    rng: np.random.Generator
    pdfs: Dict[str, Any]
    branching: Dict[str, Any]
    wait_time_mode: Dict[str, str]

    # stage_name -> ready_date -> list[DelayQueueItem]
    pending_mc: Dict[str, Dict[date, List[DelayQueueItem]]]

    # resource_name -> QueueResource
    resources: Dict[str, Any]

    stage_timing_policy: Dict[str, str] | None = None
    fixed_wait_days_by_stage: Dict[str, int] | None = None
    scenario_name: str | None = None

    stage_activity: Dict[str, Dict[str, Dict[date, int]]] = field(default_factory=dict)
        # resource_name -> arrival_date -> list[QueuePatient]
    pending_des_arrivals: Dict[str, Dict[date, List[QueuePatient]]] = field(default_factory=dict)


# --------------------------------------------------
# Sampling helpers
# --------------------------------------------------
def sample_wait_for_stage(stage_name: str, ctx: StageContext) -> int:
 
    pdf_key = STAGE_CONFIG[stage_name]["pdf_key"]
    pdf_obj = ctx.pdfs[pdf_key]

    samples = sample_empirical_ecdf(pdf_obj, rng=ctx.rng)

    return int(samples)

def get_non_des_wait_for_stage(stage_name: str, ctx: StageContext) -> int:
    stage_timing_policy = ctx.stage_timing_policy or {}
    fixed_wait_days_by_stage = ctx.fixed_wait_days_by_stage or {}

    policy = stage_timing_policy.get(stage_name, "EMPIRICAL")

    if policy == "FIXED":
        if stage_name not in fixed_wait_days_by_stage:
            raise ValueError(
                f"Stage '{stage_name}' has FIXED timing policy but no fixed wait specified."
            )
        return int(fixed_wait_days_by_stage[stage_name])

    if policy == "EMPIRICAL":
        return sample_wait_for_stage(stage_name, ctx)

    raise ValueError(f"Unknown timing policy '{policy}' for stage '{stage_name}'")
    # ---- EXAMPLE OPTIONS ----
    # Option 1: object has .sample(rng)
   # if hasattr(pdf_obj, "sample"):
    #    return int(pdf_obj.sample(ctx.rng))

    # Option 2: dict with "samples"
   # if isinstance(pdf_obj, dict) and "samples" in pdf_obj:
    #    samples = np.asarray(pdf_obj["samples"])
   #     return int(ctx.rng.choice(samples))

    # Option 3: raw list/array/Series of empirical waits
  #  try:
   #     samples = np.asarray(pdf_obj).astype(int)
  #      return int(ctx.rng.choice(samples))
 #   except Exception as e:
   #     raise ValueError(
  #          f"Could not sample wait for stage '{stage_name}' using pdf key '{pdf_key}'. "
 #           f"Please adapt sample_wait_for_stage()."
#        ) from e


def sample_mdt_decision(ctx: StageContext) -> int:
    """
    Adapt the key below if your branching dict uses a different name.
    Expects e.g. {0: 0.38, 1: 0.60, 2: 0.02}
    """
    probs = ctx.branching["biopmdt_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())
    return int(ctx.rng.choice(keys, p=p))


def sample_pathology_outcome(ctx: StageContext) -> int:
    """
    Adapt the key below if your branching dict uses a different name.
    Expects e.g. {0: 0.18, 1: 0.82}
    """
    probs = ctx.branching["pathrep_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())
    return int(ctx.rng.choice(keys, p=p))

def release_due_des_arrivals_for_day(current_date: date, ctx: StageContext) -> None:
    """
    Move DES patients whose scheduled entry date is today into the actual DES queue.
    """
    for resource_name, arrivals_by_date in ctx.pending_des_arrivals.items():
        due = arrivals_by_date.pop(current_date, [])
        for qp in due:
            ctx.resources[resource_name].add_patient(qp)

def process_all_mc_due_today_until_stable(
    current_date: date,
    ctx: StageContext,
    completed_patients: List[PatientState],
) -> int:
    """
    Repeatedly process all MC stages due today until no more same-day MC items remain.

    Returns the total number of MC items released today across all passes.
    """
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

                complete_wait_stage(
                    patient=patient,
                    stage_name=stage_name,
                    completion_date=current_date,
                    wait_days=item.sampled_wait,
                    ctx=ctx,
                )

                if patient.is_complete:
                    completed_patients.append(patient)

        total_released += released_this_pass

        if released_this_pass == 0:
            break

    return total_released

def initialize_stage_activity() -> Dict[str, Dict[str, Dict[date, int]]]:
    return {
        stage_name: {
            "daily_arrivals": {},
            "daily_in_stage": {},
            "daily_completed": {},
        }
        for stage_name in STAGE_CONFIG
    }

def initialize_pending_des_arrivals() -> Dict[str, Dict[date, List[QueuePatient]]]:
    return {
        "MRI": {},
        "Biopsy": {},
        "MRI_REPORT": {},
        "BIOPSY_MDT": {},
        "PATHOLOGY": {},
        "TREATMENT_MDT": {},
        "OUTPATIENT": {},
    }



def sample_ref_to_mri_pre_delay(ctx: StageContext) -> int:
    """
    Sample non-capacity pre-queue delay for referral -> MRI.
    This should come from the lower-quantile empirical distribution
    built in PDF_create.py.
    """
    return int(sample_empirical_ecdf(ctx.pdfs["pre_referral_to_mri_pre_delay"], ctx.rng))
# --------------------------------------------------
# Enter waiting stage
# --------------------------------------------------
def get_stage_wait_days(stage_name: str, ctx: StageContext) -> int:
    rule_mode = getattr(ctx, "stage_rule_mode", {}).get(stage_name, "DEFAULT")

    if rule_mode == "FIXED":
        return int(ctx.fixed_wait_days_by_stage[stage_name])

    # default behaviour = empirical MC
    return sample_wait_for_stage(stage_name, ctx)

def get_rule_based_wait(stage_name: str, patient: PatientState) -> int:
    """
    Deterministic/rule-based waits for PROSTAD stages.
    """
    if stage_name == "mri_to_report":
        return 1

    if stage_name == "report_to_biopmdt":
        return 0

    raise ValueError(f"No rule-based wait defined for stage '{stage_name}'")

def enter_wait_stage(patient, stage_name: str, ctx: StageContext) -> None:
    mode = ctx.wait_time_mode.get(stage_name, WAIT_MODE_MC)
    cfg = STAGE_CONFIG[stage_name]

    patient.current_stage = stage_name
    entry_date = patient.current_date
        # Record stage arrival on the date the patient enters this stage logic
    ctx.stage_activity[stage_name]["daily_arrivals"][entry_date] = (
        ctx.stage_activity[stage_name]["daily_arrivals"].get(entry_date, 0) + 1
    )

    # Only apply the hybrid pre-delay to ref->MRI when using DES
    if stage_name == "ref_to_mri" and mode == WAIT_MODE_DES:
        pre_delay = sample_ref_to_mri_pre_delay(ctx)
        entry_date = entry_date + timedelta(days=pre_delay)

        patient.data["ref_to_mri_pre_delay"] = pre_delay
        patient.data["ref_to_mri_queue_entry_date"] = entry_date

    if mode == WAIT_MODE_MC:
        timing_type = cfg.get("timing_type", "MC")

        if timing_type == "RULE":
            sampled_wait = get_rule_based_wait(stage_name, patient)
        else:
            sampled_wait = sample_wait_for_stage(stage_name, ctx)

        ready_date = entry_date + timedelta(days=sampled_wait)

        item = DelayQueueItem(
            patient=patient,
            entry_date=entry_date,
            ready_date=ready_date,
            sampled_wait=sampled_wait,
            stage_name=stage_name,
        )
        ctx.pending_mc[stage_name].setdefault(ready_date, []).append(item)

    elif mode == WAIT_MODE_DES:
        resource_name = cfg["resource"]
        if resource_name is None:
            raise ValueError(f"Stage {stage_name} is DES but has no resource")

        qp = QueuePatient(
            patient_id=patient.patient_id,
            referral_date=entry_date,
            payload={
                "patient": patient,
                "stage_name": stage_name,
                "entry_date": entry_date,
            },
        )

        ctx.pending_des_arrivals[resource_name].setdefault(entry_date, []).append(qp)

    else:
        raise ValueError(f"Unknown wait mode for stage {stage_name}: {mode}")

# --------------------------------------------------
# Route after a waiting stage completes
# --------------------------------------------------
def complete_wait_stage(
    patient: PatientState,
    stage_name: str,
    completion_date: date,
    wait_days: int,
    ctx: StageContext,
) -> None:
    patient.current_date = completion_date
    ctx.stage_activity[stage_name]["daily_completed"][completion_date] = (
        ctx.stage_activity[stage_name]["daily_completed"].get(completion_date, 0) + 1
    )

    if stage_name == "ref_to_mri":
        patient.data["wait_ref_to_mri"] = wait_days
        patient.add_event("mri_performed", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "mri_to_report", ctx)
        return

    if stage_name == "mri_to_report":
        patient.add_event("mri_report_ready", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "report_to_biopmdt", ctx)
        return

    if stage_name == "report_to_biopmdt":
        patient.add_event("MDT_occured", completion_date, wait_days=wait_days)

        mdt_outcome = sample_mdt_decision(ctx)
        patient.add_event("mdt_decision", completion_date, outcome=mdt_outcome)

        if mdt_outcome == 1:
            enter_wait_stage(patient, "biopmdt_to_biopsy", ctx)
        else:
            patient.is_complete = True
        return

    if stage_name == "biopmdt_to_biopsy":
        patient.add_event("biopsy_done", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "biopsy_to_pathrep", ctx)
        return

    if stage_name == "biopsy_to_pathrep":
        patient.add_event("Path_report_recieved", completion_date, wait_days=wait_days)

        path_outcome = sample_pathology_outcome(ctx)
        patient.add_event("Path_report_outcome", completion_date, outcome=path_outcome)

        if path_outcome == 1:
            enter_wait_stage(patient, "pathrep_to_treatmdt", ctx)
        else:
            patient.is_complete = True
        return

    if stage_name == "pathrep_to_treatmdt":
        patient.add_event(
            "Treatment_options_MDT_occured",
            completion_date,
            wait_days=wait_days,
        )
        enter_wait_stage(patient, "treatmdt_to_outpat", ctx)
        return

    if stage_name == "treatmdt_to_outpat":
        patient.add_event(
            "Outpatient_appointment_occured",
            completion_date,
            wait_days=wait_days,
        )
        patient.is_complete = True
        return

    raise ValueError(f"Unknown stage_name '{stage_name}' in complete_wait_stage().")
# --------------------------------------------------
# Daily processing helpers
# --------------------------------------------------
def process_mc_stage_for_day(
    stage_name: str,
    current_date: date,
    ctx: StageContext,
    completed_patients: List[PatientState],
) -> None:
    """
    Release all patients from an MC stage whose sampled ready_date is today.
    """
    ready_items = ctx.pending_mc[stage_name].pop(current_date, [])

    for item in ready_items:
        patient = item.patient

        complete_wait_stage(
            patient=patient,
            stage_name=stage_name,
            completion_date=current_date,
            wait_days=item.sampled_wait,
            ctx=ctx,
        )

        if patient.is_complete:
            completed_patients.append(patient)


def process_des_resource_for_day(
    resource_name: str,
    current_date: date,
    ctx: StageContext,
    completed_patients: List[PatientState],
) -> None:
    """
    Process one DES resource for the day.
    Assumes QueueResource.process_day(current_date) returns iterable of service events
    with fields:
      - event.patient   (QueuePatient)
      - event.wait_days (int)
    """
    resource = ctx.resources[resource_name]
    started_today = resource.process_day(current_date)

def process_des_resource_for_day(
    resource_name: str,
    current_date: date,
    ctx: StageContext,
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
            pre_delay = int(patient.data.get("ref_to_mri_pre_delay", 0))
            total_wait_days = pre_delay + queue_wait_days

            patient.data["ref_to_mri_queue_wait"] = queue_wait_days
            patient.data["wait_ref_to_mri"] = total_wait_days

        complete_wait_stage(
            patient=patient,
            stage_name=stage_name,
            completion_date=current_date,
            wait_days=total_wait_days,
            ctx=ctx,
        )

        if patient.is_complete:
            completed_patients.append(patient)


def create_new_patient(patient_id: int, current_date: date) -> PatientState:
    patient = PatientState(
        patient_id=patient_id,
        start_date=current_date,
        current_date=current_date,
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_received", current_date)
    return patient


def initialize_pending_mc() -> Dict[str, Dict[date, List[DelayQueueItem]]]:
    return {stage_name: {} for stage_name in STAGE_CONFIG}

def count_mc_in_stage(
    pending_mc: Dict[str, Dict[date, List[DelayQueueItem]]],
    stage_name: str,
) -> int:
    return sum(len(items) for items in pending_mc[stage_name].values())

def snapshot_stage_occupancy(current_date: date, ctx: StageContext) -> None:
    wait_time_mode = ctx.wait_time_mode or {}

    for stage_name, cfg in STAGE_CONFIG.items():
        mode = wait_time_mode.get(stage_name, WAIT_MODE_MC)

        if mode == WAIT_MODE_MC:
            in_stage = count_mc_in_stage(ctx.pending_mc, stage_name)

        elif mode == WAIT_MODE_DES:
            resource_name = cfg["resource"]
            if resource_name is None:
                in_stage = 0
            else:
                in_stage = ctx.resources[resource_name].queue_length()

        else:
            in_stage = 0

        ctx.stage_activity[stage_name]["daily_in_stage"][current_date] = in_stage

