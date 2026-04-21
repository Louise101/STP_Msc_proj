from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List

import hashlib
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
        "resource": "MRI_PROSTAD",
        "pdf_key": "pre_referral_to_mri",
        "completion_event": "mri_performed",
    },
    "mri_to_report": {
        "resource": None,
        "pdf_key": "pre_mri_to_mrireport",
        "completion_event": "mri_report_ready",
    },
    "report_to_biopmdt": {
        "resource": None,
        "pdf_key": "pre_mrirep_to_biopsymdt",
        "completion_event": "MDT_occured",
    },
    "biopmdt_to_biopsy": {
        "resource": None,
        "pdf_key": "pre_biopmdt_to_biop",
        "completion_event": "biopsy_done",
    },
    "biopsy_to_pathrep": {
        "resource": None,
        "pdf_key": "pre_biop_to_pathrep",
        "completion_event": "Path_report_recieved",
    },
    "pathrep_to_treatmdt": {
        "resource": None,
        "pdf_key": "pre_pathrep_to_treatmdt",
        "completion_event": "Treatment_options_MDT_occured",
    },
    "treatmdt_to_outpat": {
        "resource": None,
        "pdf_key": "pre_treatmdt_to_outpat",
        "completion_event": "Outpatient_appointment_occured",
    },
}

WAIT_STREAM_BY_STAGE = {
    "ref_to_mri": "wait_ref_to_mri",
    "mri_to_report": "wait_mri_to_report",
    "report_to_biopmdt": "wait_report_to_biopmdt",
    "biopmdt_to_biopsy": "wait_biopmdt_to_biopsy",
    "biopsy_to_pathrep": "wait_biopsy_to_pathrep",
    "pathrep_to_treatmdt": "wait_pathrep_to_treatmdt",
    "treatmdt_to_outpat": "wait_treatmdt_to_outpat",
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
    pending_mc: Dict[str, Dict[date, List[DelayQueueItem]]]
    resources: Dict[str, Any]
    base_seed: int

    stage_timing_policy: Dict[str, str] | None = None
    fixed_wait_days_by_stage: Dict[str, int] | None = None
    scenario_name: str | None = None
    stage_activity: Dict[str, Dict[str, Dict[date, int]]] = field(default_factory=dict)
    pending_des_arrivals: Dict[str, Dict[date, List[QueuePatient]]] = field(default_factory=dict)


# --------------------------------------------------
# RNG helpers
# --------------------------------------------------
def make_stable_int_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_patient_rng(base_seed: int, patient_id: int, stream_name: str) -> np.random.Generator:
    seed = make_stable_int_seed(base_seed, patient_id, stream_name)
    return np.random.default_rng(seed)


# --------------------------------------------------
# Sampling helpers
# --------------------------------------------------
def sample_wait_for_stage(stage_name: str, patient: PatientState, ctx: StageContext) -> int:
    pdf_key = STAGE_CONFIG[stage_name]["pdf_key"]
    pdf_obj = ctx.pdfs[pdf_key]

    stream_name = WAIT_STREAM_BY_STAGE[stage_name]
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, stream_name)

    sampled = sample_empirical_ecdf(pdf_obj, rng=rng)
    return int(sampled)


def sample_mdt_decision(patient: PatientState, ctx: StageContext) -> int:
    probs = ctx.branching["biopmdt_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())

    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "branch_mdt")
    return int(rng.choice(keys, p=p))


def sample_pathology_outcome(patient: PatientState, ctx: StageContext) -> int:
    probs = ctx.branching["pathrep_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())

    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "branch_pathology")
    return int(rng.choice(keys, p=p))


def sample_ref_to_mri_pre_delay(patient: PatientState, ctx: StageContext) -> int:
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "pre_delay_ref_to_mri")
    return int(sample_empirical_ecdf(ctx.pdfs["pre_referral_to_mri_pre_delay"], rng))


# --------------------------------------------------
# Rule-based waits
# --------------------------------------------------
def get_rule_based_wait(stage_name: str, patient: PatientState) -> int:
    if stage_name == "mri_to_report":
        return 1
    if stage_name == "report_to_biopmdt":
        return 0
    raise ValueError(f"No rule-based wait defined for stage '{stage_name}'")


# --------------------------------------------------
# Enter waiting stage
# --------------------------------------------------
def enter_wait_stage(patient: PatientState, stage_name: str, ctx: StageContext) -> None:
    mode = ctx.wait_time_mode.get(stage_name, WAIT_MODE_MC)
    cfg = STAGE_CONFIG[stage_name]

    patient.current_stage = stage_name
    entry_date = patient.current_date

    ctx.stage_activity[stage_name]["daily_arrivals"][entry_date] = (
        ctx.stage_activity[stage_name]["daily_arrivals"].get(entry_date, 0) + 1
    )

    # Only PROSTAD MRI DES uses pre-delay
    if stage_name == "ref_to_mri" and mode == WAIT_MODE_DES:
        pre_delay = sample_ref_to_mri_pre_delay(patient, ctx)
        entry_date = entry_date #+ timedelta(days=pre_delay)

        patient.data["ref_to_mri_pre_delay"] = pre_delay
        patient.data["ref_to_mri_queue_entry_date"] = entry_date

    if mode == WAIT_MODE_MC:
        policy = (ctx.stage_timing_policy or {}).get(stage_name, "EMPIRICAL")

        if policy == "FIXED":
            sampled_wait = get_rule_based_wait(stage_name, patient)
        else:
            sampled_wait = sample_wait_for_stage(stage_name, patient, ctx)

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
# Daily processing helpers
# --------------------------------------------------
def release_due_des_arrivals_for_day(current_date: date, ctx: StageContext) -> None:
    for resource_name, arrivals_by_date in ctx.pending_des_arrivals.items():
        due = arrivals_by_date.pop(current_date, [])
        for qp in due:
            ctx.resources[resource_name].add_patient(qp)


def process_all_mc_due_today_until_stable(
    current_date: date,
    ctx: StageContext,
    completed_patients: List[PatientState],
) -> int:
    # This is unused by the combined engine directly, but kept for compatibility
    total_released = 0

    while True:
        released_this_pass = 0

        for stage_name in STAGE_CONFIG:
            ready_items = ctx.pending_mc[stage_name].pop(current_date, [])
            if not ready_items:
                continue

            released_this_pass += len(ready_items)
            total_released += len(ready_items)

        if released_this_pass == 0:
            break

    return total_released


def process_des_resource_for_day(
    resource_name: str,
    current_date: date,
    ctx: StageContext,
    completed_patients: List[PatientState],
) -> None:
    # Not used directly by combined engine; kept for compatibility
    resource = ctx.resources[resource_name]
    resource.process_day(current_date)


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


def initialize_pending_des_arrivals() -> Dict[str, Dict[date, List[QueuePatient]]]:
    return {
        "MRI_PROSTAD": {},
    }


def initialize_stage_activity() -> Dict[str, Dict[str, Dict[date, int]]]:
    return {
        stage_name: {
            "daily_arrivals": {},
            "daily_in_stage": {},
            "daily_completed": {},
        }
        for stage_name in STAGE_CONFIG
    }


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
            in_stage = ctx.resources[resource_name].queue_length() if resource_name else 0

        else:
            in_stage = 0

        ctx.stage_activity[stage_name]["daily_in_stage"][current_date] = in_stage