from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any
import hashlib

import numpy as np

from core.patient import PatientState
from core.queueing import QueuePatient
from core.sampling import sample_empirical_ecdf
from engine.pathway_definitions import STAGE_CONFIG, WAIT_MODE_DES, WAIT_MODE_MC, WAIT_STREAM_BY_STAGE


@dataclass
class DelayQueueItem:
    """Represents one MC-style wait that is due to complete on a future date."""

    patient: PatientState
    entry_date: date
    ready_date: date
    sampled_wait: int
    stage_name: str


@dataclass
class StageContext:
    """Mutable state shared across the day-loop and stage transition helpers."""

    rng: np.random.Generator
    pdfs: dict[str, Any]
    branching: dict[str, Any]
    wait_time_mode: dict[str, str]
    pending_mc: dict[str, dict[date, list[DelayQueueItem]]]
    resources: dict[str, Any]
    base_seed: int

    stage_timing_policy: dict[str, str] | None = None
    fixed_wait_days_by_stage: dict[str, int] | None = None
    scenario_name: str | None = None
    stage_activity: dict[str, dict[str, dict[date, int]]] = field(default_factory=dict)
    pending_des_arrivals: dict[str, dict[date, list[QueuePatient]]] = field(default_factory=dict)


def make_stable_int_seed(*parts: object) -> int:
    """Build a deterministic seed from arbitrary components."""
    seed_string = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(seed_string.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_patient_rng(base_seed: int, patient_id: int, stream_name: str) -> np.random.Generator:
    """Return a deterministic per-patient RNG stream."""
    seed = make_stable_int_seed(base_seed, patient_id, stream_name)
    return np.random.default_rng(seed)


def initialize_pending_mc() -> dict[str, dict[date, list[DelayQueueItem]]]:
    """Create an empty MC waiting structure for every pathway stage."""
    return {stage_name: {} for stage_name in STAGE_CONFIG}


def initialize_pending_des_arrivals() -> dict[str, dict[date, list[QueuePatient]]]:
    """Create the delayed DES-arrival buffers used by queue-based stages."""
    return {"MRI_PROSTAD": {}}


def initialize_stage_activity() -> dict[str, dict[str, dict[date, int]]]:
    """Create a stage activity container used for later analysis."""
    return {
        stage_name: {
            "daily_arrivals": {},
            "daily_in_stage": {},
            "daily_completed": {},
        }
        for stage_name in STAGE_CONFIG
    }


def sample_wait_for_stage(stage_name: str, patient: PatientState, ctx: StageContext) -> int:
    """Sample one empirical wait for a named stage."""
    pdf_key = STAGE_CONFIG[stage_name]["pdf_key"]
    stream_name = WAIT_STREAM_BY_STAGE[stage_name]
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, stream_name)
    return int(sample_empirical_ecdf(ctx.pdfs[pdf_key], rng=rng))


def sample_ref_to_mri_pre_delay(patient: PatientState, ctx: StageContext) -> int:
    """Sample the non-capacity component of the PROSTAD MRI delay."""
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "pre_delay_ref_to_mri")
    return int(sample_empirical_ecdf(ctx.pdfs["pre_referral_to_mri_pre_delay"], rng=rng))


def sample_mdt_decision(patient: PatientState, ctx: StageContext) -> int:
    """Sample the biopsy MDT branching decision."""
    probs = ctx.branching["biopmdt_outcome"]
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "branch_mdt")
    return int(rng.choice(list(probs.keys()), p=list(probs.values())))


def sample_pathology_outcome(patient: PatientState, ctx: StageContext) -> int:
    """Sample the pathology outcome after biopsy."""
    probs = ctx.branching["pathrep_outcome"]
    rng = make_patient_rng(ctx.base_seed, patient.patient_id, "branch_pathology")
    return int(rng.choice(list(probs.keys()), p=list(probs.values())))


def get_rule_based_wait(stage_name: str) -> int:
    """Return deterministic waits for stages that are rule-driven in PROSTAD."""
    if stage_name == "mri_to_report":
        return 1
    if stage_name == "report_to_biopmdt":
        return 0
    raise ValueError(f"No rule-based wait defined for stage '{stage_name}'")


def release_due_des_arrivals_for_day(current_date: date, ctx: StageContext) -> None:
    """Move DES arrivals scheduled for today into the live queue resource."""
    for resource_name, arrivals_by_date in ctx.pending_des_arrivals.items():
        due_today = arrivals_by_date.pop(current_date, [])
        for queue_patient in due_today:
            ctx.resources[resource_name].add_patient(queue_patient)


def count_mc_in_stage(pending_mc: dict[str, dict[date, list[DelayQueueItem]]], stage_name: str) -> int:
    """Count how many MC items are currently pending in a stage."""
    return sum(len(items) for items in pending_mc[stage_name].values())


def snapshot_stage_occupancy(current_date: date, ctx: StageContext) -> None:
    """Record how many patients are in each stage at the end of the day."""
    for stage_name, config in STAGE_CONFIG.items():
        mode = ctx.wait_time_mode.get(stage_name, WAIT_MODE_MC)
        if mode == WAIT_MODE_MC:
            in_stage = count_mc_in_stage(ctx.pending_mc, stage_name)
        elif mode == WAIT_MODE_DES:
            resource_name = config["resource"]
            in_stage = ctx.resources[resource_name].queue_length() if resource_name else 0
        else:
            in_stage = 0

        ctx.stage_activity[stage_name]["daily_in_stage"][current_date] = in_stage
