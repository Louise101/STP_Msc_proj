from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from patient_state import PatientState
from queue_resource import QueuePatient
from sampling import sample_empirical_ecdf, sample_outcome

WAIT_MODE_MC = "MC"
WAIT_MODE_DES = "DES"


@dataclass
class StageContext:
    rng: np.random.Generator
    pdfs: Dict[str, Any]
    branching: Dict[str, Any]
    wait_time_mode: Dict[str, str]
    pending_arrivals: Dict[str, Dict[date, list[QueuePatient]]]
    resources: Dict[str, Any]
    biopsy_ready_delay_days: int = 1


STAGE_SEQUENCE = [
    "ref_to_mri",
    "mri_to_report",
    "report_to_biopmdt",
    "biopmdt_to_biopsy",
    "downstream_tail",
]

def next_weekday(d: date) -> date:
    # 0=Mon ... 6=Sun
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def advance_patient_until_pause_or_end(patient: PatientState, ctx: StageContext) -> None:
    while not patient.is_complete:
        handler = STAGE_HANDLERS[patient.current_stage]
        result = handler(patient, ctx)

        if result == "PAUSED_FOR_DES":
            return

        if result == "COMPLETE":
            return
        
def resume_after_mri(patient: PatientState, mri_day: date, queue_wait: int) -> None:
    patient.data["wait_ref_to_mri"] = queue_wait
    patient.data["mri_date"] = mri_day
    patient.current_date = mri_day
    patient.current_stage = "mri_to_report"
    patient.add_event("mri_performed", mri_day, wait_days=queue_wait)


def resume_after_biopsy(patient: PatientState, biopsy_day: date, ctx: StageContext) -> None:
    #residual = sample_empirical_ecdf(ctx.pdfs["biopsy_residual_samples"], ctx.rng)
    residual = sample_empirical_ecdf(ctx.pdfs["pre_biopmdt_to_biop"], ctx.rng)
    final_biopsy_day = biopsy_day + timedelta(days=residual)

    total_wait = (final_biopsy_day - patient.data["biopmdt_date"]).days
    patient.data["biopmdt_to_biopsy_queue_wait"] = (biopsy_day - patient.data["biopmdt_date"]).days
    patient.data["biopmdt_to_biopsy_residual_wait"] = residual
    patient.data["wait_biopmdt_to_biopsy"] = total_wait
    patient.data["biopsy_date"] = final_biopsy_day

    patient.current_date = final_biopsy_day
    patient.current_stage = "downstream_tail"
    patient.add_event("biopsy_done", final_biopsy_day, wait_days=total_wait)

#edit to add week day constraint and logging 
def process_ref_to_mri(patient: PatientState, ctx: StageContext) -> str:
    mode = ctx.wait_time_mode.get("ref_to_mri", WAIT_MODE_MC)

    if mode == WAIT_MODE_MC:
        wait_days = sample_empirical_ecdf(ctx.pdfs["pre_referral_to_mri"], ctx.rng)
        mri_date_raw = patient.current_date + timedelta(days=wait_days)
        mri_date= next_weekday(mri_date_raw)

        patient.data["wait_ref_to_mri"] = wait_days
        patient.data["mri_date"] = mri_date
        patient.current_date = mri_date
        patient.current_stage = "mri_to_report"

        patient.add_event("mri_performed", mri_date, wait_days=wait_days)
        return "ADVANCED"

    if mode == WAIT_MODE_DES:
        ctx.resources["MRI"].add_patient(
            QueuePatient(
                patient_id=patient.patient_id,
                referral_date=patient.current_date,
                payload={"patient": patient},
            )
        )
        return "PAUSED_FOR_DES"

    raise ValueError(f"Unknown mode for ref_to_mri: {mode}")



def process_mri_to_report(patient: PatientState, ctx: StageContext) -> str:
    mode = ctx.wait_time_mode.get("mri_to_report", WAIT_MODE_MC)

    if mode == WAIT_MODE_MC:
        wait_days = sample_empirical_ecdf(ctx.pdfs["pre_mri_to_mrireport"], ctx.rng)
    else:
        wait_days = 1

    report_date_raw = patient.current_date + timedelta(days=wait_days)
    report_date = next_weekday(report_date_raw)

    patient.data["wait_mri_to_report"] = wait_days
    patient.data["report_date"] = report_date
    patient.current_date = report_date
    patient.current_stage = "report_to_biopmdt"

    patient.add_event("mri_report_ready", report_date, wait_days=wait_days)
    return "ADVANCED"



def process_report_to_biopmdt(patient: PatientState, ctx: StageContext) -> str:
    mode = ctx.wait_time_mode.get("report_to_biopmdt", WAIT_MODE_MC)

    if mode == WAIT_MODE_MC:
        wait_days = sample_empirical_ecdf(ctx.pdfs["pre_mrirep_to_biopsymdt"], ctx.rng)
    else:
        wait_days = 0

    biopmdt_date_raw = patient.current_date + timedelta(days=wait_days)
    biopmdt_date=next_weekday(biopmdt_date_raw)

    patient.data["wait_report_to_biopmdt"] = wait_days
    patient.data["biopmdt_date"] = biopmdt_date
    patient.current_date = biopmdt_date

    patient.add_event("MDT_occured", biopmdt_date, wait_days=wait_days)

    # branching decision
    raw_outcome = sample_outcome(ctx.branching["biopmdt_outcome"], rng=ctx.rng)

    # normalize outcome to int if possible
    try:
        outcome = int(raw_outcome)
    except (TypeError, ValueError):
        outcome = raw_outcome

    patient.data["biopmdt_outcome"] = outcome
    patient.add_event("mdt_decision", biopmdt_date, outcome=outcome)


    if outcome != 1:
        patient.is_complete = True
        patient.exit_reason = "no_biopsy_after_mdt"
        patient.add_event("pathway_exit", biopmdt_date)
        return "COMPLETE"

    patient.current_stage = "biopmdt_to_biopsy"
    return "ADVANCED"



def process_biopmdt_to_biopsy(patient: PatientState, ctx: StageContext) -> str:
    mode = ctx.wait_time_mode.get("biopmdt_to_biopsy", WAIT_MODE_MC)

    if mode == WAIT_MODE_MC:
        wait_days = sample_empirical_ecdf(ctx.pdfs["pre_biopmdt_to_biop"], ctx.rng)
        biopsy_date_raw = patient.current_date + timedelta(days=wait_days)
        biopsy_date = next_weekday(biopsy_date_raw)

        patient.data["wait_biopmdt_to_biopsy"] = wait_days
        patient.data["biopsy_date"] = biopsy_date
        patient.current_date = biopsy_date
        patient.current_stage = "downstream_tail"

        patient.add_event("biopsy_done", biopsy_date, wait_days=wait_days)
        return "ADVANCED"

    if mode == WAIT_MODE_DES:
        biopsy_ready_date = patient.current_date + timedelta(days=ctx.biopsy_ready_delay_days)

        ctx.pending_arrivals["Biopsy"].setdefault(biopsy_ready_date, []).append(
            QueuePatient(
                patient_id=patient.patient_id,
                referral_date=biopsy_ready_date,
                payload={"patient": patient},
            )
        )
        return "PAUSED_FOR_DES"

    raise ValueError(f"Unknown mode for biopmdt_to_biopsy: {mode}")


#def process_downstream_tail(patient: PatientState, ctx: StageContext) -> str:
#    """
 #   Temporary placeholder:
  #  keep biopsy->pathology->treatmdt->outpatient as one MC block for now.
   # """
    # You can port your current single_walk logic here gradually.
   # patient.is_complete = True
    #patient.add_event("pathway_end", patient.current_date)
    #return "COMPLETE"


def process_downstream_tail(patient: PatientState, ctx: StageContext) -> str:
    current_date = patient.current_date

    # biopsy -> pathology report
    wait_days = sample_empirical_ecdf(ctx.pdfs["pre_biop_to_pathrep"], ctx.rng)
    pathrep_date = next_weekday(current_date + timedelta(days=wait_days))

    patient.data["wait_biopsy_to_pathrep"] = wait_days
    patient.data["pathrep_date"] = pathrep_date
    patient.current_date = pathrep_date
    patient.add_event("Path_report_recieved", pathrep_date, wait_days=wait_days)

    # pathology outcome
    path_outcome = sample_outcome(ctx.branching["pathrep_outcome"], rng=ctx.rng)
    try:
        path_outcome = int(path_outcome)
    except (TypeError, ValueError):
        pass

    patient.data["pathrep_outcome"] = path_outcome
    patient.add_event("Path_report_outcome", pathrep_date, outcome=path_outcome)

    # assume 1 = cancer, 0 = no cancer
    if path_outcome != 1:
        patient.is_complete = True
        patient.exit_reason = "no_cancer_after_biopsy"
        patient.add_event("pathway_end", pathrep_date)
        return "COMPLETE"

    # path report -> treatment MDT
    wait_days = sample_empirical_ecdf(ctx.pdfs["pre_pathrep_to_treatmdt"], ctx.rng)
    treatmdt_date = next_weekday(pathrep_date + timedelta(days=wait_days))

    patient.data["wait_pathrep_to_treatmdt"] = wait_days
    patient.data["treatmdt_date"] = treatmdt_date
    patient.current_date = treatmdt_date
    patient.add_event("Treatment_options_MDT_occured", treatmdt_date, wait_days=wait_days)

    # treatment MDT -> outpatient
    wait_days = sample_empirical_ecdf(ctx.pdfs["pre_treatmdt_to_outpat"], ctx.rng)
    outpat_date = next_weekday(treatmdt_date + timedelta(days=wait_days))

    patient.data["wait_treatmdt_to_outpat"] = wait_days
    patient.data["outpat_date"] = outpat_date
    patient.current_date = outpat_date
    patient.add_event("Outpatient_appointment_occured", outpat_date, wait_days=wait_days)

    patient.is_complete = True
    patient.add_event("pathway_end", outpat_date)
    return "COMPLETE"





STAGE_HANDLERS = {
    "ref_to_mri": process_ref_to_mri,
    "mri_to_report": process_mri_to_report,
    "report_to_biopmdt": process_report_to_biopmdt,
    "biopmdt_to_biopsy": process_biopmdt_to_biopsy,
    "downstream_tail": process_downstream_tail,
}