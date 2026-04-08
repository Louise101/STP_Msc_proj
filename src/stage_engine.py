from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from patient_state import PatientState
from queue_resource import QueuePatient, DelayQueueItem
from sampling import sample_empirical_ecdf, sample_outcome

WAIT_MODE_MC = "MC"
WAIT_MODE_DES = "DES"


#@dataclass
#class StageContext:
 #   rng: np.random.Generator
    #rng_downstream: np.random.Generator
  #  pdfs: Dict[str, Any]
   # branching: Dict[str, Any]
    #wait_time_mode: Dict[str, str]
   # pending_arrivals: Dict[str, Dict[date, list[QueuePatient]]]
   # resources: Dict[str, Any]
    #biopsy_ready_delay_days: int = 1

@dataclass
class StageContext:
    rng: np.random.Generator
    pdfs: dict
    branching: dict
    wait_time_mode: dict

    # MC delay queues:
    # stage_name -> ready_date -> list[DelayQueueItem]
    pending_mc: Dict[str, Dict[date, list]]

    # DES resources:
    resources: Dict[str, Any]

#STAGE_SEQUENCE = [
 #   "ref_to_mri",
  #  "mri_to_report",
   # "report_to_biopmdt",
    #"biopmdt_to_biopsy",
  #  "downstream_tail",
#]

STAGE_SEQUENCE = [
    "ref_to_mri",
    "mri_to_report",
    "report_to_biopmdt",
    "biopmdt_to_biopsy",
    "biopsy_to_pathrep",
    "pathrep_to_treatmdt",
    "treatmdt_to_outpat",
]

STAGE_CONFIG = {
    "ref_to_mri": {
        "resource": "MRI",
        "pdf_key": "ref_to_mri",
        "completion_event": "mri_performed",
    },
    "mri_to_report": {
        "resource": None,
        "pdf_key": "mri_to_report",
        "completion_event": "mri_report_ready",
    },
    "report_to_biopmdt": {
        "resource": None,
        "pdf_key": "report_to_biopmdt",
        "completion_event": "MDT_occured",
    },
    "biopmdt_to_biopsy": {
        "resource": "Biopsy",
        "pdf_key": "biopmdt_to_biopsy",
        "completion_event": "biopsy_done",
    },
    "biopsy_to_pathrep": {
        "resource": None,
        "pdf_key": "biopsy_to_pathrep",
        "completion_event": "Path_report_recieved",
    },
    "pathrep_to_treatmdt": {
        "resource": None,
        "pdf_key": "pathrep_to_treatmdt",
        "completion_event": "Treatment_options_MDT_occured",
    },
    "treatmdt_to_outpat": {
        "resource": None,
        "pdf_key": "treatmdt_to_outpat",
        "completion_event": "Outpatient_appointment_occured",
    },
}

def sample_wait_for_stage(stage_name: str, ctx: StageContext) -> int:
    pdf_key = STAGE_CONFIG[stage_name]["pdf_key"]
    pdf_obj = ctx.pdfs[pdf_key]

    # adapt this to however your PDFs are stored
    return pdf_obj.sample(ctx.rng)

def sample_mdt_decision(ctx: StageContext) -> int:
    probs = ctx.branching["biopmdt_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())
    return int(ctx.rng.choice(keys, p=p))

def sample_pathology_outcome(ctx: StageContext) -> int:
    probs = ctx.branching["pathrep_outcome"]
    keys = list(probs.keys())
    p = list(probs.values())
    return int(ctx.rng.choice(keys, p=p))

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
        
#def resume_after_mri(patient: PatientState, mri_day: date, queue_wait: int) -> None:
 #   patient.data["wait_ref_to_mri"] = queue_wait
  #  patient.data["mri_date"] = mri_day
   # patient.current_date = mri_day
    #patient.current_stage = "mri_to_report"
   # patient.add_event("mri_performed", mri_day, wait_days=queue_wait)


#def resume_after_biopsy(patient: PatientState, biopsy_day: date, ctx: StageContext) -> None:
    #residual = sample_empirical_ecdf(ctx.pdfs["biopsy_residual_samples"], ctx.rng)
 #   residual = sample_empirical_ecdf(ctx.pdfs["pre_biopmdt_to_biop"], ctx.rng)
  #  final_biopsy_day = biopsy_day #+ timedelta(days=residual)

   # total_wait = (final_biopsy_day - patient.data["biopmdt_date"]).days
   # #patient.data["biopmdt_to_biopsy_queue_wait"] = (biopsy_day - patient.data["biopmdt_date"]).days
  #  patient.data["biopmdt_to_biopsy_residual_wait"] = residual
   # patient.data["wait_biopmdt_to_biopsy"] = total_wait
    #patient.data["biopsy_date"] = final_biopsy_day

  #  patient.current_date = final_biopsy_day
   # patient.current_stage = "biopsy_to_pathrep"
    #patient.add_event("biopsy_done", final_biopsy_day, wait_days=total_wait)



#function to push patient into stage
def enter_wait_stage(patient, stage_name: str, ctx: StageContext) -> None:
    mode = ctx.wait_time_mode.get(stage_name, WAIT_MODE_MC)
    cfg = STAGE_CONFIG[stage_name]

    patient.current_stage = stage_name

    if mode == WAIT_MODE_MC:
        sampled_wait = sample_wait_for_stage(stage_name, ctx)
        ready_date = patient.current_date + timedelta(days=sampled_wait)

        item = DelayQueueItem(
            patient=patient,
            entry_date=patient.current_date,
            ready_date=ready_date,
            sampled_wait=sampled_wait,
            stage_name=stage_name,
        )
        ctx.pending_mc[stage_name].setdefault(ready_date, []).append(item)

    elif mode == WAIT_MODE_DES:
        resource_name = cfg["resource"]
        if resource_name is None:
            raise ValueError(f"Stage {stage_name} is DES but has no resource")

        ctx.resources[resource_name].add_patient(
            QueuePatient(
                patient_id=patient.patient_id,
                referral_date=patient.current_date,
                payload={
                    "patient": patient,
                    "stage_name": stage_name,
                    "entry_date": patient.current_date,
                },
            )
        )

    else:
        raise ValueError(f"Unknown wait mode for stage {stage_name}: {mode}")
    

#core funtion to complete stages

def complete_wait_stage(patient, stage_name: str, completion_date, wait_days: int, ctx: StageContext) -> None:
    patient.current_date = completion_date

    if stage_name == "ref_to_mri":
        patient.add_event("mri_performed", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "mri_to_report", ctx)
        return

    if stage_name == "mri_to_report":
        patient.add_event("mri_report_ready", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "report_to_biopmdt", ctx)
        return

    if stage_name == "report_to_biopmdt":
        patient.add_event("MDT_occured", completion_date, wait_days=wait_days)

        outcome = sample_mdt_decision(ctx)
        patient.add_event("mdt_decision", completion_date, outcome=outcome)

        # Example assumption: 1 means biopsy, others complete
        if outcome == 1:
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

        outcome = sample_pathology_outcome(ctx)
        patient.add_event("Path_report_outcome", completion_date, outcome=outcome)

        # Example assumption: 1 = cancer, 0 = not cancer
        if outcome == 1:
            enter_wait_stage(patient, "pathrep_to_treatmdt", ctx)
        else:
            patient.is_complete = True
        return

    if stage_name == "pathrep_to_treatmdt":
        patient.add_event("Treatment_options_MDT_occured", completion_date, wait_days=wait_days)
        enter_wait_stage(patient, "treatmdt_to_outpat", ctx)
        return

    if stage_name == "treatmdt_to_outpat":
        patient.add_event("Outpatient_appointment_occured", completion_date, wait_days=wait_days)
        patient.is_complete = True
        return

    raise ValueError(f"Unknown stage: {stage_name}")

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
        #patient.current_stage = "downstream_tail"
        patient.current_stage = "biopsy_to_pathrep"


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


def process_biopsy_to_pathrep(patient: PatientState, ctx: StageContext) -> str:
   # mode = ctx.wait_time_mode.get("biopsy_to_pathrep", WAIT_MODE_MC)

    #if mode == WAIT_MODE_MC:
    wait_days = sample_empirical_ecdf(ctx.pdfs["pre_biop_to_pathrep"], ctx.rng)
    pathrep_date_raw = patient.current_date + timedelta(days=wait_days)
    pathrep_date = next_weekday(pathrep_date_raw)

    patient.data["wait_biop_to_pathrep"] = wait_days
    patient.data["pathrep_date"] = pathrep_date
    patient.current_date = pathrep_date
    

    # pathology outcome
    path_outcome = sample_outcome(
        ctx.branching["pathrep_outcome"],
        rng=ctx.rng_downstream if hasattr(ctx, "rng_downstream") else ctx.rng,
    )

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

    patient.current_stage = "downstream_tail"
   
    return "ADVANCED"

    #if mode == WAIT_MODE_DES:
     #   biopsy_ready_date = patient.current_date + timedelta(days=ctx.biopsy_ready_delay_days)

      #  ctx.pending_arrivals["Biopsy"].setdefault(biopsy_ready_date, []).append(
       #     QueuePatient(
        #        patient_id=patient.patient_id,
         #       referral_date=biopsy_ready_date,
          #      payload={"patient": patient},
           # )
        #)
        #return "PAUSED_FOR_DES"

    raise ValueError(f"Unknown mode for biopsy_to_pathrep: {mode}")

def process_downstream_tail(patient: PatientState, ctx: StageContext) -> str:
    """
    Temporary placeholder:
    keep biopsy->pathology->treatmdt->outpatient as one MC block for now.
    """
    #You can port your current single_walk logic here gradually.
    patient.is_complete = True
    patient.add_event("pathway_end", patient.current_date)
    return "COMPLETE"


def process_downstream_tail2(patient, ctx):
    """
    MC downstream tail after biopsy:
      biopsy -> pathology report
      pathology outcome
      if cancer:
          path report -> treatment MDT
          treatment MDT -> outpatient
      else:
          end pathway
    Uses ctx.rng_downstream so downstream random draws do not perturb
    upstream / biopsy behaviour.
    """
    current_date = patient.current_date

    # -----------------------------
    # Biopsy -> pathology report
    # -----------------------------
    wait_days = sample_empirical_ecdf(
        ctx.pdfs["pre_biop_to_pathrep"],
        ctx.rng_downstream,
    )
    pathrep_date = current_date + timedelta(days=wait_days)

    patient.data["wait_biopsy_to_pathrep"] = wait_days
    patient.data["pathrep_date"] = pathrep_date
    patient.current_date = pathrep_date
    patient.add_event("Path_report_recieved", pathrep_date, wait_days=wait_days)

    # -----------------------------
    # Pathology outcome
    # -----------------------------
    path_outcome = sample_outcome(
        ctx.branching["pathrep_outcome"],
        rng=ctx.rng_downstream,
    )

    try:
        path_outcome = int(path_outcome)
    except (TypeError, ValueError):
        pass

    patient.data["pathrep_outcome"] = path_outcome
    patient.add_event("Path_report_outcome", pathrep_date, outcome=path_outcome)

    # assume 1 = cancer, 0 = not cancer
    if path_outcome != 1:
        patient.is_complete = True
        patient.exit_reason = "no_cancer_after_biopsy"
        patient.add_event("pathway_end", pathrep_date)
        return "COMPLETE"

    # -----------------------------
    # Path report -> treatment MDT
    # -----------------------------
    wait_days = sample_empirical_ecdf(
        ctx.pdfs["pre_pathrep_to_treatmdt"],
        ctx.rng_downstream,
    )
    treatmdt_date = pathrep_date + timedelta(days=wait_days)

    patient.data["wait_pathrep_to_treatmdt"] = wait_days
    patient.data["treatmdt_date"] = treatmdt_date
    patient.current_date = treatmdt_date
    patient.add_event("Treatment_options_MDT_occured", treatmdt_date, wait_days=wait_days)

    # -----------------------------
    # Treatment MDT -> outpatient
    # -----------------------------
    wait_days = sample_empirical_ecdf(
        ctx.pdfs["pre_treatmdt_to_outpat"],
        ctx.rng_downstream,
    )
    outpat_date = treatmdt_date + timedelta(days=wait_days)

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
    "biopsy_to_pathrep": process_biopsy_to_pathrep,
    "downstream_tail": process_downstream_tail,
}

