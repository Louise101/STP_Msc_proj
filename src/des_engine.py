from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional, Any, Dict, List

import numpy as np

from patient_state import PatientState
from queue_resource import QueuePatient, QueueResource
from sampling import sample_poisson_weekday_only, sample_empirical_ecdf
from PDF_create import build_pdfs, build_branching
from stage_engine import StageContext, WAIT_MODE_MC, WAIT_MODE_DES, advance_patient_until_pause_or_end, resume_after_mri, resume_after_biopsy

#WAIT_MODE_MC = "MC"
#WAIT_MODE_DES = "DES"



@dataclass
class EngineConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    mri_capacity_by_weekday: Dict[int, int]
    biopsy_capacity_by_weekday: Dict[int, int]
    biopsy_ready_delay_days: int = 1
    seed: int = 1234
    wait_time_mode: Dict[str, str] | None = None


def run_day_loop_with_stage_engine(
    cfg: EngineConfig,
    single_walk_fn: None,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    wait_time_mode = cfg.wait_time_mode or {}
    rng = rng or np.random.default_rng(cfg.seed)
    pdfs = build_pdfs()
    branching = build_branching()

    mri_resource = QueueResource("MRI", cfg.mri_capacity_by_weekday)

    biopsy_resource = QueueResource("Biopsy",cfg.biopsy_capacity_by_weekday or {})


    pending_arrivals = {
        "Biopsy": {},
    }

    ctx = StageContext(
        rng = rng,
        pdfs = pdfs,
        branching= branching, 
        wait_time_mode=wait_time_mode,
        pending_arrivals=pending_arrivals,
        resources= {
            "MRI":  mri_resource,
            "Biopsy": biopsy_resource
        },
        biopsy_ready_delay_days= cfg.biopsy_ready_delay_days,
    )

    completed_patients: List[PatientState] = []
    daily_referrals: Dict[date, int] = {}




    #patient_results: List[Any] = []
   # daily_referrals: Dict[date, int] = {}

    # Holds patients until their biopsy-ready date
    #pending_biopsy_arrivals: Dict[date, List[QueuePatient]] = {}

    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        # 0) MOVE READY PATIENTS INTO BIOPSY QUEUE
        ready_today = pending_arrivals["Biopsy"].pop(current_date, [])
        biopsy_resource.add_patients(ready_today)

        # 1) ARRIVALS
        n_new = sample_poisson_weekday_only(
            lam_per_workday=cfg.lam_per_workday,
            weekday=current_date.weekday(),
            rng=rng,
        )
        daily_referrals[current_date] = n_new

        for _ in range(n_new):
            patient = PatientState(
                patient_id=next_pid,
                start_date=current_date,
                current_date=current_date,
                current_stage="ref_to_mri",
            )

            next_pid += 1
            patient.add_event("referral_recieved", current_date)
            advance_patient_until_pause_or_end(patient, ctx)

            if patient.is_complete:
                completed_patients.append(patient)

        # 2) ROUTE ARRIVALS TO MRI OR RUN DIRECTLY
        #if wait_time_mode.get("ref_to_mri") == WAIT_MODE_DES:
         #   mri_resource.add_patients(new_patients)
        #else:
         #   for p in new_patients:
          #      out = single_walk_fn(
           #         patient_id=p.patient_id,
            #        start_date=p.referral_date,
             #       rng=rng,
              #      overrides={},
               # )
                #patient_results.append(out)

            #mri_resource.daily_started[current_date] = 0
            #mri_resource.daily_queue_len[current_date] = 0
            #mri_resource.daily_waits[current_date] = []

        # 3) MRI SERVICE
        # if in MC mode, resource stats missing
        if wait_time_mode.get("ref_to_mri", WAIT_MODE_MC) == WAIT_MODE_DES:
            mri_started_today = mri_resource.process_day(current_date)

            for event in mri_started_today:
                patient = event.patient.payload["patient"]
                resume_after_mri(patient, current_date, event.wait_days)
                advance_patient_until_pause_or_end(patient, ctx)
                #wait_days = event.wait_days
                #start_day = event.start_date

                if patient.is_complete:
                    completed_patients.append(patient)
        else:
            mri_resource.daily_started[current_date]=0
            mri_resource.daily_queue_len[current_date]=0
            mri_resource.daily_waits[current_date] = []

              #  overrides = {
               #     "wait_ref_to_mri": wait_days,
                #    "mri_date": start_day,
                #}

                #if wait_time_mode.get("mri_to_report") == WAIT_MODE_DES:
                #    overrides["wait_mri_to_report"] = 1
                 #   overrides["report_date"] = start_day + timedelta(days=1)

               # if wait_time_mode.get("report_to_biopmdt") == WAIT_MODE_DES:
                #    overrides["wait_report_to_biopmdt"] = 0
                 #   overrides["biopmdt_date"] = overrides.get(
                  #      "report_date",
                   #     start_day + timedelta(days=1),
                    #)

                # If biopsy is DES, hold patient until biopsy-ready date
             #   if wait_time_mode.get("biopmdt_to_biopsy") == WAIT_MODE_DES:
              #      biopmdt_date = overrides.get("biopmdt_date")
               #     if biopmdt_date is None:
                #        raise ValueError(
                 #           "biopmdt_date must be available before entering biopsy DES queue."
                  #      )

                   # biopsy_ready_delay = cfg.biopsy_ready_delay_days
                    #biopsy_ready_date = biopmdt_date + timedelta(days=biopsy_ready_delay)

                   # pending_biopsy_arrivals.setdefault(biopsy_ready_date, []).append(
                    #    QueuePatient(
                     #       patient_id=p.patient_id,
                      #      referral_date=biopsy_ready_date,
                       #     payload={
                        #        "start_date": p.referral_date,
                         #       "overrides": overrides,
                      #      },
                       # )
                    #)
               # else:
                #    out = single_walk_fn(
                 #       patient_id=p.patient_id,
                  #      start_date=p.referral_date,
                   #     rng=rng,
                    #    overrides=overrides,
                   # )
                   # patient_results.append(out)

        # Populate empty biopsy stats in MC mode
     #   if wait_time_mode.get("biopmdt_to_biopsy") != WAIT_MODE_DES:
     #       biopsy_resource.daily_started[current_date] = 0
      #      biopsy_resource.daily_queue_len[current_date] = 0
       #     biopsy_resource.daily_waits[current_date] = []

        # 4) BIOPSY SERVICE
        if wait_time_mode.get("biopmdt_to_biopsy", WAIT_MODE_MC) == WAIT_MODE_DES:
            biopsy_started_today = biopsy_resource.process_day(current_date)

            for event in biopsy_started_today:
                patient = event.patient.payload["patient"]
                resume_after_biopsy(patient, current_date,ctx)
                advance_patient_until_pause_or_end(patient, ctx)

                if patient.is_complete:
                    completed_patients.append(patient)
        else:
            biopsy_resource.daily_started[current_date]= 0
            biopsy_resource.daily_queue_len[current_date] = 0
            biopsy_resource.daily_waits[current_date] = []
               # biopsy_day = event.start_date

                #original_start_date = p.payload["start_date"]
                #overrides = dict(p.payload["overrides"])


              #  pdfs = build_pdfs()
                #total_wait = (biopsy_day - overrides["biopmdt_date"]).days
               # queue_wait = (biopsy_day - p.referral_date).days

                # Sample residual delay (from empirical distribution)
               # residual = sample_empirical_ecdf(pdfs["biopsy_residual_samples"], rng)
                #total_wait = queue_wait + residual

                #overrides["wait_biopmdt_to_biopsy"] = total_wait
                #overrides["biopsy_date"] = biopsy_day

               # out = single_walk_fn(
                #    patient_id=p.patient_id,
                 #   start_date=original_start_date,
                  #  rng=rng,
                   # overrides=overrides,
                #)
                #patient_results.append(out)

        # 5) ADVANCE DAY
        current_date += timedelta(days=1)

    mri_waits = [w for waits in mri_resource.daily_waits.values() for w in waits]
    biopsy_waits = [w for waits in biopsy_resource.daily_waits.values() for w in waits]

    patient_results = []
    for p in completed_patients:
        total_days = (p.current_date - p.start_date).days
        patient_results.append((p.events, total_days))

    pending_biopsy_count = sum(len(v) for v in pending_arrivals["Biopsy"].values())
    summary_stats = {
        "pending_biopsy_count": pending_biopsy_count,
        "total_days": cfg.n_days,
        "total_patients_completed": len(patient_results),
        "lambda_target": cfg.lam_per_workday,
        "capacity_by_resource": {
            "MRI": cfg.mri_capacity_by_weekday,
            "Biopsy": cfg.biopsy_capacity_by_weekday or {},
        },
        "final_queue_length_by_resource": {
            "MRI": mri_resource.queue_length(),
            "Biopsy": biopsy_resource.queue_length(),
        },
        "mean_queue_wait_days_by_resource": {
            "MRI": float(np.mean(mri_waits)) if mri_waits else None,
            "Biopsy": float(np.mean(biopsy_waits)) if biopsy_waits else None,
        },
        "median_queue_wait_days_by_resource": {
            "MRI": float(np.median(mri_waits)) if mri_waits else None,
            "Biopsy": float(np.median(biopsy_waits)) if biopsy_waits else None,
        },
        "final_pending_by_resource": {
            "Biopsy": pending_biopsy_count,
        },
        "final_backlog_by_resource": {
            "Biopsy": biopsy_resource.queue_length() + pending_biopsy_count,
        }
        }

    return {
        "patient_results": patient_results,
        "daily_referrals": daily_referrals,
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
        },
        "summary_stats": summary_stats,
    }

