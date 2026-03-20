from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional, Any, Dict, List

import numpy as np

from queue_resource import QueuePatient, QueueResource
from sampling import sample_poisson_weekday_only

WAIT_MODE_MC = "MC"
WAIT_MODE_DES = "DES"


@dataclass
class EngineConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    mri_capacity_by_weekday: Dict[int, int]
    seed: int = 1234
    wait_time_mode: Dict[str, str] | None = None


def run_day_loop_with_mri_queue(
    cfg: EngineConfig,
    single_walk_fn: Callable[..., Any],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Day-loop engine:
      - weekday-only Poisson arrivals
      - optional DES queue for the ref->MRI stage
      - otherwise MC sampling inside single_walk_fn
    """

    wait_time_mode = cfg.wait_time_mode or {"ref_to_mri": WAIT_MODE_MC}
    rng = rng or np.random.default_rng(cfg.seed)

    mri_resource = QueueResource(
        name="MRI",
        capacity_by_weekday=cfg.mri_capacity_by_weekday,
    )

    patient_results: List[Any] = []
    daily_referrals: Dict[date, int] = {}

    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        # 1) ARRIVALS
        n_new = sample_poisson_weekday_only(
            lam_per_workday=cfg.lam_per_workday,
            weekday=current_date.weekday(),
            rng=rng,
        )
        daily_referrals[current_date] = n_new

        new_patients = [
            QueuePatient(patient_id=next_pid + i, referral_date=current_date)
            for i in range(n_new)
        ]
        next_pid += n_new

        # 2) ROUTE ARRIVALS
        if wait_time_mode.get("ref_to_mri") == WAIT_MODE_DES:
            mri_resource.add_patients(new_patients)
        else:
            for p in new_patients:
                out = single_walk_fn(
                    patient_id=p.patient_id,
                    start_date=p.referral_date,
                    rng=rng,
                    overrides={},
                )
                patient_results.append(out)

            # populate empty resource stats for this day in MC mode
            mri_resource.daily_started[current_date] = 0
            mri_resource.daily_queue_len[current_date] = 0
            mri_resource.daily_waits[current_date] = []

        # 3) RESOURCE SERVICE
        if wait_time_mode.get("ref_to_mri") == WAIT_MODE_DES:
            started_today = mri_resource.process_day(current_date)

            for event in started_today:
                p = event.patient
                wait_days = event.wait_days
                start_day = event.start_date

                overrides = {
                    "wait_ref_to_mri": wait_days,
                    "mri_date": start_day,
                }

                out = single_walk_fn(
                    patient_id=p.patient_id,
                    start_date=p.referral_date,
                    rng=rng,
                    overrides=overrides,
                )
                patient_results.append(out)

        current_date += timedelta(days=1)

    all_waits = [w for waits in mri_resource.daily_waits.values() for w in waits]

    summary_stats = {
        "total_days": cfg.n_days,
        "total_patients_completed": len(patient_results),
        "lambda_target": cfg.lam_per_workday,
        "capacity_by_resource": {
            "MRI": cfg.mri_capacity_by_weekday,
        },
        "final_queue_length_by_resource": {
            "MRI": mri_resource.queue_length(),
        },
        "mean_queue_wait_days_by_resource": {
            "MRI": float(np.mean(all_waits)) if all_waits else None,
        },
        "median_queue_wait_days_by_resource": {
            "MRI": float(np.median(all_waits)) if all_waits else None,
        },
    }

    return {
        "patient_results": patient_results,
        "daily_referrals": daily_referrals,
        "resources": {
            "MRI": {
                "daily_started": mri_resource.daily_started,
                "daily_queue_len": mri_resource.daily_queue_len,
                "daily_waits": mri_resource.daily_waits,
            }
        },
        "summary_stats": summary_stats,
    }