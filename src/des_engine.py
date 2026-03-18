from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from collections import deque
from typing import Callable, Optional, Any, Dict, List, Deque, Tuple

import numpy as np

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

    # stage switches
    wait_time_mode: Dict[str, str] = None  # e.g. {"ref_to_mri": "DES", ...}


@dataclass
class Patient:
    pid: int
    referral_date: date


def is_weekday(d: date) -> bool:
    return d.weekday() < 5

def get_mri_capacity_for_day(current_date):
    """
    Return MRI slots available on a given day.
    Python weekday: Monday=0, Tuesday=1, ..., Sunday=6
    """
    if current_date.weekday() == 1:   # Tuesday
        return 6
    return 0


def run_day_loop_with_mri_queue(
    cfg: EngineConfig,
    single_walk_fn: Callable[..., Any],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Day-loop engine:
      - Poisson weekday-only arrivals
      - Optional DES MRI queue producing wait_ref_to_mri (ref->MRI)
      - Otherwise uses MC sampling inside single_walk_fn

    Assumptions:
      - single_walk_fn accepts (patient_id, referral_date, rng, overrides=dict)
      - overrides can contain {"wait_ref_to_mri": int, "mri_date": date}
        (You can adjust names to match your code.)
    """

    if cfg.wait_time_mode is None:
        cfg.wait_time_mode = {"ref_to_mri": WAIT_MODE_MC}

    rng = rng or np.random.default_rng(cfg.seed)

    # ---- State ----
    mri_queue: Deque[Patient] = deque()
    patient_results: List[Any] = []
    daily_referrals: Dict[date, int] = {}
    daily_mri_started: Dict[date, int] = {}
    daily_mri_queue_len: Dict[date, int] = {}
    daily_mri_waits: Dict[date, List[int]] = {}  # waits of patients who got MRI that day

    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        # 1) ARRIVALS
        n_new = sample_poisson_weekday_only(cfg.lam_per_workday, current_date.weekday(), rng=rng)
        daily_referrals[current_date] = n_new

        new_patients = []
        for _i in range(n_new):
            p = Patient(pid=next_pid, referral_date=current_date)
            next_pid += 1
            new_patients.append(p)

        # 2) ROUTE arrivals depending on MRI wait-time mode
        # If ref->MRI is DES, patients enter MRI queue now.
        # If ref->MRI is MC, we run them immediately using MC sampling.
        if cfg.wait_time_mode.get("ref_to_mri", WAIT_MODE_MC) == WAIT_MODE_DES:
            mri_queue.extend(new_patients)
        else:
            # run immediately, no queue
            for p in new_patients:
                out = single_walk_fn(
                    patient_id=p.pid,
                    start_date=p.referral_date,
                    rng=rng,
                    overrides={},# no override
                )
                patient_results.append(out)

        # 3) MRI SERVICE (only matters if DES mode)
        mri_started_today = 0
        waits_today: List[int] = []

        if cfg.wait_time_mode.get("ref_to_mri", WAIT_MODE_MC) == WAIT_MODE_DES:
            capacity = cfg.mri_capacity_by_weekday.get(current_date.weekday(), 0)

            while capacity > 0 and mri_queue:
                p = mri_queue.popleft()
                capacity -= 1
                mri_started_today += 1

                wait_days = (current_date - p.referral_date).days
                waits_today.append(wait_days)

                overrides = {
                    "wait_ref_to_mri": wait_days,
                    "mri_date": current_date,
                }

                out = single_walk_fn(
                    patient_id=p.pid,
                    start_date=p.referral_date,
                    rng=rng,
                    overrides=overrides,
                )
                patient_results.append(out)

        daily_mri_started[current_date] = mri_started_today
        daily_mri_queue_len[current_date] = len(mri_queue)
        daily_mri_waits[current_date] = waits_today

        # 4) ADVANCE DAY
        current_date += timedelta(days=1)

    # ---- summary stats ----
    all_waits = [w for waits in daily_mri_waits.values() for w in waits]
    summary = {
        "total_days": cfg.n_days,
        "total_patients_completed": len(patient_results),
        "lambda_target": cfg.lam_per_workday,
        "mri_slots_per_workday": cfg.mri_capacity_by_weekday,
        "final_mri_queue_length": len(mri_queue),
        "mean_mri_queue_wait_days": float(np.mean(all_waits)) if all_waits else None,
        "median_mri_queue_wait_days": float(np.median(all_waits)) if all_waits else None,
    }

    return {
        "patient_results": patient_results,
        "daily_referrals": daily_referrals,
        "daily_mri_started": daily_mri_started,
        "daily_mri_queue_len": daily_mri_queue_len,
        "daily_mri_waits": daily_mri_waits,
        "summary_stats": summary,
    }