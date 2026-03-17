from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional, Any, List, Dict

import numpy as np

from sampling import sample_poisson_weekday_only
from single_walk_mdt_day import trace_one_patient_mdtday
from PDF_create import build_pdfs, build_branching


@dataclass
class DayLoopConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    seed: int = 1234


def run_day_loop(
    cfg: DayLoopConfig,
    single_walk_fn: Callable[..., Any],
    *,
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    
    rng = rng or np.random.default_rng(cfg.seed)
    pdfs = build_pdfs()
    branching = build_branching()

    patient_results: List[Any] = []
    daily_referrals: Dict[date, int] = {}

    current_date = cfg.start_date
    next_pid = 1


    
    

    for _ in range(cfg.n_days):

        weekday = current_date.weekday()

        n_new = sample_poisson_weekday_only(
            lam_per_workday=cfg.lam_per_workday,
            weekday=weekday,
            rng=rng,
        )

        daily_referrals[current_date] = n_new

        for _ in range(n_new):

            pid = next_pid
            next_pid += 1

            result = trace_one_patient_mdtday(
                patient_id=pid,
                start_date=current_date,
                rng=rng,
            )

            patient_results.append(result)


        current_date += timedelta(days=1)

       # ---------- summary stats ----------
    weekday_counts = [
        count
        for d, count in daily_referrals.items()
        if d.weekday() < 5
    ]

    weekend_counts = [
        count
        for d, count in daily_referrals.items()
        if d.weekday() >= 5
    ]

    summary_stats = {
        "total_days": cfg.n_days,
        "total_patients": len(patient_results),
        "mean_weekday_referrals": float(np.mean(weekday_counts)) if weekday_counts else 0,
        "mean_weekend_referrals": float(np.mean(weekend_counts)) if weekend_counts else 0,
        "lambda_target": cfg.lam_per_workday,
    }

    return {
        "patient_results": patient_results,
        "daily_referrals": daily_referrals,
        "summary_stats": summary_stats,
    }