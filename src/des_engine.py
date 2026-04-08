from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, Any, Dict, List, Tuple

import numpy as np

from patient_state import PatientState
from queue_resource import QueuePatient, QueueResource
from sampling import sample_poisson_weekday_only
from PDF_create import build_pdfs, build_branching
from stage_engine2 import (
    StageContext,
    WAIT_MODE_MC,
    WAIT_MODE_DES,
    STAGE_CONFIG,
    initialize_pending_mc,
    enter_wait_stage,
    process_des_resource_for_day,
    process_all_mc_due_today_until_stable,
    create_new_patient,
)



@dataclass
class EngineConfig:
    start_date: date
    n_days: int
    lam_per_workday: float
    mri_capacity_by_weekday: Dict[int, int]
    biopsy_capacity_by_weekday: Dict[int, int]
    biopsy_ready_delay_days: int = 0
    seed: int = 1234
    wait_time_mode: Dict[str, str] | None = None
    stage_rule_mode: Dict[str, str] | None = None
    fixed_wait_days_by_stage:Dict[str, str] | None = None

    scenario_name : str | None = None

    stage_timing_policy: Dict[str, str] | None = None

    initial_biopsy_queue_n: int = 1
    initial_biopsy_pending_n: int = 2

    biopsy_capacity_dropout_prob_by_weekday: Dict[int, float] = field(
        default_factory=lambda: {3: 0.15, 4: 0.2}
    )

    biopsy_backlog_capacity_tiers: List[Tuple[int, Dict[int, int]]] = field(
        default_factory=lambda: [
            (5, {3: 2, 4: 1}),
            (10, {3: 2, 4: 2}),
        ]
    )




def run_day_loop_with_stage_engine(
    cfg: EngineConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    rng = rng or np.random.default_rng(cfg.seed)
    wait_time_mode = cfg.wait_time_mode or {}

    pdfs = build_pdfs()
    branching = build_branching()

    mri_resource = QueueResource("MRI", cfg.mri_capacity_by_weekday)
    biopsy_resource = QueueResource("Biopsy", cfg.biopsy_capacity_by_weekday or {})

    pending_mc = initialize_pending_mc()

    ctx = StageContext(
        rng=rng,
        pdfs=pdfs,
        branching=branching,
        wait_time_mode=wait_time_mode,
        pending_mc=pending_mc,
        resources={
            "MRI": mri_resource,
            "Biopsy": biopsy_resource,
        },
        stage_timing_policy = cfg.stage_timing_policy or {},
        fixed_wait_days_by_stage=cfg.fixed_wait_days_by_stage or {},
        scenario_name= None,
    )

    completed_patients = []
    daily_referrals = {}
    next_pid = 1
    current_date = cfg.start_date

    for _ in range(cfg.n_days):
        # 1) Add new referrals first
        n_new = sample_poisson_weekday_only(
            lam_per_workday=cfg.lam_per_workday,
            weekday=current_date.weekday(),
            rng=rng,
        )
        daily_referrals[current_date] = n_new

        for _ in range(n_new):
            patient = create_new_patient(next_pid, current_date)
            next_pid += 1
            enter_wait_stage(patient, "ref_to_mri", ctx)

        # 2) Process DES resources for today
        des_resources_to_process = set()
        for stage_name, stage_cfg in STAGE_CONFIG.items():
            if wait_time_mode.get(stage_name, WAIT_MODE_MC) == WAIT_MODE_DES:
                resource_name = stage_cfg["resource"]
                if resource_name is not None:
                    des_resources_to_process.add(resource_name)

        for resource_name in des_resources_to_process:
            process_des_resource_for_day(resource_name, current_date, ctx, completed_patients)

        # 3) Repeatedly process all MC work due today until stable
        process_all_mc_due_today_until_stable(current_date, ctx, completed_patients)

        # 4) Advance day
        current_date += timedelta(days=1)

    patient_results = []
    for p in completed_patients:
        total_days = (p.current_date - p.start_date).days
        patient_results.append((p.events, total_days))

    summary_stats = {
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

