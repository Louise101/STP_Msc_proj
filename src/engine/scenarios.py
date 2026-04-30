from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from engine.combined_engine import CombinedEngineConfig
from engine.pathway_definitions import STAGE_CONFIG, WAIT_MODE_DES, WAIT_MODE_MC


DEFAULT_START_DATE = date(2026, 1, 5)
DEFAULT_N_DAYS = 365
DEFAULT_LAM_PER_WORKDAY = 1.1010830324909748
DEFAULT_PROSTAD_OBSERVED_PROPORTION = 0.5098039215686274
DEFAULT_PROSTAD_MRI_CAPACITY = {1: 3}
DEFAULT_BIOPSY_CAPACITY = {
    0: 0,
    1: 3,
    2: 0,
    3: 3,
    4: 0,
    5: 0,
    6: 0,
}


def all_stages() -> list[str]:
    """Return the ordered stage names from the pathway definition."""
    return list(STAGE_CONFIG.keys())


@dataclass
class ScenarioTemplate:
    """High-level template describing how to build a named scenario.

    This is the single place to define and extend scenarios. To create a new
    experiment, add one new entry to ``SCENARIO_LIBRARY`` instead of writing a
    new runner script.
    """

    name: str
    p_prostad: float
    mri_capacity_by_weekday_prostad: dict[int, int] = field(default_factory=lambda: DEFAULT_PROSTAD_MRI_CAPACITY.copy())
    biopsy_capacity_by_weekday: dict[int, int] = field(default_factory=lambda: DEFAULT_BIOPSY_CAPACITY.copy())
    baseline_wait_time_mode: dict[str, str] = field(default_factory=dict)
    prostad_wait_time_mode: dict[str, str] = field(default_factory=dict)
    baseline_stage_timing_policy: dict[str, str] = field(default_factory=dict)
    prostad_stage_timing_policy: dict[str, str] = field(default_factory=dict)
    baseline_fixed_wait_days_by_stage: dict[str, int] = field(default_factory=dict)
    prostad_fixed_wait_days_by_stage: dict[str, int] = field(default_factory=dict)
    


DEFAULT_BASELINE_WAIT_MODES = {stage: WAIT_MODE_MC for stage in all_stages()}
DEFAULT_PROSTAD_WAIT_MODES = {
    "ref_to_mri": WAIT_MODE_DES,
    "mri_to_report": WAIT_MODE_MC,
    "report_to_biopmdt": WAIT_MODE_MC,
    "biopmdt_to_biopsy": WAIT_MODE_MC,
    "biopsy_to_pathrep": WAIT_MODE_MC,
    "pathrep_to_treatmdt": WAIT_MODE_MC,
    "treatmdt_to_outpat": WAIT_MODE_MC,
}
DEFAULT_BASELINE_TIMING = {
    "mri_to_report": "EMPIRICAL",
    "report_to_biopmdt": "EMPIRICAL",
}
DEFAULT_PROSTAD_TIMING = {
    "mri_to_report": "FIXED",
    "report_to_biopmdt": "FIXED",
}
DEFAULT_PROSTAD_FIXED_WAITS = {
    "mri_to_report": 1,
    "report_to_biopmdt": 0,
}

BASELINE_WITH_DES_BIOPSY_WAIT_MODES = DEFAULT_BASELINE_WAIT_MODES.copy()
BASELINE_WITH_DES_BIOPSY_WAIT_MODES["biopmdt_to_biopsy"] = WAIT_MODE_DES

PROSTAD_WITH_DES_BIOPSY_WAIT_MODES = DEFAULT_PROSTAD_WAIT_MODES.copy()
PROSTAD_WITH_DES_BIOPSY_WAIT_MODES["biopmdt_to_biopsy"] = WAIT_MODE_DES


SCENARIO_LIBRARY: dict[str, ScenarioTemplate] = {
    "ALL_BASELINE": ScenarioTemplate(
        name="ALL_BASELINE",
        p_prostad=0.0,
        baseline_wait_time_mode=DEFAULT_BASELINE_WAIT_MODES.copy(),
        prostad_wait_time_mode=DEFAULT_PROSTAD_WAIT_MODES.copy(),
        baseline_stage_timing_policy=DEFAULT_BASELINE_TIMING.copy(),
        prostad_stage_timing_policy=DEFAULT_PROSTAD_TIMING.copy(),
        prostad_fixed_wait_days_by_stage=DEFAULT_PROSTAD_FIXED_WAITS.copy(),
    ),
    "OBS_MIX": ScenarioTemplate(
        name="OBS_MIX",
        p_prostad=DEFAULT_PROSTAD_OBSERVED_PROPORTION,
        baseline_wait_time_mode=DEFAULT_BASELINE_WAIT_MODES.copy(),
        prostad_wait_time_mode=DEFAULT_PROSTAD_WAIT_MODES.copy(),
        baseline_stage_timing_policy=DEFAULT_BASELINE_TIMING.copy(),
        prostad_stage_timing_policy=DEFAULT_PROSTAD_TIMING.copy(),
        prostad_fixed_wait_days_by_stage=DEFAULT_PROSTAD_FIXED_WAITS.copy(),
    ),
    "OBS_MIX_DES_BIOPSY": ScenarioTemplate(
        name="OBS_MIX_DES_BIOPSY",
        p_prostad=DEFAULT_PROSTAD_OBSERVED_PROPORTION,
        mri_capacity_by_weekday_prostad=DEFAULT_PROSTAD_MRI_CAPACITY.copy(),
        biopsy_capacity_by_weekday=DEFAULT_BIOPSY_CAPACITY.copy(),
        baseline_wait_time_mode=BASELINE_WITH_DES_BIOPSY_WAIT_MODES.copy(),
        prostad_wait_time_mode=PROSTAD_WITH_DES_BIOPSY_WAIT_MODES.copy(),
        baseline_stage_timing_policy=DEFAULT_BASELINE_TIMING.copy(),
        prostad_stage_timing_policy=DEFAULT_PROSTAD_TIMING.copy(),
        prostad_fixed_wait_days_by_stage=DEFAULT_PROSTAD_FIXED_WAITS.copy(),
    ),
    "ALL_PROSTAD": ScenarioTemplate(
        name="ALL_PROSTAD",
        p_prostad=1.0,
        mri_capacity_by_weekday_prostad=DEFAULT_PROSTAD_MRI_CAPACITY.copy(),
        biopsy_capacity_by_weekday=DEFAULT_BIOPSY_CAPACITY.copy(),
        baseline_wait_time_mode=BASELINE_WITH_DES_BIOPSY_WAIT_MODES.copy(),
        prostad_wait_time_mode=PROSTAD_WITH_DES_BIOPSY_WAIT_MODES.copy(),
        baseline_stage_timing_policy=DEFAULT_BASELINE_TIMING.copy(),
        prostad_stage_timing_policy=DEFAULT_PROSTAD_TIMING.copy(),
        prostad_fixed_wait_days_by_stage=DEFAULT_PROSTAD_FIXED_WAITS.copy(),
    ),
}


def register_scenario(template: ScenarioTemplate) -> None:
    """Add or replace a scenario template in the central scenario library."""
    SCENARIO_LIBRARY[template.name] = template


def build_combined_config(
    scenario_name: str,
    *,
    start_date: date = DEFAULT_START_DATE,
    n_days: int = DEFAULT_N_DAYS,
    lam_per_workday: float = DEFAULT_LAM_PER_WORKDAY,
    seed: int = 1234,
    overrides: dict[str, Any] | None = None,
) -> CombinedEngineConfig:
    """Build a concrete engine config from a named scenario template."""
    if scenario_name not in SCENARIO_LIBRARY:
        valid = ", ".join(sorted(SCENARIO_LIBRARY))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid options: {valid}")

    template = SCENARIO_LIBRARY[scenario_name]
    config = CombinedEngineConfig(
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        p_prostad=template.p_prostad,
        mri_capacity_by_weekday_prostad=template.mri_capacity_by_weekday_prostad.copy(),
        biopsy_capacity_by_weekday=template.biopsy_capacity_by_weekday.copy(),
        seed=seed,
        baseline_wait_time_mode=template.baseline_wait_time_mode.copy(),
        prostad_wait_time_mode=template.prostad_wait_time_mode.copy(),
        baseline_stage_timing_policy=template.baseline_stage_timing_policy.copy(),
        prostad_stage_timing_policy=template.prostad_stage_timing_policy.copy(),
        baseline_fixed_wait_days_by_stage=template.baseline_fixed_wait_days_by_stage.copy(),
        prostad_fixed_wait_days_by_stage=template.prostad_fixed_wait_days_by_stage.copy(),
        scenario_name=scenario_name,
    )

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config


def generate_daily_referrals(
    start_date: date,
    n_days: int,
    lam_per_workday: float,
    seed: int,
) -> dict[date, int]:
    """Generate one reproducible referral stream to share across scenarios."""
    rng = np.random.default_rng(seed)
    referrals: dict[date, int] = {}
    current_date = start_date
    for _ in range(n_days):
        referrals[current_date] = int(rng.poisson(lam_per_workday)) if current_date.weekday() < 5 else 0
        current_date += timedelta(days=1)
    return referrals


