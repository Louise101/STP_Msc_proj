from __future__ import annotations

"""Scenario registry for the combined engine.

This is the single place to define and choose scenarios to run.
Add new scenarios here rather than scattering config blocks across runner scripts.
"""

from dataclasses import dataclass
from datetime import date
from typing import Callable

from .pathway_definitions import STAGE_SEQUENCE, WAIT_MODE_DES, WAIT_MODE_MC


DEFAULT_START_DATE = date(2026, 1, 5)
DEFAULT_N_DAYS = 365
DEFAULT_LAMBDA = 1.7528735632183907
DEFAULT_PROSTAD_PROPORTION = 0.5098039215686274
DEFAULT_MRI_CAPACITY_PROSTAD = {1: 4}


@dataclass(slots=True)
class CombinedEngineConfig:
    """Configuration used by the combined engine."""

    start_date: date
    n_days: int
    lam_per_workday: float
    p_prostad: float
    mri_capacity_by_weekday_prostad: dict[int, int]
    seed: int = 1234
    scenario_name: str | None = None
    baseline_wait_time_mode: dict[str, str] | None = None
    prostad_wait_time_mode: dict[str, str] | None = None
    baseline_stage_timing_policy: dict[str, str] | None = None
    prostad_stage_timing_policy: dict[str, str] | None = None
    baseline_fixed_wait_days_by_stage: dict[str, int] | None = None
    prostad_fixed_wait_days_by_stage: dict[str, int] | None = None


def make_all_mc_modes() -> dict[str, str]:
    return {stage: WAIT_MODE_MC for stage in STAGE_SEQUENCE}


def make_prostad_modes() -> dict[str, str]:
    modes = make_all_mc_modes()
    modes["ref_to_mri"] = WAIT_MODE_DES
    return modes


def default_baseline_timing_policy() -> dict[str, str]:
    return {
        "mri_to_report": "EMPIRICAL",
        "report_to_biopmdt": "EMPIRICAL",
    }


def default_prostad_timing_policy() -> dict[str, str]:
    return {
        "mri_to_report": "FIXED",
        "report_to_biopmdt": "FIXED",
    }


def default_prostad_fixed_waits() -> dict[str, int]:
    return {
        "mri_to_report": 1,
        "report_to_biopmdt": 0,
    }


def build_combined_config(
    scenario_name: str,
    *,
    start_date: date = DEFAULT_START_DATE,
    n_days: int = DEFAULT_N_DAYS,
    lam_per_workday: float = DEFAULT_LAMBDA,
    p_prostad: float | None = None,
    seed: int = 1234,
    mri_capacity_by_weekday_prostad: dict[int, int] | None = None,
) -> CombinedEngineConfig:
    """Build a named scenario for the combined engine.

    Supported scenarios
    -------------------
    ALL_BASELINE
        Every patient follows baseline rules.
    OBS_MIX
        Patients are assigned to baseline vs PROSTAD pathways using the observed
        PROSTAD-period proportion.
    ALL_PROSTAD
        Every patient follows PROSTAD rules.
    """
    if p_prostad is None:
        if scenario_name == "ALL_BASELINE":
            p_prostad = 0.0
        elif scenario_name == "OBS_MIX":
            p_prostad = DEFAULT_PROSTAD_PROPORTION
        elif scenario_name == "ALL_PROSTAD":
            p_prostad = 1.0
        else:
            raise ValueError(f"Unknown scenario name: {scenario_name}")

    return CombinedEngineConfig(
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        p_prostad=p_prostad,
        mri_capacity_by_weekday_prostad=mri_capacity_by_weekday_prostad or DEFAULT_MRI_CAPACITY_PROSTAD,
        seed=seed,
        scenario_name=scenario_name,
        baseline_wait_time_mode=make_all_mc_modes(),
        prostad_wait_time_mode=make_prostad_modes(),
        baseline_stage_timing_policy=default_baseline_timing_policy(),
        prostad_stage_timing_policy=default_prostad_timing_policy(),
        baseline_fixed_wait_days_by_stage={},
        prostad_fixed_wait_days_by_stage=default_prostad_fixed_waits(),
    )


def available_scenarios() -> list[str]:
    """Return the currently supported scenario names."""
    return ["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"]
