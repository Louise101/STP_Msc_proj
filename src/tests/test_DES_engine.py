from datetime import date
import numpy as np
import pytest

from sampling import sample_poisson, sample_poisson_weekday_only
from des_engine import (
    EngineConfig,
    run_day_loop_with_mri_queue,
    WAIT_MODE_MC,
    WAIT_MODE_DES,
)
from single_walk_mdt_day import trace_one_patient_mdtday


def fake_single_walk_fn(patient_id, start_date, rng=None, overrides=None):
    """
    Simple stub function for testing the DES engine in isolation.
    Returns whatever override values were passed in.
    """
    overrides = overrides or {}
    return {
        "patient_id": patient_id,
        "start_date": start_date,
        "wait_ref_to_mri": overrides.get("wait_ref_to_mri", "MC_USED"),
        "mri_date": overrides.get("mri_date", None),
    }


# =========================================================
# sampling.py tests
# =========================================================

def test_sample_poisson_negative_lambda_raises():
    with pytest.raises(ValueError):
        sample_poisson(-1.0)


@pytest.mark.parametrize("weekday", [5, 6])  # Saturday, Sunday
def test_sample_poisson_weekday_only_returns_zero_on_weekends(weekday):
    rng = np.random.default_rng(123)
    out = sample_poisson_weekday_only(
        lam_per_workday=2.0,
        weekday=weekday,
        rng=rng,
    )
    assert out == 0


@pytest.mark.parametrize("weekday", [0, 1, 2, 3, 4])  # Monday-Friday
def test_sample_poisson_weekday_only_returns_nonnegative_int_on_weekdays(weekday):
    rng = np.random.default_rng(123)
    out = sample_poisson_weekday_only(
        lam_per_workday=2.0,
        weekday=weekday,
        rng=rng,
    )
    assert isinstance(out, int)
    assert out >= 0


# =========================================================
# DES engine tests
# =========================================================

def test_mc_mode_runs_patients_immediately_without_des_override():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),   # Monday
        n_days=5,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},   # Tuesday only, irrelevant in MC mode
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    assert isinstance(results, dict)
    assert "patient_results" in results
    assert "summary_stats" in results
    assert "resources" in results
    assert "MRI" in results["resources"]
    assert len(results["patient_results"]) > 0

    for p in results["patient_results"]:
        assert p["wait_ref_to_mri"] == "MC_USED"
        assert p["mri_date"] is None

    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] == 0
    assert sum(results["resources"]["MRI"]["daily_started"].values()) == 0


def test_mc_mode_populates_zero_mri_stats_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=7,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["MRI"]["daily_started"]
    daily_queue_len = results["resources"]["MRI"]["daily_queue_len"]
    daily_waits = results["resources"]["MRI"]["daily_waits"]

    assert len(daily_started) == cfg.n_days
    assert len(daily_queue_len) == cfg.n_days
    assert len(daily_waits) == cfg.n_days

    for d in daily_started:
        assert daily_started[d] == 0
        assert daily_queue_len[d] == 0
        assert daily_waits[d] == []


def test_des_mode_only_starts_mri_on_configured_tuesday():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),   # Monday
        n_days=14,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},   # Tuesday only
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["MRI"]["daily_started"].items():
        if n > 0:
            assert d.weekday() == 1


def test_des_mode_never_exceeds_configured_daily_capacity():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=28,
        lam_per_workday=4.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["MRI"]["daily_started"].items():
        expected_cap = cfg.mri_capacity_by_weekday.get(d.weekday(), 0)
        assert n <= expected_cap


def test_des_mode_queue_builds_when_arrivals_exceed_capacity():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=21,
        lam_per_workday=5.0,
        mri_capacity_by_weekday={1: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] > 0


def test_des_mode_zero_capacity_means_no_patients_processed_and_queue_grows():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=10,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    assert sum(results["resources"]["MRI"]["daily_started"].values()) == 0
    assert len(results["patient_results"]) == 0
    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] > 0


def test_des_override_waits_are_nonnegative_ints():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 1},
        seed=1,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    completed = results["patient_results"]
    assert len(completed) > 0

    for p in completed:
        assert isinstance(p["wait_ref_to_mri"], int)
        assert p["wait_ref_to_mri"] >= 0
        assert p["mri_date"] is not None


def test_daily_queue_length_is_never_negative():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=30,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    for qlen in results["resources"]["MRI"]["daily_queue_len"].values():
        assert qlen >= 0


def test_des_stub_total_completed_equals_total_mri_started():
    """
    In DES mode with the stub, each MRI start immediately produces one result.
    """
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=20,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    total_started = sum(results["resources"]["MRI"]["daily_started"].values())
    assert len(results["patient_results"]) == total_started


def test_daily_mri_waits_match_number_started_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=20,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["MRI"]["daily_started"]
    daily_waits = results["resources"]["MRI"]["daily_waits"]

    for d in daily_started:
        assert len(daily_waits[d]) == daily_started[d]


def test_summary_wait_stats_match_daily_waits():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=20,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    all_waits = [
        w
        for waits in results["resources"]["MRI"]["daily_waits"].values()
        for w in waits
    ]

    mean_wait = results["summary_stats"]["mean_queue_wait_days_by_resource"]["MRI"]
    median_wait = results["summary_stats"]["median_queue_wait_days_by_resource"]["MRI"]

    if all_waits:
        assert mean_wait == pytest.approx(np.mean(all_waits))
        assert median_wait == pytest.approx(np.median(all_waits))
    else:
        assert mean_wait is None
        assert median_wait is None


# =========================================================
# Deterministic engine-ordering test using monkeypatch
# =========================================================

def test_tuesday_only_capacity_with_fixed_weekday_arrivals(monkeypatch):
    """
    Force exactly 1 arrival on each weekday and 0 on weekends.

    With current engine ordering:
      arrivals happen first, then MRI service on the same day.
    So on Tuesday, both Monday and Tuesday arrivals are in queue before service.
    """
    def fixed_arrivals(lam_per_workday, weekday, rng=None):
        return 0 if weekday >= 5 else 1

    monkeypatch.setattr("des_engine.sample_poisson_weekday_only", fixed_arrivals)

    cfg = EngineConfig(
        start_date=date(2024, 1, 1),   # Monday
        n_days=7,
        lam_per_workday=999.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    assert results["resources"]["MRI"]["daily_started"][date(2024, 1, 2)] == 2

    for d, n in results["resources"]["MRI"]["daily_started"].items():
        if d != date(2024, 1, 2):
            assert n == 0

    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] == 3


def test_multi_day_capacity_config_is_respected(monkeypatch):
    def fixed_arrivals(lam_per_workday, weekday, rng=None):
        return 2 if weekday < 5 else 0

    monkeypatch.setattr("des_engine.sample_poisson_weekday_only", fixed_arrivals)

    cfg = EngineConfig(
        start_date=date(2024, 1, 1),  # Monday
        n_days=7,
        lam_per_workday=999.0,
        mri_capacity_by_weekday={0: 1, 2: 2},  # Monday=1, Wednesday=2
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["MRI"]["daily_started"]

    assert daily_started[date(2024, 1, 1)] == 1
    assert daily_started[date(2024, 1, 3)] == 2

    for d, n in daily_started.items():
        if d not in {date(2024, 1, 1), date(2024, 1, 3)}:
            assert n == 0


# =========================================================
# Smoke tests with real pathway function
# =========================================================

def test_real_single_walk_runs_in_mc_mode():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=5,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_mri_queue(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)


def test_real_single_walk_runs_in_des_mode():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=10,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_mri_queue(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)