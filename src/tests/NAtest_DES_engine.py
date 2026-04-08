from datetime import date
import numpy as np
import pytest

from sampling import sample_poisson, sample_poisson_weekday_only
from des_engine import (
    EngineConfig,
    run_day_loop_with_stage_engine,
    WAIT_MODE_MC,
    WAIT_MODE_DES,
)
from single_walk_mdt_day import trace_one_patient_mdtday

from config_final_biopsy_des import FINAL_BIOPSY_DES_CFG
from des_engine import run_day_loop_with_stage_engine

def extract_events(result):
    """result is (events, total_days)"""
    return result[0]


def event_by_name(events, name):
    for e in events:
        if e.get("event") == name:
            return e
    return None


def wait_from_event(result, event_name):
    events = extract_events(result)
    event = event_by_name(events, event_name)
    if event is None:
        return None
    return event.get("wait_days")


def date_from_event(result, event_name):
    events = extract_events(result)
    event = event_by_name(events, event_name)
    if event is None:
        return None
    return event.get("date")


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
        "wait_mri_to_report": overrides.get("wait_mri_to_report", "MC_USED"),
        "report_date": overrides.get("report_date", None),
        "wait_report_to_biopmdt": overrides.get("wait_report_to_biopmdt", "MC_USED"),
        "biopmdt_date": overrides.get("biopmdt_date", None),
        "wait_biopmdt_to_biopsy": overrides.get("wait_biopmdt_to_biopsy", "MC_USED"),
        "biopsy_date": overrides.get("biopsy_date", None),
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
        start_date=date(2024, 1, 1),
        n_days=5,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    assert isinstance(results, dict)
    assert "patient_results" in results
    assert "summary_stats" in results
    assert len(results["patient_results"]) > 0

    for p in results["patient_results"]:
        assert wait_from_event(p, "mri_performed") is not None

    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] == 0


def test_mc_mode_populates_zero_mri_stats_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=7,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

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
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results =run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["MRI"]["daily_started"].items():
        if n > 0:
            assert d.weekday() == 1


def test_des_mode_never_exceeds_configured_daily_capacity():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=28,
        lam_per_workday=4.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["MRI"]["daily_started"].items():
        expected_cap = cfg.mri_capacity_by_weekday.get(d.weekday(), 0)
        assert n <= expected_cap


def test_des_mode_queue_builds_when_arrivals_exceed_capacity():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=21,
        lam_per_workday=5.0,
        mri_capacity_by_weekday={1: 2},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] > 0


def test_des_mode_zero_capacity_means_no_patients_processed_and_queue_grows():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=10,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    assert sum(results["resources"]["MRI"]["daily_started"].values()) == 0
    assert len(results["patient_results"]) == 0
    assert results["summary_stats"]["final_queue_length_by_resource"]["MRI"] > 0


def test_des_override_waits_are_nonnegative_ints():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 1},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=1,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    completed = results["patient_results"]
    assert len(completed) > 0

    for p in completed:
        w = wait_from_event(p, "mri_performed")
        assert isinstance(w, int)
        assert w >= 0
        assert date_from_event(p, "mri_performed") is not None

def test_daily_queue_length_is_never_negative():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=30,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

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
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    total_started = sum(results["resources"]["MRI"]["daily_started"].values())
    assert len(results["patient_results"]) == total_started


def test_daily_mri_waits_match_number_started_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=20,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

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
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

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

def test_biopsy_des_only_starts_on_configured_days():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=30,
        lam_per_workday=1.5,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},  # Thu, Fri
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["Biopsy"]["daily_started"].items():
        if n > 0:
            assert d.weekday() in {3, 4}

def test_biopsy_des_never_exceeds_daily_capacity():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=60,
        lam_per_workday=2.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["Biopsy"]["daily_started"].items():
        expected_cap = cfg.biopsy_capacity_by_weekday.get(d.weekday(), 0)
        assert n <= expected_cap

def test_biopsy_queue_length_is_never_negative():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.5,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for qlen in results["resources"]["Biopsy"]["daily_queue_len"].values():
        assert qlen >= 0

def test_daily_biopsy_waits_match_number_started_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.5,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["Biopsy"]["daily_started"]
    daily_waits = results["resources"]["Biopsy"]["daily_waits"]

    for d in daily_started:
        assert len(daily_waits[d]) == daily_started[d]

def test_biopsy_des_starts_are_recorded_when_biopsy_des_enabled():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.5,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    total_biopsy_started = sum(results["resources"]["Biopsy"]["daily_started"].values())
    assert total_biopsy_started > 0

def test_biopsy_override_waits_are_nonnegative_ints():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.2,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        biopsy_wait = wait_from_event(p, "biopsy_done")
        if biopsy_wait is not None:
            assert isinstance(biopsy_wait, int)
            assert biopsy_wait >= 0

def test_biopsy_total_wait_matches_biopsy_date_minus_biopmdt_date():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.2,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        events = extract_events(p)
        biopsy = event_by_name(events, "biopsy_done")
        mdt = event_by_name(events, "MDT_occured")

        if biopsy is not None and mdt is not None:
            expected_wait = (biopsy["date"] - mdt["date"]).days
            assert biopsy["wait_days"] == expected_wait


def test_biopsy_summary_wait_stats_match_daily_waits():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=40,
        lam_per_workday=1.5,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    all_waits = [
        w
        for waits in results["resources"]["Biopsy"]["daily_waits"].values()
        for w in waits
    ]

    mean_wait = results["summary_stats"]["mean_queue_wait_days_by_resource"]["Biopsy"]
    median_wait = results["summary_stats"]["median_queue_wait_days_by_resource"]["Biopsy"]

    if all_waits:
        assert mean_wait == pytest.approx(np.mean(all_waits))
        assert median_wait == pytest.approx(np.median(all_waits))
    else:
        assert mean_wait is None
        assert median_wait is None

def test_biopsy_mc_mode_populates_zero_biopsy_stats_each_day():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["Biopsy"]["daily_started"]
    daily_queue_len = results["resources"]["Biopsy"]["daily_queue_len"]
    daily_waits = results["resources"]["Biopsy"]["daily_waits"]

    assert len(daily_started) == cfg.n_days
    assert len(daily_queue_len) == cfg.n_days
    assert len(daily_waits) == cfg.n_days

    for d in daily_started:
        assert daily_started[d] == 0
        assert daily_queue_len[d] == 0
        assert daily_waits[d] == []




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
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

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
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    daily_started = results["resources"]["MRI"]["daily_started"]

    assert daily_started[date(2024, 1, 1)] == 1
    assert daily_started[date(2024, 1, 3)] == 2

    for d, n in daily_started.items():
        if d not in {date(2024, 1, 1), date(2024, 1, 3)}:
            assert n == 0

def test_biopsy_des_respects_configured_sessions_with_fixed_arrivals(monkeypatch):
    def fixed_arrivals(lam_per_workday, weekday, rng=None):
        return 1 if weekday < 5 else 0

    monkeypatch.setattr("des_engine.sample_poisson_weekday_only", fixed_arrivals)

    cfg = EngineConfig(
        start_date=date(2024, 1, 1),   # Monday
        n_days=21,
        lam_per_workday=999.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 1},  # Thu + Fri
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for d, n in results["resources"]["Biopsy"]["daily_started"].items():
        if n > 0:
            assert d.weekday() in {3, 4}
            assert n <= cfg.biopsy_capacity_by_weekday[d.weekday()]


# =========================================================
# Smoke tests with real pathway function
# =========================================================

def test_real_single_walk_runs_with_biopsy_des_mode():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=20,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
            "biopmdt_to_biopsy": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)

def test_real_single_walk_runs_in_mc_mode():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=5,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_MC},
    )

    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)


def test_real_single_walk_runs_in_des_mode():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=10,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={"ref_to_mri": WAIT_MODE_DES},
    )

    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)

def test_des_mri_to_report_sets_report_one_day_after_mri():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        mri_date = date_from_event(p, "mri_performed")
        report_date = date_from_event(p, "mri_report_ready")

        if mri_date is not None and report_date is not None:
            assert (report_date - mri_date).days == 1


def test_des_report_to_biopmdt_sets_biopmdt_same_day_as_report():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        report_date = date_from_event(p, "mri_report_ready")
        mdt_date = date_from_event(p, "MDT_occured")

        if report_date is not None and mdt_date is not None:
            assert (mdt_date - report_date).days == 0


def test_des_mri_to_biopmdt_total_delay_is_one_day_when_both_overrides_active():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        mri_date = date_from_event(p, "mri_performed")
        mdt_date = date_from_event(p, "MDT_occured")

        if mri_date is not None and mdt_date is not None:
            assert (mdt_date - mri_date).days == 1

def test_mixed_mode_des_mri_to_report_but_report_to_biopmdt_remains_mc():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_MC,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        report_wait = wait_from_event(p, "mri_report_ready")
        if report_wait is not None:
            assert report_wait == 1


def test_des_report_to_biopmdt_can_set_date_even_without_des_report_override():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=14,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, fake_single_walk_fn)

    for p in results["patient_results"]:
        report_date = date_from_event(p, "mri_report_ready")
        mdt_date = date_from_event(p, "MDT_occured")

        if report_date is not None and mdt_date is not None:
            assert (mdt_date - report_date).days == 0

            
def test_real_single_walk_runs_with_des_report_and_biopmdt_overrides():
    cfg = EngineConfig(
        start_date=date(2024, 1, 1),
        n_days=10,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={1: 6},
        biopsy_capacity_by_weekday={3: 2, 4: 2},
        seed=42,
        wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_DES,
            "report_to_biopmdt": WAIT_MODE_DES,
        },
    )

    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    assert isinstance(results, dict)

    #biopsy tests




def test_biopsy_des_runs_with_mc_upstream():
    cfg = FINAL_BIOPSY_DES_CFG
    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)

    biopsy_started = sum(results["resources"]["Biopsy"]["daily_started"].values())

    assert biopsy_started > 0
    assert any(len(v) > 0 for v in results["resources"]["Biopsy"]["daily_waits"].values())

def test_biopsy_queue_lengths_never_negative():
    cfg = FINAL_BIOPSY_DES_CFG
    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)

    for q in results["resources"]["Biopsy"]["daily_queue_len"].values():
        assert q >= 0

def test_biopsy_daily_started_never_exceeds_max_possible_capacity():
    cfg = FINAL_BIOPSY_DES_CFG
    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)

    max_base = max(cfg.biopsy_capacity_by_weekday.values(), default=0)
    tier_maxes = [
        max(cap.values(), default=0)
        for _, cap in getattr(cfg, "biopsy_backlog_capacity_tiers", [])
    ]
    max_possible = max([max_base] + tier_maxes)

    for started in results["resources"]["Biopsy"]["daily_started"].values():
        assert started <= max_possible

def test_final_backlog_matches_queue_plus_pending():
    cfg = FINAL_BIOPSY_DES_CFG
    results = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)
    summary = results["summary_stats"]

    assert (
        summary["final_backlog_by_resource"]["Biopsy"]
        == summary["final_queue_length_by_resource"]["Biopsy"]
        + summary["final_pending_by_resource"]["Biopsy"]
    )
        