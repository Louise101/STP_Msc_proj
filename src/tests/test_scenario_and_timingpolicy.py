from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

import stage_engine2
from stage_engine2 import (
    StageContext,
    WAIT_MODE_MC,
    WAIT_MODE_DES,
    enter_wait_stage,
    process_mc_stage_for_day,
    initialize_pending_mc,
)

# Only import this once you have created scenarios.py
import scenarios


# =========================================================
# Fakes
# =========================================================

@dataclass
class FakeServiceEvent:
    patient: object
    wait_days: int


class FakeQueueResource:
    def __init__(self, name, capacity_by_weekday):
        self.name = name
        self.capacity_by_weekday = capacity_by_weekday or {}
        self.queue = []
        self.daily_started = {}
        self.daily_queue_len = {}
        self.daily_waits = {}

    def add_patient(self, queue_patient):
        self.queue.append(queue_patient)

    def add_patients(self, queue_patients):
        self.queue.extend(queue_patients)

    def queue_length(self):
        return len(self.queue)

    def process_day(self, current_date):
        cap = self.capacity_by_weekday.get(current_date.weekday(), 0)
        self.daily_queue_len[current_date] = len(self.queue)

        started = []
        waits = []

        for _ in range(min(cap, len(self.queue))):
            qp = self.queue.pop(0)
            wait_days = (current_date - qp.referral_date).days
            waits.append(wait_days)
            started.append(FakeServiceEvent(patient=qp, wait_days=wait_days))

        self.daily_started[current_date] = len(started)
        self.daily_waits[current_date] = waits
        return started


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def ctx(rng):
    return StageContext(
        rng=rng,
        pdfs={
            "pre_referral_to_mri": [5, 5, 5],
            "pre_mri_to_mrireport": [2, 2, 2],
            "pre_mrirep_to_biopsymdt": [3, 3, 3],
            "pre_biopmdt_to_biop": [4, 4, 4],
            "pre_biop_to_pathrep": [6, 6, 6],
            "pre_pathrep_to_treatmdt": [2, 2, 2],
            "pre_treatmdt_to_outpat": [7, 7, 7],
        },
        branching={
            "biopmdt_outcome": {0: 1.0},
            "pathrep_outcome": {1: 1.0},
        },
        wait_time_mode={},
        pending_mc=initialize_pending_mc(),
        resources={
            "MRI": FakeQueueResource("MRI", {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
            "Biopsy": FakeQueueResource("Biopsy", {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
        },
        stage_timing_policy={},
        fixed_wait_days_by_stage={},
        scenario_name=None,
    )


# =========================================================
# Timing policy tests
# =========================================================

def test_get_non_des_wait_for_stage_empirical(monkeypatch, ctx):
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 9)

    ctx.stage_timing_policy = {
        "mri_to_report": "EMPIRICAL",
    }

    wait = stage_engine2.get_non_des_wait_for_stage("mri_to_report", ctx)
    assert wait == 9


def test_get_non_des_wait_for_stage_fixed(ctx):
    ctx.stage_timing_policy = {
        "mri_to_report": "FIXED",
    }
    ctx.fixed_wait_days_by_stage = {
        "mri_to_report": 1,
    }

    wait = stage_engine2.get_non_des_wait_for_stage("mri_to_report", ctx)
    assert wait == 1


def test_get_non_des_wait_for_stage_defaults_to_empirical(monkeypatch, ctx):
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 11)

    wait = stage_engine2.get_non_des_wait_for_stage("mri_to_report", ctx)
    assert wait == 11


def test_get_non_des_wait_for_stage_raises_for_unknown_policy(ctx):
    ctx.stage_timing_policy = {
        "mri_to_report": "BANANA",
    }

    with pytest.raises(ValueError, match="Unknown timing policy"):
        stage_engine2.get_non_des_wait_for_stage("mri_to_report", ctx)


# =========================================================
# enter_wait_stage behaviour with timing policy
# =========================================================

def test_enter_wait_stage_mc_empirical_uses_empirical_wait(monkeypatch, ctx):
    monkeypatch.setattr(stage_engine2, "get_non_des_wait_for_stage", lambda stage_name, ctx: 8)

    ctx.wait_time_mode = {
        "mri_to_report": WAIT_MODE_MC,
    }

    patient = stage_engine2.create_new_patient(1, date(2026, 1, 5))
    patient.current_date = date(2026, 1, 10)

    enter_wait_stage(patient, "mri_to_report", ctx)

    ready_date = date(2026, 1, 18)
    assert ready_date in ctx.pending_mc["mri_to_report"]
    item = ctx.pending_mc["mri_to_report"][ready_date][0]
    assert item.sampled_wait == 8


def test_enter_wait_stage_mc_fixed_uses_fixed_wait(monkeypatch, ctx):
    ctx.wait_time_mode = {
        "mri_to_report": WAIT_MODE_MC,
    }
    ctx.stage_timing_policy = {
        "mri_to_report": "FIXED",
    }
    ctx.fixed_wait_days_by_stage = {
        "mri_to_report": 1,
    }

    patient = stage_engine2.create_new_patient(2, date(2026, 1, 5))
    patient.current_date = date(2026, 1, 20)

    enter_wait_stage(patient, "mri_to_report", ctx)

    ready_date = date(2026, 1, 21)
    assert ready_date in ctx.pending_mc["mri_to_report"]
    item = ctx.pending_mc["mri_to_report"][ready_date][0]
    assert item.sampled_wait == 1


def test_enter_wait_stage_des_does_not_use_fixed_or_empirical_wait(monkeypatch, ctx):
    called = {"n": 0}

    def fake_get_non_des_wait(stage_name, ctx):
        called["n"] += 1
        return 99

    monkeypatch.setattr(stage_engine2, "get_non_des_wait_for_stage", fake_get_non_des_wait)

    ctx.wait_time_mode = {
        "ref_to_mri": WAIT_MODE_DES,
    }
    ctx.stage_timing_policy = {
        "ref_to_mri": "FIXED",
    }
    ctx.fixed_wait_days_by_stage = {
        "ref_to_mri": 1,
    }

    patient = stage_engine2.create_new_patient(3, date(2026, 1, 5))
    enter_wait_stage(patient, "ref_to_mri", ctx)

    assert called["n"] == 0
    assert len(ctx.resources["MRI"].queue) == 1


# =========================================================
# Fixed-rule scenario behaviour
# =========================================================

def test_all_mc_fixed_rule_applies_even_without_des(monkeypatch, ctx):
    """
    If everything is MC, fixed stage rules should still apply.
    Example: MRI performed, then MRI report ready exactly 1 day later.
    """
    ctx.wait_time_mode = {
        stage: WAIT_MODE_MC for stage in stage_engine2.STAGE_CONFIG
    }
    ctx.stage_timing_policy = {
        "mri_to_report": "FIXED",
        "report_to_biopmdt": "FIXED",
    }
    ctx.fixed_wait_days_by_stage = {
        "mri_to_report": 1,
        "report_to_biopmdt": 0,
    }

    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 99)
    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)

    patient = stage_engine2.create_new_patient(4, date(2026, 1, 5))

    # Start directly after MRI for a focused test
    patient.current_date = date(2026, 1, 10)
    enter_wait_stage(patient, "mri_to_report", ctx)

    completed = []

    # day +1 should trigger mri_report_ready
    process_mc_stage_for_day("mri_to_report", date(2026, 1, 11), ctx, completed)
    assert any(e["event"] == "mri_report_ready" for e in patient.events)

    # report_to_biopmdt fixed to 0, so it should now be due same day
    process_mc_stage_for_day("report_to_biopmdt", date(2026, 1, 11), ctx, completed)

    event_names = [e["event"] for e in patient.events]
    assert "MDT_occured" in event_names
    assert "mdt_decision" in event_names
    assert patient.is_complete is True


# =========================================================
# Scenario builder tests
# =========================================================

def test_build_scenario_all_mc_baseline():
    cfg = scenarios.build_scenario_config(
        name="ALL_MC_BASELINE",
        start_date=date(2026, 1, 5),
        n_days=100,
        lam_per_workday=2.0,
    )

    assert cfg.scenario_name == "ALL_MC_BASELINE"
    assert all(mode == WAIT_MODE_MC for mode in cfg.wait_time_mode.values())
    assert all(policy == "EMPIRICAL" for policy in cfg.stage_timing_policy.values())
    assert cfg.fixed_wait_days_by_stage == {}


def test_build_scenario_hybrid_baseline():
    cfg = scenarios.build_scenario_config(
        name="HYBRID_BASELINE",
        start_date=date(2026, 1, 5),
        n_days=100,
        lam_per_workday=2.0,
    )

    assert cfg.scenario_name == "HYBRID_BASELINE"
    assert cfg.wait_time_mode["ref_to_mri"] == WAIT_MODE_DES
    assert cfg.wait_time_mode["biopmdt_to_biopsy"] == WAIT_MODE_DES
    assert cfg.wait_time_mode["mri_to_report"] == WAIT_MODE_MC
    assert cfg.stage_timing_policy["mri_to_report"] == "EMPIRICAL"


def test_build_scenario_prostad():
    cfg = scenarios.build_scenario_config(
        name="PROSTAD",
        start_date=date(2026, 1, 5),
        n_days=100,
        lam_per_workday=2.0,
    )

    assert cfg.scenario_name == "PROSTAD"
    assert cfg.wait_time_mode["ref_to_mri"] == WAIT_MODE_DES
    assert cfg.wait_time_mode["biopmdt_to_biopsy"] == WAIT_MODE_DES

    assert cfg.stage_timing_policy["mri_to_report"] == "FIXED"
    assert cfg.stage_timing_policy["report_to_biopmdt"] == "FIXED"
    assert cfg.fixed_wait_days_by_stage["mri_to_report"] == 1
    assert cfg.fixed_wait_days_by_stage["report_to_biopmdt"] == 0

    assert cfg.mri_capacity_by_weekday[1] == 4


def test_build_scenario_unknown_raises():
    with pytest.raises(ValueError, match="Unknown scenario name"):
        scenarios.build_scenario_config(
            name="NOT_A_REAL_SCENARIO",
            start_date=date(2026, 1, 5),
            n_days=100,
            lam_per_workday=2.0,
        )

def test_prostad_fixed_rule_chain_same_day(monkeypatch, ctx):
    ctx.wait_time_mode = {
        stage: WAIT_MODE_MC for stage in stage_engine2.STAGE_CONFIG
    }
    ctx.stage_timing_policy = {
        "mri_to_report": "FIXED",
        "report_to_biopmdt": "FIXED",
    }
    ctx.fixed_wait_days_by_stage = {
        "mri_to_report": 1,
        "report_to_biopmdt": 0,
    }

    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)

    patient = stage_engine2.create_new_patient(5, date(2026, 1, 5))
    patient.current_date = date(2026, 1, 10)

    stage_engine2.enter_wait_stage(patient, "mri_to_report", ctx)

    completed = []
    stage_engine2.process_all_mc_due_today_until_stable(
        date(2026, 1, 11),
        ctx,
        completed,
    )

    event_names = [e["event"] for e in patient.events]
    assert "mri_report_ready" in event_names
    assert "MDT_occured" in event_names
    assert patient.is_complete is True

def test_deterministic_prostad_fixed_timing_dates(monkeypatch, ctx):
    ctx.wait_time_mode = {
        stage: WAIT_MODE_MC for stage in stage_engine2.STAGE_CONFIG
    }

    ctx.stage_timing_policy = {
        "mri_to_report": "FIXED",
        "report_to_biopmdt": "FIXED",
    }

    ctx.fixed_wait_days_by_stage = {
        "mri_to_report": 1,
        "report_to_biopmdt": 0,
    }

    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)

    patient = stage_engine2.create_new_patient(99, date(2026, 1, 5))
    patient.current_date = date(2026, 1, 10)

    stage_engine2.enter_wait_stage(patient, "mri_to_report", ctx)

    completed = []
    stage_engine2.process_all_mc_due_today_until_stable(
        date(2026, 1, 11),
        ctx,
        completed,
    )

    events = {e["event"]: e for e in patient.events}

    assert events["mri_report_ready"]["date"] == date(2026, 1, 11)
    assert events["MDT_occured"]["date"] == date(2026, 1, 11)
    assert events["mdt_decision"]["date"] == date(2026, 1, 11)
    assert patient.is_complete is True

