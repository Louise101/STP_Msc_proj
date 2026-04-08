from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest
import numpy as np

import stage_engine2
import des_engine

from patient_state import PatientState


# =========================================================
# Helpers / fakes
# =========================================================

@dataclass
class FakeServiceEvent:
    patient: any
    wait_days: int


class FakeQueueResource:
    """
    Minimal fake QueueResource for testing DES behaviour.
    Processes up to capacity for the weekday.
    """
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
def basic_ctx(rng):
    """
    A minimal StageContext for direct stage_engine2 tests.
    """
    return stage_engine2.StageContext(
    rng=rng,
    pdfs={},
    branching={
        "biopmdt_outcome": {1: 1.0},
        "pathrep_outcome": {1: 1.0},
    },
    wait_time_mode={},
    pending_mc=stage_engine2.initialize_pending_mc(),
    resources={
        "MRI": FakeQueueResource("MRI", {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
        "Biopsy": FakeQueueResource("Biopsy", {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
    },
    stage_timing_policy={},
    fixed_wait_days_by_stage={},
    scenario_name=None,
)

# =========================================================
# Stage engine tests
# =========================================================

def test_initialize_pending_mc_has_all_stage_keys():
    pending = stage_engine2.initialize_pending_mc()

    assert set(pending.keys()) == set(stage_engine2.STAGE_CONFIG.keys())
    for stage_name in stage_engine2.STAGE_CONFIG:
        assert pending[stage_name] == {}


def test_enter_wait_stage_mc_puts_patient_into_pending(monkeypatch, basic_ctx):
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 5)

    patient = PatientState(
        patient_id=1,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )

    basic_ctx.wait_time_mode = {"ref_to_mri": stage_engine2.WAIT_MODE_MC}

    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    ready_date = date(2026, 1, 10)
    items = basic_ctx.pending_mc["ref_to_mri"][ready_date]

    assert len(items) == 1
    assert items[0].patient is patient
    assert items[0].sampled_wait == 5
    assert items[0].stage_name == "ref_to_mri"
    assert patient.current_stage == "ref_to_mri"


def test_enter_wait_stage_des_puts_patient_into_resource_queue(basic_ctx):
    patient = PatientState(
        patient_id=2,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )

    basic_ctx.wait_time_mode = {"ref_to_mri": stage_engine2.WAIT_MODE_DES}

    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    mri_queue = basic_ctx.resources["MRI"].queue
    assert len(mri_queue) == 1
    qp = mri_queue[0]

    assert qp.patient_id == 2
    assert qp.payload["patient"] is patient
    assert qp.payload["stage_name"] == "ref_to_mri"
    assert qp.payload["entry_date"] == date(2026, 1, 5)


def test_process_mc_stage_for_day_releases_ready_patients(monkeypatch, basic_ctx):
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 2)

    patient = PatientState(
        patient_id=3,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_recieved", date(2026, 1, 5))

    basic_ctx.wait_time_mode = {
        "ref_to_mri": stage_engine2.WAIT_MODE_MC,
        "mri_to_report": stage_engine2.WAIT_MODE_MC,
    }

    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    completed = []
    stage_engine2.process_mc_stage_for_day(
        "ref_to_mri",
        date(2026, 1, 7),
        basic_ctx,
        completed,
    )

    event_names = [e["event"] for e in patient.events]
    assert "mri_performed" in event_names
    assert patient.current_date == date(2026, 1, 7)

    # After completing ref_to_mri, patient should have entered mri_to_report
    assert patient.current_stage == "mri_to_report"
    assert date(2026, 1, 9) in basic_ctx.pending_mc["mri_to_report"]


def test_full_mc_pathway_completes_patient(monkeypatch, basic_ctx):
    """
    End-to-end stage_engine2 test:
    all waits are 1 day, MDT always -> biopsy, pathology always -> cancer.
    """
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 1)
    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 1)
    monkeypatch.setattr(stage_engine2, "sample_pathology_outcome", lambda ctx: 1)

    basic_ctx.wait_time_mode = {
        stage: stage_engine2.WAIT_MODE_MC
        for stage in stage_engine2.STAGE_CONFIG
    }

    start = date(2026, 1, 5)
    patient = PatientState(
        patient_id=4,
        start_date=start,
        current_date=start,
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_recieved", start)

    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    completed = []

    for day_offset in range(1, 10):
        current_day = start + timedelta(days=day_offset)
        for stage_name in stage_engine2.STAGE_CONFIG:
            stage_engine2.process_mc_stage_for_day(
                stage_name, current_day, basic_ctx, completed
            )

    assert patient.is_complete is True
    assert patient in completed

    event_names = [e["event"] for e in patient.events]
    assert event_names == [
        "referral_recieved",
        "mri_performed",
        "mri_report_ready",
        "MDT_occured",
        "mdt_decision",
        "biopsy_done",
        "Path_report_recieved",
        "Path_report_outcome",
        "Treatment_options_MDT_occured",
        "Outpatient_appointment_occured",
    ]


def test_des_resource_processing_releases_patient(monkeypatch, basic_ctx):
    """
    Tests DES processing of ref_to_mri.
    """
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 0)

    basic_ctx.wait_time_mode = {
        "ref_to_mri": stage_engine2.WAIT_MODE_DES,
        "mri_to_report": stage_engine2.WAIT_MODE_MC,
    }

    current_day = date(2026, 1, 6)

    patient = PatientState(
        patient_id=5,
        start_date=current_day - timedelta(days=1),
        current_date=current_day - timedelta(days=1),
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_recieved", current_day - timedelta(days=1))

    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    completed = []
    stage_engine2.process_des_resource_for_day("MRI", current_day, basic_ctx, completed)

    event_names = [e["event"] for e in patient.events]
    assert "mri_performed" in event_names
    assert patient.current_stage == "mri_to_report"

    # because mri_to_report is MC with wait 0, it should be due today
    assert current_day in basic_ctx.pending_mc["mri_to_report"]


# =========================================================
# des_engine run loop tests
# =========================================================

def test_run_day_loop_all_mc_single_referral_completes(monkeypatch):
    """
    Tests des_engine.run_day_loop_with_stage_engine with deterministic behaviour:
    - 1 referral on first day only
    - all waits 0 days
    - MDT outcome = no biopsy pathway end
    """
    # Patch deterministic daily arrivals: 1 on first weekday only, then 0
    call_count = {"n": 0}

    def fake_poisson(lam_per_workday, weekday, rng):
        call_count["n"] += 1
        return 1 if call_count["n"] == 1 else 0

    monkeypatch.setattr(des_engine, "sample_poisson_weekday_only", fake_poisson)

    # Patch build functions in des_engine module
    monkeypatch.setattr(des_engine, "build_pdfs", lambda: {})
    monkeypatch.setattr(
        des_engine,
        "build_branching",
        lambda: {
            "biopmdt_outcome": {0: 1.0},   # always no biopsy
            "pathrep_outcome": {1: 1.0},
        },
    )

    # Patch stage engine sampling
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 0)
    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)
    monkeypatch.setattr(stage_engine2, "sample_pathology_outcome", lambda ctx: 1)

    cfg = des_engine.EngineConfig(
        start_date=date(2026, 1, 5),  # Monday
        n_days=3,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        biopsy_capacity_by_weekday={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        wait_time_mode={
            stage: stage_engine2.WAIT_MODE_MC
            for stage in stage_engine2.STAGE_CONFIG
        },
        seed=123,
    )

    result = des_engine.run_day_loop_with_stage_engine(cfg)

    assert result["summary_stats"]["total_patients_completed"] == 1
    assert result["daily_referrals"][date(2026, 1, 5)] == 1

    events, total_days = result["patient_results"][0]
    event_names = [e["event"] for e in events]

    assert event_names == [
        "referral_recieved",
        "mri_performed",
        "mri_report_ready",
        "MDT_occured",
        "mdt_decision",
    ]
    assert total_days >= 0


def test_run_day_loop_mixed_des_mc(monkeypatch):
    """
    Mixed test:
    - ref_to_mri is DES
    - later stages are MC
    - 1 referral on day 1 only
    - MRI capacity 1/day
    - MDT outcome no biopsy
    """
    call_count = {"n": 0}

    def fake_poisson(lam_per_workday, weekday, rng):
        call_count["n"] += 1
        return 1 if call_count["n"] == 1 else 0

    monkeypatch.setattr(des_engine, "sample_poisson_weekday_only", fake_poisson)
    monkeypatch.setattr(des_engine, "build_pdfs", lambda: {})
    monkeypatch.setattr(
        des_engine,
        "build_branching",
        lambda: {
            "biopmdt_outcome": {0: 1.0},
            "pathrep_outcome": {1: 1.0},
        },
    )

    # Replace QueueResource inside des_engine with fake
    monkeypatch.setattr(des_engine, "QueueResource", FakeQueueResource)

    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 0)
    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)
    monkeypatch.setattr(stage_engine2, "sample_pathology_outcome", lambda ctx: 1)

    wait_modes = {
        stage: stage_engine2.WAIT_MODE_MC
        for stage in stage_engine2.STAGE_CONFIG
    }
    wait_modes["ref_to_mri"] = stage_engine2.WAIT_MODE_DES

    cfg = des_engine.EngineConfig(
        start_date=date(2026, 1, 5),  # Monday
        n_days=3,
        lam_per_workday=1.0,
        mri_capacity_by_weekday={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        biopsy_capacity_by_weekday={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        wait_time_mode=wait_modes,
        seed=123,
    )

    result = des_engine.run_day_loop_with_stage_engine(cfg)

    assert result["summary_stats"]["total_patients_completed"] == 1

    events, total_days = result["patient_results"][0]
    event_names = [e["event"] for e in events]

    assert event_names == [
        "referral_recieved",
        "mri_performed",
        "mri_report_ready",
        "MDT_occured",
        "mdt_decision",
    ]

    # MRI was DES, so resource stats should show activity
    mri_started = result["resources"]["MRI"]["daily_started"]
    assert any(v == 1 for v in mri_started.values())

def test_same_day_zero_wait_mc_cascade(monkeypatch, basic_ctx):
    monkeypatch.setattr(stage_engine2, "sample_wait_for_stage", lambda stage_name, ctx: 0)
    monkeypatch.setattr(stage_engine2, "sample_mdt_decision", lambda ctx: 0)

    basic_ctx.wait_time_mode = {
        stage: stage_engine2.WAIT_MODE_MC
        for stage in stage_engine2.STAGE_CONFIG
    }

    patient = stage_engine2.create_new_patient(1, date(2026, 1, 5))
    stage_engine2.enter_wait_stage(patient, "ref_to_mri", basic_ctx)

    completed = []
    stage_engine2.process_all_mc_due_today_until_stable(
        date(2026, 1, 5),
        basic_ctx,
        completed,
    )

    assert patient.is_complete is True
    assert patient in completed

    event_names = [e["event"] for e in patient.events]
    assert event_names == [
        "referral_recieved",
        "mri_performed",
        "mri_report_ready",
        "MDT_occured",
        "mdt_decision",
    ]