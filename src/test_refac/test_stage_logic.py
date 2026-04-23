from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.patient import PatientState
from core.queueing import QueuePatient
from engine.pathway_definitions import STAGE_CONFIG, WAIT_MODE_DES, WAIT_MODE_MC
from engine.stage_logic import (
    DelayQueueItem,
    count_mc_in_stage,
    get_rule_based_wait,
    initialize_pending_des_arrivals,
    initialize_pending_mc,
    initialize_stage_activity,
    make_patient_rng,
    make_stable_int_seed,
    release_due_des_arrivals_for_day,
    sample_mdt_decision,
    sample_pathology_outcome,
    sample_wait_for_stage,
    snapshot_stage_occupancy,
)


def make_patient(patient_id: int = 1) -> PatientState:
    patient = PatientState(
        patient_id=patient_id,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )
    patient.add_event("referral_received", date(2026, 1, 5))
    return patient


def make_context() -> SimpleNamespace:
    return SimpleNamespace(
        base_seed=123,
        pdfs={
            "pre_referral_to_mri": pd.Series([10, 20, 30]),
            "pre_mri_to_mrireport": pd.Series([1, 2, 3]),
            "pre_mrirep_to_biopsymdt": pd.Series([0, 1, 2]),
            "pre_biopmdt_to_biop": pd.Series([5, 10, 15]),
            "pre_biop_to_pathrep": pd.Series([7, 8, 9]),
            "pre_pathrep_to_treatmdt": pd.Series([11, 12, 13]),
            "pre_treatmdt_to_outpat": pd.Series([14, 15, 16]),
        },
        branching={
            "biopmdt_outcome": {0: 0.2, 1: 0.8},
            "pathrep_outcome": {0: 0.3, 1: 0.7},
        },
        wait_time_mode={stage: WAIT_MODE_MC for stage in STAGE_CONFIG},
        pending_mc=initialize_pending_mc(),
        resources={},
        stage_activity=initialize_stage_activity(),
        pending_des_arrivals=initialize_pending_des_arrivals(),
    )


class DummyQueueResource:
    def __init__(self, queue_len=0):
        self.items = []
        self._queue_len = queue_len

    def add_patient(self, patient):
        self.items.append(patient)

    def queue_length(self):
        return self._queue_len


def test_make_stable_int_seed_is_reproducible():
    seed_1 = make_stable_int_seed(123, 1, "stream")
    seed_2 = make_stable_int_seed(123, 1, "stream")

    assert seed_1 == seed_2
    assert isinstance(seed_1, int)


def test_make_stable_int_seed_changes_when_inputs_change():
    seed_1 = make_stable_int_seed(123, 1, "stream")
    seed_2 = make_stable_int_seed(123, 2, "stream")
    seed_3 = make_stable_int_seed(123, 1, "other_stream")

    assert seed_1 != seed_2
    assert seed_1 != seed_3


def test_make_patient_rng_is_reproducible():
    rng_1 = make_patient_rng(123, 5, "wait_ref_to_mri")
    rng_2 = make_patient_rng(123, 5, "wait_ref_to_mri")

    draws_1 = rng_1.integers(0, 100, size=5)
    draws_2 = rng_2.integers(0, 100, size=5)

    assert np.array_equal(draws_1, draws_2)


def test_make_patient_rng_differs_across_stream_names():
    rng_1 = make_patient_rng(123, 5, "stream_a")
    rng_2 = make_patient_rng(123, 5, "stream_b")

    draws_1 = rng_1.integers(0, 100, size=5)
    draws_2 = rng_2.integers(0, 100, size=5)

    assert not np.array_equal(draws_1, draws_2)


def test_initialize_pending_mc_creates_empty_dict_for_each_stage():
    pending = initialize_pending_mc()

    assert set(pending.keys()) == set(STAGE_CONFIG.keys())
    assert all(isinstance(v, dict) for v in pending.values())
    assert all(v == {} for v in pending.values())


def test_initialize_pending_des_arrivals_creates_expected_resource_map():
    pending = initialize_pending_des_arrivals()

    assert pending == {"MRI_PROSTAD": {}}


def test_initialize_stage_activity_creates_expected_structure():
    activity = initialize_stage_activity()

    assert set(activity.keys()) == set(STAGE_CONFIG.keys())

    for stage_name, stage_data in activity.items():
        assert set(stage_data.keys()) == {"daily_arrivals", "daily_in_stage", "daily_completed"}
        assert stage_data["daily_arrivals"] == {}
        assert stage_data["daily_in_stage"] == {}
        assert stage_data["daily_completed"] == {}


def test_sample_wait_for_stage_is_reproducible_for_same_patient_and_stage():
    patient = make_patient(patient_id=7)
    ctx = make_context()

    wait_1 = sample_wait_for_stage("ref_to_mri", patient, ctx)
    wait_2 = sample_wait_for_stage("ref_to_mri", patient, ctx)

    assert wait_1 == wait_2
    assert wait_1 in {10, 20, 30}


def test_sample_wait_for_stage_uses_stage_specific_pdf():
    patient = make_patient(patient_id=7)
    ctx = make_context()

    wait = sample_wait_for_stage("mri_to_report", patient, ctx)

    assert wait in {1, 2, 3}


def test_sample_mdt_decision_is_reproducible():
    patient = make_patient(patient_id=8)
    ctx = make_context()

    draw_1 = sample_mdt_decision(patient, ctx)
    draw_2 = sample_mdt_decision(patient, ctx)

    assert draw_1 == draw_2
    assert draw_1 in {0, 1}


def test_sample_pathology_outcome_is_reproducible():
    patient = make_patient(patient_id=9)
    ctx = make_context()

    draw_1 = sample_pathology_outcome(patient, ctx)
    draw_2 = sample_pathology_outcome(patient, ctx)

    assert draw_1 == draw_2
    assert draw_1 in {0, 1}


def test_get_rule_based_wait_for_mri_to_report():
    assert get_rule_based_wait("mri_to_report") == 1


def test_get_rule_based_wait_for_report_to_biopmdt():
    assert get_rule_based_wait("report_to_biopmdt") == 0


def test_get_rule_based_wait_raises_for_unknown_stage():
    with pytest.raises(ValueError, match="No rule-based wait defined"):
        get_rule_based_wait("ref_to_mri")


def test_release_due_des_arrivals_for_day_moves_only_todays_patients():
    today = date(2026, 1, 5)
    tomorrow = date(2026, 1, 6)

    patient_today = QueuePatient(patient_id=1, referral_date=today)
    patient_tomorrow = QueuePatient(patient_id=2, referral_date=tomorrow)

    resource = DummyQueueResource()
    ctx = SimpleNamespace(
        pending_des_arrivals={
            "MRI_PROSTAD": {
                today: [patient_today],
                tomorrow: [patient_tomorrow],
            }
        },
        resources={"MRI_PROSTAD": resource},
    )

    release_due_des_arrivals_for_day(today, ctx)

    assert resource.items == [patient_today]
    assert today not in ctx.pending_des_arrivals["MRI_PROSTAD"]
    assert tomorrow in ctx.pending_des_arrivals["MRI_PROSTAD"]
    assert ctx.pending_des_arrivals["MRI_PROSTAD"][tomorrow] == [patient_tomorrow]


def test_count_mc_in_stage_counts_across_all_ready_dates():
    patient = make_patient()
    pending_mc = initialize_pending_mc()
    pending_mc["ref_to_mri"] = {
        date(2026, 1, 6): [
            DelayQueueItem(patient, date(2026, 1, 5), date(2026, 1, 6), 1, "ref_to_mri"),
            DelayQueueItem(patient, date(2026, 1, 5), date(2026, 1, 6), 1, "ref_to_mri"),
        ],
        date(2026, 1, 7): [
            DelayQueueItem(patient, date(2026, 1, 5), date(2026, 1, 7), 2, "ref_to_mri"),
        ],
    }

    count = count_mc_in_stage(pending_mc, "ref_to_mri")

    assert count == 3


def test_snapshot_stage_occupancy_counts_mc_stage_from_pending_mc():
    patient = make_patient()
    ctx = make_context()

    ctx.pending_mc["ref_to_mri"] = {
        date(2026, 1, 6): [
            DelayQueueItem(patient, date(2026, 1, 5), date(2026, 1, 6), 1, "ref_to_mri"),
            DelayQueueItem(patient, date(2026, 1, 5), date(2026, 1, 6), 1, "ref_to_mri"),
        ]
    }

    snapshot_stage_occupancy(date(2026, 1, 5), ctx)

    assert ctx.stage_activity["ref_to_mri"]["daily_in_stage"][date(2026, 1, 5)] == 2


def test_snapshot_stage_occupancy_counts_des_stage_from_resource_queue():
    ctx = make_context()
    ctx.wait_time_mode["ref_to_mri"] = WAIT_MODE_DES
    ctx.resources["MRI_PROSTAD"] = DummyQueueResource(queue_len=4)

    snapshot_stage_occupancy(date(2026, 1, 5), ctx)

    assert ctx.stage_activity["ref_to_mri"]["daily_in_stage"][date(2026, 1, 5)] == 4


def test_snapshot_stage_occupancy_defaults_unknown_mode_to_zero():
    ctx = make_context()
    ctx.wait_time_mode["ref_to_mri"] = "UNKNOWN_MODE"

    snapshot_stage_occupancy(date(2026, 1, 5), ctx)

    assert ctx.stage_activity["ref_to_mri"]["daily_in_stage"][date(2026, 1, 5)] == 0


def test_snapshot_stage_occupancy_writes_all_stages_for_given_date():
    ctx = make_context()
    ctx.resources["MRI_PROSTAD"] = DummyQueueResource(queue_len=0)

    snapshot_stage_occupancy(date(2026, 1, 5), ctx)

    for stage_name in STAGE_CONFIG:
        assert date(2026, 1, 5) in ctx.stage_activity[stage_name]["daily_in_stage"]