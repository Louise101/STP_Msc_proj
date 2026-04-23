from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.patient import PatientState
from engine.combined_engine import (
    CombinedEngineConfig,
    complete_wait_stage_combined,
    create_new_combined_patient,
    enter_stage_for_patient,
    get_patient_stage_rules,
    run_day_loop_combined_engine,
    sample_patient_pathway,
)
from engine.pathway_definitions import WAIT_MODE_DES, WAIT_MODE_MC


def make_config(**overrides) -> CombinedEngineConfig:
    """Build a minimal engine config for tests."""
    cfg = CombinedEngineConfig(
        start_date=date(2026, 1, 5),
        n_days=10,
        lam_per_workday=1.0,
        p_prostad=0.5,
        mri_capacity_by_weekday_prostad={1: 4},
        seed=123,
        baseline_wait_time_mode={
            "ref_to_mri": WAIT_MODE_MC,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        prostad_wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        baseline_stage_timing_policy={
            "mri_to_report": "EMPIRICAL",
            "report_to_biopmdt": "EMPIRICAL",
        },
        prostad_stage_timing_policy={
            "mri_to_report": "FIXED",
            "report_to_biopmdt": "FIXED",
        },
        baseline_fixed_wait_days_by_stage={},
        prostad_fixed_wait_days_by_stage={
            "mri_to_report": 1,
            "report_to_biopmdt": 0,
        },
        scenario_name="TEST",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_patient(pathway_type="BASELINE") -> PatientState:
    patient = PatientState(
        patient_id=1,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )
    patient.pathway_type = pathway_type
    patient.data["pathway_type"] = pathway_type
    patient.add_event("referral_received", date(2026, 1, 5))
    return patient


def make_minimal_ctx():
    """Minimal fake StageContext-like object for direct helper tests."""
    return SimpleNamespace(
        stage_activity={
            "ref_to_mri": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "mri_to_report": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "report_to_biopmdt": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "biopmdt_to_biopsy": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "biopsy_to_pathrep": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "pathrep_to_treatmdt": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
            "treatmdt_to_outpat": {"daily_arrivals": {}, "daily_completed": {}, "daily_in_stage": {}},
        },
        pending_mc={
            "ref_to_mri": {},
            "mri_to_report": {},
            "report_to_biopmdt": {},
            "biopmdt_to_biopsy": {},
            "biopsy_to_pathrep": {},
            "pathrep_to_treatmdt": {},
            "treatmdt_to_outpat": {},
        },
        pending_des_arrivals={"MRI_PROSTAD": {}},
        resources={"MRI_PROSTAD": object()},
        pdfs={},
        branching={
            "biopmdt_outcome": {0: 1.0},
            "pathrep_outcome": {0: 1.0},
        },
        base_seed=123,
    )


def test_create_new_combined_patient_initialises_expected_state():
    patient = create_new_combined_patient(7, date(2026, 1, 5))

    assert patient.patient_id == 7
    assert patient.start_date == date(2026, 1, 5)
    assert patient.current_date == date(2026, 1, 5)
    assert patient.current_stage == "ref_to_mri"
    assert patient.has_event("referral_received")


def test_sample_patient_pathway_is_reproducible_for_same_patient_and_seed():
    patient_1 = create_new_combined_patient(1, date(2026, 1, 5))
    patient_2 = create_new_combined_patient(1, date(2026, 1, 5))

    route_1 = sample_patient_pathway(patient_1, base_seed=123, p_prostad=0.5)
    route_2 = sample_patient_pathway(patient_2, base_seed=123, p_prostad=0.5)

    assert route_1 == route_2
    assert route_1 in {"BASELINE", "PROSTAD"}


def test_sample_patient_pathway_all_baseline_when_probability_zero():
    patient = create_new_combined_patient(1, date(2026, 1, 5))

    route = sample_patient_pathway(patient, base_seed=123, p_prostad=0.0)

    assert route == "BASELINE"


def test_sample_patient_pathway_all_prostad_when_probability_one():
    patient = create_new_combined_patient(1, date(2026, 1, 5))

    route = sample_patient_pathway(patient, base_seed=123, p_prostad=1.0)

    assert route == "PROSTAD"


def test_get_patient_stage_rules_returns_baseline_rules_for_baseline_patient():
    cfg = make_config()
    patient = make_patient("BASELINE")

    wait_modes, timing_policy, fixed_waits = get_patient_stage_rules(patient, cfg)

    assert wait_modes["ref_to_mri"] == WAIT_MODE_MC
    assert timing_policy["mri_to_report"] == "EMPIRICAL"
    assert fixed_waits == {}


def test_get_patient_stage_rules_returns_prostad_rules_for_prostad_patient():
    cfg = make_config()
    patient = make_patient("PROSTAD")

    wait_modes, timing_policy, fixed_waits = get_patient_stage_rules(patient, cfg)

    assert wait_modes["ref_to_mri"] == WAIT_MODE_DES
    assert timing_policy["mri_to_report"] == "FIXED"
    assert fixed_waits["mri_to_report"] == 1


def test_complete_wait_stage_combined_exits_after_non_biopsy_mdt(monkeypatch):
    from engine import combined_engine as mod

    patient = make_patient("BASELINE")
    ctx = make_minimal_ctx()
    cfg = make_config()

    monkeypatch.setattr(mod, "sample_mdt_decision", lambda patient, ctx: 0)

    complete_wait_stage_combined(
        patient=patient,
        stage_name="report_to_biopmdt",
        completion_date=date(2026, 1, 10),
        wait_days=5,
        ctx=ctx,
        cfg=cfg,
    )

    assert patient.is_complete is True
    assert patient.exit_reason == "no_biopsy_after_mdt"
    assert patient.has_event("MDT_occured")
    assert patient.has_event("mdt_decision")


def test_complete_wait_stage_combined_exits_after_negative_pathology(monkeypatch):
    from engine import combined_engine as mod

    patient = make_patient("BASELINE")
    ctx = make_minimal_ctx()
    cfg = make_config()

    monkeypatch.setattr(mod, "sample_pathology_outcome", lambda patient, ctx: 0)

    complete_wait_stage_combined(
        patient=patient,
        stage_name="biopsy_to_pathrep",
        completion_date=date(2026, 1, 15),
        wait_days=7,
        ctx=ctx,
        cfg=cfg,
    )

    assert patient.is_complete is True
    assert patient.exit_reason == "no_cancer_on_pathology"
    assert patient.has_event("Path_report_recieved")
    assert patient.has_event("Path_report_outcome")


def test_complete_wait_stage_combined_marks_full_pathway_complete():
    patient = make_patient("BASELINE")
    ctx = make_minimal_ctx()
    cfg = make_config()

    complete_wait_stage_combined(
        patient=patient,
        stage_name="treatmdt_to_outpat",
        completion_date=date(2026, 1, 20),
        wait_days=10,
        ctx=ctx,
        cfg=cfg,
    )

    assert patient.is_complete is True
    assert patient.exit_reason == "full_pathway_complete"
    assert patient.has_event("Outpatient_appointment_occured")


def test_run_day_loop_combined_engine_zero_referrals_runs_safely():
    cfg = make_config(n_days=5, lam_per_workday=0.0, p_prostad=0.0)
    daily_override = {
        date(2026, 1, 5): 0,
        date(2026, 1, 6): 0,
        date(2026, 1, 7): 0,
        date(2026, 1, 8): 0,
        date(2026, 1, 9): 0,
    }

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    assert result["summary_stats"]["total_patients_completed"] == 0
    assert result["event_log"].empty
    assert sum(result["daily_referrals"].values()) == 0
    assert len(result["all_patients_objects"]) == 0


def test_run_day_loop_combined_engine_returns_expected_top_level_keys():
    cfg = make_config(n_days=3, p_prostad=0.0)
    daily_override = {
        date(2026, 1, 5): 1,
        date(2026, 1, 6): 0,
        date(2026, 1, 7): 0,
    }

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    expected_keys = {
        "patient_results",
        "all_patient_results",
        "event_log",
        "daily_referrals",
        "completed_patients_objects",
        "all_patients_objects",
        "resources",
        "stage_activity",
        "summary_stats",
    }
    assert expected_keys.issubset(result.keys())


def test_run_day_loop_combined_engine_respects_daily_referral_override():
    cfg = make_config(n_days=3, p_prostad=0.0)
    daily_override = {
        date(2026, 1, 5): 2,
        date(2026, 1, 6): 1,
        date(2026, 1, 7): 0,
    }

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    assert result["daily_referrals"] == daily_override
    assert len(result["all_patients_objects"]) == 3


def test_run_day_loop_combined_engine_assigns_all_baseline_when_p_prostad_zero():
    cfg = make_config(n_days=3, p_prostad=0.0)
    daily_override = {
        date(2026, 1, 5): 2,
        date(2026, 1, 6): 1,
        date(2026, 1, 7): 0,
    }

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    assert all(p.pathway_type == "BASELINE" for p in result["all_patients_objects"])


def test_run_day_loop_combined_engine_assigns_all_prostad_when_p_prostad_one():
    cfg = make_config(n_days=3, p_prostad=1.0)
    daily_override = {
        date(2026, 1, 5): 2,
        date(2026, 1, 6): 1,
        date(2026, 1, 7): 0,
    }

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    assert all(p.pathway_type == "PROSTAD" for p in result["all_patients_objects"])


def test_run_day_loop_combined_engine_is_reproducible_for_same_seed():
    cfg1 = make_config(seed=123, n_days=5, p_prostad=0.5)
    cfg2 = make_config(seed=123, n_days=5, p_prostad=0.5)

    daily_override = {
        date(2026, 1, 5): 1,
        date(2026, 1, 6): 1,
        date(2026, 1, 7): 0,
        date(2026, 1, 8): 1,
        date(2026, 1, 9): 0,
    }

    result_1 = run_day_loop_combined_engine(cfg1, daily_referrals_override=daily_override)
    result_2 = run_day_loop_combined_engine(cfg2, daily_referrals_override=daily_override)

    assert result_1["daily_referrals"] == result_2["daily_referrals"]
    assert result_1["summary_stats"] == result_2["summary_stats"]
    pd.testing.assert_frame_equal(result_1["event_log"], result_2["event_log"])


def test_completed_patient_event_dates_are_monotonic():
    cfg = make_config(seed=123, n_days=10, p_prostad=0.5)
    daily_override = {cfg.start_date + pd.Timedelta(days=i): 1 if i < 5 else 0 for i in range(10)}
    # convert keys back to date objects
    daily_override = {k.date() if hasattr(k, "date") else k: v for k, v in daily_override.items()}

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    for patient in result["completed_patients_objects"]:
        dates = [event["date"] for event in patient.events if "date" in event]
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


def test_completed_patients_have_valid_exit_reasons():
    cfg = make_config(seed=123, n_days=10, p_prostad=0.5)
    daily_override = {cfg.start_date + pd.Timedelta(days=i): 1 if i < 5 else 0 for i in range(10)}
    daily_override = {k.date() if hasattr(k, "date") else k: v for k, v in daily_override.items()}

    result = run_day_loop_combined_engine(cfg, daily_referrals_override=daily_override)

    valid_exit_reasons = {
        "no_biopsy_after_mdt",
        "no_cancer_on_pathology",
        "full_pathway_complete",
    }

    for patient in result["completed_patients_objects"]:
        assert patient.exit_reason in valid_exit_reasons