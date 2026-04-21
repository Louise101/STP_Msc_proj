from __future__ import annotations

from datetime import date
import numpy as np

from combined_des_engine import CombinedEngineConfig, run_day_loop_combined_engine
from combined_stage_engine import WAIT_MODE_MC, WAIT_MODE_DES

def build_test_config(seed: int = 123, lam: float = 0.5) -> CombinedEngineConfig:
    return CombinedEngineConfig(
        start_date=date(2026, 1, 5),
        n_days=60,
        lam_per_workday=lam,
        p_prostad=0.5,
        mri_capacity_by_weekday_prostad={1: 4},
        seed=seed,
        scenario_name="TEST",
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
        prostad_fixed_wait_days_by_stage={
            "mri_to_report": 1,
            "report_to_biopmdt": 0,
        },
    )

def get_patient_event_names(patient) -> list[str]:
    return [e["event"] for e in patient.events]

def test_reproducibility_fixed_seed():
    cfg1 = build_test_config(seed=42)
    cfg2 = build_test_config(seed=42)

    res1 = run_day_loop_combined_engine(cfg1)
    res2 = run_day_loop_combined_engine(cfg2)

    assert res1["summary_stats"]["total_patients_completed"] == res2["summary_stats"]["total_patients_completed"]
    assert res1["event_log"].equals(res2["event_log"])

def test_zero_referrals():
    cfg = build_test_config(seed=1, lam=0.0)
    res = run_day_loop_combined_engine(cfg)

    assert res["summary_stats"]["total_patients_completed"] == 0
    assert len(res["all_patients_objects"]) == 0

def test_waits_non_negative():
    cfg = build_test_config(seed=2)
    res = run_day_loop_combined_engine(cfg)

    waits = []
    for patient in res["all_patients_objects"]:
        for event in patient.events:
            if "wait_days" in event and event["wait_days"] is not None:
                waits.append(event["wait_days"])

    assert len(waits) > 0
    assert all(w >= 0 for w in waits)

def test_event_order_is_logical():
    cfg = build_test_config(seed=3)
    res = run_day_loop_combined_engine(cfg)

    required_order = [
        "referral_received",
        "mri_performed",
        "mri_report_ready",
        "MDT_occured",
    ]

    for patient in res["all_patients_objects"]:
        names = [e["event"] for e in patient.events]

        positions = {}
        for ev in required_order:
            if ev in names:
                positions[ev] = names.index(ev)

        ordered_positions = [positions[ev] for ev in required_order if ev in positions]
        assert ordered_positions == sorted(ordered_positions)

def test_completed_patients_have_terminal_event():
    cfg = build_test_config(seed=4)
    res = run_day_loop_combined_engine(cfg)

    for patient in res["completed_patients_objects"]:
        names = get_patient_event_names(patient)
        assert (
            "Outpatient_appointment_occured" in names
            or "Path_report_outcome" in names
            or "mdt_decision" in names
        )

def test_prostad_fixed_rule_applied():
    cfg = build_test_config(seed=5)
    res = run_day_loop_combined_engine(cfg)

    for patient in res["all_patients_objects"]:
        if patient.data.get("pathway_type") == "PROSTAD":
            events = {e["event"]: e for e in patient.events}
            if "mri_report_ready" in events and "mri_performed" in events:
                diff = (events["mri_report_ready"]["date"] - events["mri_performed"]["date"]).days
                assert diff >= 0