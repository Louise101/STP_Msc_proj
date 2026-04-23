from datetime import date

from core.patient import PatientState


def make_patient() -> PatientState:
    """Helper to build a standard patient object for tests."""
    return PatientState(
        patient_id=1,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )


def test_patient_initialises_with_expected_fields():
    patient = make_patient()

    assert patient.patient_id == 1
    assert patient.start_date == date(2026, 1, 5)
    assert patient.current_date == date(2026, 1, 5)
    assert patient.current_stage == "ref_to_mri"

    assert patient.pathway_type is None
    assert patient.events == []
    assert patient.data == {}
    assert patient.is_complete is False
    assert patient.exit_reason is None


def test_add_event_appends_basic_event_record():
    patient = make_patient()

    patient.add_event("referral_received", date(2026, 1, 5))

    assert len(patient.events) == 1
    assert patient.events[0] == {
        "patient_id": 1,
        "event": "referral_received",
        "date": date(2026, 1, 5),
    }


def test_add_event_preserves_extra_keyword_fields():
    patient = make_patient()

    patient.add_event(
        "mri_performed",
        date(2026, 1, 10),
        wait_days=5,
        stage_name="ref_to_mri",
        outcome=None,
    )

    assert len(patient.events) == 1
    assert patient.events[0]["patient_id"] == 1
    assert patient.events[0]["event"] == "mri_performed"
    assert patient.events[0]["date"] == date(2026, 1, 10)
    assert patient.events[0]["wait_days"] == 5
    assert patient.events[0]["stage_name"] == "ref_to_mri"
    assert "outcome" in patient.events[0]
    assert patient.events[0]["outcome"] is None


def test_has_event_returns_true_when_event_exists():
    patient = make_patient()
    patient.add_event("referral_received", date(2026, 1, 5))
    patient.add_event("mri_performed", date(2026, 1, 10))

    assert patient.has_event("referral_received") is True
    assert patient.has_event("mri_performed") is True


def test_has_event_returns_false_when_event_missing():
    patient = make_patient()
    patient.add_event("referral_received", date(2026, 1, 5))

    assert patient.has_event("biopsy_done") is False


def test_total_days_in_system_returns_elapsed_days():
    patient = make_patient()
    patient.current_date = date(2026, 1, 20)

    assert patient.total_days_in_system() == 15


def test_total_days_in_system_can_be_zero():
    patient = make_patient()

    assert patient.total_days_in_system() == 0


def test_events_and_data_are_independent_between_patients():
    patient_1 = PatientState(
        patient_id=1,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )
    patient_2 = PatientState(
        patient_id=2,
        start_date=date(2026, 1, 5),
        current_date=date(2026, 1, 5),
        current_stage="ref_to_mri",
    )

    patient_1.add_event("referral_received", date(2026, 1, 5))
    patient_1.data["pathway_type"] = "BASELINE"

    assert patient_2.events == []
    assert patient_2.data == {}