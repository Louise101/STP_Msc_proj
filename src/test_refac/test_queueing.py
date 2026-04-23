from datetime import date

from core.queueing import QueuePatient, QueueServiceEvent, QueueResource


def make_patient(patient_id: int, referral_date: date) -> QueuePatient:
    """Helper to create a queue patient for tests."""
    return QueuePatient(
        patient_id=patient_id,
        referral_date=referral_date,
        payload={"test_key": f"value_{patient_id}"},
    )


def test_queue_patient_initialises_correctly():
    patient = make_patient(1, date(2026, 1, 5))

    assert patient.patient_id == 1
    assert patient.referral_date == date(2026, 1, 5)
    assert patient.payload == {"test_key": "value_1"}


def test_queue_service_event_initialises_correctly():
    patient = make_patient(1, date(2026, 1, 5))
    event = QueueServiceEvent(
        patient=patient,
        wait_days=3,
        start_date=date(2026, 1, 8),
    )

    assert event.patient is patient
    assert event.wait_days == 3
    assert event.start_date == date(2026, 1, 8)


def test_add_patient_increases_queue_length():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 4})
    patient = make_patient(1, date(2026, 1, 5))

    resource.add_patient(patient)

    assert resource.queue_length() == 1
    assert list(resource.queue)[0] is patient


def test_add_patients_preserves_order():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 4})
    p1 = make_patient(1, date(2026, 1, 5))
    p2 = make_patient(2, date(2026, 1, 6))
    p3 = make_patient(3, date(2026, 1, 7))

    resource.add_patients([p1, p2, p3])

    assert resource.queue_length() == 3
    assert [p.patient_id for p in resource.queue] == [1, 2, 3]


def test_get_capacity_for_day_returns_configured_capacity():
    resource = QueueResource(
        name="MRI",
        capacity_by_weekday={
            0: 2,  # Monday
            1: 4,  # Tuesday
        },
    )

    assert resource.get_capacity_for_day(date(2026, 1, 5)) == 2  # Monday
    assert resource.get_capacity_for_day(date(2026, 1, 6)) == 4  # Tuesday


def test_get_capacity_for_day_returns_zero_when_unconfigured():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 4})

    assert resource.get_capacity_for_day(date(2026, 1, 5)) == 0  # Monday not configured
    assert resource.get_capacity_for_day(date(2026, 1, 10)) == 0  # Saturday not configured


def test_process_day_respects_capacity():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})  # Tuesday capacity = 2
    current_date = date(2026, 1, 6)  # Tuesday

    p1 = make_patient(1, date(2026, 1, 1))
    p2 = make_patient(2, date(2026, 1, 2))
    p3 = make_patient(3, date(2026, 1, 3))

    resource.add_patients([p1, p2, p3])
    started = resource.process_day(current_date)

    assert len(started) == 2
    assert resource.queue_length() == 1
    assert [event.patient.patient_id for event in started] == [1, 2]


def test_process_day_is_fifo():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 3})
    current_date = date(2026, 1, 6)  # Tuesday

    p1 = make_patient(1, date(2026, 1, 1))
    p2 = make_patient(2, date(2026, 1, 2))
    p3 = make_patient(3, date(2026, 1, 3))

    resource.add_patients([p1, p2, p3])
    started = resource.process_day(current_date)

    assert [event.patient.patient_id for event in started] == [1, 2, 3]


def test_process_day_calculates_wait_days_correctly():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})
    current_date = date(2026, 1, 6)  # Tuesday

    p1 = make_patient(1, date(2026, 1, 1))  # wait = 5
    p2 = make_patient(2, date(2026, 1, 4))  # wait = 2

    resource.add_patients([p1, p2])
    started = resource.process_day(current_date)

    assert started[0].wait_days == 5
    assert started[1].wait_days == 2


def test_process_day_records_daily_metrics():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})
    current_date = date(2026, 1, 6)  # Tuesday

    p1 = make_patient(1, date(2026, 1, 1))
    p2 = make_patient(2, date(2026, 1, 2))
    p3 = make_patient(3, date(2026, 1, 3))

    resource.add_patients([p1, p2, p3])
    resource.process_day(current_date)

    assert resource.daily_started[current_date] == 2
    assert resource.daily_queue_len[current_date] == 1
    assert resource.daily_waits[current_date] == [5, 4]


def test_process_day_with_zero_capacity_starts_no_patients():
    resource = QueueResource(name="MRI", capacity_by_weekday={})
    current_date = date(2026, 1, 6)

    p1 = make_patient(1, date(2026, 1, 1))
    resource.add_patient(p1)

    started = resource.process_day(current_date)

    assert started == []
    assert resource.queue_length() == 1
    assert resource.daily_started[current_date] == 0
    assert resource.daily_queue_len[current_date] == 1
    assert resource.daily_waits[current_date] == []


def test_process_day_with_empty_queue_records_zero_metrics():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 3})
    current_date = date(2026, 1, 6)  # Tuesday

    started = resource.process_day(current_date)

    assert started == []
    assert resource.daily_started[current_date] == 0
    assert resource.daily_queue_len[current_date] == 0
    assert resource.daily_waits[current_date] == []


def test_queue_state_persists_across_multiple_days():
    resource = QueueResource(
        name="MRI",
        capacity_by_weekday={
            1: 1,  # Tuesday
            2: 1,  # Wednesday
        },
    )

    p1 = make_patient(1, date(2026, 1, 1))
    p2 = make_patient(2, date(2026, 1, 2))

    resource.add_patients([p1, p2])

    day1 = date(2026, 1, 6)  # Tuesday
    started_day1 = resource.process_day(day1)

    assert len(started_day1) == 1
    assert started_day1[0].patient.patient_id == 1
    assert resource.queue_length() == 1

    day2 = date(2026, 1, 7)  # Wednesday
    started_day2 = resource.process_day(day2)

    assert len(started_day2) == 1
    assert started_day2[0].patient.patient_id == 2
    assert resource.queue_length() == 0


def test_process_day_preserves_payload_on_started_patient():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 1})
    current_date = date(2026, 1, 6)

    patient = QueuePatient(
        patient_id=42,
        referral_date=date(2026, 1, 1),
        payload={"patient": "obj", "stage_name": "ref_to_mri"},
    )
    resource.add_patient(patient)

    started = resource.process_day(current_date)

    assert len(started) == 1
    assert started[0].patient.payload["patient"] == "obj"
    assert started[0].patient.payload["stage_name"] == "ref_to_mri"