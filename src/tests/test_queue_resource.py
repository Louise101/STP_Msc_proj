from datetime import date

from queue_resource import QueuePatient, QueueResource, QueueServiceEvent


def test_add_patient_increases_queue_length():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 6})

    patient = QueuePatient(patient_id=1, referral_date=date(2024, 1, 1))
    resource.add_patient(patient)

    assert resource.queue_length() == 1


def test_add_patients_increases_queue_length():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 6})

    patients = [
        QueuePatient(patient_id=1, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=2, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=3, referral_date=date(2024, 1, 1)),
    ]
    resource.add_patients(patients)

    assert resource.queue_length() == 3


def test_get_capacity_for_day_returns_configured_value():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 6})  # Tuesday only

    assert resource.get_capacity_for_day(date(2024, 1, 2)) == 6   # Tuesday
    assert resource.get_capacity_for_day(date(2024, 1, 1)) == 0   # Monday
    assert resource.get_capacity_for_day(date(2024, 1, 6)) == 0   # Saturday


def test_process_day_respects_capacity():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})  # Tuesday=2

    patients = [
        QueuePatient(patient_id=1, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=2, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=3, referral_date=date(2024, 1, 1)),
    ]
    resource.add_patients(patients)

    started = resource.process_day(date(2024, 1, 2))  # Tuesday

    assert len(started) == 2
    assert resource.queue_length() == 1


def test_process_day_returns_service_event_objects():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 1})
    resource.add_patient(QueuePatient(patient_id=1, referral_date=date(2024, 1, 1)))

    started = resource.process_day(date(2024, 1, 2))

    assert len(started) == 1
    assert isinstance(started[0], QueueServiceEvent)


def test_process_day_is_fifo():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})

    p1 = QueuePatient(patient_id=1, referral_date=date(2024, 1, 1))
    p2 = QueuePatient(patient_id=2, referral_date=date(2024, 1, 1))
    p3 = QueuePatient(patient_id=3, referral_date=date(2024, 1, 1))

    resource.add_patients([p1, p2, p3])

    started = resource.process_day(date(2024, 1, 2))

    assert started[0].patient.patient_id == 1
    assert started[1].patient.patient_id == 2
    assert resource.queue[0].patient_id == 3


def test_wait_days_are_calculated_correctly():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 1})

    patient = QueuePatient(patient_id=1, referral_date=date(2024, 1, 1))  # Monday
    resource.add_patient(patient)

    started = resource.process_day(date(2024, 1, 2))  # Tuesday

    assert started[0].wait_days == 1
    assert started[0].start_date == date(2024, 1, 2)


def test_same_day_service_gives_zero_wait():
    resource = QueueResource(name="MRI", capacity_by_weekday={0: 1})  # Monday

    patient = QueuePatient(patient_id=1, referral_date=date(2024, 1, 1))
    resource.add_patient(patient)

    started = resource.process_day(date(2024, 1, 1))

    assert started[0].wait_days == 0


def test_zero_capacity_day_serves_nobody():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 1})  # Tuesday only

    resource.add_patient(QueuePatient(patient_id=1, referral_date=date(2024, 1, 1)))
    started = resource.process_day(date(2024, 1, 1))  # Monday

    assert started == []
    assert resource.queue_length() == 1


def test_empty_queue_serves_nobody():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})

    started = resource.process_day(date(2024, 1, 2))

    assert started == []
    assert resource.queue_length() == 0


def test_daily_stats_are_recorded_correctly():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})

    resource.add_patients([
        QueuePatient(patient_id=1, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=2, referral_date=date(2024, 1, 1)),
        QueuePatient(patient_id=3, referral_date=date(2024, 1, 1)),
    ])

    current_day = date(2024, 1, 2)
    started = resource.process_day(current_day)

    assert len(started) == 2
    assert resource.daily_started[current_day] == 2
    assert resource.daily_queue_len[current_day] == 1
    assert resource.daily_waits[current_day] == [1, 1]


def test_stats_record_zero_when_no_service_occurs():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 2})

    current_day = date(2024, 1, 1)  # Monday, no capacity
    started = resource.process_day(current_day)

    assert started == []
    assert resource.daily_started[current_day] == 0
    assert resource.daily_queue_len[current_day] == 0
    assert resource.daily_waits[current_day] == []


def test_payload_is_preserved_through_service_event():
    resource = QueueResource(name="MRI", capacity_by_weekday={1: 1})

    patient = QueuePatient(
        patient_id=1,
        referral_date=date(2024, 1, 1),
        payload={"priority": "urgent", "source": "test"},
    )
    resource.add_patient(patient)

    started = resource.process_day(date(2024, 1, 2))
    event = started[0]

    assert event.patient.payload["priority"] == "urgent"
    assert event.patient.payload["source"] == "test"