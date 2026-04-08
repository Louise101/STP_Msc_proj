from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Deque, Dict, Iterable, List


@dataclass
class QueuePatient:
    patient_id: int
    referral_date: date
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueServiceEvent:
    patient: QueuePatient
    wait_days: int
    start_date: date

#define MC mode queue
@dataclass
class DelayQueueItem:
    patient: Any
    entry_date: date
    ready_date: date
    sampled_wait: int
    stage_name: str


@dataclass
class QueueResource:
    name: str
    capacity_by_weekday: Dict[int, int]
    queue: Deque[QueuePatient] = field(default_factory=deque)

    daily_started: Dict[date, int] = field(default_factory=dict)
    daily_queue_len: Dict[date, int] = field(default_factory=dict)
    daily_waits: Dict[date, List[int]] = field(default_factory=dict)

    def add_patient(self, patient: QueuePatient) -> None:
        """Add one patient to the back of the FIFO queue."""
        self.queue.append(patient)

    def add_patients(self, patients: Iterable[QueuePatient]) -> None:
        """Add multiple patients to the back of the FIFO queue."""
        self.queue.extend(patients)

    def queue_length(self) -> int:
        """Return current queue length."""
        return len(self.queue)

    def get_capacity_for_day(self, current_date: date) -> int:
        """
        Return available capacity for a given day.

        Python weekday convention:
            Monday=0, Tuesday=1, ..., Sunday=6
        """
        return int(self.capacity_by_weekday.get(current_date.weekday(), 0))

    def process_day(self, current_date: date) -> List[QueueServiceEvent]:
        """
        Process patients up to available capacity for the day.

        Patients are served in FIFO order.

        Returns
        -------
        List[QueueServiceEvent]
            One event per patient started today.
        """
        capacity = self.get_capacity_for_day(current_date)
        started_today: List[QueueServiceEvent] = []
        waits_today: List[int] = []

        while capacity > 0 and self.queue:
            patient = self.queue.popleft()
            wait_days = (current_date - patient.referral_date).days

            event = QueueServiceEvent(
                patient=patient,
                wait_days=wait_days,
                start_date=current_date,
            )
            started_today.append(event)
            waits_today.append(wait_days)

            capacity -= 1

        self.daily_started[current_date] = len(started_today)
        self.daily_queue_len[current_date] = len(self.queue)
        self.daily_waits[current_date] = waits_today

        return started_today