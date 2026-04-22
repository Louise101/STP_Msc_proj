from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Deque, Iterable


@dataclass
class QueuePatient:
    """A lightweight queue wrapper used by DES resources.

    ``referral_date`` here means *queue entry date* for that specific resource,
    not necessarily the original pathway referral date.
    """

    patient_id: int
    referral_date: date
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueServiceEvent:
    """One patient being started by a DES resource on a specific day."""

    patient: QueuePatient
    wait_days: int
    start_date: date


@dataclass
class QueueResource:
    """Simple FIFO queue with weekday-specific capacity.

    This is intentionally generic so the same class can represent MRI capacity,
    biopsy capacity, outpatient capacity, and any future scheduled service.
    """

    name: str
    capacity_by_weekday: dict[int, int]
    queue: Deque[QueuePatient] = field(default_factory=deque)

    # Daily audit outputs captured for later analysis.
    daily_started: dict[date, int] = field(default_factory=dict)
    daily_queue_len: dict[date, int] = field(default_factory=dict)
    daily_waits: dict[date, list[int]] = field(default_factory=dict)

    def add_patient(self, patient: QueuePatient) -> None:
        """Add one patient to the back of the queue."""
        self.queue.append(patient)

    def add_patients(self, patients: Iterable[QueuePatient]) -> None:
        """Add many patients to the back of the queue in FIFO order."""
        self.queue.extend(patients)

    def queue_length(self) -> int:
        """Return the current queue length."""
        return len(self.queue)

    def get_capacity_for_day(self, current_date: date) -> int:
        """Return the configured capacity for the given weekday."""
        return int(self.capacity_by_weekday.get(current_date.weekday(), 0))

    def process_day(self, current_date: date) -> list[QueueServiceEvent]:
        """Serve as many queued patients as the day's capacity allows.

        Patients are always processed in FIFO order.
        The function also records queue metrics used later for plotting and
        bottleneck analysis.
        """
        capacity = self.get_capacity_for_day(current_date)
        started_today: list[QueueServiceEvent] = []
        waits_today: list[int] = []

        while capacity > 0 and self.queue:
            patient = self.queue.popleft()
            wait_days = (current_date - patient.referral_date).days

            started_today.append(
                QueueServiceEvent(
                    patient=patient,
                    wait_days=wait_days,
                    start_date=current_date,
                )
            )
            waits_today.append(wait_days)
            capacity -= 1

        self.daily_started[current_date] = len(started_today)
        self.daily_queue_len[current_date] = len(self.queue)
        self.daily_waits[current_date] = waits_today
        return started_today
