from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Deque, Dict, Iterable, List



#queue wrapper used by DES resources.
    #referral_date =  queue entry date for that specific resource,

@dataclass
class QueuePatient:
    patient_id: int
    referral_date: date
    payload: Dict[str, Any] = field(default_factory=dict)


#patient started by a DES resource on a specific day
@dataclass
class QueueServiceEvent:
    patient: QueuePatient
    wait_days: int
    start_date: date




#FIFO queue with weekday-specific capacity

@dataclass
class QueueResource:
    name: str
    capacity_by_weekday: Dict[int, int]
    queue: Deque[QueuePatient] = field(default_factory=deque)

    daily_started: Dict[date, int] = field(default_factory=dict)
    daily_queue_len: Dict[date, int] = field(default_factory=dict)
    daily_waits: Dict[date, List[int]] = field(default_factory=dict)

    #Add one patient to the back of the FIFO queue
    def add_patient(self, patient: QueuePatient) -> None:
        self.queue.append(patient)

    #Add multiple patients to the back of the FIFO queue
    def add_patients(self, patients: Iterable[QueuePatient]) -> None:
        self.queue.extend(patients)

    #Return current queue length
    def queue_length(self) -> int:
        return len(self.queue)

    
    #Return available capacity for a given day.
            #Monday=0, Tuesday=1, ..., Sunday=6
  
    def get_capacity_for_day(self, current_date: date) -> int:
        return int(self.capacity_by_weekday.get(current_date.weekday(), 0))

    

        #Process patients up to available capacity for the day.

    def process_day(self, current_date: date) -> List[QueueServiceEvent]:
    
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