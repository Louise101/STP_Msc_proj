from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List


@dataclass
class PatientState:
    patient_id: int
    start_date: date
    current_date: date
    current_stage: str

    pathway_type: str | None = None

    events: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    is_complete: bool = False
    exit_reason: str | None = None

    def add_event(self, event: str, event_date: date, **kwargs) -> None:
        row = {
            "patient_id": self.patient_id,
            "event": event,
            "date": event_date,
        }
        row.update(kwargs)
        self.events.append(row)