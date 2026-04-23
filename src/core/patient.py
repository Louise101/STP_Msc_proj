from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

"""Object that represents one patient moving through the simulated pathway.
 events = a chronological audit trail used for validation and analysis.
 data = small pieces of derived state that make to help with routing and debugging 
    """


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



    """Append one event record to the patient event log."""
    def add_event(self, event: str, event_date: date, **kwargs) -> None:
        row = {
            "patient_id": self.patient_id,
            "event": event,
            "date": event_date,
        }
        row.update(kwargs)
        self.events.append(row)

    """Return True if this patient has already recorded the given event."""
    def has_event(self, event_name: str) -> bool:
        return any(event.get("event") == event_name for event in self.events)
    
    
    """Total elapsed days from pathway start to the patient's current date."""
    #@property
    def total_days_in_system(self) -> int:
        return (self.current_date - self.start_date).days