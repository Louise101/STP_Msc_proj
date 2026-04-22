from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class PatientState:
    """Represents one patient moving through the simulated pathway.

    The engine keeps two kinds of patient data:
    1. ``events``: a chronological audit trail used for validation and analysis.
    2. ``data``: small pieces of derived state that make routing or debugging easier.

    ``data`` is intentionally flexible because different scenarios may attach
    different intermediate values, for example pre-queue delay, pathway type,
    or stage-specific waits.
    """

    patient_id: int
    start_date: date
    current_date: date
    current_stage: str

    pathway_type: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    is_complete: bool = False
    exit_reason: str | None = None

    def add_event(self, event: str, event_date: date, **kwargs: Any) -> None:
        """Append one event record to the patient event log."""
        row = {
            "patient_id": self.patient_id,
            "event": event,
            "date": event_date,
        }
        row.update(kwargs)
        self.events.append(row)

    def has_event(self, event_name: str) -> bool:
        """Return True if the patient has already recorded the given event."""
        return any(event.get("event") == event_name for event in self.events)
