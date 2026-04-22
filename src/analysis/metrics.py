from __future__ import annotations

"""Reusable analysis helpers for stage waits, pathway lengths, and flow counts."""

import numpy as np
import pandas as pd
from typing import Any

from core.event_log import EVENT_TO_STAGE
from engine.pathway_definitions import STAGE_EVENT_PAIRS,FULL_PATHWAY_END_EVENT


MILESTONE_EVENTS = [
    "referral_received",
    "mri_performed",
    "mri_report_ready",
    "MDT_occured",
    "biopsy_done",
    "Path_report_recieved",
    "Path_report_outcome",
    "Treatment_options_MDT_occured",
    "Outpatient_appointment_occured",
]

STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Path Report",
    "pathrep_to_treatmdt": "Path Report → Treat MDT",
    "treatmdt_to_outpat": "Treat MDT → Outpatient",
}


def safe_pct_change(new: float, old: float) -> float:
    """Return percent change or NaN if the baseline is missing / zero."""
    if old in (0, 0.0, None) or pd.isna(old):
        return np.nan
    return (new - old) / old * 100

def extract_stage_waits(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    rows = []

    stage_pairs = [
        ("referral_received", "mri_performed", "ref_to_mri"),
        ("mri_performed", "mri_report_ready", "mri_to_report"),
        ("mri_report_ready", "MDT_occured", "report_to_biopmdt"),
        ("MDT_occured", "biopsy_done", "biopmdt_to_biopsy"),
        ("biopsy_done", "Path_report_recieved", "biopsy_to_pathrep"),
        ("Path_report_recieved", "Treatment_options_MDT_occured", "pathrep_to_treatmdt"),
        ("Treatment_options_MDT_occured", "Outpatient_appointment_occured", "treatmdt_to_outpat"),
    ]

    for patient in result["all_patients_objects"]:
        event_dates = {}
        for event in patient.events:
            event_dates[event["event"]] = event["date"]

        for start_event, end_event, stage_name in stage_pairs:
            if start_event in event_dates and end_event in event_dates:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "patient_id": patient.patient_id,
                        "stage": stage_name,
                        "wait_days": (event_dates[end_event] - event_dates[start_event]).days,
                    }
                )

    return pd.DataFrame(rows)





def summarise_flow_counts(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    """Count milestone events in one simulation result."""
    event_log = result.get("event_log")
    rows = []
    for event_name in MILESTONE_EVENTS:
        count = int((event_log["event"] == event_name).sum()) if event_log is not None and not event_log.empty else 0
        rows.append({"scenario": scenario_name, "seed": seed, "event": event_name, "count": count})
    return pd.DataFrame(rows)


def extract_full_pathway_lengths(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    """Extract total pathway lengths for patients reaching outpatient."""
    rows: list[dict] = []
    for patient in result["completed_patients_objects"]:
        if not patient.has_event("Outpatient_appointment_occured"):
            continue
        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "patient_id": patient.patient_id,
                "pathway_type": patient.data.get("pathway_type"),
                "total_days": patient.total_days_in_system(),
            }
        )
    return pd.DataFrame(rows)

def extract_pathway_lengths(result: dict[str, Any], scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    """Keep only patients who completed the full pathway to outpatient."""
    rows: list[dict[str, Any]] = []
    for patient in result.get("completed_patients_objects", []):
        event_names = {event["event"] for event in patient.events}
        if FULL_PATHWAY_END_EVENT not in event_names:
            continue
        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "patient_id": patient.patient_id,
                "pathway_type": patient.data.get("pathway_type"),
                "total_days": (patient.current_date - patient.start_date).days,
            }
        )
    return pd.DataFrame(rows)


def summarise_pathway_lengths(pathway_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise full-pathway completion times by scenario."""
    if pathway_df.empty:
        return pd.DataFrame(columns=["scenario", "n", "mean_days", "median_days", "p90_days", "pct_within_62"])
    return (
        pathway_df.groupby("scenario")["total_days"]
        .agg(
            n="count",
            mean_days="mean",
            median_days="median",
            p90_days=lambda x: np.percentile(x, 90),
            pct_within_62=lambda x: (x <= 62).mean() * 100,
        )
        .reset_index()
    )
