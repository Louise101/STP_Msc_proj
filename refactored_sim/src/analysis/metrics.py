from __future__ import annotations

"""Reusable analysis helpers for stage waits, pathway lengths, and flow counts."""

import numpy as np
import pandas as pd

from core.event_log import EVENT_TO_STAGE
from engine.stage_logic import STAGE_EVENT_MAP


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


def extract_stage_waits(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    """Extract observed stage waits from a simulation result.

    Stage waits are reconstructed from sequential events in each patient's event
    log rather than trusting stage-specific helper columns.
    """
    rows: list[dict] = []
    for events, _ in result["patient_results"]:
        if not events:
            continue
        sorted_events = sorted(events, key=lambda item: item["date"])
        for first, second in zip(sorted_events[:-1], sorted_events[1:]):
            stage_name = STAGE_EVENT_MAP.get((first["event"], second["event"]))
            if stage_name is None:
                continue
            rows.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    "patient_id": first.get("patient_id"),
                    "stage": stage_name,
                    "wait_days": (second["date"] - first["date"]).days,
                }
            )
    return pd.DataFrame(rows)


def summarise_stage_waits(stage_wait_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise a long-format stage wait table."""
    if stage_wait_df.empty:
        return pd.DataFrame(columns=["scenario", "stage", "n", "mean_wait", "median_wait", "p90_wait"])
    return (
        stage_wait_df.groupby(["scenario", "stage"])
        .agg(
            n=("wait_days", "count"),
            mean_wait=("wait_days", "mean"),
            median_wait=("wait_days", "median"),
            p90_wait=("wait_days", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )


def summarise_stage_activity(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    """Summarise stage arrivals, occupancy, and completions for one run."""
    rows: list[dict] = []
    for stage_name, metrics in result.get("stage_activity", {}).items():
        arrivals = metrics.get("daily_arrivals", {}) or {}
        in_stage = metrics.get("daily_in_stage", {}) or {}
        completed = metrics.get("daily_completed", {}) or {}

        arrivals_vals = list(arrivals.values())
        in_stage_vals = list(in_stage.values())
        completed_vals = list(completed.values())
        total_arrivals = int(sum(arrivals_vals))
        total_completed = int(sum(completed_vals))

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "stage": stage_name,
                "total_arrivals": total_arrivals,
                "mean_daily_arrivals": float(np.mean(arrivals_vals)) if arrivals_vals else 0.0,
                "peak_daily_arrivals": max(arrivals_vals) if arrivals_vals else 0,
                "mean_in_stage": float(np.mean(in_stage_vals)) if in_stage_vals else 0.0,
                "peak_in_stage": max(in_stage_vals) if in_stage_vals else 0,
                "total_completed": total_completed,
                "completion_ratio": (total_completed / total_arrivals) if total_arrivals > 0 else np.nan,
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
