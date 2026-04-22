from __future__ import annotations

import numpy as np
import pandas as pd

from engine.pathway_definitions import FULL_PATHWAY_END_EVENT, STAGE_EVENT_PAIRS


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


def safe_pct_change(new: float, old: float) -> float:
    """Return percentage change or NaN when the denominator is missing/zero."""
    if old in (0, 0.0, None) or pd.isna(old):
        return np.nan
    return (new - old) / old * 100.0


def flatten_wait_values(daily_waits: dict) -> list[float]:
    """Flatten the nested daily wait storage produced by queue resources."""
    flat: list[float] = []
    for value in (daily_waits or {}).values():
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            flat.extend(x for x in value if pd.notna(x))
        elif pd.notna(value):
            flat.append(value)
    return flat


def count_event_occurrences(result: dict, event_name: str) -> int:
    """Count how many times an event appears in the event log."""
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return 0
    return int((event_log["event"] == event_name).sum())


def extract_stage_waits(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    """Rebuild stage waits from consecutive event dates in patient results."""
    rows: list[dict] = []
    for events, _ in result["patient_results"]:
        if not events:
            continue
        events_sorted = sorted(events, key=lambda row: row["date"])
        for first, second in zip(events_sorted, events_sorted[1:]):
            stage_name = STAGE_EVENT_PAIRS.get((first["event"], second["event"]))
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
    """Summarise stage waits by scenario and stage."""
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
    """Summarise daily stage activity into one row per stage."""
    rows: list[dict] = []
    for stage_name, metrics in result.get("stage_activity", {}).items():
        arrivals = metrics.get("daily_arrivals", {}) or {}
        in_stage = metrics.get("daily_in_stage", {}) or {}
        completed = metrics.get("daily_completed", {}) or {}

        arrival_values = list(arrivals.values())
        in_stage_values = list(in_stage.values())
        completed_values = list(completed.values())
        total_arrivals = sum(arrival_values)
        total_completed = sum(completed_values)

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "stage": stage_name,
                "total_arrivals": total_arrivals,
                "mean_daily_arrivals": float(np.mean(arrival_values)) if arrival_values else 0.0,
                "peak_daily_arrivals": max(arrival_values) if arrival_values else 0,
                "mean_in_stage": float(np.mean(in_stage_values)) if in_stage_values else 0.0,
                "peak_in_stage": max(in_stage_values) if in_stage_values else 0,
                "total_completed": total_completed,
                "completion_ratio": total_completed / total_arrivals if total_arrivals else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise_stage_weekly_arrivals(result: dict, scenario_name: str, seed: int) -> pd.DataFrame: 
    """Convert daily stage arrivals into week-indexed counts."""
    rows: list[pd.DataFrame] = []
    for stage_name, metrics in result.get("stage_activity", {}).items():
        arrivals = metrics.get("daily_arrivals", {}) or {}
        if not arrivals:
            continue

        df = pd.DataFrame({
            "date": pd.to_datetime(list(arrivals.keys())),
            "n_arrivals": list(arrivals.values()),
        })
        df["week_start"] = df["date"].dt.to_period("W").apply(lambda period: period.start_time)
        weekly = (
            df.groupby("week_start", as_index=False)["n_arrivals"]
            .sum()
            .rename(columns={"n_arrivals": "weekly_arrivals"})
        )
        weekly["scenario"] = scenario_name
        weekly["seed"] = seed
        weekly["stage"] = stage_name
        rows.append(weekly)

    if not rows:
        return pd.DataFrame(columns=["week_start", "weekly_arrivals", "scenario", "seed", "stage"])
    return pd.concat(rows, ignore_index=True)


def summarise_flow_counts(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    """Count milestone events for one simulation run."""
    rows = [
        {
            "scenario": scenario_name,
            "seed": seed,
            "event": event_name,
            "count": count_event_occurrences(result, event_name),
        }
        for event_name in MILESTONE_EVENTS
    ]
    return pd.DataFrame(rows)


def summarise_resource_pressure(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
    """Summarise queue and wait behaviour for each resource."""
    rows: list[dict] = []
    for resource_name, metrics in result.get("resources", {}).items():
        queue_values = list((metrics.get("daily_queue_len", {}) or {}).values())
        wait_values = flatten_wait_values(metrics.get("daily_waits", {}) or {})
        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "resource": resource_name,
                "mean_queue_len": float(np.mean(queue_values)) if queue_values else 0.0,
                "peak_queue_len": max(queue_values) if queue_values else 0,
                "mean_wait": float(np.mean(wait_values)) if wait_values else 0.0,
                "peak_wait": max(wait_values) if wait_values else 0,
                "n_wait_observations": len(wait_values),
            }
        )
    return pd.DataFrame(rows)


def extract_full_pathway_lengths(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    """Extract pathway lengths for patients who reached full outpatient completion."""
    rows: list[dict] = []
    #test - ccan delete
    for patient in result["completed_patients_objects"][:20]:
        if patient.has_event(FULL_PATHWAY_END_EVENT):
            print(
                patient.patient_id,
                patient.start_date,
                patient.current_date,
                (patient.current_date - patient.start_date).days,
                [e["event"] for e in patient.events],
            )
# end test
    for patient in result.get("completed_patients_objects", []):
        if not patient.has_event(FULL_PATHWAY_END_EVENT):
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
    """Summarise full-pathway durations by scenario."""
    if pathway_df.empty:
        return pd.DataFrame(columns=["scenario", "n", "mean_days", "median_days", "p90_days", "pct_within_62"])
    return (
        pathway_df.groupby("scenario")["total_days"]
        .agg(
            n=("count"),
            mean_days=("mean"),
            median_days=("median"),
            p90_days=(lambda x: np.percentile(x, 90)),
            pct_within_62=(lambda x: (x <= 62).mean() * 100),
        )
        .reset_index()
    )
