from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

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



#Flatten the nested daily wait storage produced by queue resources
def flatten_wait_values(daily_waits: dict) -> list[float]:
    flat: list[float] = []
    for value in (daily_waits or {}).values():
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            flat.extend(x for x in value if pd.notna(x))
        elif pd.notna(value):
            flat.append(value)
    return flat

#Count how many times an event appears in the event log.
def count_event_occurrences(result: dict, event_name: str) -> int:
    event_log = result.get("event_log")
    if event_log is None or event_log.empty:
        return 0
    return int((event_log["event"] == event_name).sum())

#Rebuild stage waits from consecutive event dates in patient results.
def extract_stage_waits(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
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

#Summarise stage waits by scenario and stage.
def summarise_stage_waits(stage_wait_df: pd.DataFrame) -> pd.DataFrame:
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

#Summarise daily stage activity into one row per stage.
def summarise_stage_activity(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
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

#Convert daily stage arrivals into week-indexed counts.
def summarise_stage_weekly_arrivals(result: dict, scenario_name: str, seed: int) -> pd.DataFrame: 
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

#Count milestone events for one simulation run.
def summarise_flow_counts(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
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

def summarise_flow_counts_across_seeds(flow_df: pd.DataFrame) -> pd.DataFrame:
    return (
        flow_df.groupby(["scenario", "event"])
        .agg(
            mean_count=("count", "mean"),
            std_count=("count", "std"),
        )
        .reset_index()
    )

#Summarise queue pressure for the MRI_PROSTAD resource only.
def summarise_mri_resource(result: dict[str, Any], scenario_name: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for resource_name, metrics in result.get("resources", {}).items():
        if resource_name != "MRI_PROSTAD":
            continue

        queue_vals = list((metrics.get("daily_queue_len", {}) or {}).values())
        wait_vals = flatten_wait_values(metrics.get("daily_waits", {}) or {})

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "resource": resource_name,
                "mean_queue_len": float(np.mean(queue_vals)) if queue_vals else 0.0,
                "peak_queue_len": max(queue_vals) if queue_vals else 0,
                "mean_wait": float(np.mean(wait_vals)) if wait_vals else 0.0,
                "peak_wait": max(wait_vals) if wait_vals else 0,
                "n_wait_observations": len(wait_vals),
            }
        )

    return pd.DataFrame(rows)

#Summarise queue and wait behaviour for each resource.
def summarise_resource_pressure(result: dict, scenario_name: str, seed: int) -> pd.DataFrame:
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

#Extract pathway lengths for patients who reached full outpatient completion.
def extract_full_pathway_lengths(result: dict, scenario_name: str, seed: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for patient in result["completed_patients_objects"][:20]:
        if patient.has_event(FULL_PATHWAY_END_EVENT):
            print(
                patient.patient_id,
                patient.start_date,
                patient.current_date,
                (patient.current_date - patient.start_date).days,
                [e["event"] for e in patient.events],
            )

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

#Summarise full-pathway durations by scenario.
def summarise_pathway_lengths(pathway_df: pd.DataFrame) -> pd.DataFrame:
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

#Summarise full-pathway durations by scenario.
def summarise_pathway_stats(pathway_df: pd.DataFrame) -> pd.DataFrame:
    if pathway_df.empty:
        return pd.DataFrame(columns=["scenario", "n", "mean_days", "median_days", "p90_days", "pct_within_62"])
    return (
        pathway_df.groupby("scenario")["total_days"]
        .agg(
            n=("count"),
            mean_days=("mean"),
            median_days=("median"),
            std_days=("std"),
            min_days=("min"),
            max_days=("max"),
            p25=(lambda x: np.percentile(x, 25)),
            p75=(lambda x: np.percentile(x, 75)),
            p90_days=(lambda x: np.percentile(x, 90)),
            pct_within_62=(lambda x: (x <= 62).mean() * 100),
        )
        .reset_index()
    )

#Within OBS_MIX, compare full-pathway durations by patient pathway type.
def summarise_mixed_pathway_type(pathway_df: pd.DataFrame) -> pd.DataFrame:
    if pathway_df.empty:
        return pd.DataFrame(columns=["pathway_type", "n", "mean_days", "median_days", "p90", "pct_within_62"])
    return (
        pathway_df.groupby("pathway_type")["total_days"]
        .agg(
            n=("count"),
            mean_days=("mean"),
            median_days=("median"),
            p75=(lambda x: np.percentile(x, 75)),
            p90=(lambda x: np.percentile(x, 90)),
            pct_within_62=(lambda x: (x <= 62).mean() * 100),
        )
        .reset_index()
    )

