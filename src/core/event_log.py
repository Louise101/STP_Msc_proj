from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
import pandas as pd


EVENT_LOG_COLUMNS = [
    "patient_id",
    "event",
    "date",
    "day",
    "stage_name",
    "outcome",
    "source_engine",
]

EVENT_TO_STAGE = {
    "referral_recieved": "ref_to_mri",   
    "referral_received": "ref_to_mri",   

    "mri_performed": "ref_to_mri",
    "mri_report_ready": "mri_to_report",
    "MDT_occured": "report_to_biopmdt",
    "mdt_decision": "report_to_biopmdt",
    "biopsy_done": "biopmdt_to_biopsy",
    "Path_report_recieved": "biopsy_to_pathrep",
    "Path_report_outcome": "biopsy_to_pathrep",
    "Treatment_options_MDT_occured": "pathrep_to_treatmdt",
    "Outpatient_appointment_occured": "treatmdt_to_outpat",
}



#Convert many date-like inputs to a normalized pandas Timestamp
def to_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value).normalize()
    return pd.to_datetime(value).normalize()



#Convert a flat list of event dictionaries into a standard event log
def event_records_to_dataframe(
    records: list[dict],
    source_engine: str | None = None,
    start_date: date | None = None,
) -> pd.DataFrame:
    
    df = pd.DataFrame(records).copy()
    if df.empty:
        return pd.DataFrame(columns=EVENT_LOG_COLUMNS)

    for column in EVENT_LOG_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df["date"] = df["date"].apply(to_timestamp)

    if source_engine is not None:
        df["source_engine"] = source_engine

    df["stage_name"] = df["stage_name"].where(
        df["stage_name"].notna(),
        df["event"].map(EVENT_TO_STAGE),
    )

    if start_date is not None:
        start_ts = to_timestamp(start_date)
        missing_day = df["day"].isna() & df["date"].notna()
        df.loc[missing_day, "day"] = (df.loc[missing_day, "date"] - start_ts).dt.days

    return (
        df[EVENT_LOG_COLUMNS]
        .sort_values(["patient_id", "date", "event"], na_position="last")
        .reset_index(drop=True)
    )



#Convert engine patient results to the standard long-format event log
def patient_results_to_event_log(
    patient_results: list,
    source_engine: str = "COMBINED_ENGINE",
    start_date: date | None = None,
) -> pd.DataFrame:
    records: list[dict] = []

    for item in patient_results:
        if isinstance(item, dict):
            event_list = item.get("events", [])
            outer_patient_id = item.get("patient_id")
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            event_list = item[0]
            outer_patient_id = None
        else:
            continue

        if not isinstance(event_list, list):
            continue

        for event in event_list:
            if not isinstance(event, dict):
                continue

            event_name = event.get("event")
            records.append(
                {
                    "patient_id": event.get("patient_id", outer_patient_id),
                    "event": event_name,
                    "date": event.get("date"),
                    "day": event.get("day"),
                    "stage_name": event.get("stage_name", EVENT_TO_STAGE.get(event_name)),
                    "outcome": event.get("outcome"),
                    "source_engine": source_engine,
                }
            )

    return event_records_to_dataframe(records, source_engine=source_engine, start_date=start_date)


#Save event log to CSV
def save_event_log(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
