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

# Map event names to the stage they belong to
EVENT_TO_STAGE = {
    "referral_recieved": "ref_to_mri",   # keep your current spelling for compatibility
    "referral_received": "ref_to_mri",   # also allow corrected spelling

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


def _to_timestamp(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    if isinstance(x, datetime):
        return pd.Timestamp(x).normalize()
    if isinstance(x, date):
        return pd.Timestamp(x).normalize()
    return pd.to_datetime(x).normalize()


def event_records_to_dataframe(records, source_engine=None, start_date=None):
    """
    Convert a list of event dicts into a clean long-format event log.
    """
    df = pd.DataFrame(records).copy()

    if df.empty:
        return pd.DataFrame(columns=EVENT_LOG_COLUMNS)

    for col in EVENT_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = df["date"].apply(_to_timestamp)

    if source_engine is not None:
        df["source_engine"] = source_engine

    # Fill stage_name from event if missing
    df["stage_name"] = df["stage_name"].where(
        df["stage_name"].notna(),
        df["event"].map(EVENT_TO_STAGE)
    )

    # Fill day if missing and start_date provided
    if start_date is not None:
        start_ts = _to_timestamp(start_date)
        missing_day = df["day"].isna() & df["date"].notna()
        df.loc[missing_day, "day"] = (
            df.loc[missing_day, "date"] - start_ts
        ).dt.days

    df = df[EVENT_LOG_COLUMNS].sort_values(
        ["patient_id", "date", "event"],
        na_position="last"
    ).reset_index(drop=True)

    return df


def save_event_log(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved event log to: {output_path}")


def patient_results_to_event_log(
    patient_results,
    source_engine="STAGE_ENGINE",
    start_date=None,
):
    """
    Convert patient_results into a standard event log.

    Expected structure:
      each item is like:
          (event_list, total_days)
      where event_list is a list of dicts
    """
    all_records = []

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

        for ev in event_list:
            if not isinstance(ev, dict):
                continue

            event_name = ev.get("event")
            event_date = ev.get("date")
            patient_id = ev.get("patient_id", outer_patient_id)

            record = {
                "patient_id": patient_id,
                "event": event_name,
                "date": event_date,
                "day": ev.get("day"),
                "stage_name": ev.get("stage_name", EVENT_TO_STAGE.get(event_name)),
                "outcome": ev.get("outcome"),
                "source_engine": source_engine,
            }
            all_records.append(record)

    return event_records_to_dataframe(
        all_records,
        source_engine=source_engine,
        start_date=start_date,
    )