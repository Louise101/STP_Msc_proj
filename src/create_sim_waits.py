from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

#EVENTS_FILE = OUTPUT_DIR / "baseline_events.csv"
#SIM_WAITS_OUT = OUTPUT_DIR / "base_sim_waits.csv"

EVENTS_FILE = OUTPUT_DIR / "prostad_events.csv"
SIM_WAITS_OUT = OUTPUT_DIR / "pros_sim_waits.csv"


# --------------------------------------------------
# Map each wait to its start and end event
# EDIT THESE IF YOUR EVENT NAMES DIFFER
# --------------------------------------------------
WAIT_EVENT_MAP = {
    "wait_ref_to_mri": ("referral_received", "mri_performed"),
    "wait_mri_to_report": ("mri_performed", "mri_report_ready"),
    "wait_report_to_biopmdt": ("mri_report_ready", "MDT_occured"),
    "wait_biopmdt_to_biopsy": ("MDT_occured", "biopsy_done"),
    "wait_biopsy_to_pathrep": ("biopsy_done", "Path_report_recieved"),
    "wait_pathrep_to_treatmdt": ("Path_report_recieved", "Treatment_options_MDT_occured"),
    "wait_treatmdt_to_outpat": ("Treatment_options_MDT_occured", "Outpatient_appointment_occured"),
}


def load_events(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp).copy()

    required = {"patient_id", "event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prefer real dates if present, otherwise fall back to simulation day
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if "day" in df.columns:
        df["day"] = pd.to_numeric(df["day"], errors="coerce")
    else:
        df["day"] = pd.NA

    df["patient_id"] = df["patient_id"].astype(str)
    return df


def get_first_event_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sort_cols = ["patient_id", "event"]
    if "day" in df.columns:
        sort_cols.append("day")
    if "date" in df.columns:
        sort_cols.append("date")

    df = df.sort_values(sort_cols)

    # Keep first occurrence of each event per patient
    firsts = df.drop_duplicates(subset=["patient_id", "event"], keep="first")
    return firsts


def build_waits_from_events(df: pd.DataFrame) -> pd.DataFrame:
    firsts = get_first_event_per_patient(df)

    # Wide tables indexed by patient_id
    wide_date = firsts.pivot(index="patient_id", columns="event", values="date")
    wide_day = firsts.pivot(index="patient_id", columns="event", values="day")

    # Keep patient_id as index for correct alignment
    patient_index = wide_date.index.union(wide_day.index)
    out = pd.DataFrame(index=patient_index)

    # Optional: keep actual event dates in output too
    for event_col in wide_date.columns:
        out[f"date_{event_col}"] = wide_date[event_col]

    for event_col in wide_day.columns:
        out[f"day_{event_col}"] = wide_day[event_col]

    # Create waits
    for wait_col, (start_event, end_event) in WAIT_EVENT_MAP.items():
        # Date-based wait
        if start_event in wide_date.columns and end_event in wide_date.columns:
            out[f"{wait_col}_date"] = (
                wide_date[end_event] - wide_date[start_event]
            ).dt.days
        else:
            out[f"{wait_col}_date"] = pd.NA

        # Day-based wait
        if start_event in wide_day.columns and end_event in wide_day.columns:
            out[f"{wait_col}_day"] = wide_day[end_event] - wide_day[start_event]
        else:
            out[f"{wait_col}_day"] = pd.NA

        # Final combined wait
        out[wait_col] = out[f"{wait_col}_date"]
        missing_mask = out[wait_col].isna()
        out.loc[missing_mask, wait_col] = out.loc[missing_mask, f"{wait_col}_day"]

    # Pathway indicators
    out["had_biopsy"] = out["wait_biopmdt_to_biopsy"].notna().astype(int)
    out["had_pathrep"] = out["wait_biopsy_to_pathrep"].notna().astype(int)
    out["had_treat_mdt"] = out["wait_pathrep_to_treatmdt"].notna().astype(int)
    out["had_outpatient"] = out["wait_treatmdt_to_outpat"].notna().astype(int)

    # Totals
    out["total_time_to_biopsy"] = out[
        [
            "wait_ref_to_mri",
            "wait_mri_to_report",
            "wait_report_to_biopmdt",
            "wait_biopmdt_to_biopsy",
        ]
    ].sum(axis=1, min_count=1)

    out["total_time_to_pathrep"] = out[
        [
            "wait_ref_to_mri",
            "wait_mri_to_report",
            "wait_report_to_biopmdt",
            "wait_biopmdt_to_biopsy",
            "wait_biopsy_to_pathrep",
        ]
    ].sum(axis=1, min_count=1)

    out["total_time_to_treatmdt"] = out[
        [
            "wait_ref_to_mri",
            "wait_mri_to_report",
            "wait_report_to_biopmdt",
            "wait_biopmdt_to_biopsy",
            "wait_biopsy_to_pathrep",
            "wait_pathrep_to_treatmdt",
        ]
    ].sum(axis=1, min_count=1)

    out["total_time_to_outpatient"] = out[
        [
            "wait_ref_to_mri",
            "wait_mri_to_report",
            "wait_report_to_biopmdt",
            "wait_biopmdt_to_biopsy",
            "wait_biopsy_to_pathrep",
            "wait_pathrep_to_treatmdt",
            "wait_treatmdt_to_outpat",
        ]
    ].sum(axis=1, min_count=1)

    return out.reset_index()


def main():
    events_df = load_events(EVENTS_FILE)
    waits_df = build_waits_from_events(events_df)

    waits_df.to_csv(SIM_WAITS_OUT, index=False)
    print(f"Saved to {SIM_WAITS_OUT}")
    print(waits_df.head())


if __name__ == "__main__":
    main()