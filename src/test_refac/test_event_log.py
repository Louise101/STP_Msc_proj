from datetime import date, datetime
from pathlib import Path

import pandas as pd

from core.event_log import (
    EVENT_LOG_COLUMNS,
    EVENT_TO_STAGE,
    event_records_to_dataframe,
    patient_results_to_event_log,
    save_event_log,
    to_timestamp,
)


def test_to_timestamp_with_date():
    result = to_timestamp(date(2026, 1, 5))
    assert result == pd.Timestamp("2026-01-05")


def test_to_timestamp_with_datetime_normalises_time():
    result = to_timestamp(datetime(2026, 1, 5, 14, 30, 45))
    assert result == pd.Timestamp("2026-01-05")


def test_to_timestamp_with_timestamp_normalises_time():
    result = to_timestamp(pd.Timestamp("2026-01-05 23:59:00"))
    assert result == pd.Timestamp("2026-01-05")


def test_to_timestamp_with_string():
    result = to_timestamp("2026-01-05")
    assert result == pd.Timestamp("2026-01-05")


def test_to_timestamp_with_missing_value_returns_nat():
    result = to_timestamp(pd.NA)
    assert pd.isna(result)


def test_event_records_to_dataframe_returns_empty_dataframe_with_expected_columns():
    df = event_records_to_dataframe([])

    assert list(df.columns) == EVENT_LOG_COLUMNS
    assert df.empty


def test_event_records_to_dataframe_fills_missing_columns_and_maps_stage_name():
    records = [
        {
            "patient_id": 1,
            "event": "mri_performed",
            "date": date(2026, 1, 10),
        }
    ]

    df = event_records_to_dataframe(records, source_engine="TEST_ENGINE")

    assert len(df) == 1
    assert df.loc[0, "patient_id"] == 1
    assert df.loc[0, "event"] == "mri_performed"
    assert df.loc[0, "date"] == pd.Timestamp("2026-01-10")
    assert df.loc[0, "stage_name"] == "ref_to_mri"
    assert df.loc[0, "source_engine"] == "TEST_ENGINE"
    assert pd.isna(df.loc[0, "day"])
    assert pd.isna(df.loc[0, "outcome"])


def test_event_records_to_dataframe_preserves_existing_stage_name():
    records = [
        {
            "patient_id": 1,
            "event": "mri_performed",
            "date": date(2026, 1, 10),
            "stage_name": "custom_stage",
        }
    ]

    df = event_records_to_dataframe(records)

    assert df.loc[0, "stage_name"] == "custom_stage"


def test_event_records_to_dataframe_calculates_day_from_start_date():
    records = [
        {
            "patient_id": 1,
            "event": "referral_received",
            "date": date(2026, 1, 5),
        },
        {
            "patient_id": 1,
            "event": "mri_performed",
            "date": date(2026, 1, 12),
        },
    ]

    df = event_records_to_dataframe(records, start_date=date(2026, 1, 5))

    assert df.loc[0, "day"] == 0
    assert df.loc[1, "day"] == 7


def test_event_records_to_dataframe_does_not_overwrite_existing_day():
    records = [
        {
            "patient_id": 1,
            "event": "mri_performed",
            "date": date(2026, 1, 10),
            "day": 99,
        }
    ]

    df = event_records_to_dataframe(records, start_date=date(2026, 1, 5))

    assert df.loc[0, "day"] == 99


def test_event_records_to_dataframe_sorts_by_patient_id_date_and_event():
    records = [
        {"patient_id": 2, "event": "mri_performed", "date": date(2026, 1, 10)},
        {"patient_id": 1, "event": "mri_report_ready", "date": date(2026, 1, 11)},
        {"patient_id": 1, "event": "mri_performed", "date": date(2026, 1, 10)},
    ]

    df = event_records_to_dataframe(records)

    assert list(df["patient_id"]) == [1, 1, 2]
    assert list(df["event"]) == ["mri_performed", "mri_report_ready", "mri_performed"]


def test_event_records_to_dataframe_maps_legacy_referral_misspelling():
    records = [
        {
            "patient_id": 1,
            "event": "referral_recieved",
            "date": date(2026, 1, 5),
        }
    ]

    df = event_records_to_dataframe(records)

    assert df.loc[0, "stage_name"] == "ref_to_mri"


def test_patient_results_to_event_log_handles_tuple_structure():
    patient_results = [
        (
            [
                {"patient_id": 1, "event": "referral_received", "date": date(2026, 1, 5)},
                {"patient_id": 1, "event": "mri_performed", "date": date(2026, 1, 10)},
            ],
            5,
        )
    ]

    df = patient_results_to_event_log(
        patient_results,
        source_engine="TEST_ENGINE",
        start_date=date(2026, 1, 5),
    )

    assert len(df) == 2
    assert list(df["event"]) == ["referral_received", "mri_performed"]
    assert list(df["day"]) == [0, 5]
    assert list(df["stage_name"]) == ["ref_to_mri", "ref_to_mri"]
    assert all(df["source_engine"] == "TEST_ENGINE")


def test_patient_results_to_event_log_handles_dict_structure_and_outer_patient_id():
    patient_results = [
        {
            "patient_id": 7,
            "events": [
                {"event": "referral_received", "date": date(2026, 1, 5)},
                {"event": "mri_performed", "date": date(2026, 1, 8)},
            ],
        }
    ]

    df = patient_results_to_event_log(patient_results, start_date=date(2026, 1, 5))

    assert len(df) == 2
    assert list(df["patient_id"]) == [7, 7]
    assert list(df["day"]) == [0, 3]


def test_patient_results_to_event_log_ignores_malformed_items():
    patient_results = [
        "not_valid",
        123,
        {"patient_id": 1, "events": "not_a_list"},
        ([{"event": "referral_received", "date": date(2026, 1, 5)}], 0),
    ]

    df = patient_results_to_event_log(patient_results)

    assert len(df) == 1
    assert df.loc[0, "event"] == "referral_received"


def test_patient_results_to_event_log_ignores_non_dict_events():
    patient_results = [
        (
            [
                {"patient_id": 1, "event": "referral_received", "date": date(2026, 1, 5)},
                "bad_event",
            ],
            0,
        )
    ]

    df = patient_results_to_event_log(patient_results)

    assert len(df) == 1
    assert df.loc[0, "event"] == "referral_received"


def test_save_event_log_creates_parent_folder_and_writes_csv(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "event": "referral_received",
                "date": pd.Timestamp("2026-01-05"),
                "day": 0,
                "stage_name": "ref_to_mri",
                "outcome": pd.NA,
                "source_engine": "TEST_ENGINE",
            }
        ]
    )

    output_path = tmp_path / "nested" / "folder" / "event_log.csv"
    save_event_log(df, output_path)

    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    assert len(loaded) == 1
    assert loaded.loc[0, "event"] == "referral_received"
    assert loaded.loc[0, "stage_name"] == "ref_to_mri"