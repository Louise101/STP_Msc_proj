from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_prep.empirical_inputs import (
    REAL_STAGE_SPECS,
    STAGE_FILE_SPECS,
    build_branching,
    build_pdfs,
    calc_wait,
    clean_waits,
    days_to_next_weekday,
    load_dates,
    load_real_stage_waits,
    parse_date_series,
)


def test_parse_date_series_uk():
    series = pd.Series(["05/01/2026", "12/02/2026"])
    parsed = parse_date_series(series, "uk")

    assert parsed.iloc[0] == pd.Timestamp("2026-01-05")
    assert parsed.iloc[1] == pd.Timestamp("2026-02-12")


def test_parse_date_series_us():
    series = pd.Series(["01/05/26", "02/12/26"])
    parsed = parse_date_series(series, "us")

    assert parsed.iloc[0] == pd.Timestamp("2026-01-05")
    assert parsed.iloc[1] == pd.Timestamp("2026-02-12")


def test_parse_date_series_default():
    series = pd.Series(["2026-01-05", "2026-02-12"])
    parsed = parse_date_series(series, "other")

    assert parsed.iloc[0] == pd.Timestamp("2026-01-05")
    assert parsed.iloc[1] == pd.Timestamp("2026-02-12")


def test_days_to_next_weekday_include_today_true():
    dates = pd.Series(pd.to_datetime(["2026-01-07", "2026-01-08"]))  # Wed, Thu
    result = days_to_next_weekday(dates, target_weekday=2, include_today=True)  # Wednesday

    assert list(result) == [0, 6]


def test_days_to_next_weekday_include_today_false():
    dates = pd.Series(pd.to_datetime(["2026-01-07", "2026-01-08"]))  # Wed, Thu
    result = days_to_next_weekday(dates, target_weekday=2, include_today=False)

    assert list(result) == [7, 6]


def test_clean_waits_removes_nan_and_negatives():
    series = pd.Series([5, np.nan, -1, 0, 7])
    cleaned = clean_waits(series)

    assert list(cleaned) == [5, 0, 7]
    assert cleaned.dtype.kind in {"i", "u"}


def test_clean_waits_respects_min_valid():
    series = pd.Series([0, 1, 2, 3, 4])
    cleaned = clean_waits(series, min_valid=2)

    assert list(cleaned) == [2, 3, 4]


def test_calc_wait_computes_day_difference_and_cleans():
    df = pd.DataFrame(
        {
            "start": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"]),
            "end": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-15"]),
        }
    )

    waits = calc_wait(df, "start", "end")

    assert list(waits) == [2, 5]  # negative wait row removed


def test_load_dates_parses_requested_columns(tmp_path: Path):
    csv_path = tmp_path / "test_dates.csv"
    df = pd.DataFrame(
        {
            "a": ["05/01/2026", "06/01/2026"],
            "b": ["07/01/2026", "08/01/2026"],
            "other": [1, 2],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_dates(csv_path, ["a", "b"])

    assert pd.api.types.is_datetime64_any_dtype(loaded["a"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["b"])
    assert list(loaded["other"]) == [1, 2]


def test_load_real_stage_waits_reads_stage_data_and_drops_negative_waits(tmp_path: Path):
    # Create all required files listed in REAL_STAGE_SPECS
    for scenario, specs in REAL_STAGE_SPECS.items():
        for stage, (filename, start_col, end_col, style) in specs.items():
            if style == "uk":
                df = pd.DataFrame(
                    {
                        start_col: ["05/01/2026", "10/01/2026"],
                        end_col: ["07/01/2026", "09/01/2026"],  # second row negative
                    }
                )
            else:
                df = pd.DataFrame(
                    {
                        start_col: ["01/05/26", "01/10/26"],
                        end_col: ["01/07/26", "01/09/26"],  # second row negative
                    }
                )
            df.to_csv(tmp_path / filename, index=False)

    result = load_real_stage_waits(tmp_path)

    assert not result.empty
    assert set(result.columns) == {"scenario", "stage", "wait_days"}
    assert set(result["scenario"].unique()) == {"pre", "pros"}
    assert (result["wait_days"] >= 0).all()
    assert set(result["stage"].unique()).issubset(
        {
            "ref_to_mri",
            "mri_to_report",
            "report_to_biopmdt",
            "biopmdt_to_biopsy",
            "biopsy_to_pathrep",
            "pathrep_to_treatmdt",
            "treatmdt_to_outpat",
        }
    )


def test_build_pdfs_returns_expected_keys_and_integer_waits(tmp_path: Path):
    # Create all required stage files from STAGE_FILE_SPECS
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        df = pd.DataFrame(
            {
                start_col: ["05/01/2026", "10/01/2026"],
                end_col: ["07/01/2026", "15/01/2026"],
            }
        )
        df.to_csv(tmp_path / filename, index=False)

    pdfs = build_pdfs(tmp_path)

    assert set(pdfs.keys()) == set(STAGE_FILE_SPECS.keys())

    for key, series in pdfs.items():
        assert isinstance(series, pd.Series)
        assert len(series) == 2
        assert (series >= 0).all()
        assert series.dtype.kind in {"i", "u"}


def test_build_branching_builds_probability_dicts(tmp_path: Path):
    biop_df = pd.DataFrame({"Outcome code": [1, 1, 0, 2, 1]})
    path_df = pd.DataFrame({"Outcome code": [1, 0, 1, 1]})

    biop_df.to_csv(tmp_path / "pre_biop_dec.csv", index=False)
    path_df.to_csv(tmp_path / "pre_pathrep_outcome.csv", index=False)

    branching = build_branching(tmp_path)

    assert set(branching.keys()) == {"biopmdt_outcome", "pathrep_outcome"}

    assert set(branching["biopmdt_outcome"].keys()) == {0, 1, 2}
    assert set(branching["pathrep_outcome"].keys()) == {0, 1}

    assert pytest.approx(sum(branching["biopmdt_outcome"].values())) == 1.0
    assert pytest.approx(sum(branching["pathrep_outcome"].values())) == 1.0

    assert all(isinstance(k, int) for k in branching["biopmdt_outcome"].keys())
    assert all(isinstance(v, float) for v in branching["biopmdt_outcome"].values())


def test_build_branching_handles_non_numeric_values_as_missing(tmp_path: Path):
    biop_df = pd.DataFrame({"Outcome code": [1, "bad", 0, 1]})
    path_df = pd.DataFrame({"Outcome code": [1, "bad", 1]})

    biop_df.to_csv(tmp_path / "pre_biop_dec.csv", index=False)
    path_df.to_csv(tmp_path / "pre_pathrep_outcome.csv", index=False)

    branching = build_branching(tmp_path)

    assert set(branching["biopmdt_outcome"].keys()) == {0, 1}
    assert set(branching["pathrep_outcome"].keys()) == {1}
    assert pytest.approx(sum(branching["biopmdt_outcome"].values())) == 1.0
    assert pytest.approx(sum(branching["pathrep_outcome"].values())) == 1.0