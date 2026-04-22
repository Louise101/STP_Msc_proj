from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

BIOPSY_MDT_WEEKDAY = 2   # Wednesday
TREATMENT_MDT_WEEKDAY = 4  # Friday


# One central table describing the empirical input files used by the model.
STAGE_FILE_SPECS: dict[str, tuple[str, str, str]] = {
    "pre_referral_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI"),
    "pre_mri_to_mrireport": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported"),
    "pre_mrirep_to_biopsymdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT"),
    "pre_biopmdt_to_biop": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy"),
    "pre_biop_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report"),
    "pre_pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)"),
    "pre_treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt"),
}


def load_dates(path: Path, date_cols: list[str], *, dayfirst: bool = True) -> pd.DataFrame:
    """Load a CSV and parse one or more date columns."""
    df = pd.read_csv(path).copy()
    for column in date_cols:
        df[column] = pd.to_datetime(df[column], dayfirst=dayfirst, errors="coerce")
    return df


def clean_waits(series: pd.Series, min_valid: int = 0) -> pd.Series:
    """Drop NaNs and any waits below ``min_valid``."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[clean >= min_valid]
    return clean.astype(int)


def calc_wait(df: pd.DataFrame, start_col: str, end_col: str, min_valid: int = 0) -> pd.Series:
    """Calculate a day difference between two date columns and clean it."""
    waits = (df[end_col] - df[start_col]).dt.days
    return clean_waits(waits, min_valid=min_valid)


def lower_quantile_samples(series: pd.Series, quantile: float = 0.1, min_valid: int = 0) -> pd.Series:
    """Return the lower tail of an empirical wait distribution.

    This is currently used to approximate the non-capacity component of the
    referral-to-MRI delay before a patient joins the PROSTAD MRI queue.
    """
    clean = clean_waits(series, min_valid=min_valid)
    if clean.empty:
        return clean
    cutoff = clean.quantile(quantile)
    return clean[clean <= cutoff].astype(int)


def days_to_next_weekday(
    series: pd.Series,
    target_weekday: int,
    *,
    include_today: bool = True,
) -> pd.Series:
    """Return the number of days until the next chosen weekday."""
    weekday = series.dt.weekday
    if include_today:
        return (target_weekday - weekday) % 7
    return ((target_weekday - (weekday + 1)) % 7) + 1


def build_pdfs(data_dir: Path = DATA_DIR, exclude_np053_ref_to_mri: bool = True) -> dict[str, pd.Series]:
    """Build all empirical wait vectors used by the simulation.

    The output dictionary is the one consumed by the engine. Keeping all input
    construction here means every scenario uses the same underlying data prep.
    """
    stage_frames: dict[str, pd.DataFrame] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        stage_frames[pdf_key] = load_dates(data_dir / filename, [start_col, end_col])

    if exclude_np053_ref_to_mri:
        ref_df = stage_frames["pre_referral_to_mri"]
        if "Subject number" in ref_df.columns:
            stage_frames["pre_referral_to_mri"] = ref_df.loc[ref_df["Subject number"] != "NP053"].copy()

    pdfs: dict[str, pd.Series] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        pdfs[pdf_key] = calc_wait(stage_frames[pdf_key], start_col, end_col)

    # Lower-tail approximation for the non-capacity delay before entering the MRI queue.
    pdfs["pre_referral_to_mri_pre_delay"] = lower_quantile_samples(
        pdfs["pre_referral_to_mri"],
        quantile=0.1,
        min_valid=0,
    )

    # Derived residual queue distributions for MDT-linked stages.
    mdt_df = stage_frames["pre_mrirep_to_biopsymdt"].dropna(
        subset=["Date MRI reported", "Date of Prostate MRI MDT"]
    ).copy()
    mdt_df["observed_wait"] = (mdt_df["Date of Prostate MRI MDT"] - mdt_df["Date MRI reported"]).dt.days
    mdt_df["calendar_wait"] = days_to_next_weekday(mdt_df["Date MRI reported"], BIOPSY_MDT_WEEKDAY)
    mdt_df["queue_wait"] = mdt_df["observed_wait"] - mdt_df["calendar_wait"]
    pdfs["queue_mrirep_to_biopsymdt"] = clean_waits(mdt_df["queue_wait"])

    treat_df = stage_frames["pre_pathrep_to_treatmdt"].dropna(
        subset=["Date of pathology report", "Date of MDT (treatment options)"]
    ).copy()
    treat_df["observed_wait"] = (treat_df["Date of MDT (treatment options)"] - treat_df["Date of pathology report"]).dt.days
    treat_df["calendar_wait"] = days_to_next_weekday(treat_df["Date of pathology report"], TREATMENT_MDT_WEEKDAY)
    treat_df["queue_wait"] = treat_df["observed_wait"] - treat_df["calendar_wait"]
    pdfs["queue_pathrep_to_treatmdt"] = clean_waits(treat_df["queue_wait"])

    return pdfs


def build_branching(data_dir: Path = DATA_DIR) -> dict[str, dict[int, float]]:
    """Build branching probabilities from the pre-PROSTAD outcome files."""
    biop_df = pd.read_csv(data_dir / "pre_biop_dec.csv")
    path_df = pd.read_csv(data_dir / "pre_pathrep_outcome.csv")

    biop_df["Outcome code"] = pd.to_numeric(biop_df["Outcome code"], errors="coerce").astype("Int64")
    path_df["Outcome code"] = pd.to_numeric(path_df["Outcome code"], errors="coerce").astype("Int64")

    biop_probs = (
        biop_df["Outcome code"]
        .value_counts(normalize=True)
        .dropna()
        .to_dict()
    )
    path_probs = (
        path_df["Outcome code"]
        .value_counts(normalize=True)
        .dropna()
        .to_dict()
    )

    return {
        "biopmdt_outcome": {int(k): float(v) for k, v in biop_probs.items()},
        "pathrep_outcome": {int(k): float(v) for k, v in path_probs.items()},
    }
