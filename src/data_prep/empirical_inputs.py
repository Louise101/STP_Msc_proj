from __future__ import annotations
from pathlib import Path
import pandas as pd



BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

BIOPSY_MDT_WEEKDAY = 2   # Wednesday
TREATMENT_MDT_WEEKDAY = 4  # Friday

# central table describing the empirical input files used by the model.
STAGE_FILE_SPECS: dict[str, tuple[str, str, str]] = {
    "pre_referral_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI"),
    "pre_mri_to_mrireport": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported"),
    "pre_mrirep_to_biopsymdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT"),
    "pre_biopmdt_to_biop": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy"),
    "pre_biop_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report"),
    "pre_pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)"),
    "pre_treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt"),
}


def load_dates(path: Path, date_cols: list[str]) -> pd.DataFrame:
    """
    Load a CSV and parse the supplied date columns.

    Pre-PROSTAD files appear to be UK-style day-first dates, but some may not
    exactly match one strict format, so use dayfirst=True rather than a single
    rigid format string.
    """
    df = pd.read_csv(path)

    for c in date_cols:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

    return df


def days_to_next_weekday(d: pd.Series, target_weekday: int, include_today: bool = True) -> pd.Series:
    """
    Days from each date in d to the next target weekday.
    Monday=0 ... Sunday=6
    """
    wd = d.dt.weekday
    if include_today:
        return (target_weekday - wd) % 7
    return ((target_weekday - (wd + 1)) % 7) + 1


def clean_waits(series: pd.Series, min_valid: int = 0) -> pd.Series:
    """
    Remove NaNs and waits below min_valid.
    Default keeps 0+ and removes negatives only.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().copy()
    s = s[s >= min_valid]
    return s.astype(int)


def calc_wait(df: pd.DataFrame, start_col: str, end_col: str, min_valid: int = 0) -> pd.Series:
    """
    Calculate wait in days between two date columns and clean it.
    """
    waits = (df[end_col] - df[start_col]).dt.days
    return clean_waits(waits, min_valid=min_valid)



"""Build all empirical PDFs used by the simulation.
    """
def build_pdfs(data_dir: Path = DATA_DIR) -> dict[str, pd.Series]:

    stage_frames: dict[str, pd.DataFrame] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        stage_frames[pdf_key] = load_dates(data_dir / filename, [start_col, end_col])

    pdfs: dict[str, pd.Series] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        pdfs[pdf_key] = calc_wait(stage_frames[pdf_key], start_col, end_col)

    return pdfs



"""Build branching probabilities from the pre-PROSTAD outcome files."""
def build_branching(data_dir: Path = DATA_DIR) -> dict[str, dict[int, float]]:
   
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


