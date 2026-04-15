from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BIOPMDT_WD = 2   # Wednesday
TREATMDT_WD = 4  # Friday


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


def lower_quantile_samples(series: pd.Series, quantile: float = 0.5, min_valid: int = 0) -> pd.Series:
    """
    Take the lower part of an empirical distribution.
    Used for non-capacity / pre-queue delay approximation.
    """
    s = clean_waits(series, min_valid=min_valid)
    if s.empty:
        return s

    cutoff = s.quantile(quantile)
    return s[s <= cutoff].astype(int)


def build_pdfs(data_dir: Path = DATA_DIR, exclude_np053_ref_to_mri: bool = True) -> dict[str, pd.Series]:
    # --------------------------------------------------
    # Load files
    # --------------------------------------------------
    pre_referral = load_dates(
        data_dir / "pre_refferal.csv",
        ["Date of referral to pathway"],
    )

    pre_ref_to_mri = load_dates(
        data_dir / "pre_ref_to_mri.csv",
        ["Date of referral to pathway", "Date of MRI"],
    )

    pre_mri_to_mrireport = load_dates(
        data_dir / "pre_mri_to_mrirep.csv",
        ["Date of MRI", "Date MRI reported"],
    )

    pre_mrireport_to_biopmdt = load_dates(
        data_dir / "pre_mrirep_to_biopmdt.csv",
        ["Date MRI reported", "Date of Prostate MRI MDT"],
    )

    pre_biopmdt_to_biop = load_dates(
        data_dir / "pre_biopmdt_to_biop.csv",
        ["Date of Prostate MRI MDT", "Date of Biopsy"],
    )

    pre_biop_to_pathrep = load_dates(
        data_dir / "pre_biop_to_pathrep.csv",
        ["Date of Biopsy", "Date of pathology report"],
    )

    pre_pathrep_to_treatmdt = load_dates(
        data_dir / "pre_pathrep_to_treatmdt.csv",
        ["Date of pathology report", "Date of MDT (treatment options)"],
    )

    pre_treatmdt_to_outpat = load_dates(
        data_dir / "pre_treatmdt_to_outpat.csv",
        ["Date of MDT (treatment options)", "Date of outpat appt"],
    )

    # --------------------------------------------------
    # Optional sensitivity exclusion
    # --------------------------------------------------
    if exclude_np053_ref_to_mri and "Subject number" in pre_ref_to_mri.columns:
        pre_ref_to_mri = pre_ref_to_mri.loc[
            pre_ref_to_mri["Subject number"] != "NP053"
        ].copy()

    # --------------------------------------------------
    # Core empirical waits
    # --------------------------------------------------
    pdf_pre_ref_to_mri = calc_wait(
        pre_ref_to_mri,
        "Date of referral to pathway",
        "Date of MRI",
        min_valid=0,
    )

    pdf_pre_mri_to_mrireport = calc_wait(
        pre_mri_to_mrireport,
        "Date of MRI",
        "Date MRI reported",
        min_valid=0,
    )

    pdf_pre_mrireport_to_biopmdt = calc_wait(
        pre_mrireport_to_biopmdt,
        "Date MRI reported",
        "Date of Prostate MRI MDT",
        min_valid=0,
    )

    pdf_pre_biopmdt_to_biop = calc_wait(
        pre_biopmdt_to_biop,
        "Date of Prostate MRI MDT",
        "Date of Biopsy",
        min_valid=0,
    )

    pdf_pre_biop_to_pathrep = calc_wait(
        pre_biop_to_pathrep,
        "Date of Biopsy",
        "Date of pathology report",
        min_valid=0,
    )

    pdf_pre_pathrep_to_treatmdt = calc_wait(
        pre_pathrep_to_treatmdt,
        "Date of pathology report",
        "Date of MDT (treatment options)",
        min_valid=0,
    )

    pdf_pre_treatmdt_to_outpat = calc_wait(
        pre_treatmdt_to_outpat,
        "Date of MDT (treatment options)",
        "Date of outpat appt",
        min_valid=0,
    )

    # --------------------------------------------------
    # NEW: residual / pre-queue delay for referral -> MRI
    # Use lower half of pre-PROSTAD empirical distribution
    # to represent non-capacity scheduling/admin delay.
    # --------------------------------------------------
    pdf_pre_ref_to_mri_pre_delay = lower_quantile_samples(
        pdf_pre_ref_to_mri,
        quantile=0.1,
        min_valid=0,
    )

    # --------------------------------------------------
    # Queue residual PDFs for MDT-linked stages
    # --------------------------------------------------
    df = pre_mrireport_to_biopmdt.dropna(
        subset=["Date MRI reported", "Date of Prostate MRI MDT"]
    ).copy()

    df["w_obs"] = (df["Date of Prostate MRI MDT"] - df["Date MRI reported"]).dt.days
    df["s_obs"] = days_to_next_weekday(df["Date MRI reported"], BIOPMDT_WD, include_today=True)
    df["q_obs"] = df["w_obs"] - df["s_obs"]

    queue_pdf_mrirep_to_biopmdt = clean_waits(df["q_obs"], min_valid=0)

    df2 = pre_pathrep_to_treatmdt.dropna(
        subset=["Date of pathology report", "Date of MDT (treatment options)"]
    ).copy()

    df2["w_obs"] = (df2["Date of MDT (treatment options)"] - df2["Date of pathology report"]).dt.days
    df2["s_obs"] = days_to_next_weekday(df2["Date of pathology report"], TREATMDT_WD, include_today=True)
    df2["q_obs"] = df2["w_obs"] - df2["s_obs"]

    queue_pdf_pathrep_to_treatmdt = clean_waits(df2["q_obs"], min_valid=0)

    # --------------------------------------------------
    # Biopsy residual samples
    # --------------------------------------------------
    #biopsy_residual_fp = data_dir / "biopsy_residual_samples_orig.csv"
    #biopsy_residual_df = pd.read_csv(biopsy_residual_fp)

    # Try to extract a sensible numeric series regardless of column name
    #numeric_cols = biopsy_residual_df.select_dtypes(include=[np.number]).columns.tolist()
    #if not numeric_cols:
     #   raise ValueError(
      #      f"No numeric columns found in {biopsy_residual_fp.name}. "
       #     "Expected a residual delay column."
        #)

    #biopsy_residual_samples = clean_waits(biopsy_residual_df[numeric_cols[0]], min_valid=0)

    return {
        "pre_referral_to_mri": pdf_pre_ref_to_mri,
        "pre_referral_to_mri_pre_delay": pdf_pre_ref_to_mri_pre_delay,  # NEW
        "pre_mri_to_mrireport": pdf_pre_mri_to_mrireport,
        "pre_mrirep_to_biopsymdt": pdf_pre_mrireport_to_biopmdt,
        "pre_biopmdt_to_biop": pdf_pre_biopmdt_to_biop,
        "pre_biop_to_pathrep": pdf_pre_biop_to_pathrep,
        "pre_pathrep_to_treatmdt": pdf_pre_pathrep_to_treatmdt,
        "pre_treatmdt_to_outpat": pdf_pre_treatmdt_to_outpat,
        "queue_mrirep_to_biopsymdt": queue_pdf_mrirep_to_biopmdt,
        "queue_pathrep_to_treatmdt": queue_pdf_pathrep_to_treatmdt,
       # "biopsy_residual_samples": biopsy_residual_samples,
    }


def build_branching(data_dir: Path = DATA_DIR) -> dict[str, dict[int, float]]:
    """
    Build branching probabilities from pre-PROSTAD outcome files.
    """
    biop_mdt_dec = pd.read_csv(data_dir / "pre_biop_dec.csv")
    biop_mdt_dec["Outcome code"] = pd.to_numeric(
        biop_mdt_dec["Outcome code"], errors="coerce"
    ).astype("Int64")

    biop_dec_branch_probs = (
        biop_mdt_dec["Outcome code"]
        .value_counts(normalize=True)
        .dropna()
        .to_dict()
    )
    biop_dec_branch_probs = {int(k): float(v) for k, v in biop_dec_branch_probs.items()}

    path_dec = pd.read_csv(data_dir / "pre_pathrep_outcome.csv")
    path_dec["Outcome code"] = pd.to_numeric(
        path_dec["Outcome code"], errors="coerce"
    ).astype("Int64")

    path_dec_branch_probs = (
        path_dec["Outcome code"]
        .value_counts(normalize=True)
        .dropna()
        .to_dict()
    )
    path_dec_branch_probs = {int(k): float(v) for k, v in path_dec_branch_probs.items()}

    return {
        "biopmdt_outcome": biop_dec_branch_probs,
        "pathrep_outcome": path_dec_branch_probs,
    }


if __name__ == "__main__":
    pdfs = build_pdfs()
    branching = build_branching()

    print("Built PDFs:", list(pdfs.keys()))
    for k, v in pdfs.items():
        print(f"{k}: n={len(v)}, min={v.min() if len(v) else 'NA'}, max={v.max() if len(v) else 'NA'}")

    print("Built branching sets:", list(branching.keys()))