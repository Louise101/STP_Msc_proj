from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import shapiro

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

STAGE_FILES = {
    "pre_ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "uk"),
    "pre_mri_to_mrirep": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported", "uk"),
    "pre_mrirep_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT", "uk"),
    "pre_biopmdt_to_biop": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy", "uk"),
    "pre_biop_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report", "uk"),
    "pre_pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)", "uk"),
    "pre_treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt", "uk"),
}

def parse_dates(series: pd.Series, style: str) -> pd.Series:
    if style == "uk":
        return pd.to_datetime(series, dayfirst=True, errors="coerce")
    if style == "us":
        return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def check_stage_file(name: str, filename: str, col_start: str, col_end: str, style: str) -> dict:
    df = pd.read_csv(DATA_DIR / filename)

    d1 = parse_dates(df[col_start], style)
    d2 = parse_dates(df[col_end], style)
    waits = (d2 - d1).dt.days

    missing_start = d1.isna().sum()
    missing_end = d2.isna().sum()
    negative_waits = (waits < 0).sum()
    zero_waits = (waits == 0).sum()
    valid_waits = waits.dropna()
    valid_waits = valid_waits[valid_waits >= 0]

    q1 = valid_waits.quantile(0.25) if len(valid_waits) else np.nan
    q3 = valid_waits.quantile(0.75) if len(valid_waits) else np.nan
    iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
    upper_fence = q3 + 1.5 * iqr if pd.notna(iqr) else np.nan
    outliers_iqr = (valid_waits > upper_fence).sum() if pd.notna(upper_fence) else np.nan

    if 3 <= len(valid_waits) <= 5000:
        sample = valid_waits.sample(min(len(valid_waits), 500), random_state=1)
        shapiro_p = shapiro(sample).pvalue
    else:
        shapiro_p = np.nan

    return {
        "stage": name,
        "n_rows": len(df),
        "missing_start_dates": int(missing_start),
        "missing_end_dates": int(missing_end),
        "negative_waits": int(negative_waits),
        "zero_waits": int(zero_waits),
        "valid_waits_n": int(len(valid_waits)),
        "mean_wait": float(valid_waits.mean()) if len(valid_waits) else np.nan,
        "median_wait": float(valid_waits.median()) if len(valid_waits) else np.nan,
        "min_wait": float(valid_waits.min()) if len(valid_waits) else np.nan,
        "max_wait": float(valid_waits.max()) if len(valid_waits) else np.nan,
        "iqr_outliers": int(outliers_iqr) if pd.notna(outliers_iqr) else np.nan,
        "shapiro_p": shapiro_p,
        "distribution_hint": "likely non-normal/skewed" if pd.notna(shapiro_p) and shapiro_p < 0.05 else "not clearly non-normal",
    }

def main():
    rows = []
    for stage, spec in STAGE_FILES.items():
        rows.append(check_stage_file(stage, *spec))

    summary = pd.DataFrame(rows)
    print("\n=== PRE-SIMULATION DATA CHECKS ===")
    print(summary.round(3).to_string(index=False))

    out_file = DATA_DIR.parent / "outputs" / "data_check_summary.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

if __name__ == "__main__":
    main()