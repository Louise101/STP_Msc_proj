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


REAL_STAGE_SPECS = {
    "pre": {
        "ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "uk"),
        "mri_to_report": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported", "uk"),
        "report_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT", "uk"),
        "biopmdt_to_biopsy": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy", "uk"),
        "biopsy_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report", "uk"),
        "pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)", "uk"),
        "treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt", "uk"),
    },
    "pros": {
        "ref_to_mri": ("pros_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "us"),
        "mri_to_report": ("pros_mri_to_mriclin.csv", "Date of MRI", "Date of clinic", "us"),
        "biopmdt_to_biopsy": ("pros_mriclin_to_biop.csv", "Date of clinic", "Date of biopsy", "us"),
        "biopsy_to_pathrep": ("pros_biop_to_pathrep.csv", "Date of biopsy", "Date of pathology report", "us"),
        "pathrep_to_treatmdt": ("pros_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT to discuss treatment options", "us"),
        "treatmdt_to_outpat": ("pros_treatmdt_to_outpat.csv", "Date of MDT to discuss treatment options", "Date of OPD appt", "us"),
    },
}

STAGE_LABELS = {
    "ref_to_mri": "Referral â MRI",
    "mri_to_report": "MRI â Report",
    "report_to_biopmdt": "Report â Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT â Biopsy",
    "biopsy_to_pathrep": "Biopsy â Path Report",
    "pathrep_to_treatmdt": "Path Report â Treat MDT",
    "treatmdt_to_outpat": "Treat MDT â Outpatient",
}

#Parse real-world date columns that use mixed UK and US formats.
def parse_date_series(series: pd.Series, style: str) -> pd.Series:
    if style == "uk":
        return pd.to_datetime(series, dayfirst=True, errors="coerce")
    if style == "us":
        return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    return pd.to_datetime(series, errors="coerce")



def load_real_stage_waits(data_dir: Path) -> pd.DataFrame:
    rows = []

    for real_scenario, specs in REAL_STAGE_SPECS.items():
        for stage, (fname, col1, col2, style) in specs.items():
            df = pd.read_csv(data_dir / fname).copy()
            d1 = parse_date_series(df[col1], style)
            d2 = parse_date_series(df[col2], style)
            waits = (d2 - d1).dt.days
            waits = waits[(waits.notna()) & (waits >= 0)]

            for w in waits:
                rows.append({
                    "scenario": real_scenario,
                    "stage": stage,
                    "wait_days": float(w),
                })

    return pd.DataFrame(rows)


#Load a CSV and parse the supplied date columns.
def load_dates(path: Path, date_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in date_cols:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

    return df


#Days from each date in d to the next target weekday.
    #Monday=0 ... Sunday=6
def days_to_next_weekday(d: pd.Series, target_weekday: int, include_today: bool = True) -> pd.Series:
    wd = d.dt.weekday
    if include_today:
        return (target_weekday - wd) % 7
    return ((target_weekday - (wd + 1)) % 7) + 1

#Remove NaNs and waits below min_valid.
def clean_waits(series: pd.Series, min_valid: int = 0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().copy()
    s = s[s >= min_valid]
    return s.astype(int)

#Calculate wait in days between two date columns and clean it.
def calc_wait(df: pd.DataFrame, start_col: str, end_col: str, min_valid: int = 0) -> pd.Series:
    waits = (df[end_col] - df[start_col]).dt.days
    return clean_waits(waits, min_valid=min_valid)



#Build all empirical PDFs used by the simulation.
def build_pdfs(data_dir: Path = DATA_DIR) -> dict[str, pd.Series]:

    stage_frames: dict[str, pd.DataFrame] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        stage_frames[pdf_key] = load_dates(data_dir / filename, [start_col, end_col])

    pdfs: dict[str, pd.Series] = {}
    for pdf_key, (filename, start_col, end_col) in STAGE_FILE_SPECS.items():
        pdfs[pdf_key] = calc_wait(stage_frames[pdf_key], start_col, end_col)

    return pdfs



#Build branching probabilities from the pre-PROSTAD outcome files
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


