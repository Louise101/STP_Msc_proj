from __future__ import annotations

from pathlib import Path

import pandas as pd


# One central specification for stage-level real data parsing.
REAL_STAGE_SPECS = {
    "BASELINE_REAL": {
        "ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "uk"),
        "mri_to_report": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported", "uk"),
        "report_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT", "uk"),
        "biopmdt_to_biopsy": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy", "uk"),
        "biopsy_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report", "uk"),
        "pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)", "uk"),
        "treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt", "uk"),
    },
    "OBS_MIX_REAL": {
        "ref_to_mri": ("pros_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "us"),
        "mri_to_report": ("pros_mri_to_mriclin.csv", "Date of MRI", "Date of clinic", "us"),
        # Proxy placeholder retained because no direct post-report standalone MDT timestamp exists.
        "report_to_biopmdt": ("pros_mri_to_mriclin.csv", "Date of clinic", "Date of clinic", "us"),
        "biopmdt_to_biopsy": ("pros_mriclin_to_biop.csv", "Date of clinic", "Date of biopsy", "us"),
        "biopsy_to_pathrep": ("pros_biop_to_pathrep.csv", "Date of biopsy", "Date of pathology report", "us"),
        "pathrep_to_treatmdt": ("pros_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT to discuss treatment options", "us"),
        "treatmdt_to_outpat": ("pros_treatmdt_to_outpat.csv", "Date of MDT to discuss treatment options", "Date of OPD appt", "us"),
    },
}


def parse_date_series(series: pd.Series, style: str) -> pd.Series:
    """Parse dates according to the dataset's known date convention."""
    if style == "uk":
        return pd.to_datetime(series, dayfirst=True, errors="coerce")
    if style == "us":
        return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_real_stage_waits(data_dir: Path) -> pd.DataFrame:
    """Load stage-level real wait distributions for baseline and observed-mix data."""
    rows: list[dict] = []
    for scenario_name, specs in REAL_STAGE_SPECS.items():
        for stage_name, (filename, start_col, end_col, style) in specs.items():
            df = pd.read_csv(data_dir / filename)
            start_dates = parse_date_series(df[start_col], style)
            end_dates = parse_date_series(df[end_col], style)
            waits = (end_dates - start_dates).dt.days
            waits = waits[(waits.notna()) & (waits >= 0)]

            for wait in waits:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "stage": stage_name,
                        "wait_days": float(wait),
                    }
                )
    return pd.DataFrame(rows)


def build_real_pathway_csvs(
    pre_ref_file: str,
    pre_outpat_file: str,
    pros_ref_file: str,
    pros_outpat_file: str,
    out_pre_file: str,
    out_pros_file: str,
) -> None:
    """Create simple referral-to-outpatient pathway tables from raw CSVs."""
    def build_one(
        ref_file: str,
        out_file: str,
        output_file: str,
        ref_col: str,
        out_col: str,
        style: str,
    ) -> None:
        ref_df = pd.read_csv(ref_file).rename(columns={
            "Subject number": "patient_id",
            ref_col: "referral_date",
        })
        out_df = pd.read_csv(out_file).rename(columns={
            "Subject number": "patient_id",
            out_col: "outpatient_date",
        })

        ref_df["patient_id"] = ref_df["patient_id"].astype(str)
        out_df["patient_id"] = out_df["patient_id"].astype(str)
        ref_df["referral_date"] = parse_date_series(ref_df["referral_date"], style)
        out_df["outpatient_date"] = parse_date_series(out_df["outpatient_date"], style)

        combined = ref_df[["patient_id", "referral_date"]].merge(
            out_df[["patient_id", "outpatient_date"]],
            on="patient_id",
            how="inner",
        )
        combined["total_days"] = (combined["outpatient_date"] - combined["referral_date"]).dt.days
        combined = combined[(combined["total_days"].notna()) & (combined["total_days"] >= 0)].copy()
        combined.to_csv(output_file, index=False)

    build_one(
        ref_file=pre_ref_file,
        out_file=pre_outpat_file,
        output_file=out_pre_file,
        ref_col="Date of referral to pathway",
        out_col="Date of outpat appt",
        style="uk",
    )
    build_one(
        ref_file=pros_ref_file,
        out_file=pros_outpat_file,
        output_file=out_pros_file,
        ref_col="Date of referral to pathway",
        out_col="Date of OPD appt",
        style="us",
    )


def load_real_pathway_data(pre_path: str, pros_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-built real pathway datasets and recalculate total pathway days."""
    pre = pd.read_csv(pre_path)
    pros = pd.read_csv(pros_path)

    pre["referral_date"] = parse_date_series(pre["referral_date"], "uk")
    pre["outpatient_date"] = parse_date_series(pre["outpatient_date"], "uk")
    pros["referral_date"] = parse_date_series(pros["referral_date"], "us")
    pros["outpatient_date"] = parse_date_series(pros["outpatient_date"], "us")

    for df in (pre, pros):
        df["total_days"] = (df["outpatient_date"] - df["referral_date"]).dt.days

    pre = pre[(pre["total_days"].notna()) & (pre["total_days"] >= 0)].copy()
    pros = pros[(pros["total_days"].notna()) & (pros["total_days"] >= 0)].copy()
    return pre, pros
