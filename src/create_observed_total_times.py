import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "verification_outputs"
OUT_DIR.mkdir(exist_ok=True)

def load_dates(path: Path, date_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df

def main():
    # 1) Referral table (anchor)
    ref = load_dates(DATA_DIR / "pre_refferal.csv", ["Date of referral to pathway"])
    ref = ref.rename(columns={"Subject number": "patient_id", "Date of referral to pathway": "referral_date"})
    ref = ref[["patient_id", "referral_date"]].drop_duplicates()

    # 2) Load all stage files and pull out the “furthest date reached” for each patient
    stage_frames = []

    # Each block: load a file, standardise patient_id, keep relevant date columns, melt to long format
    files = [
        ("pre_ref_to_mri.csv", ["Date of referral to pathway", "Date of MRI"]),
        ("pre_mri_to_mrirep.csv", ["Date of MRI", "Date MRI reported"]),
        ("pre_mrirep_to_biopmdt.csv", ["Date MRI reported", "Date of Prostate MRI MDT"]),
        ("pre_biopmdt_to_biop.csv", ["Date of Prostate MRI MDT", "Date of Biopsy"]),
        ("pre_biop_to_pathrep.csv", ["Date of Biopsy", "Date of pathology report"]),
        ("pre_pathrep_to_treatmdt.csv", ["Date of pathology report", "Date of MDT (treatment options)"]),
        ("pre_treatmdt_to_outpat.csv", ["Date of MDT (treatment options)", "Date of outpat appt"]),
    ]

    for fname, dcols in files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[WARN] Missing {fname}, skipping")
            continue

        df = load_dates(fpath, dcols)
        df = df.rename(columns={"Subject number": "patient_id"})

        # Keep only patient_id and date columns that actually exist
        keep_cols = ["patient_id"] + [c for c in dcols if c in df.columns]
        df = df[keep_cols]

        # Convert to long: one row per patient-date observation
        long = df.melt(id_vars="patient_id", value_vars=[c for c in keep_cols if c != "patient_id"],
                       var_name="stage", value_name="date")
        long = long.dropna(subset=["date"])
        stage_frames.append(long)

    if not stage_frames:
        raise RuntimeError("No stage files were loaded. Check DATA_DIR filenames.")

    stages_long = pd.concat(stage_frames, ignore_index=True)

    # 3) For each patient, define end_date = latest date observed in any stage
    end_dates = stages_long.groupby("patient_id", as_index=False)["date"].max()
    end_dates = end_dates.rename(columns={"date": "end_date"})

    # 4) Merge with referral dates
    merged = ref.merge(end_dates, on="patient_id", how="left")

# 5) Compute total days
    merged["total_days"] = (merged["end_date"] - merged["referral_date"]).dt.days

# 6) Keep only valid durations
    merged = merged.dropna(subset=["end_date", "total_days"])
    merged = merged[merged["total_days"] >= 0] 

    # Keep all patients, but you may want to exclude negative durations
   # invalid = merged[~merged["total_days_valid"]]
    #if len(invalid) > 0:
     #   print(f"[WARN] Found {len(invalid)} patients with missing/negative total_days. Kept in file with flags.")

    # 7) Save
    # Save
    out = merged.sort_values("patient_id")
    out[["patient_id", "referral_date", "end_date", "total_days"]].to_csv(
        DATA_DIR / "Pre_observed_total_times.csv",
        index=False
    )

    print("Saved clean overall durations.")
    print(out["total_days"].describe())

if __name__ == "__main__":
    main()