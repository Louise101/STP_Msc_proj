import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
# ============================================================
# 1. Real data stage setup
# ============================================================
real_stage_info = {
    "wait_ref_to_mri": {
        "file": DATA_DIR / "pre_ref_to_mri.csv",
        "start_date_col": "Date of referral to pathway",
        "end_date_col": "Date of MRI",
    },
    "wait_mri_to_mrirep": {
        "file": DATA_DIR / "pre_mri_to_mrirep.csv",
        "start_date_col": "Date of MRI",
        "end_date_col": "Date MRI reported",
    },
    "wait_mrirep_to_biopmdt": {
        "file": DATA_DIR / "pre_mrirep_to_biopmdt.csv",
        "start_date_col": "Date MRI reported",
        "end_date_col": "Date of Prostate MRI MDT",
    },
    "wait_biopmdt_to_biop": {
        "file": DATA_DIR / "pre_biopmdt_to_biop.csv",
        "start_date_col": "Date of Prostate MRI MDT",
        "end_date_col": "Date of Biopsy",
    },
    "wait_biop_to_pathrep": {
        "file": DATA_DIR / "pre_biop_to_pathrep.csv",
        "start_date_col": "Date of Biopsy",
        "end_date_col": "Date of pathology report",
    },
    "wait_pathrep_to_treatmdt": {
        "file": DATA_DIR / "pre_pathrep_to_treatmdt.csv",
        "start_date_col": "Date of pathology report",
        "end_date_col": "Date of MDT (treatment options)",
    },
    "wait_treatmdt_to_outpat": {
        "file": DATA_DIR / "pre_treatmdt_to_outpat.csv",
        "start_date_col": "Date of MDT (treatment options)",
        "end_date_col": "Date of outpat appt",
    },
}

patient_id_col = "Subject number"
sim_file = BASE_DIR/"sim_waits.csv"

# ============================================================
# 2. Build real waits dataframe from separate files
# ============================================================
real_dfs = []

for wait_name, info in real_stage_info.items():
    df = pd.read_csv(info["file"])

    # Convert dates
    df[info["start_date_col"]] = pd.to_datetime(df[info["start_date_col"]], errors="coerce", dayfirst=True)
    df[info["end_date_col"]]   = pd.to_datetime(df[info["end_date_col"]], errors="coerce", dayfirst=True)

    # Calculate wait in days
    df[wait_name] = (df[info["end_date_col"]] - df[info["start_date_col"]]).dt.days

    # Keep only ID + wait
    df = df[[patient_id_col, wait_name]]

    real_dfs.append(df)

# Merge all real stage waits
real_df = real_dfs[0]
for df in real_dfs[1:]:
    real_df = real_df.merge(df, on=patient_id_col, how="outer")

# ============================================================
# 3. Load simulated waits
# ============================================================
sim_df = pd.read_csv(sim_file)

# ============================================================
# 4. Simulated column mapping
#    LEFT SIDE = your actual sim_waits.csv column name
#    RIGHT SIDE = standard name used in this script
# ============================================================
sim_stage_cols = {
    "wait_ref_to_mri": "wait_ref_to_mri",
    "wait_mri_to_report": "wait_mri_to_mrirep",
    "wait_report_to_biopmdt": "wait_mrirep_to_biopmdt",
    "wait_biopmdt_to_biopsy": "wait_biopmdt_to_biop",
    "wait_biopsy_to_pathreport": "wait_biop_to_pathrep",
    "wait_pathrep_to_treatmdt": "wait_pathrep_to_treatmdt",
    "wait_treatmdt_to_outpat": "wait_treatmdt_to_outpat",
}

# Rename simulated columns into the same names as the real data
sim_df = sim_df.rename(columns=sim_stage_cols)

# ============================================================
# 5. Ensure wait columns are numeric
# ============================================================
stage_cols = list(real_stage_info.keys())

for col in stage_cols:
    real_df[col] = pd.to_numeric(real_df[col], errors="coerce")
    sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce")

# ============================================================
# 6. Remove negative waits if any bad dates exist
# ============================================================
for col in stage_cols:
    real_df.loc[real_df[col] < 0, col] = pd.NA
    sim_df.loc[sim_df[col] < 0, col] = pd.NA

# ============================================================
# 7. Calculate total pathway time
#    Only for patients with all stages present
# ============================================================
real_complete = real_df.dropna(subset=stage_cols).copy()
sim_complete = sim_df.dropna(subset=stage_cols).copy()

real_complete["total_pathway"] = real_complete[stage_cols].sum(axis=1)
sim_complete["total_pathway"] = sim_complete[stage_cols].sum(axis=1)

# ============================================================
# 8. Compare real vs simulated means
# ============================================================
results = []

for col in stage_cols:
    real_mean = real_df[col].mean()
    sim_mean = sim_df[col].mean()
    abs_diff = sim_mean - real_mean
    pct_diff = (abs_diff / real_mean * 100) if pd.notna(real_mean) and real_mean != 0 else pd.NA

    results.append({
        "stage": col,
        "real_n": real_df[col].notna().sum(),
        "sim_n": sim_df[col].notna().sum(),
        "real_mean": real_mean,
        "sim_mean": sim_mean,
        "absolute_difference": abs_diff,
        "percent_difference": pct_diff
    })

# Total pathway comparison
real_total_mean = real_complete["total_pathway"].mean()
sim_total_mean = sim_complete["total_pathway"].mean()
total_abs_diff = sim_total_mean - real_total_mean
total_pct_diff = (total_abs_diff / real_total_mean * 100) if pd.notna(real_total_mean) and real_total_mean != 0 else pd.NA

results.append({
    "stage": "total_pathway",
    "real_n": len(real_complete),
    "sim_n": len(sim_complete),
    "real_mean": real_total_mean,
    "sim_mean": sim_total_mean,
    "absolute_difference": total_abs_diff,
    "percent_difference": total_pct_diff
})

comparison_df = pd.DataFrame(results)

# Round for readability
comparison_df = comparison_df.round({
    "real_mean": 2,
    "sim_mean": 2,
    "absolute_difference": 2,
    "percent_difference": 2
})

# ============================================================
# 9. Output
# ============================================================
print("\nReal vs simulated mean waits:\n")
print(comparison_df)

comparison_df.to_csv("mean_wait_comparison.csv", index=False)
real_df.to_csv("real_stage_waits_merged.csv", index=False)

print("\nSaved:")
print("- mean_wait_comparison.csv")
print("- real_stage_waits_merged.csv")