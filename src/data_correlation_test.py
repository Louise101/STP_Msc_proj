from pathlib import Path
import pandas as pd
import numpy as np
from functools import reduce
from scipy.stats import spearmanr, kendalltau, zscore

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PATIENT_ID_COL = "Subject number"

# ============================================================
# STAGE FILE CONFIGURATION
# ============================================================
# For each stage file, define:
# - file: csv filename
# - start_date_col: earlier date
# - end_date_col: later date
# - rename_to: name for calculated wait column
#
# Replace these with your actual filenames / date column names.

STAGE_FILES = [
    {
        "file": "pre_ref_to_mri.csv",
        "start_date_col": "Date of referral to pathway",
        "end_date_col": "Date of MRI",
        "rename_to": "wait_ref_to_mri",
    },
    {
        "file": "pre_mri_to_mrirep.csv",
        "start_date_col": "Date of MRI",
        "end_date_col": "Date MRI reported",
        "rename_to": "wait_mri_to_report",
    },
    {
        "file": "pre_mrirep_to_biopmdt.csv",
        "start_date_col": "Date MRI reported",
        "end_date_col": "Date of Prostate MRI MDT",
        "rename_to": "wait_report_to_biopmdt",
    },
    {
        "file": "pre_biopmdt_to_biop.csv",
        "start_date_col": "Date of Prostate MRI MDT",
        "end_date_col": "Date of Biopsy",
        "rename_to": "wait_biopmdt_to_biopsy",
    },
    {
        "file": "pre_biop_to_pathrep.csv",
        "start_date_col": "Date of Biopsy",
        "end_date_col": "Date of pathology report",
        "rename_to": "wait_biopsy_to_pathreport",
    },
    {
        "file": "pre_pathrep_to_treatmdt.csv",
        "start_date_col": "Date of pathology report",
        "end_date_col": "Date of MDT (treatment options)",
        "rename_to": "wait_pathrep_to_treatmdt",
    },
    {
        "file": "pre_treatmdt_to_outpat.csv",
        "start_date_col": "Date of MDT (treatment options)",
        "end_date_col": "Date of outpat appt",
        "rename_to": "wait_treatmdt_to_outpat",
    },
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_stage_file_from_dates(file_path, patient_id_col, start_date_col, end_date_col, rename_to):
    """
    Load one stage file, calculate wait in days from two date columns.
    """
    df = pd.read_csv(file_path)

    required_cols = [patient_id_col, start_date_col, end_date_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: missing columns {missing}")

    out = df[[patient_id_col, start_date_col, end_date_col]].copy()

    out[start_date_col] = pd.to_datetime(out[start_date_col], errors="coerce", dayfirst=True)
    out[end_date_col] = pd.to_datetime(out[end_date_col], errors="coerce", dayfirst=True)

    out[rename_to] = (out[end_date_col] - out[start_date_col]).dt.days

    # Keep only patient_id and calculated wait
    out = out[[patient_id_col, rename_to]]

    # If duplicate patients exist, keep first non-null wait
    out = (
        out.sort_values(by=[patient_id_col])
           .drop_duplicates(subset=[patient_id_col], keep="first")
    )

    return out


def merge_dataframes_on_patient_id(dfs, patient_id_col):
    return reduce(
        lambda left, right: pd.merge(left, right, on=patient_id_col, how="outer"),
        dfs
    )


def pairwise_correlation_table(data, cols):
    results = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]

            subset = data[[c1, c2]].dropna()

            if len(subset) < 3:
                results.append({
                    "stage_1": c1,
                    "stage_2": c2,
                    "n": len(subset),
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "kendall_tau": np.nan,
                    "kendall_p": np.nan
                })
                continue

            spearman_res = spearmanr(subset[c1], subset[c2])
            kendall_res = kendalltau(subset[c1], subset[c2])

            results.append({
                "stage_1": c1,
                "stage_2": c2,
                "n": len(subset),
                "spearman_rho": spearman_res.statistic,
                "spearman_p": spearman_res.pvalue,
                "kendall_tau": kendall_res.statistic,
                "kendall_p": kendall_res.pvalue
            })

    return pd.DataFrame(results)


def progression_speed_analysis(data, patient_id_col, wait_cols):
    speed_df = data[[patient_id_col] + wait_cols].copy()

    for col in wait_cols:
        non_missing = speed_df[col].dropna()
        if len(non_missing) > 1 and non_missing.nunique() > 1:
            speed_df[col + "_z"] = zscore(speed_df[col], nan_policy="omit")
        else:
            speed_df[col + "_z"] = np.nan

    z_cols = [col + "_z" for col in wait_cols]
    speed_df["mean_progression_z"] = speed_df[z_cols].mean(axis=1, skipna=True)

    non_null = speed_df["mean_progression_z"].dropna()
    if len(non_null) >= 3 and non_null.nunique() > 1:
        speed_df["progression_group"] = pd.qcut(
            speed_df["mean_progression_z"],
            q=3,
            labels=["fast", "medium", "slow"],
            duplicates="drop"
        )
    else:
        speed_df["progression_group"] = np.nan

    return speed_df


def fast_slow_comparison(data, early_col, later_col):
    subset = data[[early_col, later_col]].dropna().copy()

    if len(subset) < 10:
        return None

    median_early = subset[early_col].median()

    subset["early_group"] = np.where(
        subset[early_col] <= median_early,
        "fast_early",
        "slow_early"
    )

    summary = subset.groupby("early_group")[later_col].agg(
        n="count",
        mean="mean",
        median="median",
        std="std"
    ).reset_index()

    summary.insert(0, "early_stage", early_col)
    summary.insert(1, "later_stage", later_col)

    return summary


# ============================================================
# LOAD STAGE FILES AND CALCULATE WAITS
# ============================================================

stage_dfs = []
wait_cols = []

for cfg in STAGE_FILES:
    file_path = DATA_DIR / cfg["file"]

    df_stage = load_stage_file_from_dates(
        file_path=file_path,
        patient_id_col=PATIENT_ID_COL,
        start_date_col=cfg["start_date_col"],
        end_date_col=cfg["end_date_col"],
        rename_to=cfg["rename_to"]
    )

    stage_dfs.append(df_stage)
    wait_cols.append(cfg["rename_to"])

df = merge_dataframes_on_patient_id(stage_dfs, PATIENT_ID_COL)

# ============================================================
# BASIC CLEANING
# ============================================================

for col in wait_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

    n_neg = (df[col] < 0).sum()
    if n_neg > 0:
        print(f"Warning: {n_neg} negative values found in {col}. Setting to NaN.")
        df.loc[df[col] < 0, col] = np.nan

print("\nMerged data shape:", df.shape)
print("\nMissing values per wait column:")
print(df[wait_cols].isna().sum())

# Save merged waits
df.to_csv(OUTPUT_DIR / "merged_pathway_waits.csv", index=False)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

corr_results = pairwise_correlation_table(df, wait_cols)
spearman_matrix = df[wait_cols].corr(method="spearman")
kendall_matrix = df[wait_cols].corr(method="kendall")

print("\n============================================================")
print("PAIRWISE CORRELATION RESULTS")
print("============================================================")
print(corr_results.to_string(index=False))

print("\n============================================================")
print("SPEARMAN CORRELATION MATRIX")
print("============================================================")
print(spearman_matrix)

print("\n============================================================")
print("KENDALL CORRELATION MATRIX")
print("============================================================")
print(kendall_matrix)

# ============================================================
# PATIENT PROGRESSION SPEED ANALYSIS
# ============================================================

speed_df = progression_speed_analysis(df, PATIENT_ID_COL, wait_cols)

print("\n============================================================")
print("PATIENT PROGRESSION SPEED SUMMARY")
print("============================================================")
print(speed_df["progression_group"].value_counts(dropna=False))

print("\nFastest patients:")
print(
    speed_df[[PATIENT_ID_COL, "mean_progression_z", "progression_group"]]
    .sort_values("mean_progression_z")
    .head(10)
    .to_string(index=False)
)

print("\nSlowest patients:")
print(
    speed_df[[PATIENT_ID_COL, "mean_progression_z", "progression_group"]]
    .sort_values("mean_progression_z", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ============================================================
# FAST EARLY -> FAST LATER CHECK
# ============================================================

comparison_tables = []

for i in range(len(wait_cols) - 1):
    summary = fast_slow_comparison(df, wait_cols[i], wait_cols[i + 1])
    if summary is not None:
        comparison_tables.append(summary)

if comparison_tables:
    fast_slow_df = pd.concat(comparison_tables, ignore_index=True)
    print("\n============================================================")
    print("FAST EARLY / SLOW EARLY COMPARISONS")
    print("============================================================")
    print(fast_slow_df.to_string(index=False))
    fast_slow_df.to_csv(OUTPUT_DIR / "fast_slow_stage_comparisons.csv", index=False)
else:
    fast_slow_df = pd.DataFrame()

# ============================================================
# SAVE OUTPUTS
# ============================================================

corr_results.to_csv(OUTPUT_DIR / "pairwise_pathway_correlations.csv", index=False)
spearman_matrix.to_csv(OUTPUT_DIR / "spearman_matrix.csv")
kendall_matrix.to_csv(OUTPUT_DIR / "kendall_matrix.csv")
speed_df.to_csv(OUTPUT_DIR / "patient_progression_speed.csv", index=False)

print("\nSaved outputs to:", OUTPUT_DIR)
print("- merged_pathway_waits.csv")
print("- pairwise_pathway_correlations.csv")
print("- spearman_matrix.csv")
print("- kendall_matrix.csv")
print("- patient_progression_speed.csv")
if not fast_slow_df.empty:
    print("- fast_slow_stage_comparisons.csv")