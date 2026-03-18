import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load your simulated patient-level waits
sim_waits = pd.read_csv(BASE_DIR/"sim_waits.csv")

# Load total pathway times
results = pd.read_csv(BASE_DIR/"batch_results.csv")

# Merge together
df = pd.merge(results, sim_waits, on="patient_id", how="left")

# --------------------------------------------------
# 1. Define "long pathway" patients
# --------------------------------------------------

threshold = df["total_days"].quantile(0.95)  # top 5%

df["is_long"] = df["total_days"] >= threshold

print(f"\nLong pathway threshold (95th percentile): {threshold:.1f} days")

# --------------------------------------------------
# 2. Compare stage waits
# --------------------------------------------------

wait_cols = [
    "wait_ref_to_mri",
    "wait_mri_to_report",
    "wait_report_to_biopmdt",
    "wait_biopmdt_to_biopsy",
    "wait_biopsy_to_pathreport",
    "wait_pathrep_to_treatmdt",
    "wait_treatmdt_to_outpat",
]

summary = []

for col in wait_cols:
    long_mean = df.loc[df["is_long"], col].mean()
    normal_mean = df.loc[~df["is_long"], col].mean()

    summary.append({
        "stage": col,
        "long_mean": long_mean,
        "normal_mean": normal_mean,
        "difference": long_mean - normal_mean,
        "ratio": long_mean / normal_mean if normal_mean > 0 else np.nan
    })

summary_df = pd.DataFrame(summary).sort_values("difference", ascending=False)

print("\nStage contribution to long pathways:")
print(summary_df.to_string(index=False))

# --------------------------------------------------
# 3. Optional: show worst individual cases
# --------------------------------------------------

print("\nTop 10 longest patients:")
print(df.sort_values("total_days", ascending=False)[
    ["patient_id", "total_days"] + wait_cols
].head(10).to_string(index=False))