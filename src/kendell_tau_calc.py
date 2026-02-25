import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATE_FORMAT = "%d/%m/%Y"  # adjust if needed



#load in data anc compute days between stages
def load_stage_wait(csv_path: Path, id_col: str, start_col: str, end_col: str, new_col_name: str):

    df = pd.read_csv(csv_path)

    # Parse dates
    df[start_col] = pd.to_datetime(df[start_col], format=DATE_FORMAT, errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], format=DATE_FORMAT, errors="coerce")

    # Drop rows with missing dates
    df = df.dropna(subset=[start_col, end_col]).copy()

    # Calculate wait time
    df[new_col_name] = (df[end_col] - df[start_col]).dt.days

    # Keep only ID + wait
    return df[[id_col, new_col_name]]



# Referral -> MRI
ref_to_mri = load_stage_wait(
    DATA_DIR / "pre_ref_to_mri.csv",
    id_col="Subject number",
    start_col="Date of referral to pathway",
    end_col="Date of MRI",
    new_col_name="wait_ref_to_mri"
)

# MRI -> Report
mri_to_report = load_stage_wait(
    DATA_DIR / "pre_mri_to_mrirep.csv",
    id_col="Subject number",
    start_col="Date of MRI",
    end_col="Date MRI reported",
    new_col_name="wait_mri_to_report"
)

# Report -> Biopsy MDT
report_to_mdt = load_stage_wait(
    DATA_DIR / "pre_mrirep_to_biopmdt.csv",
    id_col="Subject number",
    start_col="Date MRI reported",
    end_col="Date of Prostate MRI MDT",
    new_col_name="wait_report_to_biopmdt"
)

# merge by subject number

obs = (
    ref_to_mri
    .merge(mri_to_report, on="Subject number", how="inner")
    .merge(report_to_mdt, on="Subject number", how="inner")
)

print(f"Number of complete patients used for τ calculation: {len(obs)}")


# calculate Kendall's Tau Matrix for observed

tau_obs = obs[
    ["wait_ref_to_mri",
     "wait_mri_to_report",
     "wait_report_to_biopmdt"]
].corr(method="kendall")

print("\nObserved Kendall's Tau matrix:")
print(tau_obs)

# clculate Kendall's Tau Matrix for simulated 
EVENT_FILE = DATA_DIR / "batch_events.csv"
events = pd.read_csv("batch_events.csv")

stage_map = {
    "mri_performed": "wait_ref_to_mri",
    "mri_report_ready": "wait_mri_to_report",
    "MDT_occured": "wait_report_to_biopmdt",
}

# Filter to relevant events
stage_events = events[events["event"].isin(stage_map.keys())].copy()

# Rename event types to stage names
stage_events["stage_name"] = stage_events["event"].map(stage_map)

# Pivot so each patient is one row
sim = stage_events.pivot_table(
    index="patient_id",
    columns="stage_name",
    values="wait_days",
    aggfunc="first"
).reset_index()

# Drop patients with missing values (must have all 3 stages)
sim_complete = sim.dropna(
    subset=["wait_ref_to_mri",
            "wait_mri_to_report",
            "wait_report_to_biopmdt"]
)

print(f"Number of complete simulated patients: {len(sim_complete)}")


tau_sim = sim_complete[
    ["wait_ref_to_mri",
     "wait_mri_to_report",
     "wait_report_to_biopmdt"]
].corr(method="kendall")

print("\nSimulated Kendall's Tau matrix:")

print(tau_sim)
