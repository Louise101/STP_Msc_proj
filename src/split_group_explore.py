from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

#SIM_FILE = BASE_DIR / "batch_events.csv"
SIM_FILE = OUTPUT_DIR / "batch_events.csv"

COMMON_PATIENT_ID_COL = "patient_id"
REAL_PATIENT_ID_COL = "Subject number"
SIM_PATIENT_ID_COL = "patient_id"



# ============================================================
# REAL DATA STAGE FILE CONFIG
# ============================================================

REAL_STAGE_FILES = [
    {
        "file": "pre_ref_to_mri.csv",
        "start_date_col": "Date of referral to pathway",
        "end_date_col": "Date of MRI",
        "start_event": "referral_received",
        "end_event": "mri_performed",
    },
    {
        "file": "pre_mri_to_mrirep.csv",
        "start_date_col": "Date of MRI",
        "end_date_col": "Date MRI reported",
        "start_event": "mri_performed",
        "end_event": "mri_report_ready",
    },
    {
        "file": "pre_mrirep_to_biopmdt.csv",
        "start_date_col": "Date MRI reported",
        "end_date_col": "Date of Prostate MRI MDT",
        "start_event": "mri_report_ready",
        "end_event": "MDT_occured",
    },
    {
        "file": "pre_biopmdt_to_biop.csv",
        "start_date_col": "Date of Prostate MRI MDT",
        "end_date_col": "Date of Biopsy",
        "start_event": "MDT_occured",
        "end_event": "biopsy_done",
    },
    {
        "file": "pre_biop_to_pathrep.csv",
        "start_date_col": "Date of Biopsy",
        "end_date_col": "Date of pathology report",
        "start_event": "biopsy_done",
        "end_event": "Path_report_recieved",
    },
    {
        "file": "pre_pathrep_to_treatmdt.csv",
        "start_date_col": "Date of pathology report",
        "end_date_col": "Date of MDT (treatment options)",
        "start_event": "Path_report_recieved",
        "end_event": "Treatment_options_MDT_occured",
    },
    {
        "file": "pre_treatmdt_to_outpat.csv",
        "start_date_col": "Date of MDT (treatment options)",
        "end_date_col": "Date of outpat appt",
        "start_event": "Treatment_options_MDT_occured",
        "end_event": "Outpatient_appointment_occured",
    },
]

# ============================================================
# LOADERS
# ============================================================

def load_simulated_event_log(
    file_path: Path,
    sim_patient_id_col: str,
    common_patient_id_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if sim_patient_id_col not in df.columns:
        raise ValueError(
            f"{file_path.name}: missing simulated patient ID column '{sim_patient_id_col}'"
        )

    required = [sim_patient_id_col, "event", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: missing columns {missing}")

    df = df.rename(columns={sim_patient_id_col: common_patient_id_col})

    # Simulated dates are ISO format YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")

    return df[[common_patient_id_col, "event", "date"]].copy()

def stage_file_to_event_rows(
    file_path: Path,
    real_patient_id_col: str,
    common_patient_id_col: str,
    start_date_col: str,
    end_date_col: str,
    start_event: str,
    end_event: str,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if start_date_col == end_date_col:
        raise ValueError(
            f"{file_path.name}: start_date_col and end_date_col are identical: {start_date_col!r}"
        )

    required = [real_patient_id_col, start_date_col, end_date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: missing columns {missing}")

    tmp = df[[real_patient_id_col, start_date_col, end_date_col]].copy()
    tmp = tmp.rename(columns={real_patient_id_col: common_patient_id_col})

    tmp[start_date_col] = pd.to_datetime(
        tmp[start_date_col].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True,
    )
    tmp[end_date_col] = pd.to_datetime(
        tmp[end_date_col].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True,
    )

    start_rows = tmp[[common_patient_id_col, start_date_col]].rename(
        columns={start_date_col: "date"}
    )
    start_rows["event"] = start_event

    end_rows = tmp[[common_patient_id_col, end_date_col]].rename(
        columns={end_date_col: "date"}
    )
    end_rows["event"] = end_event

    out = pd.concat([start_rows, end_rows], ignore_index=True)
    out = out.dropna(subset=["date"])

    return out[[common_patient_id_col, "event", "date"]]


def build_real_event_log(
    stage_files: list[dict],
    data_dir: Path,
    real_patient_id_col: str,
    common_patient_id_col: str,
) -> pd.DataFrame:
    all_rows = []

    for cfg in stage_files:
        file_path = data_dir / cfg["file"]
        event_rows = stage_file_to_event_rows(
            file_path=file_path,
            real_patient_id_col=real_patient_id_col,
            common_patient_id_col=common_patient_id_col,
            start_date_col=cfg["start_date_col"],
            end_date_col=cfg["end_date_col"],
            start_event=cfg["start_event"],
            end_event=cfg["end_event"],
        )
        all_rows.append(event_rows)

    real_events = pd.concat(all_rows, ignore_index=True)
    real_events = real_events.drop_duplicates()

    return real_events[[common_patient_id_col, "event", "date"]].copy()

# ============================================================
# EVENT HELPERS
# ============================================================

def get_referral_date(patient_df: pd.DataFrame):
    vals = patient_df.loc[
        patient_df["event"] == "referral_received",
        "date"
    ].dropna()

    if len(vals) == 0:
        return pd.NaT

    return vals.min()


def has_event_after_referral(patient_df: pd.DataFrame, event_name: str, referral_date) -> bool:
    if pd.isna(referral_date):
        return False

    vals = patient_df.loc[
        (patient_df["event"] == event_name) &
        (patient_df["date"] >= referral_date),
        "date"
    ].dropna()

    return len(vals) > 0


def get_first_event_date_after_referral(patient_df: pd.DataFrame, event_name: str, referral_date):
    if pd.isna(referral_date):
        return pd.NaT

    vals = patient_df.loc[
        (patient_df["event"] == event_name) &
        (patient_df["date"] >= referral_date),
        "date"
    ].dropna()

    if len(vals) == 0:
        return pd.NaT

    return vals.min()

# ============================================================
# SUBGROUP CLASSIFICATION
# ============================================================

def classify_subgroup_and_enddate(patient_df: pd.DataFrame):
    """
    Classify patient by furthest clinically valid stage reached and define
    pathway end date from that subgroup-defining event.

    Only events on or after referral are considered valid.
    """
    referral_date = get_referral_date(patient_df)

    if pd.isna(referral_date):
        return "other", pd.NaT

    has_outpat = has_event_after_referral(
        patient_df, "Outpatient_appointment_occured", referral_date
    )
    has_treat_mdt = has_event_after_referral(
        patient_df, "Treatment_options_MDT_occured", referral_date
    )
    has_path = has_event_after_referral(
        patient_df, "Path_report_recieved", referral_date
    )
    has_biopsy = has_event_after_referral(
        patient_df, "biopsy_done", referral_date
    )
    has_exit = has_event_after_referral(
        patient_df, "pathway_end", referral_date
    )
    has_mdt = has_event_after_referral(
        patient_df, "MDT_occured", referral_date
    )

    if has_outpat:
        end_date = get_first_event_date_after_referral(
            patient_df, "Outpatient_appointment_occured", referral_date
        )
        return "full_pathway_to_outpatient", end_date

    if has_treat_mdt:
        end_date = get_first_event_date_after_referral(
            patient_df, "Treatment_options_MDT_occured", referral_date
        )
        return "reached_treatment_mdt", end_date

    if has_path:
        end_date = get_first_event_date_after_referral(
            patient_df, "Path_report_recieved", referral_date
        )
        return "biopsy_pathology_only", end_date

    if has_biopsy:
        end_date = get_first_event_date_after_referral(
            patient_df, "biopsy_done", referral_date
        )
        return "reached_biopsy_only", end_date

    if has_exit:
        end_date = get_first_event_date_after_referral(
            patient_df, "pathway_end", referral_date
        )
        return "ended_before_biopsy", end_date

    if has_mdt:
        end_date = get_first_event_date_after_referral(
            patient_df, "MDT_occured", referral_date
        )
        return "ended_before_biopsy", end_date

    return "other", referral_date

# ============================================================
# PATIENT-LEVEL SUMMARIES
# ============================================================

def summarise_patients_from_events(event_df: pd.DataFrame, patient_id_col: str) -> pd.DataFrame:
    rows = []

    for patient_id, grp in event_df.groupby(patient_id_col):
        grp = grp.sort_values("date").copy()

        referral_date = get_referral_date(grp)
        pathway_subgroup, end_date = classify_subgroup_and_enddate(grp)

        total_days = np.nan
        if pd.notna(referral_date) and pd.notna(end_date):
            total_days = (end_date - referral_date).days
            if total_days < 0:
                total_days = np.nan

        rows.append({
            patient_id_col: patient_id,
            "referral_date": referral_date,
            "end_date": end_date,
            "total_pathway_days": total_days,
            "pathway_subgroup": pathway_subgroup,
        })

    return pd.DataFrame(rows)


def subgroup_summary(patient_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = (
        patient_df.groupby("pathway_subgroup")["total_pathway_days"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    return out.rename(columns={
        "n": f"{prefix}_n",
        "mean": f"{prefix}_mean",
        "median": f"{prefix}_median",
        "std": f"{prefix}_std",
        "min": f"{prefix}_min",
        "max": f"{prefix}_max",
    })

# ============================================================
# LOAD DATA
# ============================================================

sim_events = load_simulated_event_log(
    SIM_FILE,
    sim_patient_id_col=SIM_PATIENT_ID_COL,
    common_patient_id_col=COMMON_PATIENT_ID_COL,
)
print("\nLoaded simulated file:", SIM_FILE)
print(sim_events.loc[sim_events["patient_id"] == "VP00001"].sort_values("date").to_string(index=False))

real_events = build_real_event_log(
    REAL_STAGE_FILES,
    DATA_DIR,
    real_patient_id_col=REAL_PATIENT_ID_COL,
    common_patient_id_col=COMMON_PATIENT_ID_COL,
)

# ============================================================
# SUMMARISE TO PATIENT LEVEL
# ============================================================

real_patients = summarise_patients_from_events(real_events, COMMON_PATIENT_ID_COL)
sim_patients = summarise_patients_from_events(sim_events, COMMON_PATIENT_ID_COL)

# ============================================================
# COMPARE SUBGROUPS
# ============================================================

real_summary = subgroup_summary(real_patients, "real")
sim_summary = subgroup_summary(sim_patients, "sim")

comparison = pd.merge(real_summary, sim_summary, on="pathway_subgroup", how="outer")
comparison["n_difference"] = comparison["sim_n"] - comparison["real_n"]
comparison["mean_difference"] = comparison["sim_mean"] - comparison["real_mean"]
comparison["percent_mean_difference"] = (
    100 * comparison["mean_difference"] / comparison["real_mean"]
)

print("\nREAL subgroup counts")
print(real_patients["pathway_subgroup"].value_counts(dropna=False))

print("\nSIM subgroup counts")
print(sim_patients["pathway_subgroup"].value_counts(dropna=False))

print("\nTotal pathway comparison by subgroup")
print(comparison.to_string(index=False))

# ============================================================
# DEBUG CHECKS
# ============================================================

print("\nUnique simulated event names")
print(sorted(sim_events["event"].dropna().unique()))

zero_days = sim_patients.loc[sim_patients["total_pathway_days"] == 0]
print(f"\nSimulated patients with zero pathway days: {len(zero_days)}")
if len(zero_days) > 0:
    print(zero_days.head(10).to_string(index=False))

bad_ended = sim_patients.loc[
    sim_patients["pathway_subgroup"] == "ended_before_biopsy"
].sort_values("total_pathway_days", ascending=False)

print("\nLongest ended_before_biopsy simulated patients")
if len(bad_ended) > 0:
    print(bad_ended.head(10).to_string(index=False))

    example_id = bad_ended.iloc[0][COMMON_PATIENT_ID_COL]
    print(f"\nRaw event log for example patient: {example_id}")
    print(
        sim_events.loc[sim_events[COMMON_PATIENT_ID_COL] == example_id]
        .sort_values("date")
        .to_string(index=False)
    )

# ============================================================
# SAVE
# ============================================================

real_events.to_csv(OUTPUT_DIR / "real_event_log_rebuilt.csv", index=False)
real_patients.to_csv(OUTPUT_DIR / "real_patient_subgroups.csv", index=False)
sim_patients.to_csv(OUTPUT_DIR / "sim_patient_subgroups.csv", index=False)
comparison.to_csv(OUTPUT_DIR / "real_vs_sim_subgroup_comparison.csv", index=False)

print("\nSaved:")
print("- real_event_log_rebuilt.csv")
print("- real_patient_subgroups.csv")
print("- sim_patient_subgroups.csv")
print("- real_vs_sim_subgroup_comparison.csv")


#plots

def ecdf(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y


def plot_ecdf_by_subgroup(real_df, sim_df, subgroup, save_path=None):
    """
    Plot ECDF comparison for a single subgroup.
    """

    real_vals = real_df.loc[
        real_df["pathway_subgroup"] == subgroup,
        "total_pathway_days"
    ].dropna()

    sim_vals = sim_df.loc[
        sim_df["pathway_subgroup"] == subgroup,
        "total_pathway_days"
    ].dropna()

    if len(real_vals) == 0 or len(sim_vals) == 0:
        print(f"Skipping {subgroup} (insufficient data)")
        return

    xr, yr = ecdf(real_vals)
    xs, ys = ecdf(sim_vals)

    plt.figure()
    plt.plot(xr, yr, label=f"Real (n={len(real_vals)})")
    plt.plot(xs, ys, label=f"Sim (n={len(sim_vals)})")

    plt.xlabel("Total pathway time (days)")
    plt.ylabel("ECDF")
    plt.title(f"ECDF comparison: {subgroup}")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


from pathlib import Path

def plot_all_subgroups(real_df, sim_df, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    subgroups = sorted(
        set(real_df["pathway_subgroup"].dropna())
        | set(sim_df["pathway_subgroup"].dropna())
    )

    for subgroup in subgroups:
        if subgroup in ["other", "reached_biopsy_only", "reached_treatment_mdt"]:
            continue

        safe_name = subgroup.replace(" ", "_")

        save_path = output_dir / f"ecdf_{safe_name}.png"

        plot_ecdf_by_subgroup(
            real_df,
            sim_df,
            subgroup,
            save_path=save_path
        )

        print(f"Saved ECDF plot: {save_path}")

plot_all_subgroups(real_patients, sim_patients, OUTPUT_DIR)