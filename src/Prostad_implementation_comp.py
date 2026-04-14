from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "prostad_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# EDIT THESE PATHS / FILENAMES TO MATCH YOUR FILES
# ------------------------------------------------------------

PRE_DIR = DATA_DIR / "pre_prostad"
POST_DIR = DATA_DIR / "post_prostad"

# Simulation outputs
#SIM_WAITS_FILE = OUTPUT_DIR.parent / "pros_sim_waits.csv"         # use function below to derive sim waits from events file 
PRE_SIM_EVENTS_FILE = OUTPUT_DIR.parent / "baseline_events.csv"     # optional / useful
POST_SIM_EVENTS_FILE = OUTPUT_DIR.parent / "prostad_events.csv"   


PRE_SIM_WAITS_FILE = OUTPUT_DIR.parent / "base_sim_waits.csv"
POST_SIM_WAITS_FILE = OUTPUT_DIR.parent / "pros_sim_waits.csv"


# ------------------------------------------------------------
# TARGET THRESHOLD
# ------------------------------------------------------------

SCP_TARGET_DAYS = 62


# ============================================================
# STAGE DEFINITIONS
# ============================================================

@dataclass
class StageSpec:
    stage_name: str
    filename: str
    id_col: str
    start_date_col: str
    end_date_col: str
    output_wait_col: str
    output_start_col: str
    output_end_col: str


# ------------------------------------------------------------
# EDIT THESE TO MATCH YOUR REAL CSV STRUCTURE
# ------------------------------------------------------------
PRE_STAGE_SPECS = [
    StageSpec(
        stage_name="ref_to_mri",
        filename=DATA_DIR / "pre_ref_to_mri.csv",
        id_col="Subject number",
        start_date_col="Date of referral to pathway",
        end_date_col="Date of MRI",
        output_wait_col="wait_ref_to_mri",
        output_start_col="date_referral",
        output_end_col="date_mri",
    ),
    StageSpec(
        stage_name="mri_to_report",
        filename=DATA_DIR / "pre_mri_to_mrirep.csv",
        id_col="Subject number",
        start_date_col="Date of MRI",
        end_date_col="Date MRI reported",
        output_wait_col="wait_mri_to_report",
        output_start_col="date_mri",
        output_end_col="date_mri_report",
    ),
    StageSpec(
        stage_name="report_to_biopmdt",
        filename=DATA_DIR / "pre_mrirep_to_biopmdt.csv",
        id_col="Subject number",
        start_date_col="Date MRI reported",
        end_date_col="Date of Prostate MRI MDT",
        output_wait_col="wait_report_to_biopmdt",
        output_start_col="date_mri_report",
        output_end_col="date_biop_mdt",
    ),
    StageSpec(
        stage_name="biopmdt_to_biopsy",
        filename=DATA_DIR / "pre_biopmdt_to_biop.csv",
        id_col="Subject number",
        start_date_col="Date of Prostate MRI MDT",
        end_date_col="Date of Biopsy",
        output_wait_col="wait_biopmdt_to_biopsy",
        output_start_col="date_biop_mdt",
        output_end_col="date_biopsy",
    ),
    StageSpec(
        stage_name="biopsy_to_pathrep",
        filename=DATA_DIR / "pre_biop_to_pathrep.csv",
        id_col="Subject number",
        start_date_col="Date of Biopsy",
        end_date_col="Date of pathology report",
        output_wait_col="wait_biopsy_to_pathrep",
        output_start_col="date_biopsy",
        output_end_col="date_path_report",
    ),
    StageSpec(
        stage_name="pathrep_to_treatmdt",
        filename=DATA_DIR / "pre_pathrep_to_treatmdt.csv",
        id_col="Subject number",
        start_date_col="Date of pathology report",
        end_date_col="Date of MDT (treatment options)",
        output_wait_col="wait_pathrep_to_treatmdt",
        output_start_col="date_path_report",
        output_end_col="date_treat_mdt",
    ),
    StageSpec(
        stage_name="treatmdt_to_outpat",
        filename=DATA_DIR / "pre_treatmdt_to_outpat.csv",
        id_col="Subject number",
        start_date_col="Date of MDT (treatment options)",
        end_date_col="Date of outpat appt",
        output_wait_col="wait_treatmdt_to_outpat",
        output_start_col="date_treat_mdt",
        output_end_col="date_outpatient",
    ),
]



POST_STAGE_SPECS = [
    StageSpec(
        stage_name="ref_to_mri",
        filename=DATA_DIR / "pros_ref_to_mri.csv",
        id_col="Subject number",
        start_date_col="Date of referral to pathway",
        end_date_col="Date of MRI",
        output_wait_col="wait_ref_to_mri",
        output_start_col="date_referral",
        output_end_col="date_mri",
    ),
    StageSpec(
        stage_name="mri_to_report",
        filename=DATA_DIR /"pros_mri_to_mriclin.csv",
        id_col="Subject number",
        start_date_col="Date of MRI",
        end_date_col="Date of clinic",
        output_wait_col="wait_mri_to_report",
        output_start_col="date_mri",
        output_end_col="date_mri_clinic",
    ),
    StageSpec(
        stage_name="report_to_biopmdt",
        filename=DATA_DIR /"pros_mriclin_mri_dec.csv",
        id_col="Subject number",
        start_date_col="Date of clinic",
        end_date_col="Date of clinic_dec",
        output_wait_col="wait_report_to_biopmdt",
        output_start_col="date_mri_report",
        output_end_col="date_biop_mdt",
    ),
    StageSpec(
        stage_name="biopmdt_to_biopsy",
        filename=DATA_DIR /"pros_mriclin_to_biop.csv",
        id_col="Subject number",
        start_date_col="Date of clinic",
        end_date_col="Date of biopsy",
        output_wait_col="wait_biopmdt_to_biopsy",
        output_start_col="date_biop_mdt",
        output_end_col="date_biopsy",
    ),
    StageSpec(
        stage_name="biopsy_to_pathrep",
        filename=DATA_DIR /"pros_biop_to_pathrep.csv",
        id_col="Subject number",
        start_date_col="Date of biopsy",
        end_date_col="Date of pathology report",
        output_wait_col="wait_biopsy_to_pathrep",
        output_start_col="date_biopsy",
        output_end_col="date_path_report",
    ),
    StageSpec(
        stage_name="pathrep_to_treatmdt",
        filename=DATA_DIR /"pros_pathrep_to_treatmdt.csv",
        id_col="Subject number",
        start_date_col="Date of pathology report",
        end_date_col="Date of MDT to discuss treatment options",
        output_wait_col="wait_pathrep_to_treatmdt",
        output_start_col="date_path_report",
        output_end_col="date_treat_mdt",
    ),
    StageSpec(
        stage_name="treatmdt_to_outpat",
        filename=DATA_DIR /"pros_treatmdt_to_outpat.csv",
        id_col="Subject number",
        start_date_col="Date of MDT to discuss treatment options",
        end_date_col="Date of OPD appt",
        output_wait_col="wait_treatmdt_to_outpat",
        output_start_col="date_treat_mdt",
        output_end_col="date_outpatient",
    ),
]


# ------------------------------------------------------------
# REAL OUTCOME FILES
# EDIT THESE IF YOUR OUTCOME DATA IS STORED DIFFERENTLY
# ------------------------------------------------------------

PRE_OUTCOME_CONFIG = {
    "biop_mdt": {
        "filename": DATA_DIR / "pre_bio_dec.csv",
        "id_col": "Subject number",
        "outcome_col": "Outcome code",
    },
    "pathrep": {
        "filename": DATA_DIR / "pre_pathrep_outcome.csv",
        "id_col": "Subject number",
        "outcome_col": "Outcome code",
    },
}

POST_OUTCOME_CONFIG = {
    "biop_mdt": {
        "filename": DATA_DIR / "pros_biop_outcome.csv",
        "id_col": "Subject number",
        "outcome_col": "Outcome code",
    },
    "pathrep": {
        "filename": DATA_DIR / "pros_pathrep_outcome.csv",
        "id_col": "Subject number",
        "outcome_col": "Outcome Code",
    },
}


# ------------------------------------------------------------
# SIM COLUMN MAPPING
# EDIT TO MATCH YOUR SIM OUTPUT
# ------------------------------------------------------------

SIM_STAGE_COLS = {
    "wait_ref_to_mri": "wait_ref_to_mri",
    "wait_mri_to_mrireport": "wait_mri_to_report",
    "wait_mrireport_to_biopmdt": "wait_report_to_biopmdt",
    "wait_biopmdt_to_biopsy": "wait_biopmdt_to_biopsy",
    "wait_biopsy_to_pathreport": "wait_biopsy_to_pathrep",
    "wait_pathrep_to_treatmdt": "wait_pathrep_to_treatmdt",
    "wait_treatmdt_to_outpat": "wait_treatmdt_to_outpat",
}

STAGE_ORDER = [
    "wait_ref_to_mri",
    "wait_mri_to_report",
    "wait_report_to_biopmdt",
    "wait_biopmdt_to_biopsy",
    "wait_biopsy_to_pathrep",
    "wait_pathrep_to_treatmdt",
    "wait_treatmdt_to_outpat",
]


# ============================================================
# HELPERS
# ============================================================

def parse_date_col_pre(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def parse_date_col_post(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarise_series(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def ecdf_values(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna().sort_values().to_numpy()
    if len(x) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ============================================================
# LOAD REAL DATA
# ============================================================

def load_stage_file(data_dir: Path, spec: StageSpec, date_parser) -> pd.DataFrame:
    fp = data_dir / spec.filename
    df = pd.read_csv(fp)

    df = df.rename(columns={spec.id_col: "patient_id"}).copy()
    df["patient_id"] = df["patient_id"].astype(str)

    df[spec.output_start_col] = date_parser(df[spec.start_date_col])
    df[spec.output_end_col] = date_parser(df[spec.end_date_col])

    df[spec.output_wait_col] = (
        df[spec.output_end_col] - df[spec.output_start_col]
    ).dt.days

    keep_cols = ["patient_id", spec.output_start_col, spec.output_end_col, spec.output_wait_col]
    return df[keep_cols].drop_duplicates(subset=["patient_id"])


def build_real_patient_table(
    data_dir: Path,
    specs: list[StageSpec],
    label: str,
    date_parser
) -> pd.DataFrame:
    merged = None

    for spec in specs:
        stage_df = load_stage_file(data_dir, spec, date_parser)
        if merged is None:
            merged = stage_df.copy()
        else:
            merged = merged.merge(stage_df, on="patient_id", how="outer")

    if merged is None:
        raise ValueError(f"No stage files loaded for {label}")

    merged["dataset"] = label

    merged["had_biopsy"] = merged["wait_biopmdt_to_biopsy"].notna().astype(int)
    merged["had_pathrep"] = merged["wait_biopsy_to_pathrep"].notna().astype(int)
    merged["had_treat_mdt"] = merged["wait_pathrep_to_treatmdt"].notna().astype(int)
    merged["had_outpatient"] = merged["wait_treatmdt_to_outpat"].notna().astype(int)

    merged["total_time_to_biopsy"] = merged[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
    ]].sum(axis=1, min_count=1)

    merged["total_time_to_pathrep"] = merged[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
    ]].sum(axis=1, min_count=1)

    merged["total_time_to_treatmdt"] = merged[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
        "wait_pathrep_to_treatmdt",
    ]].sum(axis=1, min_count=1)

    merged["total_time_to_outpatient"] = merged[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
        "wait_pathrep_to_treatmdt",
        "wait_treatmdt_to_outpat",
    ]].sum(axis=1, min_count=1)

    return merged


def load_real_outcomes(base_dir: Path, outcome_config: dict) -> pd.DataFrame:
    out = None

    for key, cfg in outcome_config.items():
        fp = base_dir / cfg["filename"]
        if not fp.exists():
            print(f"[INFO] Outcome file not found, skipping: {fp}")
            continue

        df = pd.read_csv(fp).rename(columns={cfg["id_col"]: "patient_id"}).copy()
        df["patient_id"] = df["patient_id"].astype(str)

        keep = df[["patient_id", cfg["outcome_col"]]].drop_duplicates(subset=["patient_id"])

        if out is None:
            out = keep
        else:
            out = out.merge(keep, on="patient_id", how="outer")

    if out is None:
        out = pd.DataFrame(columns=["patient_id"])

    return out

def build_real_dataset(
    data_dir: Path,
    specs: list[StageSpec],
    label: str,
    outcome_config: dict,
    date_parser
) -> pd.DataFrame:
    patient_df = build_real_patient_table(data_dir, specs, label, date_parser)
    outcome_df = load_real_outcomes(data_dir, outcome_config)
    return patient_df.merge(outcome_df, on="patient_id", how="left")


# ============================================================
# LOAD SIM DATA
# ============================================================

def build_sim_dataset(sim_waits_file: Path, label: str = "sim_post") -> pd.DataFrame:
    sim_df = pd.read_csv(sim_waits_file).copy()

    if "patient_id" not in sim_df.columns:
        raise ValueError("Simulation waits file must contain 'patient_id'")

    sim_df["patient_id"] = sim_df["patient_id"].astype(str)

    # Rename sim columns into same names as real data
    sim_df = sim_df.rename(columns=SIM_STAGE_COLS)

    missing_cols = [c for c in STAGE_ORDER if c not in sim_df.columns]
    if missing_cols:
        raise ValueError(f"Simulation waits file missing expected columns: {missing_cols}")

    sim_df["dataset"] = label

    sim_df["had_biopsy"] = sim_df["wait_biopmdt_to_biopsy"].notna().astype(int)
    sim_df["had_pathrep"] = sim_df["wait_biopsy_to_pathrep"].notna().astype(int)
    sim_df["had_treat_mdt"] = sim_df["wait_pathrep_to_treatmdt"].notna().astype(int)
    sim_df["had_outpatient"] = sim_df["wait_treatmdt_to_outpat"].notna().astype(int)

    sim_df["total_time_to_biopsy"] = sim_df[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
    ]].sum(axis=1, min_count=1)

    sim_df["total_time_to_pathrep"] = sim_df[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
    ]].sum(axis=1, min_count=1)

    sim_df["total_time_to_treatmdt"] = sim_df[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
        "wait_pathrep_to_treatmdt",
    ]].sum(axis=1, min_count=1)

    sim_df["total_time_to_outpatient"] = sim_df[[
        "wait_ref_to_mri",
        "wait_mri_to_report",
        "wait_report_to_biopmdt",
        "wait_biopmdt_to_biopsy",
        "wait_biopsy_to_pathrep",
        "wait_pathrep_to_treatmdt",
        "wait_treatmdt_to_outpat",
    ]].sum(axis=1, min_count=1)

    return sim_df


# ============================================================
# STAGE-LEVEL COMPARISON
# ============================================================

def make_stage_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        stats = summarise_series(df[stage])
        row = {"dataset": dataset_name, "stage": stage}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_stage_summaries_fourway(
    pre_real_df: pd.DataFrame,
    pre_sim_df: pd.DataFrame,
    post_real_df: pd.DataFrame,
    post_sim_df: pd.DataFrame,
) -> pd.DataFrame:
    pre_real_sum = make_stage_summary(pre_real_df, "pre_real")
    pre_sim_sum = make_stage_summary(pre_sim_df, "pre_sim")
    post_real_sum = make_stage_summary(post_real_df, "post_real")
    post_sim_sum = make_stage_summary(post_sim_df, "post_sim")

    summary = (
        pre_real_sum.rename(columns=lambda c: f"{c}_pre_real" if c != "stage" else c)
        .merge(
            pre_sim_sum.rename(columns=lambda c: f"{c}_pre_sim" if c != "stage" else c),
            on="stage",
            how="outer",
        )
        .merge(
            post_real_sum.rename(columns=lambda c: f"{c}_post_real" if c != "stage" else c),
            on="stage",
            how="outer",
        )
        .merge(
            post_sim_sum.rename(columns=lambda c: f"{c}_post_sim" if c != "stage" else c),
            on="stage",
            how="outer",
        )
    )

    # Real change
    summary["real_change_mean"] = summary["mean_post_real"] - summary["mean_pre_real"]
    summary["real_change_p90"] = summary["p90_post_real"] - summary["p90_pre_real"]

    # Simulated change
    summary["sim_change_mean"] = summary["mean_post_sim"] - summary["mean_pre_sim"]
    summary["sim_change_p90"] = summary["p90_post_sim"] - summary["p90_pre_sim"]

    # How well simulation predicted the change
    summary["difference_in_change_mean"] = summary["sim_change_mean"] - summary["real_change_mean"]
    summary["difference_in_change_p90"] = summary["sim_change_p90"] - summary["real_change_p90"]

    # Baseline validation
    summary["baseline_sim_minus_real_mean"] = summary["mean_pre_sim"] - summary["mean_pre_real"]
    summary["baseline_sim_minus_real_p90"] = summary["p90_pre_sim"] - summary["p90_pre_real"]

    # Post validation
    summary["post_sim_minus_real_mean"] = summary["mean_post_sim"] - summary["mean_post_real"]
    summary["post_sim_minus_real_p90"] = summary["p90_post_sim"] - summary["p90_post_real"]

    return summary


# ============================================================
# KS TESTS
# ============================================================

def build_ks_table_fourway(
    pre_real_df: pd.DataFrame,
    pre_sim_df: pd.DataFrame,
    post_real_df: pd.DataFrame,
    post_sim_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    comparisons = [
        ("pre_real_vs_pre_sim", pre_real_df, pre_sim_df),
        ("post_real_vs_post_sim", post_real_df, post_sim_df),
        ("pre_real_vs_post_real", pre_real_df, post_real_df),
        ("pre_sim_vs_post_sim", pre_sim_df, post_sim_df),
    ]

    for stage in STAGE_ORDER:
        for name, df1, df2 in comparisons:
            vals1 = safe_numeric(df1[stage]).dropna()
            vals2 = safe_numeric(df2[stage]).dropna()

            if len(vals1) > 0 and len(vals2) > 0:
                ks = ks_2samp(vals1, vals2)
                rows.append({
                    "comparison": name,
                    "stage": stage,
                    "n1": len(vals1),
                    "n2": len(vals2),
                    "ks_statistic": ks.statistic,
                    "ks_pvalue": ks.pvalue,
                })

    return pd.DataFrame(rows)

# ============================================================
# PATHWAY SUMMARY
# ============================================================

TOTAL_COLS = [
    "total_time_to_biopsy",
    "total_time_to_pathrep",
    "total_time_to_treatmdt",
    "total_time_to_outpatient",
]

def build_total_pathway_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for col in TOTAL_COLS:
        stats = summarise_series(df[col])
        row = {"dataset": dataset_name, "measure": col}
        row.update(stats)

        # SCP proportion only really makes sense where a full pathway or
        # equivalent milestone is defined; still useful to calculate generally
        vals = safe_numeric(df[col]).dropna()
        row["prop_within_62_days"] = float((vals <= SCP_TARGET_DAYS).mean()) if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# OUTCOME COMPARISON
# ============================================================

def outcome_proportions(df: pd.DataFrame, dataset_name: str, outcome_col: str) -> pd.DataFrame:
    if outcome_col not in df.columns:
        return pd.DataFrame(columns=["dataset", "outcome_col", "outcome", "n", "prop"])

    vc = df[outcome_col].value_counts(dropna=False)
    total = vc.sum()

    out = pd.DataFrame({
        "dataset": dataset_name,
        "outcome_col": outcome_col,
        "outcome": vc.index.astype(str),
        "n": vc.values,
        "prop": vc.values / total if total else np.nan,
    })
    return out


# ============================================================
# BOTTLENECK TABLE
# ============================================================

def build_bottleneck_table(stage_summary: pd.DataFrame) -> pd.DataFrame:
    bottlenecks = stage_summary[[
        "stage",
        "mean_pre", "mean_post", "mean_sim",
        "p90_pre", "p90_post", "p90_sim",
        "real_change_mean", "sim_change_mean",
        "real_change_p90", "sim_change_p90",
        "difference_in_change_mean", "difference_in_change_p90",
    ]].copy()

    # A simple ranking by post / sim p90 as proxy for bottleneck severity
    bottlenecks["rank_post_p90"] = bottlenecks["p90_post"].rank(ascending=False, method="min")
    bottlenecks["rank_sim_p90"] = bottlenecks["p90_sim"].rank(ascending=False, method="min")

    bottlenecks["real_bottleneck_flag"] = bottlenecks["rank_post_p90"] <= 2
    bottlenecks["sim_bottleneck_flag"] = bottlenecks["rank_sim_p90"] <= 2

    return bottlenecks.sort_values("p90_post", ascending=False)


# ============================================================
# PLOTTING
# ============================================================

def plot_stage_means_fourway(stage_summary: pd.DataFrame, out_file: Path):
    x = np.arange(len(stage_summary))
    width = 0.2

    plt.figure(figsize=(13, 6))
    plt.bar(x - 1.5 * width, stage_summary["mean_pre_real"], width=width, label="Pre real")
    plt.bar(x - 0.5 * width, stage_summary["mean_pre_sim"], width=width, label="Pre sim")
    plt.bar(x + 0.5 * width, stage_summary["mean_post_real"], width=width, label="Post real")
    plt.bar(x + 1.5 * width, stage_summary["mean_post_sim"], width=width, label="Post sim")

    plt.xticks(x, stage_summary["stage"], rotation=45, ha="right")
    plt.ylabel("Mean wait (days)")
    plt.title("Stage mean waits: real vs simulation, pre vs post")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_stage_p90(stage_summary: pd.DataFrame, out_file: Path):
    plot_df = stage_summary.copy()

    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["p90_pre"], width=width, label="Pre-PROSTAD real")
    plt.bar(x, plot_df["p90_post"], width=width, label="Post-PROSTAD real")
    plt.bar(x + width, plot_df["p90_sim"], width=width, label="Simulated PROSTAD")

    plt.xticks(x, plot_df["stage"], rotation=45, ha="right")
    plt.ylabel("90th percentile wait (days)")
    plt.title("Tail waits by stage (bottleneck view)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_ecdf_fourway(pre_real_df, pre_sim_df, post_real_df, post_sim_df, col, out_file):
    plt.figure(figsize=(8, 5))

    for label, df in [
        ("Pre real", pre_real_df),
        ("Pre sim", pre_sim_df),
        ("Post real", post_real_df),
        ("Post sim", post_sim_df),
    ]:
        x, y = ecdf_values(df[col])
        if len(x) > 0:
            plt.step(x, y, where="post", label=label)

    plt.xlabel("Days")
    plt.ylabel("ECDF")
    plt.title(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_scp_compliance(pre_total: pd.DataFrame, post_total: pd.DataFrame, sim_total: pd.DataFrame, out_file: Path):
    def get_prop(df: pd.DataFrame, measure: str) -> float:
        row = df.loc[df["measure"] == measure]
        if row.empty:
            return np.nan
        return float(row["prop_within_62_days"].iloc[0])

    measure = "total_time_to_outpatient"

    values = [
        get_prop(pre_total, measure),
        get_prop(post_total, measure),
        get_prop(sim_total, measure),
    ]
    labels = ["Pre-PROSTAD real", "Post-PROSTAD real", "Simulated PROSTAD"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel(f"Proportion within {SCP_TARGET_DAYS} days")
    plt.title("SCP compliance comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# ============================================================
# OPTIONAL REAL FLOW PROXY
# ============================================================

def build_stage_entry_counts(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Approximate flow by counting how many patients enter each stage on each date.
    Useful for spotting bunching/backlogs in the real post-PROSTAD data.
    """
    mapping = [
        ("ref_to_mri", "date_referral"),
        ("mri_to_report", "date_mri"),
        ("report_to_biopmdt", "date_mri_report"),
        ("biopmdt_to_biopsy", "date_biop_mdt"),
        ("biopsy_to_pathrep", "date_biopsy"),
        ("pathrep_to_treatmdt", "date_path_report"),
        ("treatmdt_to_outpat", "date_treat_mdt"),
    ]

    rows = []
    for stage_name, date_col in mapping:
        if date_col not in df.columns:
            continue

        tmp = df[["patient_id", date_col]].dropna().copy()
        if tmp.empty:
            continue

        counts = tmp.groupby(date_col).size().reset_index(name="n_patients")
        counts = counts.rename(columns={date_col: "stage_entry_date"})
        counts["stage_name"] = stage_name
        counts["dataset"] = dataset_name
        rows.append(counts)

    if not rows:
        return pd.DataFrame(columns=["stage_entry_date", "n_patients", "stage_name", "dataset"])

    return pd.concat(rows, ignore_index=True)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading datasets...")

    pre_real_df = build_real_dataset(
        DATA_DIR,
        PRE_STAGE_SPECS,
        "pre_real",
        PRE_OUTCOME_CONFIG,
        parse_date_col_pre,
    )

    post_real_df = build_real_dataset(
        DATA_DIR,
        POST_STAGE_SPECS,
        "post_real",
        POST_OUTCOME_CONFIG,
        parse_date_col_post,
    )

    pre_sim_df = build_sim_dataset(PRE_SIM_WAITS_FILE, "pre_sim")
    post_sim_df = build_sim_dataset(POST_SIM_WAITS_FILE, "post_sim")

    stage_summary = compare_stage_summaries_fourway(
        pre_real_df,
        pre_sim_df,
        post_real_df,
        post_sim_df,
    )
    stage_summary.to_csv(OUTPUT_DIR / "stage_summary_fourway.csv", index=False)

    ks_table = build_ks_table_fourway(
        pre_real_df,
        pre_sim_df,
        post_real_df,
        post_sim_df,
    )
    ks_table.to_csv(OUTPUT_DIR / "stage_ks_fourway.csv", index=False)

    pre_real_total = build_total_pathway_summary(pre_real_df, "pre_real")
    pre_sim_total = build_total_pathway_summary(pre_sim_df, "pre_sim")
    post_real_total = build_total_pathway_summary(post_real_df, "post_real")
    post_sim_total = build_total_pathway_summary(post_sim_df, "post_sim")

    total_summary = pd.concat(
        [pre_real_total, pre_sim_total, post_real_total, post_sim_total],
        ignore_index=True,
    )
    total_summary.to_csv(OUTPUT_DIR / "total_pathway_summary_fourway.csv", index=False)

    plot_stage_means_fourway(stage_summary, OUTPUT_DIR / "stage_mean_fourway.png")

    for col in STAGE_ORDER + TOTAL_COLS:
        plot_ecdf_fourway(
            pre_real_df,
            pre_sim_df,
            post_real_df,
            post_sim_df,
            col,
            OUTPUT_DIR / f"ecdf_fourway_{col}.png",
        )

    print("Done.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    df = pd.read_csv(DATA_DIR / "pros_ref_to_mri.csv")

    df["start"] = pd.to_datetime(df["Date of referral to pathway"], format="%m/%d/%y", errors="coerce")
    df["end"] = pd.to_datetime(df["Date of MRI"], format="%m/%d/%y", errors="coerce")
    df["wait_days"] = (df["end"] - df["start"]).dt.days

    print(df[["Date of referral to pathway", "Date of MRI", "start", "end", "wait_days"]].head(20))
    print("Negative waits:", (df["wait_days"] < 0).sum())
    print("Min wait:", df["wait_days"].min())
    print("Median wait:", df["wait_days"].median())


if __name__ == "__main__":
    main()