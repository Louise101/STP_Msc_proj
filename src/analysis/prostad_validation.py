from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from analysis.validation import build_real_pathway_csvs, load_real_pathway_data


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "prostad_validation_ecdf_mri3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748

SEEDS = range(1000, 1030)

FULL_PATHWAY_EVENT = "Outpatient_appointment_occured"

STAGE_ORDER = [
    "ref_to_mri",
    "mri_to_report",
    "report_to_biopmdt",
    "biopmdt_to_biopsy",
    "biopsy_to_pathrep",
    "pathrep_to_treatmdt",
    "treatmdt_to_outpat",
]

STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → MRI Report",
    "report_to_biopmdt": "MRI clinic → Biopsy Decision",
    "biopmdt_to_biopsy": "Biopsy Decision → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Pathology",
    "pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "treatmdt_to_outpat": "Treatment MDT → Outpatient",
}

STAGE_ENDPOINT_EVENTS = {
    "ref_to_mri": "mri_performed",
    "mri_to_report": "mri_report_ready",
    "report_to_biopmdt": "MDT_occured",
    "biopmdt_to_biopsy": "biopsy_done",
    "biopsy_to_pathrep": "Path_report_recieved",
    "pathrep_to_treatmdt": "Treatment_options_MDT_occured",
    "treatmdt_to_outpat": "Outpatient_appointment_occured",
}

PROSTAD_REPORT_MEANS = {
    "ref_to_mri": 13.0,
    "mri_to_report": 14.0,
    "report_to_biopmdt": 14.0,
    "biopmdt_to_biopsy": 46.0,
    "biopsy_to_pathrep": 53.0,
    "treatmdt_to_outpat": 70.0,
}


def parse_date_series(series: pd.Series, style: str = "generic") -> pd.Series:
    s = series.astype(str).str.strip()

    if style == "uk":
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

    if style == "us":
        out = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
        missing = out.isna()
        if missing.any():
            out.loc[missing] = pd.to_datetime(
                s.loc[missing],
                dayfirst=False,
                errors="coerce",
            )
        return out

    return pd.to_datetime(s, errors="coerce")


def build_obs_mix_result(seed: int) -> dict:
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    cfg = build_combined_config(
        "OBS_MIX",
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    return run_day_loop_combined_engine(
        cfg,
        daily_referrals_override=referral_schedule,
    )


def extract_stage_waits_from_sim(result: dict, seed: int) -> pd.DataFrame:
    rows: list[dict] = []

    stage_pairs = [
        ("referral_received", "mri_performed", "ref_to_mri"),
        ("mri_performed", "mri_report_ready", "mri_to_report"),
        ("mri_report_ready", "MDT_occured", "report_to_biopmdt"),
        ("MDT_occured", "biopsy_done", "biopmdt_to_biopsy"),
        ("biopsy_done", "Path_report_recieved", "biopsy_to_pathrep"),
        ("Path_report_recieved", "Treatment_options_MDT_occured", "pathrep_to_treatmdt"),
        ("Treatment_options_MDT_occured", "Outpatient_appointment_occured", "treatmdt_to_outpat"),
    ]

    for patient in result["all_patients_objects"]:
        pathway_type = patient.data.get("pathway_type")
        event_dates = {event["event"]: event["date"] for event in patient.events}

        for start_event, end_event, stage_name in stage_pairs:
            if start_event in event_dates and end_event in event_dates:
                wait_days = (event_dates[end_event] - event_dates[start_event]).days
                if wait_days >= 0:
                    rows.append(
                        {
                            "seed": seed,
                            "patient_id": patient.patient_id,
                            "pathway_type": pathway_type,
                            "stage": stage_name,
                            "wait_days": wait_days,
                        }
                    )

    return pd.DataFrame(rows)


def extract_full_pathway_from_sim(result: dict, seed: int) -> pd.DataFrame:
    rows: list[dict] = []

    for patient in result["completed_patients_objects"]:
        pathway_type = patient.data.get("pathway_type")
        event_names = {e["event"] for e in patient.events}

        if FULL_PATHWAY_EVENT not in event_names:
            continue

        rows.append(
            {
                "seed": seed,
                "patient_id": patient.patient_id,
                "pathway_type": pathway_type,
                "total_days": (patient.current_date - patient.start_date).days,
            }
        )

    return pd.DataFrame(rows)


def get_first_event_date(patient, event_name: str):
    dates = [
        event.get("date")
        for event in patient.events
        if event.get("event") == event_name and event.get("date") is not None
    ]
    return min(dates) if dates else None


def extract_time_to_stage_from_sim(result: dict, seed: int) -> pd.DataFrame:
    rows: list[dict] = []

    for patient in result["all_patients_objects"]:
        pathway_type = patient.data.get("pathway_type")
        referral_date = get_first_event_date(patient, "referral_received")

        if referral_date is None:
            continue

        for stage, endpoint_event in STAGE_ENDPOINT_EVENTS.items():
            endpoint_date = get_first_event_date(patient, endpoint_event)

            if endpoint_date is None:
                continue

            time_to_stage = (endpoint_date - referral_date).days

            if time_to_stage >= 0:
                rows.append(
                    {
                        "seed": seed,
                        "patient_id": patient.patient_id,
                        "pathway_type": pathway_type,
                        "stage": stage,
                        "time_to_stage_days": time_to_stage,
                    }
                )

    return pd.DataFrame(rows)


def load_real_prostad_stage_waits(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    specs = {
        "ref_to_mri": ("pros_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "us"),
        "mri_to_report": ("pros_mri_to_mriclin.csv", "Date of MRI", "Date of clinic", "us"),
        "biopmdt_to_biopsy": ("pros_mriclin_to_biop.csv", "Date of clinic", "Date of biopsy", "us"),
        "biopsy_to_pathrep": ("pros_biop_to_pathrep.csv", "Date of biopsy", "Date of pathology report", "us"),
        "pathrep_to_treatmdt": ("pros_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT to discuss treatment options", "us"),
        "treatmdt_to_outpat": ("pros_treatmdt_to_outpat.csv", "Date of MDT to discuss treatment options", "Date of OPD appt", "us"),
    }

    for stage, (filename, start_col, end_col, style) in specs.items():
        df = pd.read_csv(data_dir / filename).copy()
        start_dates = parse_date_series(df[start_col], style)
        end_dates = parse_date_series(df[end_col], style)

        waits = (end_dates - start_dates).dt.days
        waits = waits[(waits.notna()) & (waits >= 0)]

        for wait in waits:
            rows.append({"stage": stage, "wait_days": float(wait)})

    rows.append({"stage": "report_to_biopmdt", "wait_days": np.nan})

    return pd.DataFrame(rows)


def load_real_baseline_stage_waits(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    specs = {
        "ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "uk"),
        "mri_to_report": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported", "uk"),
        "report_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT", "uk"),
        "biopmdt_to_biopsy": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy", "uk"),
        "biopsy_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report", "uk"),
        "pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)", "uk"),
        "treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt", "uk"),
    }

    for stage, (filename, start_col, end_col, style) in specs.items():
        df = pd.read_csv(data_dir / filename).copy()
        start_dates = parse_date_series(df[start_col], style)
        end_dates = parse_date_series(df[end_col], style)

        waits = (end_dates - start_dates).dt.days
        waits = waits[(waits.notna()) & (waits >= 0)]

        for wait in waits:
            rows.append({"stage": stage, "wait_days": float(wait)})

    return pd.DataFrame(rows)


def load_real_prostad_full_pathway(data_dir: Path) -> pd.DataFrame:
    _, real_pros_path = load_real_pathway_data(
        str(data_dir / "pre_pathway.csv"),
        str(data_dir / "pros_pathway.csv"),
    )

    real_pros_path = real_pros_path[
        real_pros_path["total_days"].notna()
        & (real_pros_path["total_days"] >= 0)
    ].copy()

    return real_pros_path[["total_days"]]


def load_real_baseline_full_pathway(data_dir: Path) -> pd.DataFrame:
    real_pre_path, _ = load_real_pathway_data(
        str(data_dir / "pre_pathway.csv"),
        str(data_dir / "pros_pathway.csv"),
    )

    real_pre_path = real_pre_path[
        real_pre_path["total_days"].notna()
        & (real_pre_path["total_days"] >= 0)
    ].copy()

    return real_pre_path[["total_days"]]


def load_real_prostad_time_to_stage(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    ref_mri = pd.read_csv(data_dir / "pros_ref_to_mri.csv").copy()
    ref_mri["patient_id"] = ref_mri["Subject number"]
    ref_mri["referral_date"] = parse_date_series(ref_mri["Date of referral to pathway"], "us")
    ref_mri["mri_date"] = parse_date_series(ref_mri["Date of MRI"], "us")

    clinic = pd.read_csv(data_dir / "pros_mri_to_mriclin.csv").copy()
    clinic["patient_id"] = clinic["Subject number"]
    clinic["clinic_date"] = parse_date_series(clinic["Date of clinic"], "us")

    biopsy = pd.read_csv(data_dir / "pros_mriclin_to_biop.csv").copy()
    biopsy["patient_id"] = biopsy["Subject number"]
    biopsy["biopsy_date"] = parse_date_series(biopsy["Date of biopsy"], "us")

    pathrep = pd.read_csv(data_dir / "pros_biop_to_pathrep.csv").copy()
    pathrep["patient_id"] = pathrep["Subject number"]
    pathrep["pathrep_date"] = parse_date_series(pathrep["Date of pathology report"], "us")

    treatmdt = pd.read_csv(data_dir / "pros_pathrep_to_treatmdt.csv").copy()
    treatmdt["patient_id"] = treatmdt["Subject number"]
    treatmdt["treatmdt_date"] = parse_date_series(
        treatmdt["Date of MDT to discuss treatment options"], "us"
    )

    outpat = pd.read_csv(data_dir / "pros_treatmdt_to_outpat.csv").copy()
    outpat["patient_id"] = outpat["Subject number"]
    outpat["outpat_date"] = parse_date_series(outpat["Date of OPD appt"], "us")

    merged = ref_mri[["patient_id", "referral_date", "mri_date"]]

    stage_date_sources = {
        "ref_to_mri": ("mri_date", merged),
        "mri_to_report": ("clinic_date", merged.merge(clinic[["patient_id", "clinic_date"]], on="patient_id", how="inner")),
        "report_to_biopmdt": ("clinic_date", merged.merge(clinic[["patient_id", "clinic_date"]], on="patient_id", how="inner")),
        "biopmdt_to_biopsy": ("biopsy_date", merged.merge(biopsy[["patient_id", "biopsy_date"]], on="patient_id", how="inner")),
        "biopsy_to_pathrep": ("pathrep_date", merged.merge(pathrep[["patient_id", "pathrep_date"]], on="patient_id", how="inner")),
        "pathrep_to_treatmdt": ("treatmdt_date", merged.merge(treatmdt[["patient_id", "treatmdt_date"]], on="patient_id", how="inner")),
        "treatmdt_to_outpat": ("outpat_date", merged.merge(outpat[["patient_id", "outpat_date"]], on="patient_id", how="inner")),
    }

    for stage, (date_col, df) in stage_date_sources.items():
        waits = (df[date_col] - df["referral_date"]).dt.days
        waits = waits[(waits.notna()) & (waits >= 0)]

        for wait in waits:
            rows.append({"stage": stage, "time_to_stage_days": float(wait)})

    return pd.DataFrame(rows)


def load_real_baseline_time_to_stage(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    ref_mri = pd.read_csv(data_dir / "pre_ref_to_mri.csv").copy()
    ref_mri["patient_id"] = ref_mri["Subject number"]
    ref_mri["referral_date"] = parse_date_series(ref_mri["Date of referral to pathway"], "uk")
    ref_mri["mri_date"] = parse_date_series(ref_mri["Date of MRI"], "uk")

    mri_report = pd.read_csv(data_dir / "pre_mri_to_mrirep.csv").copy()
    mri_report["patient_id"] = mri_report["Subject number"]
    mri_report["report_date"] = parse_date_series(mri_report["Date MRI reported"], "uk")

    biopsy_mdt = pd.read_csv(data_dir / "pre_mrirep_to_biopmdt.csv").copy()
    biopsy_mdt["patient_id"] = biopsy_mdt["Subject number"]
    biopsy_mdt["biopsy_mdt_date"] = parse_date_series(
        biopsy_mdt["Date of Prostate MRI MDT"], "uk"
    )

    biopsy = pd.read_csv(data_dir / "pre_biopmdt_to_biop.csv").copy()
    biopsy["patient_id"] = biopsy["Subject number"]
    biopsy["biopsy_date"] = parse_date_series(biopsy["Date of Biopsy"], "uk")

    pathrep = pd.read_csv(data_dir / "pre_biop_to_pathrep.csv").copy()
    pathrep["patient_id"] = pathrep["Subject number"]
    pathrep["pathrep_date"] = parse_date_series(pathrep["Date of pathology report"], "uk")

    treatmdt = pd.read_csv(data_dir / "pre_pathrep_to_treatmdt.csv").copy()
    treatmdt["patient_id"] = treatmdt["Subject number"]
    treatmdt["treatmdt_date"] = parse_date_series(
        treatmdt["Date of MDT (treatment options)"], "uk"
    )

    outpat = pd.read_csv(data_dir / "pre_treatmdt_to_outpat.csv").copy()
    outpat["patient_id"] = outpat["Subject number"]
    outpat["outpat_date"] = parse_date_series(outpat["Date of outpat appt"], "uk")

    base = ref_mri[["patient_id", "referral_date", "mri_date"]]

    stage_date_sources = {
        "ref_to_mri": ("mri_date", base),
        "mri_to_report": ("report_date", base.merge(mri_report[["patient_id", "report_date"]], on="patient_id", how="inner")),
        "report_to_biopmdt": ("biopsy_mdt_date", base.merge(biopsy_mdt[["patient_id", "biopsy_mdt_date"]], on="patient_id", how="inner")),
        "biopmdt_to_biopsy": ("biopsy_date", base.merge(biopsy[["patient_id", "biopsy_date"]], on="patient_id", how="inner")),
        "biopsy_to_pathrep": ("pathrep_date", base.merge(pathrep[["patient_id", "pathrep_date"]], on="patient_id", how="inner")),
        "pathrep_to_treatmdt": ("treatmdt_date", base.merge(treatmdt[["patient_id", "treatmdt_date"]], on="patient_id", how="inner")),
        "treatmdt_to_outpat": ("outpat_date", base.merge(outpat[["patient_id", "outpat_date"]], on="patient_id", how="inner")),
    }

    for stage, (date_col, df) in stage_date_sources.items():
        waits = (df[date_col] - df["referral_date"]).dt.days
        waits = waits[(waits.notna()) & (waits >= 0)]

        for wait in waits:
            rows.append({"stage": stage, "time_to_stage_days": float(wait)})

    return pd.DataFrame(rows)


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y


def compare_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict:
    sim_series = pd.to_numeric(sim_series, errors="coerce").dropna()
    real_series = pd.to_numeric(real_series, errors="coerce").dropna()

    if len(sim_series) == 0 or len(real_series) == 0:
        return {
            "n_sim": len(sim_series),
            "n_real": len(real_series),
            "mean_sim": np.nan,
            "mean_real": np.nan,
            "median_sim": np.nan,
            "median_real": np.nan,
            "p90_sim": np.nan,
            "p90_real": np.nan,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
        }

    ks_stat, ks_p = ks_2samp(sim_series, real_series)

    return {
        "n_sim": int(len(sim_series)),
        "n_real": int(len(real_series)),
        "mean_sim": float(sim_series.mean()),
        "mean_real": float(real_series.mean()),
        "median_sim": float(sim_series.median()),
        "median_real": float(real_series.median()),
        "p90_sim": float(np.percentile(sim_series, 90)),
        "p90_real": float(np.percentile(real_series, 90)),
        "mean_diff": float(sim_series.mean() - real_series.mean()),
        "median_diff": float(sim_series.median() - real_series.median()),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def compare_mean_difference(group_a: pd.Series, group_b: pd.Series) -> dict:
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        return {
            "difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }

    mean_diff = a.mean() - b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    se = np.sqrt((var_a / len(a)) + (var_b / len(b)))

    df_num = ((var_a / len(a)) + (var_b / len(b))) ** 2
    df_den = ((var_a / len(a)) ** 2 / (len(a) - 1)) + ((var_b / len(b)) ** 2 / (len(b) - 1))
    welch_df = df_num / df_den

    from scipy.stats import t

    t_crit = t.ppf(0.975, welch_df)

    test = ttest_ind(a, b, equal_var=False, nan_policy="omit")

    return {
        "difference": float(mean_diff),
        "ci_low": float(mean_diff - t_crit * se),
        "ci_high": float(mean_diff + t_crit * se),
        "p_value": float(test.pvalue),
    }


def build_table8_mixed_sim_comparison(
    sim_time_to_stage: pd.DataFrame,
    real_pros_time_to_stage: pd.DataFrame,
    real_standard_time_to_stage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    table_stages = [
        "ref_to_mri",
        "mri_to_report",
        "report_to_biopmdt",
        "biopmdt_to_biopsy",
        "biopsy_to_pathrep",
        "treatmdt_to_outpat",
    ]

    table_labels = {
        "ref_to_mri": "Time to MRI",
        "mri_to_report": "Time to MRI reporting",
        "report_to_biopmdt": "Time to clinical decision whether to biopsy",
        "biopmdt_to_biopsy": "Time to biopsy",
        "biopsy_to_pathrep": "Time to diagnosis",
        "treatmdt_to_outpat": "Time to outpatient appointment where patient informed of diagnosis",
    }

    for stage in table_stages:
        real_pros = real_pros_time_to_stage.loc[
            real_pros_time_to_stage["stage"] == stage,
            "time_to_stage_days",
        ]

        real_standard = real_standard_time_to_stage.loc[
            real_standard_time_to_stage["stage"] == stage,
            "time_to_stage_days",
        ]

        sim_pros = sim_time_to_stage.loc[
            (sim_time_to_stage["pathway_type"] == "PROSTAD")
            & (sim_time_to_stage["stage"] == stage),
            "time_to_stage_days",
        ]

        sim_standard = sim_time_to_stage.loc[
            (sim_time_to_stage["pathway_type"] == "BASELINE")
            & (sim_time_to_stage["stage"] == stage),
            "time_to_stage_days",
        ]

        real_diff = compare_mean_difference(real_pros, real_standard)
        sim_diff = compare_mean_difference(sim_pros, sim_standard)

        rows.append(
            {
                "stage": stage,
                "label": table_labels.get(stage, stage),
                "real_prostad_n": int(real_pros.dropna().shape[0]),
                "real_prostad_mean": float(real_pros.mean()) if len(real_pros.dropna()) else np.nan,
                "real_prostad_median": float(real_pros.median()) if len(real_pros.dropna()) else np.nan,
                "real_standard_n": int(real_standard.dropna().shape[0]),
                "real_standard_mean": float(real_standard.mean()) if len(real_standard.dropna()) else np.nan,
                "real_standard_median": float(real_standard.median()) if len(real_standard.dropna()) else np.nan,
                "real_difference_mean": real_diff["difference"],
                "real_difference_ci_low": real_diff["ci_low"],
                "real_difference_ci_high": real_diff["ci_high"],
                "real_difference_p_value": real_diff["p_value"],
                "sim_prostad_n": int(sim_pros.dropna().shape[0]),
                "sim_prostad_mean": float(sim_pros.mean()) if len(sim_pros.dropna()) else np.nan,
                "sim_prostad_median": float(sim_pros.median()) if len(sim_pros.dropna()) else np.nan,
                "sim_standard_n": int(sim_standard.dropna().shape[0]),
                "sim_standard_mean": float(sim_standard.mean()) if len(sim_standard.dropna()) else np.nan,
                "sim_standard_median": float(sim_standard.median()) if len(sim_standard.dropna()) else np.nan,
                "sim_difference_mean": sim_diff["difference"],
                "sim_difference_ci_low": sim_diff["ci_low"],
                "sim_difference_ci_high": sim_diff["ci_high"],
                "sim_difference_p_value": sim_diff["p_value"],
            }
        )

    return pd.DataFrame(rows)


def summarise_seed_level_results(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_summary_df
        .groupby(["level", "stage", "label"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            n_sim_mean=("n_sim", "mean"),
            n_sim_sd=("n_sim", "std"),
            n_real=("n_real", "first"),
            mean_sim_mean=("mean_sim", "mean"),
            mean_sim_sd=("mean_sim", "std"),
            mean_real=("mean_real", "first"),
            median_sim_mean=("median_sim", "mean"),
            median_sim_sd=("median_sim", "std"),
            median_real=("median_real", "first"),
            p90_sim_mean=("p90_sim", "mean"),
            p90_sim_sd=("p90_sim", "std"),
            p90_real=("p90_real", "first"),
            mean_diff_mean=("mean_diff", "mean"),
            mean_diff_sd=("mean_diff", "std"),
            ks_stat_mean=("ks_stat", "mean"),
            ks_stat_sd=("ks_stat", "std"),
            ks_pvalue_median=("ks_pvalue", "median"),
        )
    )


def add_seed_validation_rows(
    seed: int,
    pathway_label: str,
    sim_stage_waits: pd.DataFrame,
    sim_pathway: pd.DataFrame,
    real_stage_waits: pd.DataFrame,
    real_pathway: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for stage in STAGE_ORDER:
        sim_series = sim_stage_waits.loc[
            sim_stage_waits["stage"] == stage,
            "wait_days",
        ]

        real_series = real_stage_waits.loc[
            real_stage_waits["stage"] == stage,
            "wait_days",
        ]

        if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
            continue

        rows.append(
            {
                "seed": seed,
                "validation_group": pathway_label,
                "level": "stage",
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage),
                **compare_distributions(sim_series, real_series),
            }
        )

    pathway_summary = compare_distributions(
        sim_pathway["total_days"],
        real_pathway["total_days"],
    )

    rows.append(
        {
            "seed": seed,
            "validation_group": pathway_label,
            "level": "full_pathway",
            "stage": "full_pathway",
            "label": "Full pathway",
            **pathway_summary,
        }
    )

    return pd.DataFrame(rows)


def plot_ecdf(
    sim_series: pd.Series,
    real_series: pd.Series,
    title: str,
    out_path: Path,
    x_label: str = "Days",
    sim_label: str = "Simulated",
    real_label: str = "Real",
    sim_n_display: float | None = None,
    real_n_display: float | None = None,
) -> None:
    sim_series = pd.to_numeric(sim_series, errors="coerce").dropna()
    real_series = pd.to_numeric(real_series, errors="coerce").dropna()

    if len(sim_series) == 0 or len(real_series) == 0:
        print(f"Skipped ECDF: {title}")
        return

    sx, sy = ecdf(sim_series)
    rx, ry = ecdf(real_series)

    stats = compare_distributions(sim_series, real_series)

    sim_n_text = f"{sim_n_display:.0f}" if sim_n_display is not None else str(stats["n_sim"])
    real_n_text = f"{real_n_display:.0f}" if real_n_display is not None else str(stats["n_real"])

    stats_text = (
        f"n real = {real_n_text}, mean n sim = {sim_n_text}\n"
        f"Mean diff = {stats['mean_diff']:.1f} days\n"
        f"Median diff = {stats['median_diff']:.1f} days\n"
        f"KS stat = {stats['ks_stat']:.3f}\n"
        f"KS p = {stats['ks_pvalue']:.3f}"
    )

    plt.figure(figsize=(8, 5))
    plt.step(rx, ry, where="post", label=f"{real_label} (n={real_n_text})")
    plt.step(sx, sy, where="post", label=f"{sim_label} (mean n={sim_n_text})")
    plt.xlabel(x_label)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend(loc="lower right")

    plt.gcf().text(
        0.70,
        0.5,
        stats_text,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.subplots_adjust(right=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(
    sim_series: pd.Series,
    real_series: pd.Series,
    title: str,
    out_path: Path,
    y_label: str = "Days",
    sim_label: str = "Simulated",
    real_label: str = "Real",
) -> None:
    sim_values = pd.to_numeric(sim_series, errors="coerce").dropna().to_numpy()
    real_values = pd.to_numeric(real_series, errors="coerce").dropna().to_numpy()

    if len(sim_values) == 0 or len(real_values) == 0:
        print(f"Skipped boxplot: {title}")
        return

    stats = compare_distributions(pd.Series(sim_values), pd.Series(real_values))

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [real_values, sim_values],
        labels=[real_label, sim_label],
        showfliers=True,
        flierprops=dict(markersize=3),
    )

    plt.ylabel(y_label)
    plt.title(
        f"{title}\n"
        f"KS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3f}, "
        f"Mean diff={stats['mean_diff']:.1f} days"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_pooled_validation_plots(
    pooled_stage: pd.DataFrame,
    pooled_pathway: pd.DataFrame,
    real_stage_waits: pd.DataFrame,
    real_pathway: pd.DataFrame,
    group_name: str,
    sim_label: str,
    real_label: str,
    filename_tag: str,
) -> None:
    for stage in STAGE_ORDER:
        sim_series = pooled_stage.loc[
            pooled_stage["stage"] == stage,
            "wait_days",
        ]

        real_series = real_stage_waits.loc[
            real_stage_waits["stage"] == stage,
            "wait_days",
        ]

        if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
            continue

        sim_n_mean = round(
            pooled_stage[pooled_stage["stage"] == stage]
            .groupby("seed")["patient_id"]
            .nunique()
            .mean()
        )

        real_n = real_series.dropna().shape[0]

        plot_ecdf(
            sim_series=sim_series,
            real_series=real_series,
            title=f"Pooled multi-seed {group_name} ECDF: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"pooled_multiseed_ecdf_{filename_tag}_{stage}.png",
            x_label="Stage wait days",
            sim_label=sim_label,
            real_label=real_label,
            sim_n_display=sim_n_mean,
            real_n_display=real_n,
        )

        plot_boxplot(
            sim_series=sim_series,
            real_series=real_series,
            title=f"Pooled multi-seed {group_name} boxplot: {STAGE_LABELS.get(stage, stage)}",
            out_path=OUTPUT_DIR / f"pooled_multiseed_boxplot_{filename_tag}_{stage}.png",
            y_label="Stage wait days",
            sim_label=sim_label,
            real_label=real_label,
        )

    sim_pathway_n_mean = round(
        pooled_pathway
        .groupby("seed")["patient_id"]
        .nunique()
        .mean()
    )

    real_pathway_n = real_pathway["total_days"].dropna().shape[0]

    plot_ecdf(
        sim_series=pooled_pathway["total_days"],
        real_series=real_pathway["total_days"],
        title=f"Pooled multi-seed {group_name} ECDF: Full pathway time",
        out_path=OUTPUT_DIR / f"pooled_multiseed_ecdf_{filename_tag}_full_pathway.png",
        x_label="Total pathway days",
        sim_label=sim_label,
        real_label=real_label,
        sim_n_display=sim_pathway_n_mean,
        real_n_display=real_pathway_n,
    )

    plot_boxplot(
        sim_series=pooled_pathway["total_days"],
        real_series=real_pathway["total_days"],
        title=f"Pooled multi-seed {group_name} boxplot: Full pathway time",
        out_path=OUTPUT_DIR / f"pooled_multiseed_boxplot_{filename_tag}_full_pathway.png",
        y_label="Total pathway days",
        sim_label=sim_label,
        real_label=real_label,
    )


def main() -> None:
    build_real_pathway_csvs(
        pre_ref_file=str(DATA_DIR / "pre_ref_to_mri.csv"),
        pre_outpat_file=str(DATA_DIR / "pre_treatmdt_to_outpat.csv"),
        pros_ref_file=str(DATA_DIR / "pros_ref_to_mri.csv"),
        pros_outpat_file=str(DATA_DIR / "pros_treatmdt_to_outpat.csv"),
        out_pre_file=str(DATA_DIR / "pre_pathway.csv"),
        out_pros_file=str(DATA_DIR / "pros_pathway.csv"),
    )

    real_prostad_stage_waits = load_real_prostad_stage_waits(DATA_DIR)
    real_prostad_pathway = load_real_prostad_full_pathway(DATA_DIR)

    real_standard_stage_waits = load_real_baseline_stage_waits(DATA_DIR)
    real_standard_pathway = load_real_baseline_full_pathway(DATA_DIR)

    real_prostad_time_to_stage = load_real_prostad_time_to_stage(DATA_DIR)
    real_standard_time_to_stage = load_real_baseline_time_to_stage(DATA_DIR)

    all_stage_waits: list[pd.DataFrame] = []
    all_pathways: list[pd.DataFrame] = []
    all_time_to_stage: list[pd.DataFrame] = []

    all_prostad_seed_summaries: list[pd.DataFrame] = []
    all_standard_seed_summaries: list[pd.DataFrame] = []
    all_table8_rows: list[pd.DataFrame] = []

    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")

        result = build_obs_mix_result(seed)

        sim_stage_waits_all = extract_stage_waits_from_sim(result, seed)
        sim_pathway_all = extract_full_pathway_from_sim(result, seed)
        sim_time_to_stage = extract_time_to_stage_from_sim(result, seed)

        all_stage_waits.append(sim_stage_waits_all)
        all_pathways.append(sim_pathway_all)
        all_time_to_stage.append(sim_time_to_stage)

        sim_prostad_stage_waits = sim_stage_waits_all[
            sim_stage_waits_all["pathway_type"] == "PROSTAD"
        ].copy()

        sim_prostad_pathway = sim_pathway_all[
            sim_pathway_all["pathway_type"] == "PROSTAD"
        ].copy()

        sim_standard_stage_waits = sim_stage_waits_all[
            sim_stage_waits_all["pathway_type"] == "BASELINE"
        ].copy()

        sim_standard_pathway = sim_pathway_all[
            sim_pathway_all["pathway_type"] == "BASELINE"
        ].copy()

        prostad_seed_df = add_seed_validation_rows(
            seed=seed,
            pathway_label="PROSTAD",
            sim_stage_waits=sim_prostad_stage_waits,
            sim_pathway=sim_prostad_pathway,
            real_stage_waits=real_prostad_stage_waits,
            real_pathway=real_prostad_pathway,
        )
        all_prostad_seed_summaries.append(prostad_seed_df)

        standard_seed_df = add_seed_validation_rows(
            seed=seed,
            pathway_label="STANDARD",
            sim_stage_waits=sim_standard_stage_waits,
            sim_pathway=sim_standard_pathway,
            real_stage_waits=real_standard_stage_waits,
            real_pathway=real_standard_pathway,
        )
        all_standard_seed_summaries.append(standard_seed_df)

        table8_df = build_table8_mixed_sim_comparison(
            sim_time_to_stage=sim_time_to_stage,
            real_pros_time_to_stage=real_prostad_time_to_stage,
            real_standard_time_to_stage=real_standard_time_to_stage,
        )
        table8_df["seed"] = seed
        all_table8_rows.append(table8_df)

    all_stage_waits_df = pd.concat(all_stage_waits, ignore_index=True)
    all_pathway_df = pd.concat(all_pathways, ignore_index=True)
    all_time_to_stage_df = pd.concat(all_time_to_stage, ignore_index=True)

    all_stage_waits_df.to_csv(OUTPUT_DIR / "all_seed_stage_waits.csv", index=False)
    all_pathway_df.to_csv(OUTPUT_DIR / "all_seed_full_pathways.csv", index=False)
    all_time_to_stage_df.to_csv(OUTPUT_DIR / "all_seed_time_to_stage.csv", index=False)

    prostad_seed_summary_df = pd.concat(all_prostad_seed_summaries, ignore_index=True)
    prostad_seed_summary_df.to_csv(
        OUTPUT_DIR / "prostad_validation_seed_level_summary.csv",
        index=False,
    )

    prostad_across_seed_df = summarise_seed_level_results(prostad_seed_summary_df)
    prostad_across_seed_df.to_csv(
        OUTPUT_DIR / "prostad_validation_across_seed_summary.csv",
        index=False,
    )

    standard_seed_summary_df = pd.concat(all_standard_seed_summaries, ignore_index=True)
    standard_seed_summary_df.to_csv(
        OUTPUT_DIR / "standard_validation_seed_level_summary.csv",
        index=False,
    )

    standard_across_seed_df = summarise_seed_level_results(standard_seed_summary_df)
    standard_across_seed_df.to_csv(
        OUTPUT_DIR / "standard_validation_across_seed_summary.csv",
        index=False,
    )

    print("\n=== ACROSS-SEED PROSTAD VALIDATION SUMMARY ===")
    print(
        prostad_across_seed_df[
            [
                "label",
                "n_runs",
                "n_sim_mean",
                "n_sim_sd",
                "n_real",
                "mean_sim_mean",
                "mean_sim_sd",
                "mean_real",
                "median_sim_mean",
                "median_sim_sd",
                "median_real",
                "mean_diff_mean",
                "mean_diff_sd",
                "ks_stat_mean",
                "ks_stat_sd",
                "ks_pvalue_median",
            ]
        ].round(3).to_string(index=False)
    )

    print("\n=== ACROSS-SEED STANDARD VALIDATION SUMMARY ===")
    print(
        standard_across_seed_df[
            [
                "label",
                "n_runs",
                "n_sim_mean",
                "n_sim_sd",
                "n_real",
                "mean_sim_mean",
                "mean_sim_sd",
                "mean_real",
                "median_sim_mean",
                "median_sim_sd",
                "median_real",
                "mean_diff_mean",
                "mean_diff_sd",
                "ks_stat_mean",
                "ks_stat_sd",
                "ks_pvalue_median",
            ]
        ].round(3).to_string(index=False)
    )

    table8_seed_df = pd.concat(all_table8_rows, ignore_index=True)
    table8_seed_df.to_csv(
        OUTPUT_DIR / "table8_mixed_sim_comparison_seed_level.csv",
        index=False,
    )

    table8_across_seed_df = (
        table8_seed_df
        .groupby(["stage", "label"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            real_prostad_n=("real_prostad_n", "first"),
            real_prostad_mean=("real_prostad_mean", "first"),
            real_prostad_median=("real_prostad_median", "first"),
            real_standard_n=("real_standard_n", "first"),
            real_standard_mean=("real_standard_mean", "first"),
            real_standard_median=("real_standard_median", "first"),
            real_difference_mean=("real_difference_mean", "first"),
            real_difference_ci_low=("real_difference_ci_low", "first"),
            real_difference_ci_high=("real_difference_ci_high", "first"),
            real_difference_p_value=("real_difference_p_value", "first"),
            sim_prostad_n_mean=("sim_prostad_n", "mean"),
            sim_prostad_n_sd=("sim_prostad_n", "std"),
            sim_prostad_mean_mean=("sim_prostad_mean", "mean"),
            sim_prostad_mean_sd=("sim_prostad_mean", "std"),
            sim_prostad_median_mean=("sim_prostad_median", "mean"),
            sim_standard_n_mean=("sim_standard_n", "mean"),
            sim_standard_n_sd=("sim_standard_n", "std"),
            sim_standard_mean_mean=("sim_standard_mean", "mean"),
            sim_standard_mean_sd=("sim_standard_mean", "std"),
            sim_standard_median_mean=("sim_standard_median", "mean"),
            sim_difference_mean_mean=("sim_difference_mean", "mean"),
            sim_difference_mean_sd=("sim_difference_mean", "std"),
            sim_difference_ci_low_mean=("sim_difference_ci_low", "mean"),
            sim_difference_ci_high_mean=("sim_difference_ci_high", "mean"),
            sim_difference_p_value_median=("sim_difference_p_value", "median"),
        )
    )

    table8_across_seed_df.to_csv(
        OUTPUT_DIR / "table8_mixed_sim_comparison_across_seed.csv",
        index=False,
    )

    print("\n=== TABLE 8 ACROSS-SEED SUMMARY ===")
    print(
        table8_across_seed_df[
            [
                "label",
                "n_runs",
                "real_prostad_n",
                "real_prostad_mean",
                "real_standard_n",
                "real_standard_mean",
                "sim_prostad_n_mean",
                "sim_prostad_n_sd",
                "sim_prostad_mean_mean",
                "sim_prostad_mean_sd",
                "sim_standard_n_mean",
                "sim_standard_n_sd",
                "sim_standard_mean_mean",
                "sim_standard_mean_sd",
                "sim_difference_mean_mean",
                "sim_difference_mean_sd",
            ]
        ].round(3).to_string(index=False)
    )

    pooled_prostad_stage = all_stage_waits_df[
        all_stage_waits_df["pathway_type"] == "PROSTAD"
    ].copy()

    pooled_prostad_pathway = all_pathway_df[
        all_pathway_df["pathway_type"] == "PROSTAD"
    ].copy()

    pooled_standard_stage = all_stage_waits_df[
        all_stage_waits_df["pathway_type"] == "BASELINE"
    ].copy()

    pooled_standard_pathway = all_pathway_df[
        all_pathway_df["pathway_type"] == "BASELINE"
    ].copy()

    make_pooled_validation_plots(
        pooled_stage=pooled_prostad_stage,
        pooled_pathway=pooled_prostad_pathway,
        real_stage_waits=real_prostad_stage_waits,
        real_pathway=real_prostad_pathway,
        group_name="PROSTAD",
        sim_label="Simulated PROSTAD",
        real_label="Real PROSTAD",
        filename_tag="prostad",
    )

    make_pooled_validation_plots(
        pooled_stage=pooled_standard_stage,
        pooled_pathway=pooled_standard_pathway,
        real_stage_waits=real_standard_stage_waits,
        real_pathway=real_standard_pathway,
        group_name="standard",
        sim_label="Simulated standard",
        real_label="Real standard",
        filename_tag="standard",
    )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()