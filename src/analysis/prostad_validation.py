from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from analysis.metrics import extract_full_pathway_lengths, extract_stage_waits
from data_prep.empirical_inputs import REAL_STAGE_SPECS


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "prostad_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATHWAY_FILES = {
    "pre": ("pre_pathway.csv", "uk"),
    "pros": ("pros_pathway.csv", "generic"),
}

PROSTAD_REPORT_OVERALL = {
    "PROSTAD_mean_time_to_mri": 13.0,
    "PROSTAD_mean_time_to_mri_reporting": 14.0,
    "PROSTAD_mean_time_to_biopsy_decision": 14.0,
    "PROSTAD_mean_time_to_biopsy": 46.0,
    "PROSTAD_mean_time_to_diagnosis": 53.0,
    "PROSTAD_mean_time_to_outpatient_diagnosis": 70.0,
    "USUAL_mean_time_to_mri": 25.0,
    "USUAL_mean_time_to_mri_reporting": 33.0,
    "USUAL_mean_time_to_biopsy_decision": 38.0,
    "USUAL_mean_time_to_biopsy": 66.0,
    "USUAL_mean_time_to_diagnosis": 76.0,
    "USUAL_mean_time_to_outpatient_diagnosis": 98.0,
}

CUMULATIVE_METRIC_LABELS = {
    "time_to_mri": "Time to MRI",
    "time_to_mri_reporting": "Time to MRI reporting",
    "time_to_biopsy_decision": "Time to biopsy decision",
    "time_to_biopsy": "Time to biopsy",
    "time_to_diagnosis": "Time to diagnosis",
    "time_to_outpatient_diagnosis": "Time to outpatient diagnosis",
}

STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "mri_to_decision_combined": "MRI → clinic/decision (combined)",
    "biopmdt_to_biopsy": "Biopsy MDT/clinic → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Pathology",
    "pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "treatmdt_to_outpat": "Treatment MDT → Outpatient",
}


def parse_date_series(series: pd.Series, style: str) -> pd.Series:
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


def compare_series(sim: pd.Series, real: pd.Series) -> dict:
    sim = pd.to_numeric(sim, errors="coerce").dropna()
    real = pd.to_numeric(real, errors="coerce").dropna()

    if len(sim) == 0 or len(real) == 0:
        return {
            "n_sim": len(sim),
            "n_real": len(real),
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

    ks = ks_2samp(sim, real)

    return {
        "n_sim": int(len(sim)),
        "n_real": int(len(real)),
        "mean_sim": float(sim.mean()),
        "mean_real": float(real.mean()),
        "median_sim": float(sim.median()),
        "median_real": float(real.median()),
        "p90_sim": float(np.percentile(sim, 90)),
        "p90_real": float(np.percentile(real, 90)),
        "mean_diff": float(sim.mean() - real.mean()),
        "median_diff": float(sim.median() - real.median()),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def read_pros_mri_to_mriclin(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "pros_mri_to_mriclin.csv").copy()
    df = df.rename(columns={"Subject number": "patient_id"})

    required = ["patient_id", "Date of MRI", "Date of clinic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"pros_mri_to_mriclin.csv is missing expected columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[required].copy()


def load_real_stage_waits(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for dataset, specs in REAL_STAGE_SPECS.items():
        for stage, (fname, col1, col2, style) in specs.items():
            if dataset == "pros" and stage == "mri_to_report":
                df = read_pros_mri_to_mriclin(data_dir)
                d1 = parse_date_series(df["Date of MRI"], "us")
                d2 = parse_date_series(df["Date of clinic"], "us")
                waits = (d2 - d1).dt.days
                waits = waits[(waits.notna()) & (waits >= 0)]

                for w in waits:
                    rows.append(
                        {
                            "source_dataset": dataset,
                            "stage": "mri_to_decision_combined",
                            "wait_days": float(w),
                        }
                    )
                continue

            df = pd.read_csv(data_dir / fname).copy()
            d1 = parse_date_series(df[col1], style)
            d2 = parse_date_series(df[col2], style)
            waits = (d2 - d1).dt.days
            waits = waits[(waits.notna()) & (waits >= 0)]

            for w in waits:
                rows.append(
                    {
                        "source_dataset": dataset,
                        "stage": stage,
                        "wait_days": float(w),
                    }
                )

    return pd.DataFrame(rows)


def load_real_pathway_lengths(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for dataset, (fname, style) in PATHWAY_FILES.items():
        df = pd.read_csv(data_dir / fname).copy()
        df["referral_date"] = parse_date_series(df["referral_date"], style)
        df["outpatient_date"] = parse_date_series(df["outpatient_date"], style)
        df["total_days"] = (df["outpatient_date"] - df["referral_date"]).dt.days
        df = df[(df["total_days"].notna()) & (df["total_days"] >= 0)].copy()
        df["source_dataset"] = dataset
        frames.append(df[["source_dataset", "total_days"]])

    return pd.concat(frames, ignore_index=True)


def extract_stage_waits_with_pathway_type(
    result: dict,
    scenario_name: str,
    seed: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    event_pair_to_stage = {
        ("referral_received", "mri_performed"): "ref_to_mri",
        ("mri_performed", "mri_report_ready"): "mri_to_report",
        ("mri_report_ready", "MDT_occured"): "report_to_biopmdt",
        ("MDT_occured", "biopsy_done"): "biopmdt_to_biopsy",
        ("biopsy_done", "Path_report_recieved"): "biopsy_to_pathrep",
        ("Path_report_recieved", "Treatment_options_MDT_occured"): "pathrep_to_treatmdt",
        ("Treatment_options_MDT_occured", "Outpatient_appointment_occured"): "treatmdt_to_outpat",
    }

    for patient in result.get("completed_patients_objects", []):
        events = sorted(patient.events, key=lambda x: x["date"])
        pathway_type = patient.data.get("pathway_type")

        for i in range(len(events) - 1):
            e1 = events[i]
            e2 = events[i + 1]
            stage = event_pair_to_stage.get((e1["event"], e2["event"]))
            if stage is None:
                continue

            rows.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    "patient_id": patient.patient_id,
                    "pathway_type": pathway_type,
                    "stage": stage,
                    "wait_days": (e2["date"] - e1["date"]).days,
                }
            )

    return pd.DataFrame(rows)


def add_combined_prostad_stage(sim_stage_df: pd.DataFrame) -> pd.DataFrame:
    if sim_stage_df.empty:
        return sim_stage_df

    prostad = sim_stage_df[sim_stage_df["pathway_type"] == "PROSTAD"].copy()
    non_prostad = sim_stage_df[sim_stage_df["pathway_type"] != "PROSTAD"].copy()

    mri_report = prostad[prostad["stage"] == "mri_to_report"].copy()
    report_mdt = prostad[prostad["stage"] == "report_to_biopmdt"].copy()

    combined = mri_report.merge(
        report_mdt,
        on=["scenario", "seed", "patient_id", "pathway_type"],
        suffixes=("_mri", "_mdt"),
    )

    if not combined.empty:
        combined["stage"] = "mri_to_decision_combined"
        combined["wait_days"] = combined["wait_days_mri"] + combined["wait_days_mdt"]
        combined = combined[
            ["scenario", "seed", "patient_id", "pathway_type", "stage", "wait_days"]
        ]

    # Drop the two split stages for PROSTAD, replace with combined version
    prostad = prostad[~prostad["stage"].isin(["mri_to_report", "report_to_biopmdt"])]

    out = pd.concat([non_prostad, prostad, combined], ignore_index=True)
    return out


def extract_full_pathway_lengths_with_pathway_type(
    result: dict,
    scenario_name: str,
    seed: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    for patient in result.get("completed_patients_objects", []):
        event_names = {e["event"] for e in patient.events}
        if "Outpatient_appointment_occured" not in event_names:
            continue

        rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "patient_id": patient.patient_id,
                "pathway_type": patient.data.get("pathway_type"),
                "total_days": (patient.current_date - patient.start_date).days,
            }
        )

    return pd.DataFrame(rows)


def make_prostad_decision_to_biopsy_ecdf(obs_mix_result: dict, data_dir: Path) -> None:
    """
    ECDF for PROSTAD patients only:
      simulated MDT/clinic decision -> biopsy
      vs observed PROSTAD clinic -> biopsy.
    """

    # Simulated PROSTAD: MDT_occured -> biopsy_done
    sim_stage = extract_stage_waits_with_pathway_type(obs_mix_result, "OBS_MIX")

    sim_series = sim_stage.loc[
        (sim_stage["pathway_type"] == "PROSTAD")
        & (sim_stage["stage"] == "biopmdt_to_biopsy"),
        "wait_days",
    ]

    # Observed PROSTAD: clinic -> biopsy
    real_df = pd.read_csv(data_dir / "pros_mriclin_to_biop.csv").copy()

    real_df["clinic_date"] = parse_date_series(real_df["Date of clinic"], "us")
    real_df["biopsy_date"] = parse_date_series(real_df["Date of biopsy"], "us")

    real_series = (real_df["biopsy_date"] - real_df["clinic_date"]).dt.days
    real_series = real_series[(real_series.notna()) & (real_series >= 0)]

    print("\n=== PROSTAD DECISION/CLINIC → BIOPSY ECDF CHECK ===")
    print(f"Sim n:  {len(sim_series.dropna())}")
    print(f"Real n: {len(real_series.dropna())}")
    print(f"Sim mean:  {sim_series.mean():.3f}")
    print(f"Real mean: {real_series.mean():.3f}")

    output_path = OUTPUT_DIR / "ecdf_mixed_prostad_decision_to_biopsy.png"

    plot_ecdf(
        real=real_series,
        sim=sim_series,
        title="PROSTAD pathway in mixed simulation vs PROSTAD data: clinic/decision → biopsy",
        xlab="Wait days",
        output_path=output_path,
    )


def extract_cumulative_times_with_pathway_type(
    result: dict,
    scenario_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    milestone_map = {
        "mri_performed": "time_to_mri",
        "mri_report_ready": "time_to_mri_reporting",
        "MDT_occured": "time_to_biopsy_decision",
        "biopsy_done": "time_to_biopsy",
        "Path_report_recieved": "time_to_diagnosis",
        "Outpatient_appointment_occured": "time_to_outpatient_diagnosis",
    }

    for patient in result.get("completed_patients_objects", []):
        referral_date = patient.start_date
        pathway_type = patient.data.get("pathway_type")

        for event in patient.events:
            event_name = event.get("event")
            event_date = event.get("date")

            if event_name in milestone_map and event_date is not None:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "patient_id": patient.patient_id,
                        "pathway_type": pathway_type,
                        "metric": milestone_map[event_name],
                        "days_from_referral": (event_date - referral_date).days,
                    }
                )

    return pd.DataFrame(rows)


def calculate_observed_pre_cumulative_values(data_dir: Path) -> dict[str, float]:
    observed: dict[str, float] = {}

    ref_mri = pd.read_csv(data_dir / "pre_ref_to_mri.csv").copy()
    ref_mri = ref_mri.rename(columns={"Subject number": "patient_id"})
    ref_mri["referral_date"] = parse_date_series(ref_mri["Date of referral to pathway"], "uk")
    ref_mri["mri_date"] = parse_date_series(ref_mri["Date of MRI"], "uk")

    mri_rep = pd.read_csv(data_dir / "pre_mri_to_mrirep.csv").copy()
    mri_rep = mri_rep.rename(columns={"Subject number": "patient_id"})
    mri_rep["report_date"] = parse_date_series(mri_rep["Date MRI reported"], "uk")

    rep_mdt = pd.read_csv(data_dir / "pre_mrirep_to_biopmdt.csv").copy()
    rep_mdt = rep_mdt.rename(columns={"Subject number": "patient_id"})
    rep_mdt["mdt_date"] = parse_date_series(rep_mdt["Date of Prostate MRI MDT"], "uk")

    mdt_biop = pd.read_csv(data_dir / "pre_biopmdt_to_biop.csv").copy()
    mdt_biop = mdt_biop.rename(columns={"Subject number": "patient_id"})
    mdt_biop["biopsy_date"] = parse_date_series(mdt_biop["Date of Biopsy"], "uk")

    biop_path = pd.read_csv(data_dir / "pre_biop_to_pathrep.csv").copy()
    biop_path = biop_path.rename(columns={"Subject number": "patient_id"})
    biop_path["pathrep_date"] = parse_date_series(biop_path["Date of pathology report"], "uk")

    pathway = pd.read_csv(data_dir / "pre_pathway.csv").copy()
    pathway["referral_date"] = parse_date_series(pathway["referral_date"], "uk")
    pathway["outpatient_date"] = parse_date_series(pathway["outpatient_date"], "uk")

    waits = (ref_mri["mri_date"] - ref_mri["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_mri"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        mri_rep[["patient_id", "report_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["report_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_mri_reporting"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        rep_mdt[["patient_id", "mdt_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["mdt_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_biopsy_decision"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        mdt_biop[["patient_id", "biopsy_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["biopsy_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_biopsy"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        biop_path[["patient_id", "pathrep_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["pathrep_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_diagnosis"] = float(waits.mean()) if len(waits) else np.nan

    waits = (pathway["outpatient_date"] - pathway["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_outpatient_diagnosis"] = float(waits.mean()) if len(waits) else np.nan

    return observed


def calculate_observed_prostad_cumulative_values(data_dir: Path) -> dict[str, float]:
    observed: dict[str, float] = {}

    ref_mri = pd.read_csv(data_dir / "pros_ref_to_mri.csv").copy()
    ref_mri = ref_mri.rename(columns={"Subject number": "patient_id"})
    ref_mri["referral_date"] = parse_date_series(ref_mri["Date of referral to pathway"], "us")
    ref_mri["mri_date"] = parse_date_series(ref_mri["Date of MRI"], "us")

    mri_clinic = read_pros_mri_to_mriclin(data_dir).copy()
    mri_clinic["clinic_date"] = parse_date_series(mri_clinic["Date of clinic"], "us")

    clinic_biop = pd.read_csv(data_dir / "pros_mriclin_to_biop.csv").copy()
    clinic_biop = clinic_biop.rename(columns={"Subject number": "patient_id"})
    clinic_biop["biopsy_date"] = parse_date_series(clinic_biop["Date of biopsy"], "us")

    biop_path = pd.read_csv(data_dir / "pros_biop_to_pathrep.csv").copy()
    biop_path = biop_path.rename(columns={"Subject number": "patient_id"})
    biop_path["pathrep_date"] = parse_date_series(biop_path["Date of pathology report"], "us")

    pathway = pd.read_csv(data_dir / "pros_pathway.csv").copy()
    pathway["referral_date"] = parse_date_series(pathway["referral_date"], "generic")
    pathway["outpatient_date"] = parse_date_series(pathway["outpatient_date"], "generic")

    waits = (ref_mri["mri_date"] - ref_mri["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_mri"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        mri_clinic[["patient_id", "clinic_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["clinic_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_mri_reporting"] = float(waits.mean()) if len(waits) else np.nan

    observed["time_to_biopsy_decision"] = np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        clinic_biop[["patient_id", "biopsy_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["biopsy_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_biopsy"] = float(waits.mean()) if len(waits) else np.nan

    merged = ref_mri[["patient_id", "referral_date"]].merge(
        biop_path[["patient_id", "pathrep_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["pathrep_date"] - merged["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_diagnosis"] = float(waits.mean()) if len(waits) else np.nan

    waits = (pathway["outpatient_date"] - pathway["referral_date"]).dt.days
    waits = waits[(waits.notna()) & (waits >= 0)]
    observed["time_to_outpatient_diagnosis"] = float(waits.mean()) if len(waits) else np.nan

    return observed


def build_table8_like_comparison(obs_mix_result: dict, data_dir: Path) -> pd.DataFrame:
    sim_cum = extract_cumulative_times_with_pathway_type(obs_mix_result, "OBS_MIX")
    sim_pros = sim_cum[sim_cum["pathway_type"] == "PROSTAD"].copy()

    observed = calculate_observed_prostad_cumulative_values(data_dir)

    report_lookup = {
        "time_to_mri": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_mri"],
        "time_to_mri_reporting": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_mri_reporting"],
        "time_to_biopsy_decision": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_biopsy_decision"],
        "time_to_biopsy": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_biopsy"],
        "time_to_diagnosis": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_diagnosis"],
        "time_to_outpatient_diagnosis": PROSTAD_REPORT_OVERALL["PROSTAD_mean_time_to_outpatient_diagnosis"],
    }

    rows: list[dict] = []
    for metric, label in CUMULATIVE_METRIC_LABELS.items():
        sim_series = sim_pros.loc[sim_pros["metric"] == metric, "days_from_referral"]
        real_mean = observed.get(metric, np.nan)
        report_mean = report_lookup.get(metric, np.nan)

        rows.append(
            {
                "metric": metric,
                "label": label,
                "n_sim": int(len(sim_series)),
                "sim_mean": float(sim_series.mean()) if len(sim_series) else np.nan,
                "sim_median": float(sim_series.median()) if len(sim_series) else np.nan,
                "sim_p90": float(np.percentile(sim_series, 90)) if len(sim_series) else np.nan,
                "observed_mean": real_mean,
                "report_mean": report_mean,
                "sim_minus_observed": (
                    float(sim_series.mean()) - real_mean
                    if len(sim_series) and pd.notna(real_mean)
                    else np.nan
                ),
                "sim_minus_report": (
                    float(sim_series.mean()) - report_mean
                    if len(sim_series) and pd.notna(report_mean)
                    else np.nan
                ),
                "observed_minus_report": (
                    real_mean - report_mean
                    if pd.notna(real_mean) and pd.notna(report_mean)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_obs_mix_subgroups_stage_level(obs_mix_result: dict, data_dir: Path) -> pd.DataFrame:
    sim_stage = extract_stage_waits_with_pathway_type(obs_mix_result, "OBS_MIX")
    sim_stage = add_combined_prostad_stage(sim_stage)
    real_stage = load_real_stage_waits(data_dir)

    subgroup_map = {
        "BASELINE": "pre",
        "PROSTAD": "pros",
    }

    rows: list[dict] = []

    for pathway_type, real_dataset in subgroup_map.items():
        sim_subset = sim_stage[sim_stage["pathway_type"] == pathway_type]

        for stage in sorted(sim_subset["stage"].dropna().unique()):
            sim_series = sim_subset.loc[sim_subset["stage"] == stage, "wait_days"]
            real_series = real_stage.loc[
                (real_stage["source_dataset"] == real_dataset)
                & (real_stage["stage"] == stage),
                "wait_days",
            ]

            row = {
                "comparison": f"OBS_MIX {pathway_type} vs {real_dataset}",
                "pathway_type": pathway_type,
                "real_dataset": real_dataset,
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage),
            }
            row.update(compare_series(sim_series, real_series))
            rows.append(row)

    return pd.DataFrame(rows)


def validate_obs_mix_subgroups_full_pathway(obs_mix_result: dict, data_dir: Path) -> pd.DataFrame:
    sim_path = extract_full_pathway_lengths_with_pathway_type(obs_mix_result, "OBS_MIX")
    real_path = load_real_pathway_lengths(data_dir)

    subgroup_map = {
        "BASELINE": "pre",
        "PROSTAD": "pros",
    }

    rows: list[dict] = []

    for pathway_type, real_dataset in subgroup_map.items():
        sim_series = sim_path.loc[sim_path["pathway_type"] == pathway_type, "total_days"]
        real_series = real_path.loc[real_path["source_dataset"] == real_dataset, "total_days"]

        row = {
            "comparison": f"OBS_MIX {pathway_type} vs {real_dataset}",
            "pathway_type": pathway_type,
            "real_dataset": real_dataset,
            "stage": "full_pathway",
            "label": "Full pathway",
        }
        row.update(compare_series(sim_series, real_series))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_ecdf(real: pd.Series, sim: pd.Series, title: str, xlab: str, output_path: Path) -> None:
    real = pd.to_numeric(real, errors="coerce").dropna().sort_values().to_numpy()
    sim = pd.to_numeric(sim, errors="coerce").dropna().sort_values().to_numpy()

    if len(real) == 0 or len(sim) == 0:
        print(f"SKIPPED ECDF: {title}")
        print(f"  real n = {len(real)}")
        print(f"  sim n  = {len(sim)}")
        return

    real_y = np.arange(1, len(real) + 1) / len(real)
    sim_y = np.arange(1, len(sim) + 1) / len(sim)

    plt.figure(figsize=(7, 5))
    plt.step(real, real_y, where="post", label="Observed")
    plt.step(sim, sim_y, where="post", label="Simulated")
    plt.xlabel(xlab)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ECDF: {output_path}")


def make_stage_ecdf_plots(obs_mix_result: dict, data_dir: Path) -> None:
    """
    ECDFs comparing:
      - OBS_MIX BASELINE patients vs pre-PROSTAD observed data
      - OBS_MIX PROSTAD patients vs PROSTAD observed data

    Uses combined MRI -> clinic/decision stage for PROSTAD.
    """
    sim_stage = extract_stage_waits_with_pathway_type(obs_mix_result, "OBS_MIX")
    sim_stage = add_combined_prostad_stage(sim_stage)

    real_stage = load_real_stage_waits(data_dir)

    subgroup_map = {
        "BASELINE": "pre",
        "PROSTAD": "pros",
    }

    for pathway_type, real_dataset in subgroup_map.items():
        sim_subset = sim_stage[sim_stage["pathway_type"] == pathway_type]

        for stage in sorted(sim_subset["stage"].dropna().unique()):
            sim_series = sim_subset.loc[
                sim_subset["stage"] == stage,
                "wait_days",
            ]

            real_series = real_stage.loc[
                (real_stage["source_dataset"] == real_dataset)
                & (real_stage["stage"] == stage),
                "wait_days",
            ]

            if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
                continue

            title = (
                f"{pathway_type} pathway in mixed simulation vs "
                f"{real_dataset.upper()} data: {STAGE_LABELS.get(stage, stage)}"
            )

            output_path = (
                OUTPUT_DIR
                / f"ecdf_mixed_{pathway_type.lower()}_vs_{real_dataset}_{stage}.png"
            )

            plot_ecdf(
                real=real_series,
                sim=sim_series,
                title=title,
                xlab="Stage wait days",
                output_path=output_path,
            )

def make_cumulative_ecdf_plots(obs_mix_result: dict, data_dir: Path) -> None:
    """
    ECDFs comparing referral-to-milestone times:
      - OBS_MIX BASELINE patients vs pre-PROSTAD observed data
      - OBS_MIX PROSTAD patients vs PROSTAD observed data
    """
    sim_cum = extract_cumulative_times_with_pathway_type(obs_mix_result, "OBS_MIX")

    metric_series: dict[tuple[str, str], pd.Series] = {}

    # -----------------------------
    # BASELINE observed data
    # -----------------------------
    pre_ref = pd.read_csv(data_dir / "pre_ref_to_mri.csv").copy()
    pre_ref["patient_id"] = pre_ref["Subject number"]
    pre_ref["referral_date"] = parse_date_series(
        pre_ref["Date of referral to pathway"], "uk"
    )
    pre_ref["mri_date"] = parse_date_series(pre_ref["Date of MRI"], "uk")

    waits = (pre_ref["mri_date"] - pre_ref["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_mri")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pre_mri_rep = pd.read_csv(data_dir / "pre_mri_to_mrirep.csv").copy()
    pre_mri_rep["patient_id"] = pre_mri_rep["Subject number"]
    pre_mri_rep["report_date"] = parse_date_series(
        pre_mri_rep["Date MRI reported"], "uk"
    )

    merged = pre_ref[["patient_id", "referral_date"]].merge(
        pre_mri_rep[["patient_id", "report_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["report_date"] - merged["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_mri_reporting")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pre_rep_mdt = pd.read_csv(data_dir / "pre_mrirep_to_biopmdt.csv").copy()
    pre_rep_mdt["patient_id"] = pre_rep_mdt["Subject number"]
    pre_rep_mdt["mdt_date"] = parse_date_series(
        pre_rep_mdt["Date of Prostate MRI MDT"], "uk"
    )

    merged = pre_ref[["patient_id", "referral_date"]].merge(
        pre_rep_mdt[["patient_id", "mdt_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["mdt_date"] - merged["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_biopsy_decision")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pre_biop = pd.read_csv(data_dir / "pre_biopmdt_to_biop.csv").copy()
    pre_biop["patient_id"] = pre_biop["Subject number"]
    pre_biop["biopsy_date"] = parse_date_series(pre_biop["Date of Biopsy"], "uk")

    merged = pre_ref[["patient_id", "referral_date"]].merge(
        pre_biop[["patient_id", "biopsy_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["biopsy_date"] - merged["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_biopsy")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pre_pathrep = pd.read_csv(data_dir / "pre_biop_to_pathrep.csv").copy()
    pre_pathrep["patient_id"] = pre_pathrep["Subject number"]
    pre_pathrep["pathrep_date"] = parse_date_series(
        pre_pathrep["Date of pathology report"], "uk"
    )

    merged = pre_ref[["patient_id", "referral_date"]].merge(
        pre_pathrep[["patient_id", "pathrep_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["pathrep_date"] - merged["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_diagnosis")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pre_path = pd.read_csv(data_dir / "pre_pathway.csv").copy()
    pre_path["referral_date"] = parse_date_series(pre_path["referral_date"], "uk")
    pre_path["outpatient_date"] = parse_date_series(pre_path["outpatient_date"], "uk")

    waits = (pre_path["outpatient_date"] - pre_path["referral_date"]).dt.days
    metric_series[("BASELINE", "time_to_outpatient_diagnosis")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    # -----------------------------
    # PROSTAD observed data
    # -----------------------------
    pros_ref = pd.read_csv(data_dir / "pros_ref_to_mri.csv").copy()
    pros_ref["patient_id"] = pros_ref["Subject number"]
    pros_ref["referral_date"] = parse_date_series(
        pros_ref["Date of referral to pathway"], "us"
    )
    pros_ref["mri_date"] = parse_date_series(pros_ref["Date of MRI"], "us")

    waits = (pros_ref["mri_date"] - pros_ref["referral_date"]).dt.days
    metric_series[("PROSTAD", "time_to_mri")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pros_clinic = read_pros_mri_to_mriclin(data_dir).copy()
    pros_clinic["clinic_date"] = parse_date_series(
        pros_clinic["Date of clinic"], "us"
    )

    merged = pros_ref[["patient_id", "referral_date"]].merge(
        pros_clinic[["patient_id", "clinic_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["clinic_date"] - merged["referral_date"]).dt.days
    metric_series[("PROSTAD", "time_to_mri_reporting")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    # PROSTAD biopsy decision is represented by clinic date.
    metric_series[("PROSTAD", "time_to_biopsy_decision")] = metric_series[
        ("PROSTAD", "time_to_mri_reporting")
    ]

    pros_biop = pd.read_csv(data_dir / "pros_mriclin_to_biop.csv").copy()
    pros_biop["patient_id"] = pros_biop["Subject number"]
    pros_biop["biopsy_date"] = parse_date_series(
        pros_biop["Date of biopsy"], "us"
    )

    merged = pros_ref[["patient_id", "referral_date"]].merge(
        pros_biop[["patient_id", "biopsy_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["biopsy_date"] - merged["referral_date"]).dt.days
    metric_series[("PROSTAD", "time_to_biopsy")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pros_pathrep = pd.read_csv(data_dir / "pros_biop_to_pathrep.csv").copy()
    pros_pathrep["patient_id"] = pros_pathrep["Subject number"]
    pros_pathrep["pathrep_date"] = parse_date_series(
        pros_pathrep["Date of pathology report"], "us"
    )

    merged = pros_ref[["patient_id", "referral_date"]].merge(
        pros_pathrep[["patient_id", "pathrep_date"]],
        on="patient_id",
        how="inner",
    )
    waits = (merged["pathrep_date"] - merged["referral_date"]).dt.days
    metric_series[("PROSTAD", "time_to_diagnosis")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    pros_path = pd.read_csv(data_dir / "pros_pathway.csv").copy()
    pros_path["referral_date"] = parse_date_series(
        pros_path["referral_date"], "generic"
    )
    pros_path["outpatient_date"] = parse_date_series(
        pros_path["outpatient_date"], "generic"
    )

    waits = (pros_path["outpatient_date"] - pros_path["referral_date"]).dt.days
    metric_series[("PROSTAD", "time_to_outpatient_diagnosis")] = waits[
        (waits.notna()) & (waits >= 0)
    ]

    # -----------------------------
    # Plot
    # -----------------------------
    for pathway_type, real_dataset_label in [
        ("BASELINE", "pre-PROSTAD"),
        ("PROSTAD", "PROSTAD"),
    ]:
        sim_subset = sim_cum[sim_cum["pathway_type"] == pathway_type].copy()

        for metric, label in CUMULATIVE_METRIC_LABELS.items():
            sim_series = sim_subset.loc[
                sim_subset["metric"] == metric,
                "days_from_referral",
            ]

            real_series = metric_series.get(
                (pathway_type, metric),
                pd.Series(dtype=float),
            )

            if len(sim_series.dropna()) == 0 or len(real_series.dropna()) == 0:
                continue

            title = (
                f"{pathway_type} pathway in mixed simulation vs "
                f"{real_dataset_label} data: {label}"
            )

            output_path = (
                OUTPUT_DIR
                / f"ecdf_mixed_{pathway_type.lower()}_{metric}.png"
            )

            plot_ecdf(
                real=real_series,
                sim=sim_series,
                title=title,
                xlab="Days from referral",
                output_path=output_path,
            )


def get_first_event_date(patient, event_name: str):
    dates = [
        event.get("date")
        for event in patient.events
        if event.get("event") == event_name and event.get("date") is not None
    ]
    return min(dates) if dates else None


def extract_decision_to_biopsy_waits_from_sim(
    obs_mix_result: dict,
    pathway_type: str,
) -> pd.Series:
    waits = []

    for patient in obs_mix_result.get("completed_patients_objects", []):
        if patient.data.get("pathway_type") != pathway_type:
            continue

        decision_date = get_first_event_date(patient, "MDT_occured")
        biopsy_date = get_first_event_date(patient, "biopsy_done")

        if decision_date is None or biopsy_date is None:
            continue

        wait = (biopsy_date - decision_date).days

        if wait >= 0:
            waits.append(wait)

    return pd.Series(waits, dtype=float)


def make_decision_to_biopsy_ecdf(obs_mix_result: dict, data_dir: Path) -> None:
    real_stage = load_real_stage_waits(data_dir)

    comparisons = [
        {
            "pathway_type": "BASELINE",
            "real_dataset": "pre",
            "title": "BASELINE pathway in mixed simulation vs PRE data: MDT decision → Biopsy",
            "output": "ecdf_mixed_baseline_decision_to_biopsy.png",
        },
        {
            "pathway_type": "PROSTAD",
            "real_dataset": "pros",
            "title": "PROSTAD pathway in mixed simulation vs PROS data: Clinic/decision → Biopsy",
            "output": "ecdf_mixed_prostad_decision_to_biopsy.png",
        },
    ]

    for item in comparisons:
        pathway_type = item["pathway_type"]
        real_dataset = item["real_dataset"]

        sim_series = extract_decision_to_biopsy_waits_from_sim(
            obs_mix_result,
            pathway_type,
        )

        real_series = real_stage.loc[
            (real_stage["source_dataset"] == real_dataset)
            & (real_stage["stage"] == "biopmdt_to_biopsy"),
            "wait_days",
        ]

        print(f"\n=== {pathway_type} DECISION → BIOPSY ECDF ===")
        print("sim n:", len(sim_series.dropna()))
        print("real n:", len(real_series.dropna()))
        print("sim mean:", sim_series.mean())
        print("real mean:", real_series.mean())

        output_path = OUTPUT_DIR / item["output"]

        plot_ecdf(
            real=real_series,
            sim=sim_series,
            title=item["title"],
            xlab="Stage wait days",
            output_path=output_path,
        )
def real_weekly_mri_counts_summary(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "pros_ref_to_mri.csv").copy()
    df["mri_date"] = parse_date_series(df["Date of MRI"], "us")
    df = df[df["mri_date"].notna()].copy()

    df["week"] = df["mri_date"].dt.to_period("W").apply(lambda r: r.start_time)

    return (
        df.groupby("week")
        .size()
        .reset_index(name="n_mri")
        .sort_values("week")
        .reset_index(drop=True)
    )
def summarise_weekly_mri_distribution(weekly_df: pd.DataFrame) -> pd.DataFrame:
    counts = weekly_df["n_mri"]

    return pd.DataFrame({
        "metric": [
            "mean_per_week",
            "median_per_week",
            "min_per_week",
            "max_per_week",
            "weeks_lt_4",
            "weeks_eq_4",
            "weeks_gt_4",
            "n_weeks",
        ],
        "value": [
            float(counts.mean()),
            float(counts.median()),
            int(counts.min()),
            int(counts.max()),
            int((counts < 4).sum()),
            int((counts == 4).sum()),
            int((counts > 4).sum()),
            int(len(counts)),
        ],
    })


def main() -> None:
    seed = 1234
    start_date = date(2026, 1, 5)
    n_days = 365
    lam_per_workday = 1.7528735632183907

    referral_schedule = generate_daily_referrals(
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        seed=seed,
    )

    obs_mix_cfg = build_combined_config(
        "OBS_MIX",
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        seed=seed,
    )
    all_prostad_cfg = build_combined_config(
        "ALL_PROSTAD",
        start_date=start_date,
        n_days=n_days,
        lam_per_workday=lam_per_workday,
        seed=seed,
    )

    obs_mix_result = run_day_loop_combined_engine(
        obs_mix_cfg,
        daily_referrals_override=referral_schedule,
    )
    all_prostad_result = run_day_loop_combined_engine(
        all_prostad_cfg,
        daily_referrals_override=referral_schedule,
    )

    subgroup_stage_val_df = validate_obs_mix_subgroups_stage_level(obs_mix_result, DATA_DIR)
    subgroup_stage_val_df.to_csv(OUTPUT_DIR / "obs_mix_subgroup_stage_validation.csv", index=False)

    subgroup_full_df = validate_obs_mix_subgroups_full_pathway(obs_mix_result, DATA_DIR)
    subgroup_full_df.to_csv(OUTPUT_DIR / "obs_mix_subgroup_full_pathway_validation.csv", index=False)

    table8_df = build_table8_like_comparison(obs_mix_result, DATA_DIR)
    table8_df.to_csv(OUTPUT_DIR / "table8_like_prostad_comparison.csv", index=False)

    make_stage_ecdf_plots(obs_mix_result, DATA_DIR)
    make_cumulative_ecdf_plots(obs_mix_result, DATA_DIR)

    make_decision_to_biopsy_ecdf(obs_mix_result, DATA_DIR)

    weekly_real_df = real_weekly_mri_counts_summary(DATA_DIR)
    weekly_real_df.to_csv(OUTPUT_DIR / "real_prostad_weekly_mri_counts.csv", index=False)

    weekly_real_summary_df = summarise_weekly_mri_distribution(weekly_real_df)
    weekly_real_summary_df.to_csv(OUTPUT_DIR / "real_prostad_weekly_mri_summary.csv", index=False)

    print("\n=== TABLE 8-LIKE PROSTAD COMPARISON ===")
    print(table8_df.round(3).to_string(index=False))

    print("\n=== MIXED SIMULATION SUBGROUP STAGE VALIDATION ===")
    print(subgroup_stage_val_df.round(3).to_string(index=False))

    print("\n=== MIXED SIMULATION SUBGROUP FULL PATHWAY VALIDATION ===")
    print(subgroup_full_df.round(3).to_string(index=False))

    print("\n=== WEEKLY MRI SUMMARY (REAL PROSTAD DATA) ===")
    print(weekly_real_summary_df.to_string(index=False))

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()