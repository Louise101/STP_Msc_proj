from __future__ import annotations
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals
from engine.pathway_definitions import STAGE_EVENT_PAIRS
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stage_occupancy_plots_MRIcap3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748
SEEDS = range(1000, 1030)
SCENARIOS = ["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"]
STAGE_LABELS = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT/clinic → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Pathology",
    "pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "treatmdt_to_outpat": "Treatment MDT → Outpatient",
}
SCENARIO_LABELS = {
    "ALL_BASELINE": "All standard",
    "OBS_MIX": "Mixed pathway",
    "ALL_PROSTAD": "All PROSTAD",
}
PATHWAY_TYPE_LABELS = {
    "BASELINE": "Standard",
    "PROSTAD": "PROSTAD",
}
def make_referral_schedule(seed: int) -> dict:
    return generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )
def run_scenarios_for_seed(seed: int) -> dict[str, dict]:
    referral_schedule = make_referral_schedule(seed)
    outputs = {}
    for scenario in SCENARIOS:
        cfg = build_combined_config(
            scenario,
            start_date=START_DATE,
            n_days=N_DAYS,
            lam_per_workday=LAM_PER_WORKDAY,
            seed=seed,
        )
        outputs[scenario] = run_day_loop_combined_engine(
            cfg,
            daily_referrals_override=referral_schedule,
        )
    return outputs
def get_event_date(patient, event_name: str):
    dates = [
        event.get("date")
        for event in patient.events
        if event.get("event") == event_name and event.get("date") is not None
    ]
    return min(dates) if dates else None
def get_patient_pathway_type(patient) -> str:
    pathway_type = getattr(patient, "pathway_type", None)
    if pathway_type is None:
        pathway_type = patient.data.get("pathway_type")
    if pathway_type is None:
        pathway_type = patient.data.get("pathway")
    return pathway_type if pathway_type is not None else "UNKNOWN"
# ------------------------------------------------------------------
# PART 1: ALL SCENARIOS OCCUPANCY
# ------------------------------------------------------------------
def extract_daily_stage_occupancy(outputs: dict[str, dict], seed: int) -> pd.DataFrame:
    rows = []
    for scenario, result in outputs.items():
        for stage, metrics in result["stage_activity"].items():
            for day, count in metrics["daily_in_stage"].items():
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "stage": stage,
                        "date": pd.Timestamp(day),
                        "n_in_stage": int(count),
                    }
                )
    return pd.DataFrame(rows)
def summarise_daily_mean_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scenario", "stage", "date"], as_index=False)
        .agg(
            n_in_stage_mean=("n_in_stage", "mean"),
            n_in_stage_sd=("n_in_stage", "std"),
        )
    )
def plot_stage_occupancy_all_scenarios(mean_df: pd.DataFrame, stage: str) -> None:
    plot_df = mean_df[mean_df["stage"] == stage].copy()
    plt.figure(figsize=(12, 5))
    for scenario in SCENARIOS:
        subset = plot_df[plot_df["scenario"] == scenario].copy()
        if subset.empty:
            continue
        plt.plot(
            subset["date"],
            subset["n_in_stage_mean"],
            linewidth=2,
            label=SCENARIO_LABELS.get(scenario, scenario),
        )
    plt.title(
        f"Patients currently waiting: {STAGE_LABELS.get(stage, stage)}\n"
        f"Mean occupancy across {len(SEEDS)} seeds"
    )
    plt.xlabel("Date")
    plt.ylabel("Mean number in stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"occupancy_all_scenarios_multiseed_mean_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
def summarise_occupancy_all_scenarios(seed_df: pd.DataFrame) -> pd.DataFrame:
    seed_summary = (
        seed_df.groupby(["seed", "scenario", "stage"])
        .agg(
            mean_in_stage=("n_in_stage", "mean"),
            median_in_stage=("n_in_stage", "median"),
            peak_in_stage=("n_in_stage", "max"),
        )
        .reset_index()
    )
    seed_summary.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_summary_all_scenarios_seed_level.csv",
        index=False,
    )
    return (
        seed_summary.groupby(["scenario", "stage"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            mean_in_stage_mean=("mean_in_stage", "mean"),
            mean_in_stage_sd=("mean_in_stage", "std"),
            median_in_stage_mean=("median_in_stage", "mean"),
            peak_in_stage_mean=("peak_in_stage", "mean"),
            peak_in_stage_sd=("peak_in_stage", "std"),
        )
        .sort_values(["stage", "scenario"])
    )
# ------------------------------------------------------------------
# PART 2: OBS_MIX OCCUPANCY BY PATHWAY TYPE
# ------------------------------------------------------------------
def extract_occupancy_by_pathway_type(obs_mix_result: dict, seed: int) -> pd.DataFrame:
    columns = ["seed", "date", "pathway_type", "stage", "n_in_stage"]
    rows = []
    all_dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    patients = obs_mix_result.get("all_patients_objects", [])
    if not patients:
        patients = obs_mix_result.get("completed_patients_objects", [])
    for patient in patients:
        pathway_type = get_patient_pathway_type(patient)
        for (start_event, end_event), stage in STAGE_EVENT_PAIRS.items():
            start = get_event_date(patient, start_event)
            end = get_event_date(patient, end_event)
            if start is None:
                continue
            if end is None:
                if getattr(patient, "is_complete", False):
                    continue
                if getattr(patient, "current_stage", None) != stage:
                    continue
                end = pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS)
            else:
                end = pd.Timestamp(end)
            start = pd.Timestamp(start)
            if end <= start:
                continue
            in_stage_dates = all_dates[(all_dates >= start) & (all_dates < end)]
            for current_date in in_stage_dates:
                rows.append(
                    {
                        "seed": seed,
                        "date": current_date,
                        "pathway_type": pathway_type,
                        "stage": stage,
                        "n_in_stage": 1,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows, columns=columns)
    df = (
        df.groupby(["seed", "date", "pathway_type", "stage"], as_index=False)["n_in_stage"]
        .sum()
    )
    stage_names = list(STAGE_EVENT_PAIRS.values())
    pathway_types = sorted(df["pathway_type"].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [[seed], all_dates, pathway_types, stage_names],
        names=["seed", "date", "pathway_type", "stage"],
    )
    df = (
        df.set_index(["seed", "date", "pathway_type", "stage"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return df[columns]
def summarise_daily_mean_by_pathway(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pathway_type", "stage", "date"], as_index=False)
        .agg(
            n_in_stage_mean=("n_in_stage", "mean"),
            n_in_stage_sd=("n_in_stage", "std"),
        )
    )
def plot_mixed_pathway_occupancy(mean_df: pd.DataFrame, stage: str) -> None:
    plot_df = mean_df[mean_df["stage"] == stage].copy()
    plt.figure(figsize=(12, 5))
    colours = {
        "BASELINE": "tab:blue",
        "PROSTAD": "tab:orange",
    }
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = plot_df[plot_df["pathway_type"] == pathway_type].copy()
        if subset.empty:
            continue
        colour = colours.get(pathway_type)
        mean_value = subset["n_in_stage_mean"].mean()
        label = PATHWAY_TYPE_LABELS.get(pathway_type, pathway_type)
        plt.plot(
            subset["date"],
            subset["n_in_stage_mean"],
            color=colour,
            linestyle="-",
            linewidth=2,
            label=f"{label} mean occupancy",
        )
        plt.axhline(
            y=mean_value,
            color=colour,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"{label} mean = {mean_value:.1f}",
        )
    plt.title(
        f"Mixed pathway: Patients currently waiting — {STAGE_LABELS.get(stage, stage)}\n"
        f"Mean occupancy across {len(SEEDS)} seeds"
    )
    plt.xlabel("Date")
    plt.ylabel("Mean number in stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"obs_mix_occupancy_by_pathway_multiseed_mean_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
def summarise_mixed_pathway_occupancy(seed_df: pd.DataFrame) -> pd.DataFrame:
    seed_summary = (
        seed_df.groupby(["seed", "pathway_type", "stage"])
        .agg(
            mean_in_stage=("n_in_stage", "mean"),
            median_in_stage=("n_in_stage", "median"),
            peak_in_stage=("n_in_stage", "max"),
        )
        .reset_index()
    )
    seed_summary.to_csv(
        OUTPUT_DIR / "obs_mix_occupancy_by_pathway_type_summary_seed_level.csv",
        index=False,
    )
    return (
        seed_summary.groupby(["pathway_type", "stage"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            mean_in_stage_mean=("mean_in_stage", "mean"),
            mean_in_stage_sd=("mean_in_stage", "std"),
            median_in_stage_mean=("median_in_stage", "mean"),
            peak_in_stage_mean=("peak_in_stage", "mean"),
            peak_in_stage_sd=("peak_in_stage", "std"),
        )
        .reset_index()
        .sort_values(["stage", "pathway_type"])
    )
# ------------------------------------------------------------------
# PART 3: GENERIC STAGE FLOW
# ------------------------------------------------------------------
def extract_stage_flow_by_pathway_type(obs_mix_result: dict, seed: int) -> pd.DataFrame:
    columns = ["seed", "date", "pathway_type", "stage", "entered", "left", "net_flow"]
    rows = []
    all_dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    patients = obs_mix_result.get("all_patients_objects", [])
    for patient in patients:
        pathway_type = get_patient_pathway_type(patient)
        for (start_event, end_event), stage in STAGE_EVENT_PAIRS.items():
            start = get_event_date(patient, start_event)
            end = get_event_date(patient, end_event)
            if start is not None:
                rows.append(
                    {
                        "seed": seed,
                        "date": pd.Timestamp(start),
                        "pathway_type": pathway_type,
                        "stage": stage,
                        "entered": 1,
                        "left": 0,
                    }
                )
            if end is not None:
                rows.append(
                    {
                        "seed": seed,
                        "date": pd.Timestamp(end),
                        "pathway_type": pathway_type,
                        "stage": stage,
                        "entered": 0,
                        "left": 1,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    df = (
        df.groupby(["seed", "date", "pathway_type", "stage"], as_index=False)
        .agg(
            entered=("entered", "sum"),
            left=("left", "sum"),
        )
    )
    stage_names = list(STAGE_EVENT_PAIRS.values())
    pathway_types = sorted(df["pathway_type"].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [[seed], all_dates, pathway_types, stage_names],
        names=["seed", "date", "pathway_type", "stage"],
    )
    df = (
        df.set_index(["seed", "date", "pathway_type", "stage"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    df["net_flow"] = df["entered"] - df["left"]
    return df[columns]
def summarise_daily_stage_flow_by_pathway(flow_df: pd.DataFrame) -> pd.DataFrame:
    df = flow_df.copy()
    df["net_flow"] = df["entered"] - df["left"]
    return (
        df.groupby(["pathway_type", "stage", "date"], as_index=False)
        .agg(
            entered_mean=("entered", "mean"),
            left_mean=("left", "mean"),
            net_mean=("net_flow", "mean"),
        )
    )
def summarise_stage_flow_table(flow_df: pd.DataFrame) -> pd.DataFrame:
    df = flow_df.copy()
    df["net_flow"] = df["entered"] - df["left"]
    seed_summary = (
        df.groupby(["seed", "pathway_type", "stage"], as_index=False)
        .agg(
            mean_entering_per_day=("entered", "mean"),
            mean_leaving_per_day=("left", "mean"),
            mean_net_per_day=("net_flow", "mean"),
            total_entered=("entered", "sum"),
            total_left=("left", "sum"),
            total_net=("net_flow", "sum"),
        )
    )
    seed_summary.to_csv(
        OUTPUT_DIR / "obs_mix_stage_flow_by_pathway_seed_level.csv",
        index=False,
    )
    return (
        seed_summary.groupby(["pathway_type", "stage"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            mean_entering_per_day=("mean_entering_per_day", "mean"),
            mean_leaving_per_day=("mean_leaving_per_day", "mean"),
            mean_net_per_day=("mean_net_per_day", "mean"),
            total_entered_mean=("total_entered", "mean"),
            total_left_mean=("total_left", "mean"),
            total_net_mean=("total_net", "mean"),
        )
        .sort_values(["stage", "pathway_type"])
    )
def plot_stage_flow_by_pathway(flow_daily_mean_df: pd.DataFrame, stage: str) -> None:
    plot_df = flow_daily_mean_df[flow_daily_mean_df["stage"] == stage].copy()
    colours = {
        "BASELINE": "tab:blue",
        "PROSTAD": "tab:orange",
    }
    plt.figure(figsize=(12, 5))
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = plot_df[plot_df["pathway_type"] == pathway_type].copy()
        if subset.empty:
            continue
        label = PATHWAY_TYPE_LABELS.get(pathway_type, pathway_type)
        colour = colours.get(pathway_type)
        plt.plot(
            subset["date"],
            subset["entered_mean"],
            color=colour,
            linestyle="-",
            linewidth=2,
            label=f"{label} entering",
        )
        plt.plot(
            subset["date"],
            subset["left_mean"],
            color=colour,
            linestyle="--",
            linewidth=2,
            label=f"{label} leaving",
        )
    plt.title(
        f"Mixed pathway flow: entering vs leaving — {STAGE_LABELS.get(stage, stage)}\n"
        f"Mean daily flow across {len(SEEDS)} seeds"
    )
    plt.xlabel("Date")
    plt.ylabel("Mean patients per day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"obs_mix_flow_entering_leaving_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    plt.figure(figsize=(12, 5))
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = plot_df[plot_df["pathway_type"] == pathway_type].copy()
        if subset.empty:
            continue
        label = PATHWAY_TYPE_LABELS.get(pathway_type, pathway_type)
        colour = colours.get(pathway_type)
        plt.plot(
            subset["date"],
            subset["net_mean"],
            color=colour,
            linewidth=2,
            label=f"{label} entering - leaving",
        )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero line")
    plt.title(
        f"Mixed pathway net flow: entering - leaving — {STAGE_LABELS.get(stage, stage)}\n"
        "Positive values suggest accumulation in the stage"
    )
    plt.xlabel("Date")
    plt.ylabel("Mean net patients per day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"obs_mix_flow_net_difference_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
# ------------------------------------------------------------------
# PART 4: WEEKLY BIOPSY-WAIT FLOW
# ------------------------------------------------------------------
def extract_weekly_biopsy_wait_flow_by_pathway(obs_mix_result: dict, seed: int) -> pd.DataFrame:
    """
    Biopsy-wait specific flow.
    Entering biopsy wait:
      date of biopsy decision / MDT / clinic, but only for patients who later have biopsy.
    Leaving biopsy wait:
      date of biopsy_done.
    This avoids counting patients who leave the pathway at biopsy decision as if
    they were waiting for biopsy.
    """
    columns = [
        "seed",
        "week_start",
        "pathway_type",
        "stage",
        "entered_biopsy_wait",
        "had_biopsy",
        "net_biopsy_wait",
    ]
    rows = []
    patients = obs_mix_result.get("all_patients_objects", [])
    for patient in patients:
        pathway_type = get_patient_pathway_type(patient)
        biopsy_wait_start = get_event_date(patient, "MDT_occured")
        biopsy_done = get_event_date(patient, "biopsy_done")
        if biopsy_wait_start is None or biopsy_done is None:
            continue
        rows.append(
            {
                "seed": seed,
                "date": pd.Timestamp(biopsy_wait_start),
                "pathway_type": pathway_type,
                "stage": "biopmdt_to_biopsy",
                "entered_biopsy_wait": 1,
                "had_biopsy": 0,
            }
        )
        rows.append(
            {
                "seed": seed,
                "date": pd.Timestamp(biopsy_done),
                "pathway_type": pathway_type,
                "stage": "biopmdt_to_biopsy",
                "entered_biopsy_wait": 0,
                "had_biopsy": 1,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    # Week starts Monday
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    weekly = (
        df.groupby(["seed", "week_start", "pathway_type", "stage"], as_index=False)
        .agg(
            entered_biopsy_wait=("entered_biopsy_wait", "sum"),
            had_biopsy=("had_biopsy", "sum"),
        )
    )
    pathway_types = sorted(weekly["pathway_type"].dropna().unique())
    all_weeks = pd.date_range(
        START_DATE,
        pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS - 1),
        freq="W-MON",
    )
    full_index = pd.MultiIndex.from_product(
        [[seed], all_weeks, pathway_types, ["biopmdt_to_biopsy"]],
        names=["seed", "week_start", "pathway_type", "stage"],
    )
    weekly = (
        weekly.set_index(["seed", "week_start", "pathway_type", "stage"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    weekly["net_biopsy_wait"] = (
        weekly["entered_biopsy_wait"] - weekly["had_biopsy"]
    )
    return weekly[columns]
def summarise_weekly_biopsy_wait_flow_table(biopsy_weekly_df: pd.DataFrame) -> pd.DataFrame:
    df = biopsy_weekly_df.copy()
    df["net_biopsy_wait"] = df["entered_biopsy_wait"] - df["had_biopsy"]
    seed_summary = (
        df.groupby(["seed", "pathway_type", "stage"], as_index=False)
        .agg(
            mean_entering_biopsy_wait_per_week=("entered_biopsy_wait", "mean"),
            mean_had_biopsy_per_week=("had_biopsy", "mean"),
            mean_net_biopsy_wait_per_week=("net_biopsy_wait", "mean"),
            total_entered_biopsy_wait=("entered_biopsy_wait", "sum"),
            total_had_biopsy=("had_biopsy", "sum"),
            total_net_biopsy_wait=("net_biopsy_wait", "sum"),
        )
    )
    seed_summary.to_csv(
        OUTPUT_DIR / "obs_mix_weekly_biopsy_wait_flow_summary_seed_level.csv",
        index=False,
    )
    return (
        seed_summary.groupby(["pathway_type", "stage"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            mean_entering_biopsy_wait_per_week=("mean_entering_biopsy_wait_per_week", "mean"),
            mean_had_biopsy_per_week=("mean_had_biopsy_per_week", "mean"),
            mean_net_biopsy_wait_per_week=("mean_net_biopsy_wait_per_week", "mean"),
            total_entered_biopsy_wait_mean=("total_entered_biopsy_wait", "mean"),
            total_had_biopsy_mean=("total_had_biopsy", "mean"),
            total_net_biopsy_wait_mean=("total_net_biopsy_wait", "mean"),
        )
        .sort_values(["stage", "pathway_type"])
    )
def plot_weekly_biopsy_wait_flow(biopsy_weekly_df: pd.DataFrame) -> None:
    df = biopsy_weekly_df.copy()
    df["net_biopsy_wait"] = df["entered_biopsy_wait"] - df["had_biopsy"]
    weekly_mean = (
        df.groupby(["week_start", "pathway_type"], as_index=False)
        .agg(
            entered_biopsy_wait_mean=("entered_biopsy_wait", "mean"),
            had_biopsy_mean=("had_biopsy", "mean"),
            net_biopsy_wait_mean=("net_biopsy_wait", "mean"),
        )
        .sort_values(["pathway_type", "week_start"])
    )
    colours = {
        "BASELINE": "tab:blue",
        "PROSTAD": "tab:orange",
    }
    # Entering vs leaving
    plt.figure(figsize=(12, 5))
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = weekly_mean[weekly_mean["pathway_type"] == pathway_type].copy()
        if subset.empty:
            continue
        label = PATHWAY_TYPE_LABELS.get(pathway_type, pathway_type)
        colour = colours.get(pathway_type)
        plt.plot(
            subset["week_start"],
            subset["entered_biopsy_wait_mean"],
            color=colour,
            linestyle="-",
            linewidth=2,
            label=f"{label} entering biopsy wait",
        )
        plt.plot(
            subset["week_start"],
            subset["had_biopsy_mean"],
            color=colour,
            linestyle="--",
            linewidth=2,
            label=f"{label} had biopsy",
        )
    plt.title(
        "Weekly biopsy wait flow in mixed simulation\n"
        "Only includes patients who enter biopsy wait and later receive biopsy"
    )
    plt.xlabel("Week starting")
    plt.ylabel("Mean patients per week across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "obs_mix_weekly_biopsy_wait_entering_vs_had_biopsy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # Net flow
    plt.figure(figsize=(12, 5))
    for pathway_type in ["BASELINE", "PROSTAD"]:
        subset = weekly_mean[weekly_mean["pathway_type"] == pathway_type].copy()
        if subset.empty:
            continue
        label = PATHWAY_TYPE_LABELS.get(pathway_type, pathway_type)
        colour = colours.get(pathway_type)
        mean_net = subset["net_biopsy_wait_mean"].mean()
        plt.plot(
            subset["week_start"],
            subset["net_biopsy_wait_mean"],
            color=colour,
            linewidth=2,
            label=f"{label} weekly net flow",
        )
        plt.axhline(
            y=mean_net,
            color=colour,
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label=f"{label} mean net = {mean_net:.2f}",
        )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero line")
    plt.title(
        "Weekly biopsy wait net flow\n"
        "Positive values indicate more patients entering biopsy wait than receiving biopsy"
    )
    plt.xlabel("Week starting")
    plt.ylabel("Mean weekly entering - had biopsy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "obs_mix_weekly_biopsy_wait_net_flow.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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


def load_observed_stage_intervals(data_dir: Path) -> pd.DataFrame:
    rows = []

    specs = {
        "standard": {
            "ref_to_mri": ("pre_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "uk"),
            "mri_to_report": ("pre_mri_to_mrirep.csv", "Date of MRI", "Date MRI reported", "uk"),
            "report_to_biopmdt": ("pre_mrirep_to_biopmdt.csv", "Date MRI reported", "Date of Prostate MRI MDT", "uk"),
            "biopmdt_to_biopsy": ("pre_biopmdt_to_biop.csv", "Date of Prostate MRI MDT", "Date of Biopsy", "uk"),
            "biopsy_to_pathrep": ("pre_biop_to_pathrep.csv", "Date of Biopsy", "Date of pathology report", "uk"),
            "pathrep_to_treatmdt": ("pre_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT (treatment options)", "uk"),
            "treatmdt_to_outpat": ("pre_treatmdt_to_outpat.csv", "Date of MDT (treatment options)", "Date of outpat appt", "uk"),
        },
        "prostad": {
            "ref_to_mri": ("pros_ref_to_mri.csv", "Date of referral to pathway", "Date of MRI", "us"),
            "mri_to_report": ("pros_mri_to_mriclin.csv", "Date of MRI", "Date of clinic", "us"),
            "biopmdt_to_biopsy": ("pros_mriclin_to_biop.csv", "Date of clinic", "Date of biopsy", "us"),
            "biopsy_to_pathrep": ("pros_biop_to_pathrep.csv", "Date of biopsy", "Date of pathology report", "us"),
            "pathrep_to_treatmdt": ("pros_pathrep_to_treatmdt.csv", "Date of pathology report", "Date of MDT to discuss treatment options", "us"),
            "treatmdt_to_outpat": ("pros_treatmdt_to_outpat.csv", "Date of MDT to discuss treatment options", "Date of OPD appt", "us"),
        },
    }

    for observed_pathway, pathway_specs in specs.items():
        for stage, (filename, start_col, end_col, style) in pathway_specs.items():
            df = pd.read_csv(data_dir / filename).copy()

            start_dates = parse_date_series(df[start_col], style)
            end_dates = parse_date_series(df[end_col], style)

            for start, end in zip(start_dates, end_dates):
                if pd.isna(start) or pd.isna(end):
                    continue

                start = pd.Timestamp(start)
                end = pd.Timestamp(end)

                if end <= start:
                    continue

                rows.append(
                    {
                        "observed_pathway": observed_pathway,
                        "stage": stage,
                        "start_date": start,
                        "end_date": end,
                    }
                )

    return pd.DataFrame(rows)


def build_observed_stage_occupancy(observed_intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for observed_pathway, group in observed_intervals.groupby("observed_pathway"):
        cohort_start = group["start_date"].min()

        for _, row in group.iterrows():
            start_day = int((row["start_date"] - cohort_start).days)
            end_day = int((row["end_date"] - cohort_start).days)

            for day_index in range(start_day, end_day):
                rows.append(
                    {
                        "observed_pathway": observed_pathway,
                        "stage": row["stage"],
                        "day_index": day_index,
                        "n_in_stage": 1,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["observed_pathway", "stage", "day_index", "n_in_stage"]
        )

    return (
        pd.DataFrame(rows)
        .groupby(["observed_pathway", "stage", "day_index"], as_index=False)
        .agg(n_in_stage=("n_in_stage", "sum"))
    )


def build_sim_stage_occupancy_by_day_index(pathway_daily_mean_df: pd.DataFrame) -> pd.DataFrame:
    df = pathway_daily_mean_df.copy()
    df["day_index"] = (df["date"] - pd.Timestamp(START_DATE)).dt.days
    return df


def plot_observed_vs_mixed_sim_occupancy(
    observed_occupancy: pd.DataFrame,
    sim_occupancy: pd.DataFrame,
    stage: str,
) -> None:
    plt.figure(figsize=(12, 5))

    plot_specs = [
        ("standard", None, "Observed standard", "black", "-"),
        ("prostad", None, "Observed PROSTAD", "black", "--"),
        (None, "BASELINE", "Simulated standard", "tab:blue", "-"),
        (None, "PROSTAD", "Simulated PROSTAD", "tab:orange", "-"),
    ]

    for observed_pathway, pathway_type, label, colour, linestyle in plot_specs:
        if observed_pathway is not None:
            subset = observed_occupancy[
                (observed_occupancy["observed_pathway"] == observed_pathway)
                & (observed_occupancy["stage"] == stage)
            ].copy()

            y_col = "n_in_stage"

        else:
            subset = sim_occupancy[
                (sim_occupancy["pathway_type"] == pathway_type)
                & (sim_occupancy["stage"] == stage)
            ].copy()

            y_col = "n_in_stage_mean"

        if subset.empty:
            continue

        subset = subset.sort_values("day_index")

        plt.plot(
            subset["day_index"],
            subset[y_col],
            color=colour,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    plt.title(
        f"Observed vs mixed simulation stage occupancy: {STAGE_LABELS.get(stage, stage)}"
    )
    plt.xlabel("Days from cohort start")
    plt.ylabel("Patients in stage")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"observed_vs_mixed_sim_occupancy_{stage}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
def main() -> None:
    all_scenario_frames: list[pd.DataFrame] = []
    all_pathway_frames: list[pd.DataFrame] = []
    all_flow_frames: list[pd.DataFrame] = []
    all_biopsy_weekly_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")
        outputs = run_scenarios_for_seed(seed)
        scenario_df = extract_daily_stage_occupancy(outputs, seed)
        all_scenario_frames.append(scenario_df)
        obs_mix_result = outputs["OBS_MIX"]
        pathway_df = extract_occupancy_by_pathway_type(obs_mix_result, seed)
        all_pathway_frames.append(pathway_df)
        flow_df = extract_stage_flow_by_pathway_type(obs_mix_result, seed)
        all_flow_frames.append(flow_df)
        biopsy_weekly_df = extract_weekly_biopsy_wait_flow_by_pathway(
            obs_mix_result,
            seed,
        )
        all_biopsy_weekly_frames.append(biopsy_weekly_df)
    all_scenario_df = pd.concat(all_scenario_frames, ignore_index=True)
    all_scenario_df.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_all_scenarios_seed_level.csv",
        index=False,
    )
    all_scenario_daily_mean_df = summarise_daily_mean_all_scenarios(all_scenario_df)
    all_scenario_daily_mean_df.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_all_scenarios_multiseed_daily_mean.csv",
        index=False,
    )
    all_scenario_summary = summarise_occupancy_all_scenarios(all_scenario_df)
    all_scenario_summary.to_csv(
        OUTPUT_DIR / "daily_stage_occupancy_summary_all_scenarios_multiseed.csv",
        index=False,
    )
    for stage in sorted(all_scenario_daily_mean_df["stage"].unique()):
        plot_stage_occupancy_all_scenarios(all_scenario_daily_mean_df, stage)
    pathway_df = pd.concat(all_pathway_frames, ignore_index=True)
    pathway_df.to_csv(
        OUTPUT_DIR / "obs_mix_daily_occupancy_by_pathway_type_seed_level.csv",
        index=False,
    )
    pathway_daily_mean_df = summarise_daily_mean_by_pathway(pathway_df)
    pathway_daily_mean_df.to_csv(
        OUTPUT_DIR / "obs_mix_daily_occupancy_by_pathway_type_multiseed_daily_mean.csv",
        index=False,
    )
    pathway_summary = summarise_mixed_pathway_occupancy(pathway_df)
    pathway_summary.to_csv(
        OUTPUT_DIR / "obs_mix_occupancy_by_pathway_type_summary_multiseed.csv",
        index=False,
    )
    for stage in STAGE_EVENT_PAIRS.values():
        plot_mixed_pathway_occupancy(pathway_daily_mean_df, stage)
    flow_df = pd.concat(all_flow_frames, ignore_index=True)
    flow_df["net_flow"] = flow_df["entered"] - flow_df["left"]
    flow_df.to_csv(
        OUTPUT_DIR / "obs_mix_stage_flow_by_pathway_seed_level_daily.csv",
        index=False,
    )
    flow_daily_mean_df = summarise_daily_stage_flow_by_pathway(flow_df)
    flow_daily_mean_df.to_csv(
        OUTPUT_DIR / "obs_mix_stage_flow_by_pathway_multiseed_daily_mean.csv",
        index=False,
    )
    flow_summary = summarise_stage_flow_table(flow_df)
    flow_summary.to_csv(
        OUTPUT_DIR / "obs_mix_stage_flow_by_pathway_summary_multiseed.csv",
        index=False,
    )
    for stage in STAGE_EVENT_PAIRS.values():
        plot_stage_flow_by_pathway(flow_daily_mean_df, stage)
    biopsy_weekly_df = pd.concat(all_biopsy_weekly_frames, ignore_index=True)
    biopsy_weekly_df.to_csv(
        OUTPUT_DIR / "obs_mix_weekly_biopsy_wait_flow_seed_level.csv",
        index=False,
    )
    biopsy_weekly_summary = summarise_weekly_biopsy_wait_flow_table(biopsy_weekly_df)
    biopsy_weekly_summary.to_csv(
        OUTPUT_DIR / "obs_mix_weekly_biopsy_wait_flow_summary_multiseed.csv",
        index=False,
    )
    plot_weekly_biopsy_wait_flow(biopsy_weekly_df)
    print("\n=== ALL SCENARIOS MULTI-SEED STAGE OCCUPANCY SUMMARY ===")
    print(all_scenario_summary.round(3).to_string(index=False))
    print("\n=== OBS_MIX MULTI-SEED OCCUPANCY BY PATHWAY TYPE ===")
    print(pathway_summary.round(3).to_string(index=False))
    print("\n=== OBS_MIX STAGE FLOW BY PATHWAY TYPE ===")
    print(flow_summary.round(3).to_string(index=False))
    print("\n=== OBS_MIX WEEKLY BIOPSY WAIT FLOW: ACTUAL BIOPSIES ONLY ===")
    print(biopsy_weekly_summary.round(3).to_string(index=False))
    print(f"\nSaved plots and CSVs to: {OUTPUT_DIR}")

        # --------------------------------------------------
    # Observed vs mixed simulation stage occupancy
    # --------------------------------------------------
    observed_intervals = load_observed_stage_intervals(DATA_DIR)
    observed_intervals.to_csv(
        OUTPUT_DIR / "observed_stage_intervals.csv",
        index=False,
    )

    observed_occupancy = build_observed_stage_occupancy(observed_intervals)
    observed_occupancy.to_csv(
        OUTPUT_DIR / "observed_stage_occupancy_by_day_index.csv",
        index=False,
    )

    sim_occupancy_by_day = build_sim_stage_occupancy_by_day_index(pathway_daily_mean_df)
    sim_occupancy_by_day.to_csv(
        OUTPUT_DIR / "mixed_sim_stage_occupancy_by_day_index.csv",
        index=False,
    )

    for stage in STAGE_EVENT_PAIRS.values():
        plot_observed_vs_mixed_sim_occupancy(
            observed_occupancy=observed_occupancy,
            sim_occupancy=sim_occupancy_by_day,
            stage=stage,
        )
if __name__ == "__main__":
    main()