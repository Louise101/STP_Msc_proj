from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from engine.combined_engine import run_day_loop_combined_engine, CombinedEngineConfig
from data_prep.empirical_inputs import load_real_stage_waits

from analysis.validation import build_real_pathway_csvs, load_real_pathway_data
from analysis.summaries import summarise_stage_activity, summarise_flow_counts, summarise_stage_waits, summarise_flow_counts_across_seeds, summarise_pathway_lengths
from analysis.metrics import safe_pct_change, extract_stage_waits, extract_full_pathway_lengths


from engine.scenarios import build_combined_config, generate_daily_referrals

from engine.pathway_definitions import WAIT_MODE_DES, WAIT_MODE_MC, STAGE_EVENT_PAIRS


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "refac_dissertation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CALIBRATED INPUTS
# --------------------------------------------------
START_DATE = date(2026, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.7528735632183907
P_PROSTAD_OBS = 0.5098039215686274
SEEDS = list(range(1, 21))

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
def build_combined_config(name: str, p_prostad: float, seed: int) -> CombinedEngineConfig:
    return CombinedEngineConfig(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        p_prostad=p_prostad,
        mri_capacity_by_weekday_prostad={1: 4},
        seed=seed,
        scenario_name=name,
        baseline_wait_time_mode={
            "ref_to_mri": WAIT_MODE_MC,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        prostad_wait_time_mode={
            "ref_to_mri": WAIT_MODE_DES,
            "mri_to_report": WAIT_MODE_MC,
            "report_to_biopmdt": WAIT_MODE_MC,
            "biopmdt_to_biopsy": WAIT_MODE_MC,
            "biopsy_to_pathrep": WAIT_MODE_MC,
            "pathrep_to_treatmdt": WAIT_MODE_MC,
            "treatmdt_to_outpat": WAIT_MODE_MC,
        },
        baseline_stage_timing_policy={
            "mri_to_report": "EMPIRICAL",
            "report_to_biopmdt": "EMPIRICAL",
        },
        prostad_stage_timing_policy={
            "mri_to_report": "FIXED",
            "report_to_biopmdt": "FIXED",
        },
        prostad_fixed_wait_days_by_stage={
            "mri_to_report": 1,
            "report_to_biopmdt": 0,
        },
    )


def generate_daily_referrals(
    start_date: date,
    n_days: int,
    lam_per_workday: float,
    seed: int,
) -> Dict[date, int]:
    rng = np.random.default_rng(seed)
    referrals = {}
    current_date = start_date

    for _ in range(n_days):
        referrals[current_date] = int(rng.poisson(lam_per_workday)) if current_date.weekday() < 5 else 0
        current_date += timedelta(days=1)

    return referrals


def run_three_scenarios(seed: int):
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=seed,
    )

    baseline_cfg = build_combined_config("ALL_BASELINE", p_prostad=0.0, seed=seed)
    mixed_cfg = build_combined_config("OBS_MIX", p_prostad=P_PROSTAD_OBS, seed=seed)
    all_pros_cfg = build_combined_config("ALL_PROSTAD", p_prostad=1.0, seed=seed)

    baseline_res = run_day_loop_combined_engine(
        baseline_cfg,
        daily_referrals_override=referral_schedule,
    )
    mixed_res = run_day_loop_combined_engine(
        mixed_cfg,
        daily_referrals_override=referral_schedule,
    )
    all_pros_res = run_day_loop_combined_engine(
        all_pros_cfg,
        daily_referrals_override=referral_schedule,
    )

    return baseline_res, mixed_res, all_pros_res





# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
def compare_wait_distributions(sim_series: pd.Series, real_series: pd.Series) -> dict:
    ks_stat, ks_p = ks_2samp(sim_series, real_series)
    return {
        "n_sim": len(sim_series),
        "n_real": len(real_series),
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


# --------------------------------------------------
# FLAGS
# --------------------------------------------------
FAST_FLOW_ELIGIBLE_STAGES = {
    "mri_to_report",
    "report_to_biopmdt",
}


def classify_stage_from_changes(
    arrival_pct,
    in_stage_pct,
    completion_change,
    mean_in_stage_value,
    wait_pct_change=None,
    stage_name=None,
):
    if (
        pd.notna(arrival_pct) and arrival_pct > 10
        and (
            (pd.notna(in_stage_pct) and in_stage_pct > 10)
            or (pd.notna(wait_pct_change) and wait_pct_change > 10)
        )
        and (pd.isna(completion_change) or completion_change <= 0)
    ):
        return "🔴 EMERGING BOTTLENECK"

    if (
        stage_name in FAST_FLOW_ELIGIBLE_STAGES
        and pd.notna(arrival_pct) and arrival_pct > 10
        and pd.notna(wait_pct_change) and wait_pct_change < -10
        and (pd.isna(completion_change) or completion_change >= 0)
    ):
        return "🚀 FASTER FLOW"

    if (
        pd.notna(arrival_pct) and arrival_pct > 5
        and not (pd.notna(wait_pct_change) and wait_pct_change < -10)
    ):
        return "🟠 PRESSURE ↑"

    if (
        (pd.notna(wait_pct_change) and wait_pct_change < -10)
        or (pd.notna(in_stage_pct) and in_stage_pct < -10)
    ):
        return "🟢 IMPROVED"

    if pd.notna(mean_in_stage_value) and mean_in_stage_value < 0.5:
        return "⚡ FAST TRACK"

    return "⚪ STABLE"

VALIDATION_COMPARISONS = [
    {
        "sim_scenario": "ALL_BASELINE",
        "real_scenario": "BASELINE_REAL",
        "label": "ALL_BASELINE vs BASELINE_REAL",
        "note": "Direct baseline validation",
    },
    {
        "sim_scenario": "OBS_MIX",
        "real_scenario": "OBS_MIX_REAL",
        "label": "OBS_MIX vs OBS_MIX_REAL",
        "note": "Direct observed-mix / PROSTAD-period validation",
    },
    {
        "sim_scenario": "ALL_PROSTAD",
        "real_scenario": "OBS_MIX_REAL",
        "label": "ALL_PROSTAD vs OBS_MIX_REAL",
        "note": "Illustrative comparison only; full PROSTAD scenario compared against observed PROSTAD-period data",
    },
]
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    all_stage_waits = []
    all_flow_counts = []
    all_pathway_lengths = []
    all_stage_activity_rows = []
    validation_store = {}

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        baseline_res, mixed_res, all_pros_res = run_three_scenarios(seed)

        for scenario_name, result in [
            ("ALL_BASELINE", baseline_res),
            ("OBS_MIX", mixed_res),
            ("ALL_PROSTAD", all_pros_res),
        ]:
            all_stage_waits.append(extract_stage_waits(result, scenario_name, seed))
            all_flow_counts.append(summarise_flow_counts(result, scenario_name, seed))
            all_pathway_lengths.append(extract_full_pathway_lengths(result, scenario_name, seed))
            all_stage_activity_rows.append(summarise_stage_activity(result, scenario_name, seed))

        if seed == SEEDS[0]:
            validation_store["ALL_BASELINE"] = baseline_res
            validation_store["OBS_MIX"] = mixed_res
            validation_store["ALL_PROSTAD"] = all_pros_res

    stage_wait_df = pd.concat(all_stage_waits, ignore_index=True)
    flow_df = pd.concat(all_flow_counts, ignore_index=True)
    pathway_df = pd.concat(all_pathway_lengths, ignore_index=True)
    stage_activity_df = pd.concat(all_stage_activity_rows, ignore_index=True)

    # --------------------------------------------------
    # 1. PATIENT WAITS AT EACH STAGE
    # --------------------------------------------------
    sim_stage_wait_summary = summarise_stage_waits(stage_wait_df)

    #real_stage_wait_df = load_real_stage_waits(DATA_DIR)


    real_stage_wait_df = load_real_stage_waits(DATA_DIR).copy()
    real_stage_wait_df["scenario"] = real_stage_wait_df["scenario"].replace(
        {
            "pre": "BASELINE_REAL",
            "pros": "OBS_MIX_REAL",
        }
    )

    real_stage_wait_summary = summarise_stage_waits(real_stage_wait_df)
    stage_wait_summary = pd.concat(
        [sim_stage_wait_summary, real_stage_wait_summary],
        ignore_index=True
    )
    stage_wait_summary.to_csv(OUTPUT_DIR / "dissertation_stage_wait_summary.csv", index=False)

    # --------------------------------------------------
    # 2. TOTAL PATIENTS COMPLETED / NUMBER COMPLETING EACH STAGE
    # --------------------------------------------------
    flow_summary = summarise_flow_counts_across_seeds(flow_df)
    flow_summary.to_csv(OUTPUT_DIR / "dissertation_flow_summary.csv", index=False)

    # --------------------------------------------------
    # 3. TOTAL PATHWAY TIME FOR FULL PATHWAY PATIENTS
    # --------------------------------------------------
    pathway_summary = summarise_pathway_lengths(pathway_df)
    pathway_summary.to_csv(OUTPUT_DIR / "dissertation_pathway_summary.csv", index=False)

        # --------------------------------------------------
    # 4. VALIDATION TABLES
    # --------------------------------------------------
    build_real_pathway_csvs(
        pre_ref_file=str(DATA_DIR / "pre_ref_to_mri.csv"),
        pre_outpat_file=str(DATA_DIR / "pre_treatmdt_to_outpat.csv"),
        pros_ref_file=str(DATA_DIR / "pros_ref_to_mri.csv"),
        pros_outpat_file=str(DATA_DIR / "pros_treatmdt_to_outpat.csv"),
        out_pre_file=str(DATA_DIR / "pre_pathway.csv"),
        out_pros_file=str(DATA_DIR / "pros_pathway.csv"),
    )

    real_pre_path, real_pros_path = load_real_pathway_data(
        str(DATA_DIR / "pre_pathway.csv"),
        str(DATA_DIR / "pros_pathway.csv"),
    )

    real_pathway_lookup = {
        "BASELINE_REAL": real_pre_path,
        "OBS_MIX_REAL": real_pros_path,
    }

    #real_stage_wait_df = load_real_stage_waits(DATA_DIR)

    validation_rows = []

    # ---- full pathway comparisons
    for comp in VALIDATION_COMPARISONS:
        sim_scenario = comp["sim_scenario"]
        real_scenario = comp["real_scenario"]

        sim_path = extract_full_pathway_lengths(
            validation_store[sim_scenario],
            sim_scenario,
            seed=1,
        )

        real_path = real_pathway_lookup[real_scenario]

        validation_rows.append({
            "comparison": comp["label"],
            "validation_note": comp["note"],
            "level": "full_pathway",
            "stage": "full_pathway",
            **compare_wait_distributions(sim_path["total_days"], real_path["total_days"]),
        })

    # ---- stage-level comparisons
    for comp in VALIDATION_COMPARISONS:
        sim_scenario = comp["sim_scenario"]
        real_scenario = comp["real_scenario"]

        for stage in sorted(stage_wait_df["stage"].unique()):
            sim_series = stage_wait_df[
                (stage_wait_df["scenario"] == sim_scenario) &
                (stage_wait_df["stage"] == stage)
            ]["wait_days"]

            real_series = real_stage_wait_df[
                (real_stage_wait_df["scenario"] == real_scenario) &
                (real_stage_wait_df["stage"] == stage)
            ]["wait_days"]

            if len(sim_series) == 0 or len(real_series) == 0:
                validation_rows.append({
                    "comparison": comp["label"],
                    "validation_note": comp["note"],
                    "level": "stage",
                    "stage": stage,
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
                })
            else:
                validation_rows.append({
                    "comparison": comp["label"],
                    "validation_note": comp["note"],
                    "level": "stage",
                    "stage": stage,
                    **compare_wait_distributions(sim_series, real_series),
                })

    validation_summary = pd.DataFrame(validation_rows)
    validation_summary.to_csv(OUTPUT_DIR / "dissertation_validation_summary.csv", index=False)

    # --------------------------------------------------
    # 5. FULL COMPARISON TABLE WITH FLAGS
    # --------------------------------------------------
    stage_activity_summary = (
        stage_activity_df.groupby(["scenario", "stage"])
        .agg(
            mean_daily_arrivals=("mean_daily_arrivals", "mean"),
            mean_in_stage=("mean_in_stage", "mean"),
            completion_ratio=("completion_ratio", "mean"),
        )
        .reset_index()
    )

    stage_pressure = stage_activity_summary.pivot(index="stage", columns="scenario")
    stage_pressure.columns = [f"{metric}_{scenario}" for metric, scenario in stage_pressure.columns]
    stage_pressure = stage_pressure.reset_index()

    wait_summary_wide = sim_stage_wait_summary.pivot(index="stage", columns="scenario", values="mean_wait")
    wait_summary_wide.columns = [f"mean_wait_{c}" for c in wait_summary_wide.columns]
    wait_summary_wide = wait_summary_wide.reset_index()

    stage_pressure = stage_pressure.merge(wait_summary_wide, on="stage", how="left")

    stage_pressure["mix_arrival_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_daily_arrivals_OBS_MIX"), r.get("mean_daily_arrivals_ALL_BASELINE")),
        axis=1,
    )
    stage_pressure["mix_in_stage_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_in_stage_OBS_MIX"), r.get("mean_in_stage_ALL_BASELINE")),
        axis=1,
    )
    stage_pressure["mix_completion_change"] = (
        stage_pressure.get("completion_ratio_OBS_MIX") - stage_pressure.get("completion_ratio_ALL_BASELINE")
    )
    stage_pressure["mix_wait_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_wait_OBS_MIX"), r.get("mean_wait_ALL_BASELINE")),
        axis=1,
    )

    stage_pressure["allpros_arrival_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_daily_arrivals_ALL_PROSTAD"), r.get("mean_daily_arrivals_ALL_BASELINE")),
        axis=1,
    )
    stage_pressure["allpros_in_stage_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_in_stage_ALL_PROSTAD"), r.get("mean_in_stage_ALL_BASELINE")),
        axis=1,
    )
    stage_pressure["allpros_completion_change"] = (
        stage_pressure.get("completion_ratio_ALL_PROSTAD") - stage_pressure.get("completion_ratio_ALL_BASELINE")
    )
    stage_pressure["allpros_wait_pct_change"] = stage_pressure.apply(
        lambda r: safe_pct_change(r.get("mean_wait_ALL_PROSTAD"), r.get("mean_wait_ALL_BASELINE")),
        axis=1,
    )

    stage_pressure["flag_obs_mix"] = stage_pressure.apply(
        lambda r: classify_stage_from_changes(
            r.get("mix_arrival_pct_change"),
            r.get("mix_in_stage_pct_change"),
            r.get("mix_completion_change"),
            r.get("mean_in_stage_OBS_MIX"),
            wait_pct_change=r.get("mix_wait_pct_change"),
            stage_name=r.get("stage"),
        ),
        axis=1,
    )

    stage_pressure["flag_all_prostad"] = stage_pressure.apply(
        lambda r: classify_stage_from_changes(
            r.get("allpros_arrival_pct_change"),
            r.get("allpros_in_stage_pct_change"),
            r.get("allpros_completion_change"),
            r.get("mean_in_stage_ALL_PROSTAD"),
            wait_pct_change=r.get("allpros_wait_pct_change"),
            stage_name=r.get("stage"),
        ),
        axis=1,
    )

    stage_pressure.to_csv(OUTPUT_DIR / "dissertation_stage_pressure_summary.csv", index=False)

    # --------------------------------------------------
    # OPTIONAL SIMPLE FIGURE
    # --------------------------------------------------
    plot_df = stage_pressure.copy()
    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["mean_in_stage_ALL_BASELINE"], width, label="All baseline")
    plt.bar(x, plot_df["mean_in_stage_OBS_MIX"], width, label="Observed mix")
    plt.bar(x + width, plot_df["mean_in_stage_ALL_PROSTAD"], width, label="All PROSTAD")
    plt.xticks(x, plot_df["stage"], rotation=30, ha="right")
    plt.ylabel("Mean number in stage")
    plt.title("Dissertation stage pressure comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dissertation_stage_pressure.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved dissertation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()