from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.summaries import (
    extract_full_pathway_lengths,
    extract_stage_waits,
    summarise_flow_counts,
    summarise_pathway_lengths,
    summarise_resource_pressure, 
    summarise_stage_activity,
    summarise_stage_waits,
    summarise_stage_weekly_arrivals,
)
from engine.scenarios import (
    DEFAULT_LAM_PER_WORKDAY,
    DEFAULT_N_DAYS,
    DEFAULT_START_DATE,
    build_combined_config,
    generate_daily_referrals,
)
from engine.combined_engine import run_day_loop_combined_engine


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "refactored_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Run one or more named scenarios from the central scenario registry.
def run_named_scenarios(
    scenario_names: list[str],
    *,
    start_date=DEFAULT_START_DATE,
    n_days=DEFAULT_N_DAYS,
    lam_per_workday=DEFAULT_LAM_PER_WORKDAY,
    seeds: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    

    if seeds is None:
        seeds = [1234]

    all_stage_rows = []
    all_flow_rows = []
    all_resource_rows = []
    all_pathway_rows = []
    all_stage_wait_rows = []
    all_weekly_rows = []

    for seed in seeds:
        referral_schedule = generate_daily_referrals(start_date, n_days, lam_per_workday, seed)
        for scenario_name in scenario_names:
            cfg = build_combined_config(
                scenario_name,
                start_date=start_date,
                n_days=n_days,
                lam_per_workday=lam_per_workday,
                seed=seed,
            )
            result = run_day_loop_combined_engine(cfg, daily_referrals_override=referral_schedule)

            all_stage_rows.append(summarise_stage_activity(result, scenario_name, seed))
            all_flow_rows.append(summarise_flow_counts(result, scenario_name, seed))
            all_resource_rows.append(summarise_resource_pressure(result, scenario_name, seed))
            all_pathway_rows.append(extract_full_pathway_lengths(result, scenario_name, seed))
            all_stage_wait_rows.append(extract_stage_waits(result, scenario_name, seed))
            all_weekly_rows.append(summarise_stage_weekly_arrivals(result, scenario_name, seed))

    stage_df = pd.concat(all_stage_rows, ignore_index=True) if all_stage_rows else pd.DataFrame()
    flow_df = pd.concat(all_flow_rows, ignore_index=True) if all_flow_rows else pd.DataFrame()
    resource_df = pd.concat(all_resource_rows, ignore_index=True) if all_resource_rows else pd.DataFrame()
    pathway_df = pd.concat(all_pathway_rows, ignore_index=True) if all_pathway_rows else pd.DataFrame()
    stage_wait_df = pd.concat(all_stage_wait_rows, ignore_index=True) if all_stage_wait_rows else pd.DataFrame()
    weekly_df = pd.concat(all_weekly_rows, ignore_index=True) if all_weekly_rows else pd.DataFrame()

    # Save raw outputs.
    stage_df.to_csv(OUTPUT_DIR / "stage_activity_all_runs.csv", index=False)
    flow_df.to_csv(OUTPUT_DIR / "flow_counts_all_runs.csv", index=False)
    resource_df.to_csv(OUTPUT_DIR / "resource_pressure_all_runs.csv", index=False)
    pathway_df.to_csv(OUTPUT_DIR / "pathway_lengths_all_runs.csv", index=False)
    stage_wait_df.to_csv(OUTPUT_DIR / "stage_waits_all_runs.csv", index=False)
    weekly_df.to_csv(OUTPUT_DIR / "weekly_stage_arrivals_all_runs.csv", index=False)

    summaries = {
        "stage_activity": stage_df,
        "flow_counts": flow_df,
        "resource_pressure": resource_df,
        "pathway_lengths": pathway_df,
        "stage_waits": stage_wait_df,
        "weekly_stage_arrivals": weekly_df,
        "stage_wait_summary": summarise_stage_waits(stage_wait_df),
        "pathway_summary": summarise_pathway_lengths(pathway_df),
    }

    summaries["stage_wait_summary"].to_csv(OUTPUT_DIR / "stage_wait_summary.csv", index=False)
    summaries["pathway_summary"].to_csv(OUTPUT_DIR / "pathway_length_summary.csv", index=False)
    return summaries


if __name__ == "__main__":
    outputs = run_named_scenarios(
        ["ALL_BASELINE", 
         "OBS_MIX",
         "OBS_MIX_DES_BIOPSY", 
         "ALL_PROSTAD",
         ],
        seeds=list(range(1, 6)),
    )
    print("Saved refactored scenario outputs to:", OUTPUT_DIR)
