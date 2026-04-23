from pathlib import Path

import pandas as pd
import pytest

import runners.run_sim as run_sim

print(run_sim.__file__)
print([name for name in dir(run_sim) if "generate" in name or "summarise" in name])

def fake_result():
    """Minimal fake engine result object used by the orchestration tests."""
    return {
        "patient_results": [],
        "all_patient_results": [],
        "event_log": pd.DataFrame(),
        "daily_referrals": {},
        "completed_patients_objects": [],
        "all_patients_objects": [],
        "resources": {},
        "stage_activity": {},
        "summary_stats": {},
    }


def test_run_named_scenarios_returns_expected_summary_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_sim, "build_combined_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", lambda *args, **kwargs: fake_result())

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A"}]))
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A"}]))
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A"}]))
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A", "total_days": 10}]))
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A", "stage": "ref_to_mri", "wait_days": 5}]))
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame([{"scenario": "A", "stage": "ref_to_mri", "week_start": "2026-01-05", "weekly_arrivals": 2}]))
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame([{"scenario": "A", "stage": "ref_to_mri"}]))
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame([{"scenario": "A", "mean_days": 10}]))

    outputs = run_sim.run_named_scenarios(["ALL_BASELINE"], seeds=[1])

    expected_keys = {
        "stage_activity",
        "flow_counts",
        "resource_pressure",
        "pathway_lengths",
        "stage_waits",
        "weekly_stage_arrivals",
        "stage_wait_summary",
        "pathway_summary",
    }
    assert set(outputs.keys()) == expected_keys


def test_run_named_scenarios_calls_engine_once_per_scenario_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    engine_calls = []

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda *args, **kwargs: {"shared": 1})
    monkeypatch.setattr(run_sim, "build_combined_config", lambda scenario_name, **kwargs: {"scenario_name": scenario_name, **kwargs})

    def fake_run_engine(cfg, daily_referrals_override=None):
        engine_calls.append((cfg, daily_referrals_override))
        return fake_result()

    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", fake_run_engine)

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame())

    run_sim.run_named_scenarios(["ALL_BASELINE", "OBS_MIX"], seeds=[1, 2, 3])

    assert len(engine_calls) == 2 * 3


def test_run_named_scenarios_uses_same_referral_schedule_for_all_scenarios_within_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    generated = {}
    engine_seen = []

    def fake_generate_daily_referrals(start_date, n_days, lam_per_workday, seed):
        schedule = {f"seed_{seed}": seed}
        generated[seed] = schedule
        return schedule

    def fake_build_config(scenario_name, **kwargs):
        return {"scenario_name": scenario_name, **kwargs}

    def fake_run_engine(cfg, daily_referrals_override=None):
        engine_seen.append((cfg["seed"], cfg["scenario_name"], daily_referrals_override))
        return fake_result()

    monkeypatch.setattr(run_sim, "generate_daily_referrals", fake_generate_daily_referrals)
    monkeypatch.setattr(run_sim, "build_combined_config", fake_build_config)
    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", fake_run_engine)

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame())

    run_sim.run_named_scenarios(["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"], seeds=[11, 22])

    for seed, scenario_name, referral_schedule in engine_seen:
        assert referral_schedule is generated[seed]


def test_run_named_scenarios_uses_default_seed_when_none(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    seen_seeds = []

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda start_date, n_days, lam_per_workday, seed: {})
    monkeypatch.setattr(run_sim, "build_combined_config", lambda scenario_name, **kwargs: kwargs)

    def fake_run_engine(cfg, daily_referrals_override=None):
        seen_seeds.append(cfg["seed"])
        return fake_result()

    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", fake_run_engine)

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame())

    run_sim.run_named_scenarios(["ALL_BASELINE"], seeds=None)

    assert seen_seeds == [1234]


def test_run_named_scenarios_handles_empty_seed_list(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame())

    outputs = run_sim.run_named_scenarios(["ALL_BASELINE"], seeds=[])

    for key in [
        "stage_activity",
        "flow_counts",
        "resource_pressure",
        "pathway_lengths",
        "stage_waits",
        "weekly_stage_arrivals",
    ]:
        assert isinstance(outputs[key], pd.DataFrame)
        assert outputs[key].empty


def test_run_named_scenarios_writes_output_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_sim, "build_combined_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", lambda *args, **kwargs: fake_result())

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame([{"a": 1}]))
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame([{"a": 1}]))
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame([{"a": 1}]))
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame([{"total_days": 10}]))
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame([{"stage": "ref_to_mri", "wait_days": 5}]))
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame([{"week_start": "2026-01-05", "weekly_arrivals": 1}]))
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame([{"summary": 1}]))
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame([{"summary": 1}]))

    run_sim.run_named_scenarios(["ALL_BASELINE"], seeds=[1])

    expected_files = [
        "stage_activity_all_runs.csv",
        "flow_counts_all_runs.csv",
        "resource_pressure_all_runs.csv",
        "pathway_lengths_all_runs.csv",
        "stage_waits_all_runs.csv",
        "weekly_stage_arrivals_all_runs.csv",
        "stage_wait_summary.csv",
        "pathway_length_summary.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists(), f"{filename} was not created"


def test_run_named_scenarios_passes_requested_scenario_names_to_config_builder(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    seen_scenarios = []

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda *args, **kwargs: {})

    def fake_build_config(scenario_name, **kwargs):
        seen_scenarios.append(scenario_name)
        return object()

    monkeypatch.setattr(run_sim, "build_combined_config", fake_build_config)
    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", lambda *args, **kwargs: fake_result())

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame())
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame())

    run_sim.run_named_scenarios(["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"], seeds=[1])

    assert seen_scenarios == ["ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"]


def test_run_named_scenarios_returns_concatenated_dataframes(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sim, "OUTPUT_DIR", tmp_path)

    monkeypatch.setattr(run_sim, "generate_daily_referrals", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_sim, "build_combined_config", lambda scenario_name, **kwargs: {"scenario_name": scenario_name})
    monkeypatch.setattr(run_sim, "run_day_loop_combined_engine", lambda *args, **kwargs: fake_result())

    monkeypatch.setattr(run_sim, "summarise_stage_activity", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed}]))
    monkeypatch.setattr(run_sim, "summarise_flow_counts", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed}]))
    monkeypatch.setattr(run_sim, "summarise_resource_pressure", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed}]))
    monkeypatch.setattr(run_sim, "extract_full_pathway_lengths", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed, "total_days": 10}]))
    monkeypatch.setattr(run_sim, "extract_stage_waits", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed, "stage": "ref_to_mri", "wait_days": 5}]))
    monkeypatch.setattr(run_sim, "summarise_stage_weekly_arrivals", lambda result, scenario_name, seed: pd.DataFrame([{"scenario": scenario_name, "seed": seed, "week_start": "2026-01-05", "weekly_arrivals": 1}]))
    monkeypatch.setattr(run_sim, "summarise_stage_waits", lambda df: pd.DataFrame([{"rows": len(df)}]))
    monkeypatch.setattr(run_sim, "summarise_pathway_lengths", lambda df: pd.DataFrame([{"rows": len(df)}]))

    outputs = run_sim.run_named_scenarios(["ALL_BASELINE", "OBS_MIX"], seeds=[1, 2])

    assert len(outputs["stage_activity"]) == 4
    assert len(outputs["flow_counts"]) == 4
    assert len(outputs["resource_pressure"]) == 4
    assert len(outputs["pathway_lengths"]) == 4
    assert len(outputs["stage_waits"]) == 4
    assert len(outputs["weekly_stage_arrivals"]) == 4