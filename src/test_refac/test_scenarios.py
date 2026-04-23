from datetime import date, timedelta

import pytest

from engine.combined_engine import CombinedEngineConfig
from engine.pathway_definitions import STAGE_CONFIG, WAIT_MODE_DES, WAIT_MODE_MC
from engine.scenarios import (
    DEFAULT_BASELINE_TIMING,
    DEFAULT_BASELINE_WAIT_MODES,
    DEFAULT_LAM_PER_WORKDAY,
    DEFAULT_N_DAYS,
    DEFAULT_PROSTAD_FIXED_WAITS,
    DEFAULT_PROSTAD_MRI_CAPACITY,
    DEFAULT_PROSTAD_OBSERVED_PROPORTION,
    DEFAULT_PROSTAD_TIMING,
    DEFAULT_PROSTAD_WAIT_MODES,
    DEFAULT_START_DATE,
    SCENARIO_LIBRARY,
    ScenarioTemplate,
    all_stages,
    build_combined_config,
    generate_daily_referrals,
    register_scenario,
)


def test_all_stages_matches_stage_config_order():
    assert all_stages() == list(STAGE_CONFIG.keys())


def test_default_baseline_wait_modes_are_all_mc():
    assert set(DEFAULT_BASELINE_WAIT_MODES.keys()) == set(STAGE_CONFIG.keys())
    assert all(mode == WAIT_MODE_MC for mode in DEFAULT_BASELINE_WAIT_MODES.values())


def test_default_prostad_wait_modes_have_des_only_for_ref_to_mri():
    assert DEFAULT_PROSTAD_WAIT_MODES["ref_to_mri"] == WAIT_MODE_DES

    for stage_name, mode in DEFAULT_PROSTAD_WAIT_MODES.items():
        if stage_name != "ref_to_mri":
            assert mode == WAIT_MODE_MC


def test_default_timing_policies_are_as_expected():
    assert DEFAULT_BASELINE_TIMING == {
        "mri_to_report": "EMPIRICAL",
        "report_to_biopmdt": "EMPIRICAL",
    }
    assert DEFAULT_PROSTAD_TIMING == {
        "mri_to_report": "FIXED",
        "report_to_biopmdt": "FIXED",
    }


def test_default_prostad_fixed_waits_are_as_expected():
    assert DEFAULT_PROSTAD_FIXED_WAITS == {
        "mri_to_report": 1,
        "report_to_biopmdt": 0,
    }


def test_scenario_library_contains_expected_default_scenarios():
    assert {"ALL_BASELINE", "OBS_MIX", "ALL_PROSTAD"}.issubset(SCENARIO_LIBRARY.keys())


def test_register_scenario_adds_new_template():
    name = "UNIT_TEST_SCENARIO"
    template = ScenarioTemplate(
        name=name,
        p_prostad=0.25,
        baseline_wait_time_mode=DEFAULT_BASELINE_WAIT_MODES.copy(),
        prostad_wait_time_mode=DEFAULT_PROSTAD_WAIT_MODES.copy(),
    )

    original = SCENARIO_LIBRARY.get(name)

    try:
        register_scenario(template)
        assert name in SCENARIO_LIBRARY
        assert SCENARIO_LIBRARY[name].p_prostad == 0.25
    finally:
        if original is None:
            SCENARIO_LIBRARY.pop(name, None)
        else:
            SCENARIO_LIBRARY[name] = original


def test_register_scenario_replaces_existing_template():
    name = "UNIT_TEST_REPLACE"
    first = ScenarioTemplate(name=name, p_prostad=0.1)
    second = ScenarioTemplate(name=name, p_prostad=0.9)

    original = SCENARIO_LIBRARY.get(name)

    try:
        register_scenario(first)
        assert SCENARIO_LIBRARY[name].p_prostad == 0.1

        register_scenario(second)
        assert SCENARIO_LIBRARY[name].p_prostad == 0.9
    finally:
        if original is None:
            SCENARIO_LIBRARY.pop(name, None)
        else:
            SCENARIO_LIBRARY[name] = original


def test_build_combined_config_returns_config_object():
    cfg = build_combined_config("ALL_BASELINE")

    assert isinstance(cfg, CombinedEngineConfig)


def test_build_combined_config_uses_default_values():
    cfg = build_combined_config("OBS_MIX")

    assert cfg.start_date == DEFAULT_START_DATE
    assert cfg.n_days == DEFAULT_N_DAYS
    assert cfg.lam_per_workday == DEFAULT_LAM_PER_WORKDAY
    assert cfg.p_prostad == DEFAULT_PROSTAD_OBSERVED_PROPORTION
    assert cfg.mri_capacity_by_weekday_prostad == DEFAULT_PROSTAD_MRI_CAPACITY
    assert cfg.scenario_name == "OBS_MIX"


def test_build_combined_config_builds_all_baseline_correctly():
    cfg = build_combined_config("ALL_BASELINE")

    assert cfg.p_prostad == 0.0
    assert cfg.baseline_wait_time_mode["ref_to_mri"] == WAIT_MODE_MC
    assert cfg.prostad_wait_time_mode["ref_to_mri"] == WAIT_MODE_DES
    assert cfg.baseline_stage_timing_policy["mri_to_report"] == "EMPIRICAL"
    assert cfg.prostad_stage_timing_policy["mri_to_report"] == "FIXED"
    assert cfg.prostad_fixed_wait_days_by_stage["mri_to_report"] == 1


def test_build_combined_config_builds_all_prostad_correctly():
    cfg = build_combined_config("ALL_PROSTAD")

    assert cfg.p_prostad == 1.0
    assert cfg.scenario_name == "ALL_PROSTAD"


def test_build_combined_config_accepts_overrides():
    cfg = build_combined_config(
        "OBS_MIX",
        start_date=date(2030, 1, 1),
        n_days=30,
        lam_per_workday=3.5,
        seed=999,
        overrides={
            "p_prostad": 0.75,
            "scenario_name": "CUSTOM_NAME",
        },
    )

    assert cfg.start_date == date(2030, 1, 1)
    assert cfg.n_days == 30
    assert cfg.lam_per_workday == 3.5
    assert cfg.seed == 999
    assert cfg.p_prostad == 0.75
    assert cfg.scenario_name == "CUSTOM_NAME"


def test_build_combined_config_copies_template_dicts_not_shared_references():
    cfg1 = build_combined_config("OBS_MIX")
    cfg2 = build_combined_config("OBS_MIX")

    cfg1.baseline_wait_time_mode["ref_to_mri"] = "CHANGED"
    cfg1.mri_capacity_by_weekday_prostad[1] = 999

    assert cfg2.baseline_wait_time_mode["ref_to_mri"] == WAIT_MODE_MC
    assert cfg2.mri_capacity_by_weekday_prostad[1] == DEFAULT_PROSTAD_MRI_CAPACITY[1]


def test_build_combined_config_unknown_scenario_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_combined_config("NOT_A_REAL_SCENARIO")


def test_generate_daily_referrals_returns_expected_number_of_days():
    start = date(2026, 1, 5)
    n_days = 10

    referrals = generate_daily_referrals(
        start_date=start,
        n_days=n_days,
        lam_per_workday=1.5,
        seed=123,
    )

    assert len(referrals) == n_days
    assert list(referrals.keys())[0] == start
    assert list(referrals.keys())[-1] == start + timedelta(days=n_days - 1)


def test_generate_daily_referrals_is_reproducible():
    r1 = generate_daily_referrals(
        start_date=date(2026, 1, 5),
        n_days=14,
        lam_per_workday=1.75,
        seed=123,
    )
    r2 = generate_daily_referrals(
        start_date=date(2026, 1, 5),
        n_days=14,
        lam_per_workday=1.75,
        seed=123,
    )

    assert r1 == r2


def test_generate_daily_referrals_sets_weekends_to_zero():
    start = date(2026, 1, 5)  # Monday
    referrals = generate_daily_referrals(
        start_date=start,
        n_days=7,
        lam_per_workday=10.0,
        seed=123,
    )

    for day, count in referrals.items():
        if day.weekday() >= 5:
            assert count == 0


def test_generate_daily_referrals_returns_non_negative_integers():
    referrals = generate_daily_referrals(
        start_date=date(2026, 1, 5),
        n_days=14,
        lam_per_workday=2.0,
        seed=123,
    )

    for count in referrals.values():
        assert isinstance(count, int)
        assert count >= 0


def test_generate_daily_referrals_with_zero_lambda_gives_zero_everywhere():
    referrals = generate_daily_referrals(
        start_date=date(2026, 1, 5),
        n_days=10,
        lam_per_workday=0.0,
        seed=123,
    )

    assert all(count == 0 for count in referrals.values())

def test_build_combined_config_works_for_registered_custom_scenario():
    name = "CUSTOM_EXPERIMENT"
    template = ScenarioTemplate(
        name=name,
        p_prostad=0.33,
        baseline_wait_time_mode=DEFAULT_BASELINE_WAIT_MODES.copy(),
        prostad_wait_time_mode=DEFAULT_PROSTAD_WAIT_MODES.copy(),
        baseline_stage_timing_policy=DEFAULT_BASELINE_TIMING.copy(),
        prostad_stage_timing_policy=DEFAULT_PROSTAD_TIMING.copy(),
        prostad_fixed_wait_days_by_stage=DEFAULT_PROSTAD_FIXED_WAITS.copy(),
    )

    original = SCENARIO_LIBRARY.get(name)

    try:
        register_scenario(template)
        cfg = build_combined_config(name, seed=321)
        assert cfg.scenario_name == name
        assert cfg.p_prostad == 0.33
        assert cfg.seed == 321
    finally:
        if original is None:
            SCENARIO_LIBRARY.pop(name, None)
        else:
            SCENARIO_LIBRARY[name] = original