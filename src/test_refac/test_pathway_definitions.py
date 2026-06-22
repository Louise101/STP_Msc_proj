from engine.pathway_definitions import (
    FULL_PATHWAY_END_EVENT,
    STAGE_CONFIG,
    STAGE_EVENT_PAIRS,
    WAIT_MODE_DES,
    WAIT_MODE_MC,
    WAIT_STREAM_BY_STAGE,
)


def test_wait_mode_constants_are_defined():
    assert WAIT_MODE_MC == "MC"
    assert WAIT_MODE_DES == "DES"


def test_stage_config_contains_expected_stages_in_order():
    expected_stages = [
        "ref_to_mri",
        "mri_to_report",
        "report_to_biopmdt",
        "biopmdt_to_biopsy",
        "biopsy_to_pathrep",
        "pathrep_to_treatmdt",
        "treatmdt_to_outpat",
    ]
    assert list(STAGE_CONFIG.keys()) == expected_stages


def test_each_stage_config_has_required_keys():
    required_keys = {"resource", "pdf_key", "completion_event"}

    for stage_name, stage_cfg in STAGE_CONFIG.items():
        assert required_keys.issubset(stage_cfg.keys()), f"{stage_name} missing required keys"


def test_only_des_stages_have_resources():
    expected_resources = {
        "ref_to_mri": "MRI_PROSTAD",
        "biopmdt_to_biopsy": "BIOPSY",
    }

    for stage_name, stage_cfg in STAGE_CONFIG.items():
        expected = expected_resources.get(stage_name)
        assert stage_cfg["resource"] == expected


def test_pdf_keys_are_unique():
    pdf_keys = [cfg["pdf_key"] for cfg in STAGE_CONFIG.values()]
    assert len(pdf_keys) == len(set(pdf_keys))


def test_completion_events_are_unique():
    completion_events = [cfg["completion_event"] for cfg in STAGE_CONFIG.values()]
    assert len(completion_events) == len(set(completion_events))


def test_wait_stream_by_stage_has_same_stage_keys_as_stage_config():
    assert set(WAIT_STREAM_BY_STAGE.keys()) == set(STAGE_CONFIG.keys())


def test_wait_stream_names_match_expected_pattern():
    expected = {
        "ref_to_mri": "wait_ref_to_mri",
        "mri_to_report": "wait_mri_to_report",
        "report_to_biopmdt": "wait_report_to_biopmdt",
        "biopmdt_to_biopsy": "wait_biopmdt_to_biopsy",
        "biopsy_to_pathrep": "wait_biopsy_to_pathrep",
        "pathrep_to_treatmdt": "wait_pathrep_to_treatmdt",
        "treatmdt_to_outpat": "wait_treatmdt_to_outpat",
    }
    assert WAIT_STREAM_BY_STAGE == expected


def test_stage_event_pairs_cover_all_pathway_stages():
    assert set(STAGE_EVENT_PAIRS.values()) == set(STAGE_CONFIG.keys())


def test_stage_event_pairs_are_unique_by_stage():
    stages = list(STAGE_EVENT_PAIRS.values())
    assert len(stages) == len(set(stages))


def test_stage_event_pairs_match_stage_completion_events():
    expected_pairs = {
        ("referral_received", STAGE_CONFIG["ref_to_mri"]["completion_event"]): "ref_to_mri",
        (STAGE_CONFIG["ref_to_mri"]["completion_event"], STAGE_CONFIG["mri_to_report"]["completion_event"]): "mri_to_report",
        (STAGE_CONFIG["mri_to_report"]["completion_event"], STAGE_CONFIG["report_to_biopmdt"]["completion_event"]): "report_to_biopmdt",
        (STAGE_CONFIG["report_to_biopmdt"]["completion_event"], STAGE_CONFIG["biopmdt_to_biopsy"]["completion_event"]): "biopmdt_to_biopsy",
        (STAGE_CONFIG["biopmdt_to_biopsy"]["completion_event"], STAGE_CONFIG["biopsy_to_pathrep"]["completion_event"]): "biopsy_to_pathrep",
        (STAGE_CONFIG["biopsy_to_pathrep"]["completion_event"], STAGE_CONFIG["pathrep_to_treatmdt"]["completion_event"]): "pathrep_to_treatmdt",
        (STAGE_CONFIG["pathrep_to_treatmdt"]["completion_event"], STAGE_CONFIG["treatmdt_to_outpat"]["completion_event"]): "treatmdt_to_outpat",
    }
    assert STAGE_EVENT_PAIRS == expected_pairs


def test_full_pathway_end_event_matches_final_stage_completion_event():
    assert FULL_PATHWAY_END_EVENT == STAGE_CONFIG["treatmdt_to_outpat"]["completion_event"]


def test_all_stage_names_are_strings():
    assert all(isinstance(stage_name, str) for stage_name in STAGE_CONFIG.keys())


def test_all_config_values_have_expected_types():
    for stage_cfg in STAGE_CONFIG.values():
        assert isinstance(stage_cfg["pdf_key"], str)
        assert isinstance(stage_cfg["completion_event"], str)
        assert stage_cfg["resource"] is None or isinstance(stage_cfg["resource"], str)