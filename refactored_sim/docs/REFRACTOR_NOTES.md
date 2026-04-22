# Refactor notes

## What changed

This refactor treats the **combined engine** as the single engine to keep.
Older parallel modules (`des_engine.py`, `stage_engine2.py`, `scenarios.py`,
and related validation wrappers) are treated as legacy.

## One place to define scenarios

All scenario definitions now live in:

- `src/config/scenarios.py`

That file contains:
- the `CombinedEngineConfig` dataclass,
- reusable default settings,
- the central `SCENARIO_LIBRARY`,
- `build_combined_config()` to construct a scenario by name.

## Built-in scenario names

- `ALL_BASELINE`
- `OBS_MIX`
- `ALL_PROSTAD`

These cover:
- baseline-only validation,
- mixed baseline + PROSTAD pathways,
- full PROSTAD scenario analysis.

## Suggested migration of old scripts

### Keep conceptually, but replace with the refactored modules
- `Combined_des_engine.py` -> `src/engine/combined_engine.py`
- `Combined_stage_engine.py` -> `src/engine/stage_logic.py`
- `Combined_scenarios.py` -> `src/config/scenarios.py`
- `Patient_state.py` -> `src/core/patient.py`
- `Queue_resource.py` -> `src/core/queueing.py`
- `Event_log_utils.py` -> `src/core/event_log.py`
- `Sampling.py` -> `src/core/sampling.py`
- `Pdf_create.py` -> `src/data_prep/empirical_inputs.py`
- `Validate_against_real.py` -> `src/data_prep/real_data.py` + `src/analysis/validation.py`

### Strong candidates to retire or reduce to thin runner scripts
- `Des_engine.py`
- `Stage_engine2.py`
- `Scenarios.py`
- `Validate_scenarios.py`
- `Run_combined_sim.py`
- `Validate_baseline.py`
- `Validate_combined_stage_pressure.py`
- `Dissertation_results.py`
- `Multi_seed_comp.py`
- `Single_patient_walk_mdtday.py`

## Important caveat

This is a structural cleanup aimed at making the codebase easier to maintain.
Before deleting the old engine family entirely, you should compare a few key
outputs between old and new workflows:

- total completed patients,
- mean / median pathway days,
- stage wait summaries,
- key validation comparisons.

That will confirm the refactor has not changed behaviour unexpectedly.
