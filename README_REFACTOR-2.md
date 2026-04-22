Refactored simulation package
============================

This folder contains a cleaned package structure built around the **combined engine**.

What changed
------------
- The project now has **one engine family** instead of parallel old/new engines.
- Scenario setup now lives in **one central registry**: `src/config/scenarios.py`.
- Shared analysis helpers now live in `src/analysis/` instead of being copied into multiple scripts.
- Shared data preparation now lives in `src/data_prep/`.

Main entry points
-----------------
- `src/config/scenarios.py`
  - define scenarios here
  - add new scenarios here
  - generate common referral schedules here
- `src/engine/combined_engine.py`
  - the main simulation engine
- `src/runners/run_scenarios.py`
  - example runner showing how to run named scenarios from the scenario registry

Recommended migration
---------------------
1. Compare the old engine outputs against the new combined-engine outputs for:
   - ALL_BASELINE
   - ALL_PROSTAD
2. Confirm key metrics are acceptably similar:
   - mean pathway time
   - stage wait summaries
   - flow counts
   - validation plots
3. Once satisfied, retire the old engine files.

Files that are likely legacy after migration
--------------------------------------------
- des_engine.py
- stage_engine2.py
- scenarios.py
- validate_scenarios.py
- single_patient_walk_mdtday.py

Design rule for future work
---------------------------
If you want a new scenario, add it to the scenario library instead of creating another bespoke runner script.
