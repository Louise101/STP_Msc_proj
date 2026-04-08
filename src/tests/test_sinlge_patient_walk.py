import datetime as dt
import numpy as np
import pytest

from single_patient_walk import trace_one_patient


def extract_dates(log):
    return [event["date"] for event in log if "date" in event]

# check that dates don't go backwards 
def test_dates_are_monotonic():
    rng = np.random.default_rng(42)
    log, total_days = trace_one_patient(dt.date(2026, 1, 5), rng)

    dates = extract_dates(log)

    for i in range(len(dates) - 1):
        assert dates[i+1] >= dates[i]


#check that total days is correct
def test_total_days_matches_difference():
    rng = np.random.default_rng(123)
    log, total_days = trace_one_patient(dt.date(2026, 1, 5), rng)

    referral_date = log[0]["date"]
    final_date = log[-1]["date"]

    assert total_days == (final_date - referral_date).days

# check that decision node output is always valid
def test_valid_branch_outcomes():
    rng = np.random.default_rng(99)
    log, _ = trace_one_patient(dt.date(2026, 1, 5), rng)

    for event in log:
        if event["event"] == "mdt_decision":
            assert event["outcome"] in [0, 1, 2]


#check that biopsy does not occur when it is not selected
def test_no_biopsy_if_not_selected():

    # try several seeds to increase chance of non-biopsy branch
    for seed in range(10, 30):
        rng = np.random.default_rng(seed)
        log, _ = trace_one_patient(dt.date(2026, 1, 5), rng)

        mdt_event = next(e for e in log if e["event"] == "mdt_decision")

        if mdt_event["outcome"] != 1:  # not biopsy
            biopsy_events = [e for e in log if e["event"] == "biopsy_done"]
            assert len(biopsy_events) == 0

#check MDT treat doesnt occur without cancer 

import datetime as dt
import numpy as np
import pytest

from single_patient_walk import trace_one_patient


def test_treatment_mdt_only_if_cancer():
    saw_pathway_reaching_pathology = False

    for seed in range(1, 100):
        rng = np.random.default_rng(seed)
        log, _ = trace_one_patient(dt.date(2026, 1, 5), rng)

        pathrep_events = [e for e in log if e.get("event") == "Path_report_outcome"]
        treat_mdt_events = [e for e in log if e.get("event") == "Treatment_options_MDT_occured"]

        # If the patient never had a pathology outcome, this rule doesn't apply
        if len(pathrep_events) == 0:
            # Optional: you might want to assert that if no pathology, also no treatment MDT
            assert len(treat_mdt_events) == 0, f"Seed {seed}: treatment MDT occurred without pathology stage"
            continue

        assert len(pathrep_events) == 1, f"Seed {seed}: expected exactly one 'Path report outcome' event"
        saw_pathway_reaching_pathology = True

        outcome = int(pathrep_events[0]["outcome"])  # handles '0'/'1'

        if outcome != 1:  # not cancer
            assert len(treat_mdt_events) == 0, (
                f"Seed {seed}: non-cancer outcome={outcome} but found treatment MDT event(s): {treat_mdt_events}"
            )
        else:  # cancer
            assert len(treat_mdt_events) >= 1, f"Seed {seed}: cancer outcome but no treatment MDT event found"

    # Make sure the test actually exercised the pathology branch at least once
    assert saw_pathway_reaching_pathology, "No tested seeds reached the pathology stage; widen seed range."




