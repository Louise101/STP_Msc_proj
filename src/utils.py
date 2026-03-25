import numpy as np
import pandas as pd


def extract_mdt_to_biopsy_waits(patient_results):
    """
    Extract total wait from MDT_occured to biopsy_done from patient event logs.

    patient_results is assumed to be a list of tuples:
        (events_list, total_pathway_days)
    """
    waits = []

    for result in patient_results:
        if not isinstance(result, tuple) or len(result) < 1:
            continue

        events = result[0]
        if not isinstance(events, list):
            continue

        mdt_date = None
        biopsy_date = None

        for event in events:
            event_name = event.get("event")

            if event_name == "MDT_occured":
                mdt_date = event.get("date")

            elif event_name == "biopsy_done":
                biopsy_date = event.get("date")

        if mdt_date is not None and biopsy_date is not None:
            waits.append((biopsy_date - mdt_date).days)

    return np.asarray(waits)


def summarise_waits(waits, label="waits"):
    waits = np.asarray(waits)

    if len(waits) == 0:
        return pd.DataFrame([{
            "label": label,
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }])

    return pd.DataFrame([{
        "label": label,
        "n": len(waits),
        "mean": float(np.mean(waits)),
        "median": float(np.median(waits)),
        "p90": float(np.percentile(waits, 90)),
        "min": float(np.min(waits)),
        "max": float(np.max(waits)),
    }])