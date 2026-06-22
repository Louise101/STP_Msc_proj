from pathlib import Path

import pandas as pd
import pytest

from empirical_inputs import build_pdfs, build_branching


def test_build_pdfs_contains_only_non_negative_waits():
    pdfs = build_pdfs()

    assert len(pdfs) > 0

    for name, samples in pdfs.items():
        s = pd.Series(samples).dropna()

        assert len(s) > 0, f"{name} contains no valid samples"
        assert (s >= 0).all(), f"{name} contains negative waits"


def test_build_branching_probabilities_sum_to_one():
    branching = build_branching()

    assert len(branching) > 0

    for name, probs in branching.items():
        total = sum(probs.values())

        assert total == pytest.approx(1.0)
        assert all(p >= 0 for p in probs.values())
        assert all(isinstance(k, int) for k in probs.keys())


def test_build_branching_contains_expected_keys():
    branching = build_branching()

    assert "biopmdt_outcome" in branching
    assert "pathrep_outcome" in branching