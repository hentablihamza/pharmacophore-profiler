"""Pattern-pattern correlation flags redundant columns."""

from __future__ import annotations

import pandas as pd

from pharmprof.stats import pattern_correlation


def test_identical_columns_are_redundant():
    matrix = pd.DataFrame(
        {
            "mol_id": list("abcdef"),
            "p1": [1, 0, 2, 0, 1, 3],
            "p2": [1, 0, 2, 0, 1, 3],  # identical to p1
            "p3": [0, 1, 0, 1, 0, 0],
        }
    )
    corr, pairs = pattern_correlation(matrix, min_abs=0.9, method="pearson")
    assert corr.loc["p1", "p2"] == 1.0
    assert {frozenset((r.pattern_a, r.pattern_b)) for r in pairs.itertuples()} == {frozenset(("p1", "p2"))}


def test_jaccard_on_binary():
    matrix = pd.DataFrame({"p1": [1, 1, 0, 0], "p2": [1, 1, 0, 0], "p3": [0, 0, 1, 1]})
    corr, pairs = pattern_correlation(matrix, min_abs=0.99, method="jaccard")
    assert corr.loc["p1", "p2"] == 1.0
    assert corr.loc["p1", "p3"] == 0.0
    assert len(pairs) == 1
