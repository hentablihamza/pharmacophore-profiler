"""Metric implementations against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmprof.metrics import bedroc, enrichment_factor, pr_auc, roc_auc, screening_metrics
from pharmprof.stats import _bh_fdr, compare_classes, pattern_summary


def test_enrichment_factor_perfect_ranking():
    # 1000 ligands, 100 actives, all 100 actives ranked first.
    y = np.array([1] * 100 + [0] * 900)
    s = np.linspace(1, 0, 1000)
    ef = enrichment_factor(y, s, fraction=0.01)
    # top 1% = 10 molecules, all active; random rate = 0.1; EF = 1/0.1 = 10
    assert ef == pytest.approx(10.0)
    ef05 = enrichment_factor(y, s, fraction=0.005)
    # top 0.5% = 5 molecules, all active; EF = 1/0.1 = 10
    assert ef05 == pytest.approx(10.0)


def test_enrichment_factor_no_better_than_random():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100)
    s = np.zeros(len(y))  # ties: mergesort keeps input order, which is alternating
    # With constant scores the original order is kept; first 1% of an alternating
    # list is also ~50% active, so EF ~ 1.
    ef = enrichment_factor(y, s, fraction=0.01)
    assert ef == pytest.approx(1.0, abs=0.4)


def test_roc_and_pr_perfect_and_random():
    y = np.array([1] * 20 + [0] * 80)
    perfect = np.linspace(1, 0, 100)
    assert roc_auc(y, perfect) == pytest.approx(1.0)
    assert pr_auc(y, perfect) == pytest.approx(1.0)
    # Completely inverted ranking
    assert roc_auc(y, -perfect) == pytest.approx(0.0)


def test_bedroc_perfect_is_near_one():
    # Truchon & Bayly: a perfect early ranking should give BEDROC close to 1.
    y = np.array([1] * 10 + [0] * 90)
    s = np.linspace(1, 0, 100)
    value = bedroc(y, s, alpha=20.0)
    assert value == pytest.approx(1.0, abs=1e-6)


def test_bedroc_hand_computed_n10_nact2():
    # N=10, actives at ranks 1 and 2, alpha=20. Computed from the formula
    # in metrics.bedroc (Truchon & Bayly eqs. 27/32).
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1, dtype=float)
    n_total, n_act, alpha = 10, 2, 20.0
    active_ranks = np.array([1, 2])
    ra = n_act / n_total
    rie_num = float(np.sum(np.exp(-alpha * active_ranks / n_total)))
    rie_den = (n_act / n_total) * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1.0)
    rie = rie_num / rie_den
    expected = rie * ra * np.sinh(alpha / 2.0) / (
        np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra)
    )
    assert bedroc(y, s, alpha=20.0) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(1.0, abs=1e-6)


def test_screening_metrics_keys():
    y = np.array([1] * 10 + [0] * 90)
    s = np.linspace(1, 0, 100)
    m = screening_metrics(y, s)
    assert m["ef1"] == pytest.approx(10.0)
    assert m["ef0.5"] == pytest.approx(10.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert "bedroc" in m and "pr_auc" in m


def test_bh_fdr_known_example():
    # Classic BH: p = 0.01, 0.04, 0.10 with m=3.
    # Sorted raw q = p * m / i = 0.03, 0.06, 0.10; monotone from the tail
    # stays 0.03, 0.06, 0.10.
    p = np.array([0.10, 0.01, 0.04])
    q = _bh_fdr(p)
    assert q[1] == pytest.approx(0.03)
    assert q[2] == pytest.approx(0.06)
    assert q[0] == pytest.approx(0.10)


def test_compare_classes_odds_ratio_and_fisher():
    # Pattern hits 8/10 actives and 1/10 inactives.
    matrix = pd.DataFrame({"pat": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    labels = [1] * 10 + [0] * 10
    out = compare_classes(matrix, labels)
    row = out.iloc[0]
    assert row["hits_actives"] == 8
    assert row["hits_inactives"] == 1
    assert row["enrichment"] == pytest.approx((8 / 10) / (1 / 10))
    assert row["odds_ratio"] == pytest.approx((8 * 9) / (2 * 1))
    assert row["p_value"] < 0.01
    assert 0 <= row["q_value"] <= 1


def test_pattern_summary_hit_rate():
    matrix = pd.DataFrame({"mol_id": ["a", "b", "c", "d"], "p": [2, 0, 1, 0]})
    summary = pattern_summary(matrix)
    row = summary.iloc[0]
    assert row["hit_count"] == 2
    assert row["hit_rate"] == pytest.approx(0.5)
    assert row["mean"] == pytest.approx(0.75)
    assert row["max"] == 2


def test_tied_scores_do_not_follow_file_order():
    """Actives listed first with identical scores must not yield the maximum EF."""
    # 100 actives then 900 decoys, all with the same score.
    y = np.array([1] * 100 + [0] * 900)
    s = np.ones(1000)
    ef = enrichment_factor(y, s, fraction=0.01)
    # Max EF1% would be 10 if the first 10 rows (all actives) were taken.
    assert ef < 8.0
    assert ef == pytest.approx(1.0, abs=0.8)
