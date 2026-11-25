"""Virtual-screening metrics: EF, BEDROC, ROC-AUC, PR-AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Default BEDROC alpha recommended by Truchon & Bayly (2007) for early enrichment.
DEFAULT_BEDROC_ALPHA = 20.0


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask].astype(int), s[mask]


def _order_by_score(scores: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Descending score order; ties are broken randomly but reproducibly."""
    rng = np.random.default_rng(seed)
    noise = rng.random(len(scores))
    # lexsort: last key is primary
    return np.lexsort((noise, -np.asarray(scores, dtype=float)))


def enrichment_factor(y_true, y_score, fraction: float = 0.01) -> float:
    """Enrichment factor at a fraction of the ranked list.

    ``EF_x% = (actives in the top x%) / (x% of all actives expected at random)``.
    The top slice size is ``max(1, round(fraction * N))``. Ties are broken randomly with a fixed seed so file order cannot inflate early enrichment.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    y, s = _as_arrays(y_true, y_score)
    n_total = len(y)
    n_act = int(y.sum())
    if n_total == 0 or n_act == 0:
        return float("nan")
    k = max(1, int(round(fraction * n_total)))
    order = _order_by_score(s, seed=0)
    hits = int(y[order][:k].sum())
    return (hits / k) / (n_act / n_total)


def bedroc(y_true, y_score, alpha: float = DEFAULT_BEDROC_ALPHA) -> float:
    """Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007).

    Ranks are 1-based positions in the score-descending list. ``alpha=20``
    is the value recommended in the original paper for early enrichment.

    Reference: Truchon J-F, Bayly CI. J. Chem. Inf. Model. 2007, 47, 488-508.
    doi:10.1021/ci600426e
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    y, s = _as_arrays(y_true, y_score)
    n_total = len(y)
    n_act = int(y.sum())
    if n_total == 0 or n_act == 0 or n_act == n_total:
        return float("nan")
    order = _order_by_score(s, seed=0)
    ranks = np.empty(n_total, dtype=int)
    ranks[order] = np.arange(1, n_total + 1)
    active_ranks = ranks[y == 1]
    ra = n_act / n_total
    rie_num = float(np.sum(np.exp(-alpha * active_ranks / n_total)))
    rie_den = (n_act / n_total) * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1.0)
    rie = rie_num / rie_den
    # Eq. 27 / 32 in Truchon & Bayly.
    numer = rie * ra * np.sinh(alpha / 2.0)
    denom = np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra)
    return float(numer / denom)


def roc_auc(y_true, y_score) -> float:
    y, s = _as_arrays(y_true, y_score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision)."""
    y, s = _as_arrays(y_true, y_score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def screening_metrics(
    y_true,
    y_score,
    *,
    fractions: tuple[float, ...] = (0.005, 0.01),
    alpha: float = DEFAULT_BEDROC_ALPHA,
) -> pd.Series:
    """Compute the standard early-enrichment suite on a labelled ranking."""
    result = {
        "n": int(np.isfinite(np.asarray(y_true, dtype=float)).sum()),
        "n_actives": int(np.nansum(np.asarray(y_true, dtype=float))),
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "bedroc": bedroc(y_true, y_score, alpha=alpha),
        "bedroc_alpha": alpha,
    }
    for frac in fractions:
        key = f"ef{frac * 100:.1f}".replace(".0", "")
        # ef1, ef0.5 — keep a stable name
        if abs(frac - 0.01) < 1e-12:
            key = "ef1"
        elif abs(frac - 0.005) < 1e-12:
            key = "ef0.5"
        result[key] = enrichment_factor(y_true, y_score, fraction=frac)
    return pd.Series(result)
