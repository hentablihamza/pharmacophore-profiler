"""Per-pattern summaries, actives-vs-inactives enrichment, and redundancy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from pharmprof.patterns import PatternSet


def _pattern_columns(matrix: pd.DataFrame, patterns: PatternSet | None) -> list[str]:
    reserved = {"mol_id", "smiles", "valid", "label", "n_heavy"}
    if patterns is not None:
        return [p.pattern_id for p in patterns.enabled() if p.pattern_id in matrix.columns]
    return [c for c in matrix.columns if c not in reserved and pd.api.types.is_numeric_dtype(matrix[c])]


def pattern_summary(matrix: pd.DataFrame, patterns: PatternSet | None = None) -> pd.DataFrame:
    """Hit rate and count-distribution summaries for every pattern column."""
    cols = _pattern_columns(matrix, patterns)
    rows = []
    for col in cols:
        s = pd.to_numeric(matrix[col], errors="coerce")
        valid = s.dropna()
        rows.append(
            {
                "pattern_id": col,
                "n": int(valid.shape[0]),
                "n_missing": int(s.isna().sum()),
                "hit_count": int((valid > 0).sum()),
                "hit_rate": float((valid > 0).mean()) if len(valid) else float("nan"),
                "mean": float(valid.mean()) if len(valid) else float("nan"),
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                "median": float(valid.median()) if len(valid) else float("nan"),
                "p90": float(valid.quantile(0.90)) if len(valid) else float("nan"),
                "max": float(valid.max()) if len(valid) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if patterns is not None and not out.empty:
        meta = {p.pattern_id: p for p in patterns.enabled()}
        out.insert(1, "name", out["pattern_id"].map(lambda i: getattr(meta.get(i), "name", "")))
        out.insert(2, "family", out["pattern_id"].map(lambda i: getattr(meta.get(i), "family", "")))
        out.insert(3, "role", out["pattern_id"].map(lambda i: getattr(meta.get(i), "role", "")))
    return out


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values in original order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    q = np.full(n, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order]
    m = len(order)
    raw = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the tail.
    mono = np.minimum.accumulate(raw[::-1])[::-1]
    q[order] = np.clip(mono, 0, 1)
    return q


def compare_classes(
    matrix: pd.DataFrame,
    labels,
    patterns: PatternSet | None = None,
    *,
    positive: int = 1,
) -> pd.DataFrame:
    """Actives-vs-inactives comparison for each pattern.

    For each pattern a 2x2 table is built from *presence* (count > 0):

    * enrichment = (hit rate in actives) / (hit rate in inactives)
    * odds ratio with Haldane-Anscombe +0.5 correction when a cell is empty
    * two-sided Fisher's exact p-value
    * Benjamini-Hochberg q-value across patterns
    """
    cols = _pattern_columns(matrix, patterns)
    y = np.asarray(labels)
    if len(y) != len(matrix):
        raise ValueError("labels and matrix have different lengths")
    active = y == positive
    inactive = np.isin(y, [0, 1]) & ~active
    n_act = int(active.sum())
    n_inact = int(inactive.sum())
    rows = []
    for col in cols:
        s = pd.to_numeric(matrix[col], errors="coerce").to_numpy()
        a_mask = active & np.isfinite(s)
        i_mask = inactive & np.isfinite(s)
        act_hits = int((s[a_mask] > 0).sum())
        inact_hits = int((s[i_mask] > 0).sum())
        act_miss = int(a_mask.sum()) - act_hits
        inact_miss = int(i_mask.sum()) - inact_hits
        hr_a = act_hits / n_act if n_act else float("nan")
        hr_i = inact_hits / n_inact if n_inact else float("nan")
        enrichment = hr_a / hr_i if hr_i and hr_i > 0 else (np.inf if hr_a > 0 else float("nan"))
        # Haldane-Anscombe correction for the odds ratio when a cell is 0.
        table = np.array([[act_hits, act_miss], [inact_hits, inact_miss]], dtype=float)
        if (table == 0).any():
            or_val = ((act_hits + 0.5) * (inact_miss + 0.5)) / ((act_miss + 0.5) * (inact_hits + 0.5))
        else:
            or_val = (act_hits * inact_miss) / (act_miss * inact_hits)
        try:
            _, pvalue = fisher_exact(
                [[act_hits, act_miss], [inact_hits, inact_miss]], alternative="two-sided"
            )
        except ValueError:
            pvalue = float("nan")
        rows.append(
            {
                "pattern_id": col,
                "n_actives": n_act,
                "n_inactives": n_inact,
                "hits_actives": act_hits,
                "hits_inactives": inact_hits,
                "hit_rate_actives": hr_a,
                "hit_rate_inactives": hr_i,
                "enrichment": enrichment,
                "odds_ratio": float(or_val),
                "p_value": float(pvalue),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = _bh_fdr(out["p_value"].to_numpy())
    if patterns is not None and not out.empty:
        meta = {p.pattern_id: p for p in patterns.enabled()}
        out.insert(1, "name", out["pattern_id"].map(lambda i: getattr(meta.get(i), "name", "")))
        out.insert(2, "family", out["pattern_id"].map(lambda i: getattr(meta.get(i), "family", "")))
    return out.sort_values(["q_value", "p_value", "pattern_id"], ignore_index=True)


def pattern_correlation(
    matrix: pd.DataFrame,
    patterns: PatternSet | None = None,
    *,
    method: str = "pearson",
    min_abs: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pattern-pattern correlation, plus the pairs above ``min_abs``.

    ``method`` is ``pearson`` (on counts) or ``jaccard`` (on binary presence).
    Returns ``(square correlation matrix, long table of redundant pairs)``.
    """
    cols = _pattern_columns(matrix, patterns)
    data = matrix[cols].apply(pd.to_numeric, errors="coerce")
    if method == "pearson":
        corr = data.corr(method="pearson")
    elif method == "jaccard":
        binary = (data.fillna(0) > 0).astype(int)
        arr = binary.to_numpy()
        # Jaccard = (A and B) / (A or B)
        inter = arr.T @ arr
        n_i = arr.sum(axis=0)
        union = n_i[:, None] + n_i[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(union > 0, inter / union, np.nan)
        corr = pd.DataFrame(j, index=cols, columns=cols)
    else:
        raise ValueError("method must be 'pearson' or 'jaccard'")

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if not np.isfinite(val):
                continue
            if abs(val) >= min_abs:
                pairs.append({"pattern_a": a, "pattern_b": b, "correlation": float(val), "method": method})
    pairs_df = pd.DataFrame(pairs, columns=["pattern_a", "pattern_b", "correlation", "method"])
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("correlation", ascending=False, ignore_index=True)
    return corr, pairs_df


