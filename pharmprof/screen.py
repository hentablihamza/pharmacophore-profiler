"""Library filters and Gobbi Pharm2D similarity scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from tqdm.auto import tqdm

from pharmprof.io import read_molecules
from pharmprof.patterns import PatternSet


@dataclass
class FilterClause:
    """A required or forbidden constraint on a pattern or family."""

    key: str
    minimum: float = 1.0
    maximum: float | None = None
    kind: str = "pattern"  # 'pattern' or 'family'


def _resolve_columns(key: str, matrix: pd.DataFrame, patterns: PatternSet | None) -> list[str]:
    if key in matrix.columns:
        return [key]
    if patterns is not None:
        # name or family
        ids = [
            p.pattern_id
            for p in patterns.enabled()
            if p.pattern_id == key or p.name == key or p.family.lower() == key.lower()
        ]
        ids = [i for i in ids if i in matrix.columns]
        if ids:
            return ids
    matches = [c for c in matrix.columns if c == key or c.endswith("." + key) or c.endswith(key)]
    if matches:
        return matches
    raise KeyError(f"cannot resolve filter key {key!r} to a matrix column")


def apply_filter(
    matrix: pd.DataFrame,
    *,
    require: Sequence[str] | None = None,
    forbid: Sequence[str] | None = None,
    min_counts: dict[str, float] | None = None,
    max_counts: dict[str, float] | None = None,
    patterns: PatternSet | None = None,
) -> pd.Series:
    """Return a boolean mask over ``matrix`` for a required / forbidden combination.

    ``require`` / ``forbid`` entries may be pattern ids, pattern names, or
    family names. A family requirement is satisfied if *any* pattern in that
    family is present (count > 0). ``min_counts`` / ``max_counts`` apply to
    individual pattern columns (or the sum of a family).
    """
    mask = pd.Series(True, index=matrix.index)
    for key in require or []:
        cols = _resolve_columns(key, matrix, patterns)
        present = (matrix[cols].fillna(0) > 0).any(axis=1)
        mask &= present
    for key in forbid or []:
        cols = _resolve_columns(key, matrix, patterns)
        present = (matrix[cols].fillna(0) > 0).any(axis=1)
        mask &= ~present
    for key, minimum in (min_counts or {}).items():
        cols = _resolve_columns(key, matrix, patterns)
        mask &= matrix[cols].fillna(0).sum(axis=1) >= minimum
    for key, maximum in (max_counts or {}).items():
        cols = _resolve_columns(key, matrix, patterns)
        mask &= matrix[cols].fillna(0).sum(axis=1) <= maximum
    return mask


def _gobbi_fp(mol: Chem.Mol):
    # Gobbi donor definitions inspect explicit hydrogens. Without AddHs a
    # phenol yields an empty fingerprint and Tanimoto 0 against itself.
    return Generate.Gen2DFingerprint(Chem.AddHs(mol), Gobbi_Pharm2D.factory)


def pharm2d_similarity(
    query,
    library,
    *,
    aggregation: str = "max",
    progress: bool = False,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Tanimoto similarity of Gobbi Pharm2D fingerprints against a query set.

    ``aggregation`` is ``max`` or ``mean`` over query molecules. Invalid
    molecules receive a score of NaN. When ``exclude_self`` is true (default),
    a library molecule is never compared to a query with the same ``mol_id``,
    so a retrospective screen does not get a free Tanimoto of 1.0 against itself.
    """
    if aggregation not in {"max", "mean"}:
        raise ValueError("aggregation must be 'max' or 'mean'")
    qrecs = query if isinstance(query, list) else read_molecules(query)
    lrecs = library if isinstance(library, list) else read_molecules(library)
    qfps = []
    qids = []
    for r in qrecs:
        if r.mol is None:
            continue
        try:
            qfps.append(_gobbi_fp(r.mol))
            qids.append(str(r.mol_id))
        except Exception:
            continue
    if not qfps:
        raise ValueError("no valid query molecules with Gobbi fingerprints")

    rows = []
    iterable = tqdm(lrecs, desc="pharm2d", disable=not progress)
    for r in iterable:
        if r.mol is None:
            rows.append({"mol_id": r.mol_id, "smiles": r.smiles, "pharm2d_tanimoto": np.nan})
            continue
        try:
            fp = _gobbi_fp(r.mol)
            sims = [
                DataStructs.TanimotoSimilarity(fp, q)
                for q, qid in zip(qfps, qids, strict=True)
                if not (exclude_self and qid == str(r.mol_id))
            ]
            score = float(np.max(sims) if aggregation == "max" else np.mean(sims)) if sims else float("nan")
        except Exception:
            score = float("nan")
        rows.append({"mol_id": r.mol_id, "smiles": r.smiles, "pharm2d_tanimoto": score})
    return pd.DataFrame(rows)
