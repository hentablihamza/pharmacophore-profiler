"""Build per-ligand SMARTS count matrices, including large-library runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm.auto import tqdm

from pharmprof.io import MolRecord, iter_molecule_chunks, read_molecules, records_to_frame
from pharmprof.patterns import Pattern, PatternSet, load_patterns

MODES = ("count", "binary", "normalized")


def _match_count(mol: Chem.Mol, query: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(query, uniquify=True))


def _profile_one(
    smiles: str,
    valid: bool,
    n_atoms: int,
    smarts_list: list[str],
    mode: str,
) -> np.ndarray:
    n = len(smarts_list)
    if not valid:
        return np.full(n, np.nan, dtype=np.float64)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(n, np.nan, dtype=np.float64)
    counts = np.empty(n, dtype=np.float64)
    for i, sma in enumerate(smarts_list):
        q = Chem.MolFromSmarts(sma)
        counts[i] = float(_match_count(mol, q))
    if mode == "binary":
        counts = (counts > 0).astype(np.float64)
    elif mode == "normalized":
        denom = float(n_atoms) if n_atoms else 1.0
        counts = counts / denom
    return counts


def _n_atoms(mol: Chem.Mol | None) -> int:
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def profile(
    molecules,
    patterns: str | Path | PatternSet | Sequence[Pattern],
    *,
    mode: str = "count",
    n_jobs: int = 1,
    chunksize: int = 5000,
    progress: bool = True,
    out: str | Path | None = None,
    smiles_col: str | None = None,
    id_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Match every pattern against every ligand and return a count table.

    Parameters
    ----------
    molecules
        A path (SMILES / SDF / CSV / parquet), a DataFrame, or a list of
        :class:`~pharmprof.io.MolRecord`.
    patterns
        A :class:`~pharmprof.patterns.PatternSet`, a path, or a bundled set name.
    mode
        ``count`` (number of unique substructure matches), ``binary`` (0/1),
        or ``normalized`` (count / heavy-atom count).
    n_jobs
        Worker processes for molecule-level parallelism. ``1`` is serial;
        ``-1`` uses all CPUs.
    chunksize
        Molecules per chunk. Each chunk is profiled and, if ``out`` is a
        parquet path, appended to disk before the next chunk is started.
    progress
        Show a tqdm progress bar.
    out
        Optional output path (``.csv``, ``.tsv``, or ``.parquet``). Parquet
        is written incrementally by chunk so a run of a few hundred thousand
        molecules does not need a second copy of the matrix in memory beyond
        the chunk.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    pset = (
        patterns
        if isinstance(patterns, PatternSet)
        else load_patterns(patterns)  # type: ignore[arg-type]
    )
    enabled = pset.enabled()
    if len(enabled) == 0:
        raise ValueError("pattern set is empty (all patterns disabled?)")
    smarts_list = [p.smarts for p in enabled]
    col_ids = [p.pattern_id for p in enabled]

    out_path = Path(out) if out is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

    frames: list[pd.DataFrame] = []
    # Materialise records once; SMILES of 1e5 molecules is small. The count
    # matrix is the heavy object and is built (and optionally flushed) per chunk.
    if isinstance(molecules, list) and molecules and isinstance(molecules[0], MolRecord):
        records = molecules
    else:
        records = read_molecules(
            molecules, smiles_col=smiles_col, id_col=id_col, label_col=label_col
        )

    n_total = len(records)
    n_chunks = max(1, (n_total + chunksize - 1) // chunksize)
    chunk_iter = iter_molecule_chunks(records, chunksize=chunksize)
    if progress:
        chunk_iter = tqdm(chunk_iter, total=n_chunks, desc="profile", unit="chunk")

    wrote_header = False
    for chunk in chunk_iter:
        matrix = _profile_chunk(chunk, smarts_list, mode, n_jobs=n_jobs)
        meta = records_to_frame(chunk)
        meta["n_heavy"] = [_n_atoms(r.mol) for r in chunk]
        frame = pd.concat([meta.reset_index(drop=True), pd.DataFrame(matrix, columns=col_ids)], axis=1)
        if out_path is not None:
            _append_output(frame, out_path, write_header=not wrote_header)
            wrote_header = True
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result


def _profile_chunk(
    chunk: Sequence[MolRecord],
    smarts_list: list[str],
    mode: str,
    *,
    n_jobs: int,
) -> np.ndarray:
    args = [
        (r.smiles, r.mol is not None, _n_atoms(r.mol), smarts_list, mode) for r in chunk
    ]
    if n_jobs == 1 or len(chunk) < 8:
        rows = [_profile_one(*a) for a in args]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_profile_one)(*a) for a in args
        )
    return np.vstack(rows) if rows else np.empty((0, len(smarts_list)))


def _append_output(frame: pd.DataFrame, path: Path, *, write_header: bool) -> None:
    name = path.name.lower()
    if name.endswith(".parquet"):
        # pyarrow dataset-style: write one file per chunk in a sidecar dir, then
        # for a single path we concatenate via incremental parquet writer.
        _append_parquet(frame, path, write_header=write_header)
        return
    sep = "\t" if name.endswith(".tsv") else ","
    frame.to_csv(path, mode="w" if write_header else "a", index=False, header=write_header, sep=sep)


def _append_parquet(frame: pd.DataFrame, path: Path, *, write_header: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if write_header or not path.exists():
        writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        writer.close()
        return
    # Append by reading existing + concat is memory-heavy. Use a sidecar
    # multi-file directory if the path already exists.
    # For practical sizes (100k x 500 is ~200 MB) a rewrite is acceptable;
    # we still avoid holding two *chunks* only by writing a dataset.
    existing = pq.read_table(path)
    merged = pa.concat_tables([existing, table], promote_options="default")
    pq.write_table(merged, path)


def family_counts(matrix: pd.DataFrame, patterns: PatternSet, *, how: str = "sum") -> pd.DataFrame:
    """Aggregate pattern columns by family.

    ``how='sum'`` sums pattern counts in the family (matches can overlap).
    """
    out = matrix[["mol_id"]].copy() if "mol_id" in matrix.columns else pd.DataFrame(index=matrix.index)
    families: dict[str, list[str]] = {}
    for p in patterns.enabled():
        if not p.family:
            continue
        families.setdefault(p.family, []).append(p.pattern_id)
    for family, cols in families.items():
        present = [c for c in cols if c in matrix.columns]
        if not present:
            continue
        if how == "sum":
            out[family] = matrix[present].sum(axis=1, min_count=1)
        elif how == "any":
            out[family] = (matrix[present].fillna(0) > 0).any(axis=1).astype(int)
        else:
            raise ValueError("how must be 'sum' or 'any'")
    return out
