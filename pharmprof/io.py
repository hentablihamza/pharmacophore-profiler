"""Read and write molecule tables (SMILES, SDF, CSV, TSV, parquet)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

_SMILES_COL_CANDIDATES = ("smiles", "smi", "canonical_smiles", "iso_smiles", "ism")
_ID_COL_CANDIDATES = ("mol_id", "id", "name", "compound_id", "zinc_id", "chembl_id")
_LABEL_COL_CANDIDATES = ("label", "y", "active", "is_active", "activity")


@dataclass
class MolRecord:
    mol_id: str
    smiles: str
    mol: Chem.Mol | None
    label: int | None = None
    extra: dict = field(default_factory=dict)


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _stem_suffixes(path: Path) -> str:
    name = path.name.lower()
    for ext in (".sdf.gz", ".smi.gz", ".ism.gz", ".csv.gz", ".tsv.gz"):
        if name.endswith(ext):
            return ext
    return path.suffix.lower()


def read_molecules(
    source: str | Path | pd.DataFrame,
    *,
    smiles_col: str | None = None,
    id_col: str | None = None,
    label_col: str | None = None,
    sanitize: bool = True,
) -> list[MolRecord]:
    """Load molecules from a file or a DataFrame into a list of records."""
    if isinstance(source, pd.DataFrame):
        return list(_records_from_frame(source, smiles_col, id_col, label_col, sanitize))

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    ext = _stem_suffixes(path)
    if ext in {".sdf", ".sdf.gz"}:
        return list(_read_sdf(path, sanitize=sanitize, label_col=label_col))
    if ext in {".smi", ".ism", ".smi.gz", ".ism.gz"}:
        return list(_read_smi(path, sanitize=sanitize))
    if ext in {".csv", ".tsv", ".csv.gz", ".tsv.gz", ".parquet"}:
        if ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            sep = "\t" if ".tsv" in ext else ","
            df = pd.read_csv(path, sep=sep)
        return list(_records_from_frame(df, smiles_col, id_col, label_col, sanitize))
    # Fall back to SMILES-like
    return list(_read_smi(path, sanitize=sanitize))


def iter_molecule_chunks(
    source: str | Path | pd.DataFrame | Sequence[MolRecord],
    *,
    chunksize: int = 5000,
    smiles_col: str | None = None,
    id_col: str | None = None,
    label_col: str | None = None,
    sanitize: bool = True,
) -> Iterator[list[MolRecord]]:
    """Yield records in chunks so large libraries need not sit in memory twice."""
    if isinstance(source, list):
        recs = source
        for i in range(0, len(recs), chunksize):
            yield recs[i : i + chunksize]
        return
    # For files we still load then slice: SMILES files of a few 100k are small.
    recs = read_molecules(
        source, smiles_col=smiles_col, id_col=id_col, label_col=label_col, sanitize=sanitize
    )
    for i in range(0, len(recs), chunksize):
        yield recs[i : i + chunksize]


def _pick_column(columns, candidates, explicit: str | None) -> str | None:
    if explicit is not None:
        if explicit not in columns:
            raise KeyError(f"column {explicit!r} not in {list(columns)}")
        return explicit
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _mol_from_smiles(smiles: str, sanitize: bool) -> Chem.Mol | None:
    if not smiles or not str(smiles).strip():
        return None
    return Chem.MolFromSmiles(str(smiles).strip(), sanitize=sanitize)


def _canonical_smiles(mol: Chem.Mol | None, fallback: str = "") -> str:
    if mol is None:
        return fallback
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return fallback


def _records_from_frame(
    df: pd.DataFrame,
    smiles_col: str | None,
    id_col: str | None,
    label_col: str | None,
    sanitize: bool,
) -> Iterator[MolRecord]:
    smi_name = _pick_column(df.columns, _SMILES_COL_CANDIDATES, smiles_col)
    if smi_name is None:
        raise KeyError(
            f"no SMILES column found; pass smiles_col=. columns={list(df.columns)}"
        )
    id_name = _pick_column(df.columns, _ID_COL_CANDIDATES, id_col)
    lab_name = _pick_column(df.columns, _LABEL_COL_CANDIDATES, label_col)
    for i, row in df.iterrows():
        smiles = row[smi_name]
        mol = _mol_from_smiles("" if pd.isna(smiles) else str(smiles), sanitize)
        mol_id = str(row[id_name]) if id_name and not pd.isna(row[id_name]) else f"mol_{i}"
        label = None
        if lab_name is not None and not pd.isna(row[lab_name]):
            label = _coerce_label(row[lab_name])
        extra = {c: row[c] for c in df.columns if c not in {smi_name, id_name, lab_name}}
        yield MolRecord(
            mol_id=mol_id,
            smiles=_canonical_smiles(mol, str(smiles) if not pd.isna(smiles) else ""),
            mol=mol,
            label=label,
            extra=extra,
        )


def _coerce_label(value) -> int:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "active", "yes", "act", "binder"}:
            return 1
        if s in {"0", "false", "inactive", "no", "decoy", "inact", "nonbinder"}:
            return 0
    return int(bool(int(value))) if str(value).strip() not in {"", "nan"} else 0


def _read_smi(path: Path, sanitize: bool) -> Iterator[MolRecord]:
    """Read whitespace-delimited SMILES files, including DUD-E ``.ism``."""
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"mol_{i + 1}"
            extra: dict = {}
            if len(parts) > 2:
                # DUD-E actives: SMILES id chembl_id
                extra["chembl_id"] = parts[2]
                if len(parts) > 3:
                    extra["rest"] = " ".join(parts[3:])
            mol = _mol_from_smiles(smiles, sanitize)
            yield MolRecord(
                mol_id=mol_id,
                smiles=_canonical_smiles(mol, smiles),
                mol=mol,
                extra=extra,
            )


def _read_sdf(path: Path, sanitize: bool, label_col: str | None) -> Iterator[MolRecord]:
    opener = path.open("rb")
    if path.name.endswith(".gz"):
        import gzip

        opener = gzip.open(path, "rb")
    with opener as fh:
        suppl = Chem.ForwardSDMolSupplier(fh, sanitize=sanitize, removeHs=True)
        for i, mol in enumerate(suppl):
            if mol is None:
                yield MolRecord(mol_id=f"mol_{i + 1}", smiles="", mol=None)
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i + 1}"
            extra = {}
            label = None
            for k in mol.GetPropNames():
                extra[k] = mol.GetProp(k)
            if label_col and label_col in extra:
                label = _coerce_label(extra[label_col])
            yield MolRecord(
                mol_id=name or f"mol_{i + 1}",
                smiles=_canonical_smiles(mol),
                mol=mol,
                label=label,
                extra=extra,
            )


def records_to_frame(records: Sequence[MolRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"mol_id": r.mol_id, "smiles": r.smiles, "valid": r.mol is not None}
        if r.label is not None:
            row["label"] = r.label
        rows.append(row)
    return pd.DataFrame(rows)


def inchikey(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None
