"""Fetch and load the bundled DUD-E EGFR benchmark (actives + property-matched decoys)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem

from pharmprof.io import MolRecord, read_molecules

DUDE_TARGET = "egfr"
DUDE_BASE = "https://dude.docking.org/targets/egfr"
ACTIVES_URL = f"{DUDE_BASE}/actives_final.ism"
DECOYS_URL = f"{DUDE_BASE}/decoys_final.ism"

# Mysinger et al., J. Med. Chem. 2012, 55, 6582-6594.
CITATION = (
    "Mysinger MM, Carchia M, Irwin JJ, Shoichet BK. "
    "Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking. "
    "J Med Chem. 2012;55(14):6582-6594. doi:10.1021/jm300687e"
)
TARGET_NOTE = (
    "DUD-E target EGFR (UniProt P00533 / Q01279, ChEMBL CHEMBL203). "
    "Actives compiled by DUD-E from ChEMBL/literature; decoys are property-matched "
    "(~50 decoys per active, similar physicochemical properties, dissimilar 2-D topology)."
)

SUBSET_SEED = 0
SUBSET_N_DECOYS = 2500


def default_data_dir() -> Path:
    """Directory that holds bundled and downloaded EGFR files.

    Looks at ``PHARMPROF_DATA``, then a ``data/`` folder next to the package
    checkout, then ``~/.cache/pharmprof``.
    """
    env = os.environ.get("PHARMPROF_DATA")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    # repo_root/data when installed editable from the checkout
    for parent in here.parents:
        candidate = parent / "data" / "dude-egfr"
        if candidate.exists() or (parent / "pyproject.toml").exists():
            return parent / "data" / "dude-egfr"
    return Path.home() / ".cache" / "pharmprof" / "dude-egfr"


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed official URL
    tmp.replace(dest)
    return dest


def fetch_dude_egfr(*, data_dir: str | Path | None = None, force: bool = False) -> dict[str, Path]:
    """Download the official DUD-E EGFR ``actives_final.ism`` and ``decoys_final.ism``."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, url in (("actives", ACTIVES_URL), ("decoys", DECOYS_URL)):
        dest = raw / f"{kind}_final.ism"
        if force and dest.exists():
            dest.unlink()
        paths[kind] = _download(url, dest)
    return paths


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _records_to_unique_frame(records: list[MolRecord], label: int) -> pd.DataFrame:
    rows = []
    seen = set()
    for r in records:
        if r.mol is None:
            continue
        key = r.smiles
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "smiles": r.smiles,
                "mol_id": r.mol_id,
                "label": label,
                "chembl_id": r.extra.get("chembl_id", ""),
            }
        )
    return pd.DataFrame(rows)


def load_egfr(
    *,
    subset: bool = True,
    data_dir: str | Path | None = None,
    fetch: bool = True,
) -> pd.DataFrame:
    """Load EGFR actives and property-matched decoys as a labelled table.

    ``subset=True`` (default) returns the bundled set: every unique active plus
    a reproducible ``SUBSET_N_DECOYS``-decoy sample, so notebooks run offline.
    ``subset=False`` loads the official full DUD-E files (downloaded on demand
    into the cache).
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if subset:
        bundled = root / "egfr_subset.csv.gz"
        if bundled.exists():
            return pd.read_csv(bundled)
        # Fall through and build from official files if the bundle is missing.
    if fetch:
        fetch_dude_egfr(data_dir=root)
    raw = root / "raw"
    actives = read_molecules(raw / "actives_final.ism")
    decoys = read_molecules(raw / "decoys_final.ism")
    aframe = _records_to_unique_frame(actives, 1)
    dframe = _records_to_unique_frame(decoys, 0)
    if subset:
        dframe = dframe.sample(n=min(SUBSET_N_DECOYS, len(dframe)), random_state=SUBSET_SEED)
    out = pd.concat([aframe, dframe], ignore_index=True)
    out["target"] = "egfr"
    out["source"] = "DUD-E"
    return out


def write_subset(
    dest: str | Path,
    *,
    data_dir: str | Path | None = None,
    n_decoys: int = SUBSET_N_DECOYS,
    seed: int = SUBSET_SEED,
) -> pd.DataFrame:
    """Build the bundled subset from already-downloaded official files."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    fetch_dude_egfr(data_dir=root)
    raw = root / "raw"
    aframe = _records_to_unique_frame(read_molecules(raw / "actives_final.ism"), 1)
    dframe = _records_to_unique_frame(read_molecules(raw / "decoys_final.ism"), 0)
    dframe = dframe.sample(n=min(n_decoys, len(dframe)), random_state=seed)
    out = pd.concat([aframe, dframe], ignore_index=True)
    out["target"] = "egfr"
    out["source"] = "DUD-E"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".gz" or dest.name.endswith(".csv.gz"):
        out.to_csv(dest, index=False, compression="gzip")
    else:
        out.to_csv(dest, index=False)
    return out
