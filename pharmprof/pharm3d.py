"""Optional 3D pharmacophore tools: conformers, FeatureFactory sites, EmbedLib matching.

This path is substantially slower than 2D SMARTS profiling. Use it on a
shortlist, not on a 10^5-compound library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.Pharm3D import EmbedLib, Pharmacophore

from pharmprof.io import read_molecules
from pharmprof.patterns import feature_factory

try:
    from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore as Pharm3D
except ImportError:  # pragma: no cover
    Pharm3D = Pharmacophore.Pharmacophore


@dataclass
class ConformerResult:
    mol: Chem.Mol
    n_conformers: int
    rmsd_ok: bool


def generate_conformers(
    mol: Chem.Mol,
    n_confs: int = 10,
    *,
    random_seed: int = 0xC0FFEE,
    prune_rms: float = 0.5,
    add_hs: bool = True,
) -> Chem.Mol:
    """Embed ``n_confs`` ETKDG conformers. Returns a new molecule with Hs."""
    if mol is None:
        raise ValueError("cannot generate conformers for an invalid molecule")
    work = Chem.Mol(mol)
    if add_hs:
        work = Chem.AddHs(work)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.pruneRmsThresh = float(prune_rms)
    AllChem.EmbedMultipleConfs(work, numConfs=n_confs, params=params)
    if work.GetNumConformers() == 0:
        # Last-ditch single embed without pruning.
        AllChem.EmbedMolecule(work, randomSeed=int(random_seed))
    return work


def features_for_conformer(mol: Chem.Mol, factory=None, conf_id: int = -1) -> list:
    """RDKit chemical features on one conformer (requires 3D coordinates)."""
    fac = factory if factory is not None else feature_factory("basefeatures")
    return list(fac.GetFeaturesForMol(mol, confId=conf_id))


def pharmacophore_from_reference(
    mol: Chem.Mol,
    factory=None,
    *,
    families: Sequence[str] | None = None,
    tolerance: float = 1.0,
    conf_id: int = -1,
):
    """Build a Pharm3D pharmacophore from a reference ligand's feature sites.

    Distance bounds are the observed 3D distances ± ``tolerance`` (Å). Features
    can be restricted to ``families`` (e.g. Donor, Acceptor, Aromatic).
    """
    fac = factory if factory is not None else feature_factory("basefeatures")
    feats = features_for_conformer(mol, fac, conf_id=conf_id)
    if families is not None:
        allowed = {f.lower() for f in families}
        feats = [f for f in feats if f.GetFamily().lower() in allowed]
    if len(feats) < 2:
        raise ValueError(
            f"need at least two 3D features to build a pharmacophore, found {len(feats)}"
        )
    pharm = Pharm3D(feats)
    n = len(feats)
    for i in range(n):
        pi = np.array(feats[i].GetPos(confId=conf_id))
        for j in range(i + 1, n):
            pj = np.array(feats[j].GetPos(confId=conf_id))
            d = float(np.linalg.norm(pi - pj))
            lo = max(0.0, d - tolerance)
            hi = d + tolerance
            pharm.setLowerBound(i, j, lo)
            pharm.setUpperBound(i, j, hi)
    return pharm, feats


def match_pharmacophore(
    library,
    pharm,
    factory=None,
    *,
    generate: bool = True,
    n_confs: int = 5,
    random_seed: int = 0xC0FFEE,
    progress: bool = False,
) -> pd.DataFrame:
    """Try to match a Pharm3D pharmacophore against each library molecule.

    When ``generate=True`` (default) ETKDG conformers are built first.
    EmbedLib matching is the expensive step and is intended for small sets.
    """
    from tqdm.auto import tqdm

    fac = factory if factory is not None else feature_factory("basefeatures")
    recs = library if isinstance(library, list) else read_molecules(library)
    rows = []
    iterable = tqdm(recs, desc="pharm3d", disable=not progress)
    for r in iterable:
        row = {
            "mol_id": r.mol_id,
            "smiles": r.smiles,
            "matched": False,
            "n_feature_matches": 0,
            "embedded": False,
            "error": "",
        }
        if r.mol is None:
            row["error"] = "invalid molecule"
            rows.append(row)
            continue
        try:
            mol = generate_conformers(r.mol, n_confs=n_confs, random_seed=random_seed) if generate else Chem.AddHs(Chem.Mol(r.mol))
            can_match, matches = EmbedLib.MatchPharmacophoreToMol(mol, fac, pharm)
            row["n_feature_matches"] = len(matches) if matches else 0
            row["matched"] = bool(can_match)
            if can_match:
                try:
                    bounds_matrix = Chem.GetMoleculeBoundsMatrix(mol)
                    failed, _, used, _ = EmbedLib.MatchPharmacophore(matches, bounds_matrix, pharm)
                    if not failed:
                        ok, rms, embed_mol, _e = EmbedLib.EmbedPharmacophore(
                            mol, used, pharm, count=1, silent=True
                        )
                        row["embedded"] = bool(ok) or (embed_mol is not None)
                        if rms is not None:
                            try:
                                row["embed_rms"] = float(rms)
                            except (TypeError, ValueError):
                                pass
                except Exception as exc:  # EmbedLib is picky; keep the 2D/feature match.
                    row["error"] = f"embed failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)
