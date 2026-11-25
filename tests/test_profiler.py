"""Profiler matrix correctness against a hand-checked five-molecule set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from pharmprof.io import MolRecord, read_molecules
from pharmprof.patterns import load_patterns
from pharmprof.profiler import family_counts, profile

# Patterns chosen so the counts can be verified by inspection.
#   aromatic6  — exactly one benzene ring
#   hydroxyl   — any [OH], including the OH of a carboxylic acid
#   carboxyl   — C(=O)[OH]
#   primary_aromatic_amine — aniline nitrogen
HAND_PATTERNS = """pattern_id\tname\tfamily\tsmarts\trole\tsource\tlicense\tcitation\tenabled\tnotes
aromatic6\tbenzene ring\tAromatic\tc1ccccc1\tpharmacophore\thand\tMIT\t\ttrue\t
hydroxyl\thydroxyl\tDonor\t[OH]\tpharmacophore\thand\tMIT\t\ttrue\t
carboxyl\tcarboxylic acid\tNegIonizable\tC(=O)[OH]\tpharmacophore\thand\tMIT\t\ttrue\t
aniline\tprimary aromatic amine\tDonor\tc[NH2]\tpharmacophore\thand\tMIT\t\ttrue\t
"""

# smiles, expected (aromatic6, hydroxyl, carboxyl, aniline)
HAND_CASES = [
    ("c1ccccc1", "benzene", (1, 0, 0, 0)),
    ("c1ccccc1O", "phenol", (1, 1, 0, 0)),
    ("c1ccccc1C(=O)O", "benzoic_acid", (1, 1, 1, 0)),
    ("c1ccccc1N", "aniline", (1, 0, 0, 1)),
    ("CCO", "ethanol", (0, 1, 0, 0)),
]


def _pattern_file(tmp_path: Path) -> Path:
    path = tmp_path / "hand.tsv"
    path.write_text(HAND_PATTERNS)
    return path


def test_hand_checked_counts(tmp_path: Path):
    pset = load_patterns(_pattern_file(tmp_path))
    recs = [
        MolRecord(mol_id=name, smiles=smi, mol=Chem.MolFromSmiles(smi))
        for smi, name, _ in HAND_CASES
    ]
    frame = profile(recs, pset, mode="count", n_jobs=1, progress=False)
    assert list(frame["mol_id"]) == [c[1] for c in HAND_CASES]
    for i, (_, name, expected) in enumerate(HAND_CASES):
        got = (
            int(frame.loc[i, "aromatic6"]),
            int(frame.loc[i, "hydroxyl"]),
            int(frame.loc[i, "carboxyl"]),
            int(frame.loc[i, "aniline"]),
        )
        assert got == expected, f"{name}: got {got}, expected {expected}"


def test_binary_and_normalized_modes(tmp_path: Path):
    pset = load_patterns(_pattern_file(tmp_path))
    recs = [
        MolRecord(mol_id="phenol", smiles="c1ccccc1O", mol=Chem.MolFromSmiles("c1ccccc1O"))
    ]
    binary = profile(recs, pset, mode="binary", n_jobs=1, progress=False)
    assert int(binary.loc[0, "aromatic6"]) == 1
    assert int(binary.loc[0, "hydroxyl"]) == 1
    assert int(binary.loc[0, "carboxyl"]) == 0
    norm = profile(recs, pset, mode="normalized", n_jobs=1, progress=False)
    n_heavy = int(norm.loc[0, "n_heavy"])
    assert n_heavy == 7  # C6H5OH
    assert abs(float(norm.loc[0, "hydroxyl"]) - 1 / 7) < 1e-12


def test_invalid_smiles_become_nan(tmp_path: Path):
    pset = load_patterns(_pattern_file(tmp_path))
    recs = [
        MolRecord(mol_id="ok", smiles="CCO", mol=Chem.MolFromSmiles("CCO")),
        MolRecord(mol_id="bad", smiles="not-a-molecule", mol=None),
    ]
    frame = profile(recs, pset, mode="count", n_jobs=1, progress=False)
    assert frame.loc[0, "valid"]
    assert not frame.loc[1, "valid"]
    assert pd.isna(frame.loc[1, "hydroxyl"])
    assert int(frame.loc[0, "hydroxyl"]) == 1


def test_parquet_roundtrip(tmp_path: Path):
    pset = load_patterns(_pattern_file(tmp_path))
    recs = [
        MolRecord(mol_id=name, smiles=smi, mol=Chem.MolFromSmiles(smi))
        for smi, name, _ in HAND_CASES
    ]
    out = tmp_path / "counts.parquet"
    frame = profile(recs, pset, mode="count", n_jobs=1, chunksize=2, progress=False, out=out)
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert list(loaded["mol_id"]) == list(frame["mol_id"])
    assert int(loaded.loc[loaded.mol_id == "phenol", "hydroxyl"].iloc[0]) == 1


def test_family_counts_sum(tmp_path: Path):
    pset = load_patterns(_pattern_file(tmp_path))
    recs = [
        MolRecord(mol_id="benzoic", smiles="c1ccccc1C(=O)O", mol=Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    ]
    frame = profile(recs, pset, mode="count", n_jobs=1, progress=False)
    fam = family_counts(frame, pset, how="sum")
    # Donor family = hydroxyl (1) + aniline (0)
    assert int(fam.loc[0, "Donor"]) == 1
    assert int(fam.loc[0, "Aromatic"]) == 1


def test_read_smi_file(tmp_path: Path):
    path = tmp_path / "ligs.smi"
    path.write_text("CCO ethanol\nc1ccccc1 benzene\n")
    recs = read_molecules(path)
    assert [r.mol_id for r in recs] == ["ethanol", "benzene"]
    assert recs[0].mol is not None
