"""Library filter and Gobbi similarity smoke tests."""

from __future__ import annotations

from rdkit import Chem

from pharmprof.io import MolRecord
from pharmprof.patterns import load_patterns
from pharmprof.profiler import profile
from pharmprof.screen import apply_filter, pharm2d_similarity


def _recs():
    mols = {
        "phenol": "c1ccccc1O",
        "benzene": "c1ccccc1",
        "ethanol": "CCO",
        "aniline": "c1ccccc1N",
    }
    return [MolRecord(mol_id=k, smiles=v, mol=Chem.MolFromSmiles(v)) for k, v in mols.items()]


def _matrix(tmp_path):
    path = tmp_path / "p.tsv"
    path.write_text("aromatic\tc1ccccc1\nhydroxyl\t[OH]\n")
    pset = load_patterns(path)
    frame = profile(_recs(), pset, mode="count", n_jobs=1, progress=False)
    return frame, pset


def test_require_and_forbid(tmp_path):
    frame, pset = _matrix(tmp_path)
    keep = apply_filter(frame, require=["aromatic"], forbid=["hydroxyl"], patterns=pset)
    kept = set(frame.loc[keep, "mol_id"])
    # benzene and aniline have a ring but no OH; phenol is forbidden; ethanol has no ring
    assert kept == {"benzene", "aniline"}


def test_min_count(tmp_path):
    frame, pset = _matrix(tmp_path)
    keep = apply_filter(frame, min_counts={"hydroxyl": 1}, patterns=pset)
    assert set(frame.loc[keep, "mol_id"]) == {"phenol", "ethanol"}


def test_family_requirement(tmp_path):
    path = tmp_path / "p.tsv"
    path.write_text(
        "pattern_id\tname\tfamily\tsmarts\trole\tsource\tlicense\tcitation\tenabled\tnotes\n"
        "a6\tbenzene\tAromatic\tc1ccccc1\tpharmacophore\t\t\t\ttrue\t\n"
        "oh\thydroxyl\tDonor\t[OH]\tpharmacophore\t\t\t\ttrue\t\n"
    )
    pset = load_patterns(path)
    frame = profile(_recs(), pset, mode="count", n_jobs=1, progress=False)
    keep = apply_filter(frame, require=["Aromatic"], patterns=pset)
    assert "ethanol" not in set(frame.loc[keep, "mol_id"])
    assert "phenol" in set(frame.loc[keep, "mol_id"])


def test_gobbi_similarity_ranks_query_first():
    query = [MolRecord(mol_id="phenol", smiles="c1ccccc1O", mol=Chem.MolFromSmiles("c1ccccc1O"))]
    library = _recs()
    scores = pharm2d_similarity(query, library, aggregation="max", progress=False, exclude_self=False)
    # The query molecule itself should have Tanimoto 1.0 and rank first.
    import pytest

    phenol = scores.loc[scores.mol_id == "phenol", "pharm2d_tanimoto"].iloc[0]
    ethanol = scores.loc[scores.mol_id == "ethanol", "pharm2d_tanimoto"].iloc[0]
    assert phenol == pytest.approx(1.0)
    assert phenol > ethanol
