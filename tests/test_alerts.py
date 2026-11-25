"""Alert flagging against molecules with known FilterCatalog / NIBR hits."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from pharmprof.alerts import flag_alerts
from pharmprof.io import MolRecord
from pharmprof.patterns import builtin_patterns
from pharmprof.profiler import profile


def _rec(name: str, smiles: str) -> MolRecord:
    return MolRecord(mol_id=name, smiles=smiles, mol=Chem.MolFromSmiles(smiles))


def _pains_matches(smiles: str) -> list[str]:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    cat = FilterCatalog.FilterCatalog(params)
    mol = Chem.MolFromSmiles(smiles)
    return [e.GetDescription() for e in cat.GetMatches(mol)]


def test_p_benzoquinone_is_a_pains_hit():
    # p-benzoquinone is a canonical quinone PAINS. We assert it against
    # FilterCatalog itself, then that flag_alerts surfaces the same hit.
    smiles = "O=C1C=CC(=O)C=C1"
    hits = _pains_matches(smiles)
    assert hits, "expected FilterCatalog PAINS to fire on p-benzoquinone"
    recs = [_rec("quinone", smiles), _rec("benzene", "c1ccccc1")]
    frame = flag_alerts(recs, catalogs=["PAINS"], progress=False)
    q = frame.loc[frame.mol_id == "quinone"].iloc[0]
    b = frame.loc[frame.mol_id == "benzene"].iloc[0]
    assert q["n_filtercatalog_hits"] >= 1
    assert "PAINS" in q["filtercatalog_hits"]
    assert b["n_filtercatalog_hits"] == 0
    assert not b["any_alert"]


def test_nibr_flags_acid_halide():
    # Acetyl chloride is a textbook acid halide; NIBR includes an acid-halide
    # (or closely related acyl halide) pattern. We check the bundled SMARTS
    # directly so the test does not depend on a particular pattern name.
    recs = [_rec("acetyl_chloride", "CC(=O)Cl"), _rec("acetone", "CC(=O)C")]
    nibr = builtin_patterns("nibr2019")
    counts = profile(recs, nibr, mode="binary", n_jobs=1, progress=False)
    nibr_cols = [p.pattern_id for p in nibr]
    ac = counts.loc[counts.mol_id == "acetyl_chloride", nibr_cols].iloc[0]
    acc = counts.loc[counts.mol_id == "acetone", nibr_cols].iloc[0]
    assert int((ac > 0).sum()) >= 1
    # Acetone should fire far fewer (ideally zero) of the same reactive patterns.
    assert int((acc > 0).sum()) < int((ac > 0).sum()) or int((acc > 0).sum()) == 0

    frame = flag_alerts(recs, catalogs=["NIBR"], progress=False)
    assert int(frame.loc[frame.mol_id == "acetyl_chloride", "n_nibr_hits"].iloc[0]) >= 1
