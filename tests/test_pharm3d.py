"""3D conformer and feature-factory smoke test (EmbedLib is optional/slower)."""

from __future__ import annotations

from rdkit import Chem

from pharmprof.patterns import feature_factory
from pharmprof.pharm3d import (
    features_for_conformer,
    generate_conformers,
    pharmacophore_from_reference,
)


def test_generate_conformers_and_features_on_aspirin():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    conf = generate_conformers(mol, n_confs=3, random_seed=1)
    assert conf.GetNumConformers() >= 1
    factory = feature_factory("basefeatures")
    feats = features_for_conformer(conf, factory, conf_id=0)
    families = {f.GetFamily() for f in feats}
    # Aspirin should always present an aromatic ring and an acceptor.
    assert "Aromatic" in families
    assert "Acceptor" in families


def test_pharmacophore_built_from_reference():
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)O")
    conf = generate_conformers(mol, n_confs=2, random_seed=1)
    pharm, feats = pharmacophore_from_reference(
        conf, families=["Aromatic", "Acceptor", "Donor", "NegIonizable"], tolerance=1.5
    )
    assert len(feats) >= 2
    # bounds should be set between the first two sites
    lo = pharm.getLowerBound(0, 1)
    hi = pharm.getUpperBound(0, 1)
    assert hi > lo
    assert lo >= 0.0
