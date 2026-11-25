"""Pattern file loading and SMARTS validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmprof.patterns import PatternError, builtin_patterns, load_patterns


def test_builtin_basefeatures_has_27_patterns_and_8_families():
    pset = builtin_patterns("basefeatures")
    assert len(pset) == 27
    families = {p.family for p in pset}
    assert families == {
        "Donor",
        "Acceptor",
        "Aromatic",
        "Hydrophobe",
        "LumpedHydrophobe",
        "PosIonizable",
        "NegIonizable",
        "ZnBinder",
    }
    assert all(p.query is not None and p.enabled for p in pset)


def test_builtin_pmapper_has_30_patterns():
    pset = builtin_patterns("pmapper")
    assert len(pset) == 30
    assert {p.family for p in pset} == {
        "Aromatic",
        "Donor",
        "Acceptor",
        "PosIonizable",
        "NegIonizable",
        "Hydrophobe",
    }


def test_builtin_nibr_has_444_unique_ids():
    pset = builtin_patterns("nibr2019")
    assert len(pset) == 444
    assert len({p.pattern_id for p in pset}) == 444
    assert all(p.role == "alert" for p in pset)


def test_full_tsv_roundtrip(tmp_path: Path):
    path = tmp_path / "custom.tsv"
    path.write_text(
        "# comment should be ignored\n"
        "pattern_id\tname\tfamily\tsmarts\trole\tsource\tlicense\tcitation\tenabled\tnotes\n"
        "p1\thydroxyl\tDonor\t[OH]\tpharmacophore\tuser\tMIT\t\ttrue\t\n"
        "p2\toff\tAlert\tC#N\talert\tuser\tMIT\t\tfalse\tdisabled on purpose\n"
        "p3\taromatic\tAromatic\tc1ccccc1\tpharmacophore\tuser\tMIT\t\ttrue\t\n"
    )
    enabled = load_patterns(path)
    assert [p.pattern_id for p in enabled] == ["p1", "p3"]
    all_p = load_patterns(path, include_disabled=True)
    assert [p.pattern_id for p in all_p] == ["p1", "p2", "p3"]
    assert all_p.get("p2").enabled is False


def test_two_column_name_smarts(tmp_path: Path):
    path = tmp_path / "simple.tsv"
    path.write_text("hydroxyl\t[OH]\nbenzene\tc1ccccc1\n")
    pset = load_patterns(path)
    assert len(pset) == 2
    assert pset[0].name == "hydroxyl"
    assert pset[0].smarts == "[OH]"
    assert pset[1].pattern_id == "benzene"


def test_two_column_smarts_then_name(tmp_path: Path):
    path = tmp_path / "reversed.tsv"
    path.write_text("[OH]\thydroxyl\n")
    pset = load_patterns(path)
    assert pset[0].smarts == "[OH]"
    assert pset[0].name == "hydroxyl"


def test_fdef_load_basefeatures_file():
    from pharmprof.patterns import builtin_fdef

    pset = load_patterns(builtin_fdef("basefeatures"))
    assert len(pset) == 27
    families = {p.family for p in pset}
    assert "Donor" in families and "ZnBinder" in families


def test_unparseable_smarts_names_the_row(tmp_path: Path):
    path = tmp_path / "bad.tsv"
    path.write_text("ok\t[OH]\nbroken\t[[this is not smarts\n")
    with pytest.raises(PatternError) as exc:
        load_patterns(path)
    msg = str(exc.value)
    assert "row 2" in msg
    assert "broken" in msg
    assert "[[this is not smarts" in msg


def test_disabled_pattern_is_not_compiled(tmp_path: Path):
    path = tmp_path / "mixed.tsv"
    path.write_text(
        "pattern_id\tname\tfamily\tsmarts\trole\tsource\tlicense\tcitation\tenabled\tnotes\n"
        "good\tx\tDonor\t[OH]\tcustom\t\t\t\ttrue\t\n"
        "bad\ty\tAlert\t[[not-smarts\tcustom\t\t\t\tfalse\t\n"
    )
    pset = load_patterns(path, include_disabled=True)
    assert pset.get("good").query is not None
    assert pset.get("bad").query is None
    assert pset.get("bad").enabled is False


def test_merge_two_builtins():
    pset = load_patterns(["basefeatures", "pmapper"])
    assert len(pset) == 57
