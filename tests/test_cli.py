"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pharmprof.cli import app

runner = CliRunner()


def test_help_and_version():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "profile" in result.stdout
    assert "alerts" in result.stdout
    ver = runner.invoke(app, ["--version"])
    assert ver.exit_code == 0
    assert "pharmprof" in ver.stdout


def test_profile_and_stats_and_screen_and_alerts(tmp_path: Path):
    ligs = tmp_path / "ligs.smi"
    ligs.write_text("c1ccccc1O phenol 1\nc1ccccc1 benzene 0\nCCO ethanol 0\nc1ccccc1C(=O)O benzoic 1\n")
    patterns = tmp_path / "p.tsv"
    patterns.write_text("aromatic\tc1ccccc1\nhydroxyl\t[OH]\n")
    out = tmp_path / "counts.csv"
    result = runner.invoke(
        app,
        ["profile", str(ligs), "--patterns", str(patterns), "--out", str(out), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    text = out.read_text()
    assert "aromatic" in text and "hydroxyl" in text

    # labels live in a sidecar
    labels = tmp_path / "labels.csv"
    labels.write_text("mol_id,label\nphenol,1\nbenzene,0\nethanol,0\nbenzoic,1\n")
    stats_out = tmp_path / "stats.csv"
    result = runner.invoke(
        app,
        [
            "stats",
            str(out),
            "--out",
            str(stats_out),
            "--labels",
            str(labels),
            "--patterns",
            str(patterns),
        ],
    )
    assert result.exit_code == 0, result.output
    assert stats_out.exists()

    screen_out = tmp_path / "hits.csv"
    result = runner.invoke(
        app,
        [
            "screen",
            str(ligs),
            "--patterns",
            str(patterns),
            "--require",
            "aromatic",
            "--out",
            str(screen_out),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    body = screen_out.read_text()
    assert "phenol" in body
    assert "ethanol" not in body or body.count("ethanol") == 0

    alerts_out = tmp_path / "alerts.csv"
    result = runner.invoke(
        app,
        ["alerts", str(ligs), "--catalogs", "PAINS", "--out", str(alerts_out), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    assert alerts_out.exists()


def test_unparseable_patterns_fail_on_cli(tmp_path: Path):
    ligs = tmp_path / "ligs.smi"
    ligs.write_text("CCO ethanol\n")
    patterns = tmp_path / "bad.tsv"
    patterns.write_text("broken\t[[bad\n")
    out = tmp_path / "counts.csv"
    result = runner.invoke(
        app, ["profile", str(ligs), "--patterns", str(patterns), "--out", str(out), "--quiet"]
    )
    assert result.exit_code != 0
    assert "unparseable SMARTS" in (result.output + str(result.exception))
