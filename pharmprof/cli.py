"""Command-line interface for pharmacophore profiling, statistics, screening, and alerts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pharmprof import __version__
from pharmprof.alerts import CATALOG_CITATIONS, DEFAULT_ALERTS, flag_alerts
from pharmprof.data import fetch_dude_egfr, load_egfr
from pharmprof.metrics import screening_metrics
from pharmprof.patterns import list_builtins, load_patterns
from pharmprof.profiler import profile
from pharmprof.screen import apply_filter, pharm2d_similarity
from pharmprof.stats import compare_classes, pattern_correlation, pattern_summary

app = typer.Typer(
    name="pharmprof",
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Ligand-based pharmacophore profiling.\n\n"
        "Run a SMARTS pattern file over a set of ligands, write a count table, "
        "compare actives to inactives, filter a library, or flag PAINS/BRENK/NIBR alerts."
    ),
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pharmprof {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Ligand-based pharmacophore profiling."""


def _load_frame(path: Path, smiles_col: str | None, id_col: str | None, label_col: str | None):
    from pharmprof.io import read_molecules

    recs = read_molecules(path, smiles_col=smiles_col, id_col=id_col, label_col=label_col)
    return recs


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif name.endswith(".tsv") or name.endswith(".tsv.gz"):
        df.to_csv(path, index=False, sep="\t")
    else:
        df.to_csv(path, index=False)
    console.print(f"[green]Wrote[/green] {path}  ({len(df)} rows, {df.shape[1]} columns)")


def _parse_kv(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"expected KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        out[key] = float(val)
    return out


@app.command('patterns')
def patterns_list() -> None:
    """List bundled pattern sets."""
    table = Table(title="Bundled pattern sets")
    table.add_column("Name")
    table.add_column("What it is")
    table.add_column("License")
    rows = [
        ("basefeatures", "RDKit BaseFeatures pharmacophores (27 patterns, 8 families)", "BSD-3-Clause"),
        ("pmapper", "pmapper / Pharmer-style pharmacophores (30 patterns, 6 families)", "BSD-3-Clause"),
        ("nibr2019", "NIBR 2019 screening-deck alerts (444 SMARTS)", "BSD-3-Clause"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print(
        "Load any of these by name, or pass a TSV / two-column SMARTS file / .fdef path."
    )
    console.print("FilterCatalog sets (PAINS, BRENK, NIH, …) are used at runtime by [bold]pharmprof alerts[/bold].")


@app.command('profile')
def profile_cmd(
    ligands: Path = typer.Argument(..., exists=True, help="Ligand file (SMILES, SDF, CSV, TSV, parquet)."),
    patterns: list[str] = typer.Option(
        ["basefeatures"],
        "--patterns",
        "-p",
        help="Bundled set name (basefeatures, pmapper, nibr2019) and/or a TSV/.fdef path. Repeatable.",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output table (.csv, .tsv, or .parquet)."),
    mode: str = typer.Option("count", "--mode", help="count | binary | normalized (count / heavy atoms)."),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Worker processes. Use -1 for all CPUs."),
    chunksize: int = typer.Option(5000, "--chunksize", help="Molecules per chunk for large libraries."),
    smiles_col: str | None = typer.Option(None, "--smiles-col", help="SMILES column if the input is a table."),
    id_col: str | None = typer.Option(None, "--id-col", help="Molecule id column if the input is a table."),
    label_col: str | None = typer.Option(None, "--label-col", help="Optional activity label column to copy through."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable the progress bar."),
) -> None:
    """Match each SMARTS pattern against each ligand and write a count table."""
    recs = _load_frame(ligands, smiles_col, id_col, label_col)
    pset = load_patterns(patterns)
    console.print(f"Loaded {len(recs)} ligands, {len(pset.enabled())} enabled patterns from {patterns}.")
    frame = profile(
        recs,
        pset,
        mode=mode,
        n_jobs=n_jobs,
        chunksize=chunksize,
        progress=not quiet,
        out=out if out.suffix.lower() == ".parquet" else None,
    )
    if out.suffix.lower() != ".parquet":
        _write_frame(frame, out)
    else:
        # profile() already wrote parquet incrementally; rewrite a clean copy
        # only if we still have the frame (we do).
        _write_frame(frame, out)


@app.command()
def stats(
    matrix: Path = typer.Argument(..., exists=True, help="Count table written by `pharmprof profile`."),
    out: Path = typer.Option(..., "--out", "-o", help="Output statistics table."),
    labels: Path | None = typer.Option(
        None, "--labels", help="Optional table with mol_id and a label column for actives-vs-inactives."
    ),
    label_col: str = typer.Option("label", "--label-col", help="Label column in --labels or in the matrix itself."),
    patterns: str | None = typer.Option(
        None, "--patterns", "-p", help="Pattern set used to annotate names/families (optional)."
    ),
    corr: str = typer.Option(
        "pearson", "--corr", help="Redundancy method: pearson (counts) or jaccard (binary presence)."
    ),
    min_abs: float = typer.Option(
        0.9, "--min-abs", help="Absolute correlation threshold for the redundancy pair list."
    ),
) -> None:
    """Per-pattern hit rates, actives-vs-inactives enrichment, and redundancy."""
    table = _read_table(matrix)
    pset = load_patterns(patterns) if patterns else None
    summary = pattern_summary(table, pset)
    pieces = {"summary": summary}
    if label_col in table.columns:
        y = table[label_col]
        pieces["enrichment"] = compare_classes(table, y, pset)
    elif labels is not None:
        lab = _read_table(labels)
        merged = table.merge(lab[["mol_id", label_col]], on="mol_id", how="left")
        pieces["enrichment"] = compare_classes(merged, merged[label_col], pset)
    corr_mat, pairs = pattern_correlation(table, pset, method=corr, min_abs=min_abs)
    pieces["redundant_pairs"] = pairs
    # Write the main requested table (enrichment if present, otherwise summary)
    # and sidecar files for the rest.
    primary = pieces.get("enrichment", summary)
    _write_frame(primary, out)
    stem = out.with_suffix("")
    if out.name.endswith(".csv.gz"):
        stem = Path(str(out)[: -len(".csv.gz")])
    _write_frame(summary, Path(str(stem) + ".summary.csv"))
    if "enrichment" in pieces:
        _write_frame(pieces["enrichment"], Path(str(stem) + ".enrichment.csv"))
    _write_frame(pairs, Path(str(stem) + ".redundant_pairs.csv"))
    corr_path = Path(str(stem) + ".correlation.csv")
    corr_mat.to_csv(corr_path)
    console.print(f"[green]Wrote[/green] {corr_path}")


@app.command()
def screen(
    ligands: Path = typer.Argument(..., exists=True, help="Library file to filter or score."),
    out: Path = typer.Option(..., "--out", "-o", help="Output table of accepted / scored ligands."),
    patterns: list[str] = typer.Option(
        ["basefeatures"], "--patterns", "-p", help="Pattern set used with --require / --forbid."
    ),
    require: list[str] = typer.Option(
        None, "--require", "-r", help="Required pattern id, name, or family (repeatable). Any hit counts."
    ),
    forbid: list[str] = typer.Option(
        None, "--forbid", "-x", help="Forbidden pattern id, name, or family (repeatable)."
    ),
    min_count: list[str] = typer.Option(
        None, "--min", help="Minimum count as PATTERN=N (repeatable). Families are summed."
    ),
    max_count: list[str] = typer.Option(
        None, "--max", help="Maximum count as PATTERN=N (repeatable)."
    ),
    query: Path | None = typer.Option(
        None, "--query", help="Query ligands for Gobbi Pharm2D Tanimoto scoring."
    ),
    aggregation: str = typer.Option("max", "--aggregation", help="max or mean over query fingerprints."),
    labels: Path | None = typer.Option(
        None, "--labels", help="Optional labels for computing EF1%, BEDROC, ROC-AUC, PR-AUC on the ranking."
    ),
    label_col: str = typer.Option("label", "--label-col"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Filter a library by feature combinations and/or score it with Gobbi Pharm2D similarity."""
    recs = _load_frame(ligands, None, None, None)
    pset = load_patterns(patterns)
    matrix = profile(recs, pset, mode="count", n_jobs=n_jobs, progress=not quiet)
    if require or forbid or min_count or max_count:
        mask = apply_filter(
            matrix,
            require=require or [],
            forbid=forbid or [],
            min_counts=_parse_kv(min_count),
            max_counts=_parse_kv(max_count),
            patterns=pset,
        )
        matrix = matrix.loc[mask].reset_index(drop=True)
        console.print(f"Filter accepted {int(mask.sum())} / {int(mask.shape[0])} ligands.")
    if query is not None:
        # Score only the ligands that passed the filter (or the whole set).
        kept_ids = set(matrix["mol_id"].astype(str))
        kept = [r for r in recs if r.mol_id in kept_ids]
        sims = pharm2d_similarity(query, kept, aggregation=aggregation, progress=not quiet)
        matrix = matrix.merge(sims[["mol_id", "pharm2d_tanimoto"]], on="mol_id", how="left")
        matrix = matrix.sort_values("pharm2d_tanimoto", ascending=False, ignore_index=True)
        if labels is not None or "label" in matrix.columns:
            if labels is not None:
                lab = _read_table(labels)
                matrix = matrix.merge(lab[["mol_id", label_col]], on="mol_id", how="left")
            y = matrix[label_col] if label_col in matrix.columns else None
            if y is not None and y.notna().any():
                metrics = screening_metrics(y, matrix["pharm2d_tanimoto"])
                console.print("[bold]Gobbi Pharm2D ranking metrics[/bold]")
                for k, v in metrics.items():
                    console.print(f"  {k}: {v}")
    _write_frame(matrix, out)


@app.command()
def alerts(
    ligands: Path = typer.Argument(..., exists=True, help="Ligand file to flag."),
    out: Path = typer.Option(..., "--out", "-o", help="Output table of alert hits."),
    catalogs: list[str] = typer.Option(
        list(DEFAULT_ALERTS),
        "--catalogs",
        "-c",
        help="Alert catalogs. Default: PAINS,BRENK,NIH,NIBR. CHEMBL_* sets are available at runtime.",
    ),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Flag ligands with PAINS, BRENK, NIH (FilterCatalog) and/or NIBR 2019 alerts."""
    recs = _load_frame(ligands, None, None, None)
    frame = flag_alerts(recs, catalogs=catalogs, n_jobs=n_jobs, progress=not quiet)
    n_flagged = int(frame["any_alert"].sum())
    console.print(f"Flagged {n_flagged} / {len(frame)} ligands.")
    for name in catalogs:
        cite = CATALOG_CITATIONS.get(name, CATALOG_CITATIONS.get(name.upper(), ""))
        if cite:
            console.print(f"  cite {name}: {cite}")
    _write_frame(frame, out)


@app.command("fetch-data")
def fetch_data(
    full: bool = typer.Option(False, "--full", help="Also download the official full DUD-E EGFR files."),
) -> None:
    """Show or download the DUD-E EGFR benchmark used in the notebooks."""
    if full:
        paths = fetch_dude_egfr()
        for k, p in paths.items():
            console.print(f"{k}: {p} ({p.stat().st_size} bytes)")
    frame = load_egfr(subset=True, fetch=full)
    console.print(
        f"Bundled EGFR subset: {int((frame.label==1).sum())} unique actives, "
        f"{int((frame.label==0).sum())} decoys  (see data/README.md)."
    )


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if ".tsv" in name else ","
    return pd.read_csv(path, sep=sep)


# typer.testing looks at `app`. Keep list_builtins referenced for help completeness.
_ = list_builtins
