"""Structural-alert flagging via RDKit FilterCatalog and bundled NIBR SMARTS."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from tqdm.auto import tqdm

from pharmprof.io import read_molecules
from pharmprof.patterns import load_patterns
from pharmprof.profiler import profile

# Catalogs we expose. CHEMBL_* exist in RDKit but we do not vendor those SMARTS
# and we mention them only as optional runtime filters with a citation.
FILTERCATALOG_SETS = {
    "PAINS": FilterCatalogParams.FilterCatalogs.PAINS,
    "PAINS_A": FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalogParams.FilterCatalogs.PAINS_C,
    "BRENK": FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalogParams.FilterCatalogs.NIH,
    "ZINC": FilterCatalogParams.FilterCatalogs.ZINC,
}

# Optional ChEMBL-23 sets. Invoked only when the user names them; patterns stay
# inside RDKit and are not redistributed by this package.
CHEMBL_RUNTIME_SETS = {
    "CHEMBL": FilterCatalogParams.FilterCatalogs.CHEMBL,
    "CHEMBL_Glaxo": FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo,
    "CHEMBL_Dundee": FilterCatalogParams.FilterCatalogs.CHEMBL_Dundee,
    "CHEMBL_BMS": FilterCatalogParams.FilterCatalogs.CHEMBL_BMS,
    "CHEMBL_SureChEMBL": FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL,
    "CHEMBL_MLSMR": FilterCatalogParams.FilterCatalogs.CHEMBL_MLSMR,
    "CHEMBL_Inpharmatica": FilterCatalogParams.FilterCatalogs.CHEMBL_Inpharmatica,
    "CHEMBL_LINT": FilterCatalogParams.FilterCatalogs.CHEMBL_LINT,
}

CATALOG_CITATIONS = {
    "PAINS": "Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j",
    "PAINS_A": "Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j",
    "PAINS_B": "Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j",
    "PAINS_C": "Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740. doi:10.1021/jm901137j",
    "BRENK": "Brenk R, et al. ChemMedChem. 2008;3(3):435-444. doi:10.1002/cmdc.200700139",
    "NIH": "Jadhav A, et al. J Med Chem. 2010;53(1):37-51. doi:10.1021/jm901070c",
    "ZINC": "Irwin JJ, Shoichet BK. J Chem Inf Model. 2005;45(1):177-182. doi:10.1021/ci049714+",
    "NIBR": "Schuffenhauer A, et al. J Med Chem. 2020;63(23):14425-14447. doi:10.1021/acs.jmedchem.0c01332",
    "CHEMBL": "ChEMBL structural alerts (release 23), CC BY-SA 3.0; Walters P. rd_filters. https://github.com/PatWalters/rd_filters",
}

DEFAULT_ALERTS = ("PAINS", "BRENK", "NIH", "NIBR")


def _catalog(names: Sequence[str]) -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    unknown = []
    for name in names:
        key = name.upper() if name.upper() in FILTERCATALOG_SETS else name
        if key in FILTERCATALOG_SETS:
            params.AddCatalog(FILTERCATALOG_SETS[key])
        elif key in CHEMBL_RUNTIME_SETS:
            params.AddCatalog(CHEMBL_RUNTIME_SETS[key])
        else:
            unknown.append(name)
    if unknown:
        raise KeyError(
            f"unknown FilterCatalog set(s) {unknown}; "
            f"known: {sorted(FILTERCATALOG_SETS) + sorted(CHEMBL_RUNTIME_SETS)}"
        )
    return FilterCatalog.FilterCatalog(params)


def flag_alerts(
    molecules,
    *,
    catalogs: Sequence[str] = DEFAULT_ALERTS,
    n_jobs: int = 1,
    progress: bool = True,
    smiles_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Flag ligands against PAINS / BRENK / NIH (FilterCatalog) and/or NIBR SMARTS.

    FilterCatalog sets are executed inside RDKit at runtime; those SMARTS are
    not redistributed. The NIBR 2019 set is the bundled TSV (BSD-3-Clause,
    RDKit Contrib).
    """
    recs = molecules if isinstance(molecules, list) else read_molecules(
        molecules, smiles_col=smiles_col, id_col=id_col
    )
    # Normalise aliases
    normalised = []
    for c in catalogs:
        cu = c.upper()
        if cu == "NIBR" or cu == "NIBR2019":
            normalised.append("NIBR")
        elif cu in FILTERCATALOG_SETS:
            normalised.append(cu)
        elif c in CHEMBL_RUNTIME_SETS or cu in {k.upper() for k in CHEMBL_RUNTIME_SETS}:
            # preserve canonical spelling
            mapping = {k.upper(): k for k in CHEMBL_RUNTIME_SETS}
            normalised.append(mapping.get(cu, c))
        else:
            raise KeyError(
                f"unknown alert catalog {c!r}; "
                f"known: {list(DEFAULT_ALERTS) + sorted(FILTERCATALOG_SETS) + sorted(CHEMBL_RUNTIME_SETS)}"
            )

    fc_names = [n for n in normalised if n != "NIBR"]
    use_nibr = "NIBR" in normalised

    rows = []
    iterable = tqdm(recs, desc="alerts", disable=not progress)
    catalog = _catalog(fc_names) if fc_names else None
    for r in iterable:
        hits: list[dict] = []
        if catalog is not None and r.mol is not None:
            for entry in catalog.GetMatches(r.mol):
                props = set(entry.GetPropList())
                filt = entry.GetProp("FilterSet") if "FilterSet" in props else ""
                hits.append(
                    {
                        "catalog": filt or "FilterCatalog",
                        "description": entry.GetDescription(),
                        "scope": entry.GetProp("Scope") if "Scope" in props else "",
                    }
                )
        rows.append(
            {
                "mol_id": r.mol_id,
                "smiles": r.smiles,
                "valid": r.mol is not None,
                "n_filtercatalog_hits": len(hits),
                "filtercatalog_hits": "; ".join(
                    f"{h['catalog']}:{h['description']}" for h in hits
                ),
            }
        )
    out = pd.DataFrame(rows)

    if use_nibr:
        nibr = load_patterns("nibr2019")
        counts = profile(recs, nibr, mode="binary", n_jobs=n_jobs, progress=progress)
        nibr_cols = [p.pattern_id for p in nibr.enabled()]
        present = counts[nibr_cols].fillna(0)
        n_hits = present.sum(axis=1).astype(int)
        def _names(row):
            fired = [c for c in nibr_cols if row[c] > 0]
            return "; ".join(c.split(".", 1)[-1] for c in fired)

        out["n_nibr_hits"] = n_hits.to_numpy()
        out["nibr_hits"] = present.apply(_names, axis=1).to_numpy()
        # Severity: any EXCLUDE pattern is a 10+ style flag. We count EXCLUDE matches.
        exclude_cols = [
            p.pattern_id for p in nibr.enabled() if "severity=EXCLUDE" in (p.notes or "")
        ]
        out["n_nibr_exclude"] = present[exclude_cols].sum(axis=1).astype(int).to_numpy()
    else:
        out["n_nibr_hits"] = 0
        out["nibr_hits"] = ""
        out["n_nibr_exclude"] = 0

    out["n_alert_hits"] = out["n_filtercatalog_hits"] + out["n_nibr_hits"]
    out["any_alert"] = out["n_alert_hits"] > 0
    return out


def catalog_citation(name: str) -> str:
    key = "NIBR" if name.upper() in {"NIBR", "NIBR2019"} else name
    return CATALOG_CITATIONS.get(key, CATALOG_CITATIONS.get(key.upper(), ""))


