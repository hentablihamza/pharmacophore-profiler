# Provenance and licenses of bundled pattern files

The Python code in this repository is MIT-licensed (see `/LICENSE`). The
pattern files under `pharmprof/pattern_data/` are third-party scientific
data redistributed under their own terms. Each TSV starts with a comment
header that repeats the license and citation.

| File | Patterns | License | Cite |
|---|---:|---|---|
| `basefeatures.tsv`, `basefeatures.fdef` | 27 / 8 families | BSD-3-Clause (RDKit) | Gobbi and Poppinger, *Biotechnol. Bioeng.* 1998, 61, 47–54. [doi:10.1002/(SICI)1097-0290(199824)61:1\<47::AID-BIT9\>3.0.CO;2-Z](https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z). Upstream: https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef |
| `pmapper.tsv`, `pmapper.fdef` | 30 / 6 families | BSD-3-Clause (pmapper) | Koes and Camacho, *J. Chem. Inf. Model.* 2011, 51, 1307–1314. [doi:10.1021/ci200097m](https://doi.org/10.1021/ci200097m); Polishchuk, pmapper, https://github.com/DrrDom/pmapper. Feature SMARTS originated in Pharmer (Meireles) and were reimplemented in pmapper. |
| `nibr2019.tsv` | 444 alerts | BSD-3-Clause (RDKit Contrib) | Schuffenhauer et al., *J. Med. Chem.* 2020, 63, 14425–14447. [doi:10.1021/acs.jmedchem.0c01332](https://doi.org/10.1021/acs.jmedchem.0c01332). Redistributed from RDKit `Contrib/NIBRSubstructureFilters` **without** the PubChem example structures. One duplicate `PATTERN_NAME` (`acyclic_enolate`) is stored as `nibr2019.acyclic_enolate_2`. |

Full license texts:

- `licenses/RDKit-BSD-3-Clause.txt`
- `licenses/pmapper-BSD-3-Clause.txt`
- `licenses/NIBR-README.md` (upstream Contrib README)

## Patterns that are *not* vendored

PAINS, BRENK, NIH, ZINC, and the ChEMBL 23 alert sets are **not** copied into
this repository. They are executed at runtime through RDKit
`FilterCatalog` (`pharmprof alerts --catalogs PAINS,BRENK,NIH`). ChEMBL data
are [CC BY-SA 3.0](https://chembl.github.io/chembl-licensing/); if you use the
optional `CHEMBL_*` catalogs, attribute ChEMBL 23 and see
https://github.com/PatWalters/rd_filters.

Daylight SMARTS examples are not included (Daylight site terms forbid
redistribution). The Eli Lilly MedChem Rules are Apache-2.0 but are written
in Cerius2 query language, not SMARTS, and are not bundled here.
