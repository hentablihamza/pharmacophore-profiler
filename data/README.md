# Benchmark data: DUD-E EGFR

The notebooks and the default `pharmprof.data.load_egfr()` call use a **bundled subset** of the DUD-E EGFR target so they run offline. The official full files can be downloaded on demand.

## What is in the bundle

`data/dude-egfr/egfr_subset.csv.gz` — 3,042 unique RDKit-canonical SMILES:

| Class | N | How it was chosen |
|---|---:|---|
| Actives | 542 | Every unique, RDKit-parseable structure in DUD-E `actives_final.ism` (542/542 official actives were already unique) |
| Decoys | 2,500 | Reproducible random sample (`random_state=0`) of the 35,047 unique official decoys |

Columns: `smiles`, `mol_id`, `label` (1 = active, 0 = decoy), `chembl_id` (actives only), `target`, `source`.

This subset is what the README metrics and the notebooks report, unless a cell says otherwise.

## Official full set

The official DUD-E EGFR files are **not** stored in this repository (the decoy SMILES file is ~2.5 MB). Fetch them with:

```bash
pharmprof fetch-data --full
```

or in Python:

```python
from pharmprof.data import fetch_dude_egfr, load_egfr
fetch_dude_egfr()                  # writes data/dude-egfr/raw/
df = load_egfr(subset=False)       # all unique actives + all unique decoys
```

Measured on the files downloaded from `https://dude.docking.org/targets/egfr/` on 25 November 2026:

| File | Lines | Unique valid canonical SMILES |
|---|---:|---:|
| `actives_final.ism` | 542 | 542 |
| `decoys_final.ism` | 35,050 | 35,047 |

DUD-E EGFR corresponds to UniProt P00533 / Q01279 and ChEMBL target CHEMBL203. Actives were compiled by the DUD-E authors from ChEMBL and the literature. For each active, DUD-E provides property-matched decoys (about 50 per active) with similar physicochemical properties but dissimilar 2-D topology. Decoys have not been assayed against EGFR; they are presumed inactives for benchmarking.

The `.ism` format is whitespace-delimited: `SMILES  id  [CHEMBL_id]`.

## Citation and license

DUD-E is copyright 2009–2024 The Irwin Lab and the UC Regents. The data are provided by the authors for benchmarking; if you use this set, cite:

Mysinger MM, Carchia M, Irwin JJ, Shoichet BK. Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking. *J. Med. Chem.* 2012, 55, 6582–6594. [doi:10.1021/jm300687e](https://doi.org/10.1021/jm300687e)

You may also wish to cite the original DUD paper: Huang N, Shoichet BK, Irwin JJ. *J. Med. Chem.* 2006, 49, 6789–6801. [doi:10.1021/jm0608356](https://doi.org/10.1021/jm0608356)

Source: [https://dude.docking.org/targets/egfr/](https://dude.docking.org/targets/egfr/)

## A note on ligand-based methods and DUD-E

DUD-E decoys are *designed* to be topologically dissimilar to the actives. A 2-D pharmacophore fingerprint (for example Gobbi Pharm2D) is therefore expected to separate actives from decoys more cleanly than it would on a property-and-scaffold-matched set. The numbers in this repo are real measurements on this standard set; they should not be read as a claim about how the same method would perform against scaffold-matched decoys.
