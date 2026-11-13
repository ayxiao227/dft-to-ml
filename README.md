# DFT-to-ML: molecular orbital-gap prediction

Python workflows for collecting small molecules, preparing Gaussian opt/freq inputs,
validating calculation outputs, constructing RDKit graph descriptors, and comparing
regression baselines for the DFT HOMO–LUMO orbital gap in eV.

These are **orbital gaps at a specified calculation protocol**, not experimental
optical gaps or molecular excitation energies. Historical figures are exploratory;
the raw logs/datasets needed to validate their scores are not included in Git.

## Install and test

Python 3.12+ and `uv` are required. Gaussian 16 and SLURM are only needed on the HPC host.

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
```

Tests use synthetic logs and small synthetic ML labels; they do not reproduce the
historical scores or execute Gaussian. No credentials are needed for these tests.

## Generate inputs

```bash
uv run python main.py --names methane ethanol acetone --id demo --cpus 12 --memory-gb 8
uv run python main.py --csv molecules.csv --id batch01 --sample 100 --seed 42
```

CSV input requires `SMILES` (aliases `ConnectivitySMILES` and `CanonicalSMILES`
are accepted). CID, Name, IUPACName and Charge are optional. Charge is inferred
from the molecule when absent. The default filter accepts neutral, connected,
nonradical, unlabeled molecules with 1–10 heavy atoms and elements C/H/O/N/S/P/F/Cl.
Canonical duplicates are removed during validated ingestion. A requested sample
must fit the validated pool; omitting `--sample` processes all valid rows.

Outputs:

- `molecules/gjf_files/<ID>/*.gjf` and `manifest.csv`
- `scripts/txt_files/<ID>/*.txt`

The manifest maps canonical molecular identities to exact log filenames. Keep it
with the calculations. IDs must be safe path components. Use a new ID for a new
experiment; existing manifests are protected from overwrite. Failed generations
are recorded and the CLI exits unsuccessfully if any inputs failed.

The basis policy in `src/structure_automation.py` is preserved: explicit BSE basis
blocks, matching ECPs when needed, B3LYP and GD3BJ by default. Geometry and frequency
use the same basis. Review charge, spin, conformer and basis choices before expensive
runs. This CLI is for the small-molecule dataset, not a general metal-complex workflow.

## HPC transfer and execution

Transfer GJF files to `/home/<user>/<remote-folder>/gjf_files/<ID>/`, where defaults
are `xaustin` and `dft_ml`; override with `--user` and `--remote-folder`.
Submit each generated script with `sbatch`. The script defaults are Great Lakes
specific (account `nszym1`, Gaussian module and scratch root); customize the
`create_txt_file` configuration for another cluster.

All generation modes now use the same remote layout. The old per-molecule AX_DFT
layout and interactive menu are replaced; do not combine old scripts with new inputs.

Scripts allocate matching CPU counts and two GB of memory overhead above Gaussian.
Outputs, including binary checkpoints, go to `log_files/<ID>/<SLURM_JOB_ID>/`.
Scratch is deleted only after successful execution and transfer. If a job fails,
inspect its retained scratch path. Do not automatically resubmit without checking
outputs. Multiple reruns of one molecule require explicitly selecting the intended
run before parsing.

## Validate labels and create descriptors

```bash
uv run python -m src.dft_parser downloaded_logs molecules/gjf_files/demo/manifest.csv data/labels.csv
uv run python -m src.descriptors data/labels.csv data/features.csv
```

The parser requires a final normal termination, completed optimization, orbital
energies, and frequencies without values below -10 cm^-1. It writes
`parse_rejections.csv` in the log directory, including missing and unmatched logs.
It fails if no labels pass. This is a conservative **single opt/freq job** policy;
monatomic species, other job types and linked/restarted calculations need explicit
review. Small imaginary modes within the tolerance still merit scientific review.

For historical filenames only, use `--legacy-names --id OLD_ID` with the original
metadata. Duplicate filename mappings are errors. This option cannot recover
molecules lost to historical name collisions. New filenames require manifests.

Graph descriptors are calculated from SMILES, not optimized 3D coordinates. Neither
the parser nor descriptors verify that the optimized geometry retained its original
connectivity; this remains a scientific QA step. Do not mix calculation protocols
without auditing the original logs/input files.

## Evaluate

```bash
uv run python -m src.ml_analysis data/features.csv --name batch01 --feature-set conj --trees 200 --jobs 1
uv run python -m src.ml_analysis data/features.csv --cv --trees 200 --jobs 1
```

Every run writes to a unique `results/runs/<timestamp>_<label>_<id>/` directory,
including dataset SHA-256, code/environment metadata, settings, split membership,
and result tables. Holdout runs also save predictions and plots. Duplicate canonical
molecules cause evaluation to fail; resolve their identities/labels before training.

Models are mean/size baselines, ridge, histogram gradient boosting and random forest.
CV comparisons use the same complete rows and folds across feature sets. Split seed
is 42. These random splits measure performance within the sampled distribution,
not performance on new scaffolds. Feature development based on held-out residuals
makes that holdout part of development. Freeze an untouched final test set before
making final performance claims. CV summaries are descriptive, not independent
confirmation after model/feature selection.

## Architecture and follow-up

- `schema.py`: schema normalization, molecular identities, identifier validation.
- `csv_conversion.py` / `pubchem_download.py`: ingestion and shared filtering.
- `structure_automation.py`: geometry/basis generation, scheduler scripts, manifests.
- `dft_parser.py`: calculation acceptance and label extraction with rejection reports.
- `descriptors.py`: explicit descriptor generation; imports perform no file I/O.
- `run_metadata.py` / `ml_analysis.py`: dataset checks, provenance, evaluation.

See [AUDIT.md](AUDIT.md) for priorities, limitations and the remaining improvement plan.
Historical notebooks are exploratory records, not maintained workflow entry points.
