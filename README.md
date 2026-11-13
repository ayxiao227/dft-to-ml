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
The standalone parser fails if no labels pass; the prospective AL bridge records
an all-rejected batch as a valid terminal event. This is a conservative **single opt/freq job** policy;
monatomic species, other job types and linked/restarted calculations need explicit
review. Small imaginary modes within the tolerance still merit scientific review.

For historical filenames only, use `--legacy-names --id OLD_ID` with the original
metadata. Duplicate filename mappings are errors. This option cannot recover
molecules lost to historical name collisions. New filenames require manifests.

Legacy graph descriptors use accepted optimized SMILES, not 3D coordinates. The
parser reconstructs final organic connectivity and checks it against the input,
including specified stereochemistry. AL uses a separately versioned **input-only**
feature path; see [active learning](docs/active_learning.md). Do not mix protocols.

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

## Organic dataset integrity (updated contract)

This pipeline accepts **neutral, closed-shell organic-dataset opt/freq jobs only**.
SPE-only calculations are excluded, including normally terminated ones. Arbitrary
linked jobs and restarts are rejected for manual review; Gaussian's automatic
opt/freq continuation is recognized. Transition-metal reconstruction is out of scope.

New manifests contain `molecule_id`, `run_id`, `input_smiles`, `input_path` (relative
to the manifest), `input_sha256`, `protocol_id`, and generation options. Preserve
these columns and the original GJF files. Results retain both input and optimized
SMILES, log hashes, optimized XYZ paths, identity decisions and frequencies. CSV
metadata joins must use `(molecule_id, run_id)` with `validate="one_to_one"`.
Multiple log attempts with the same basename require selecting the intended run.

The parser reconstructs organic connectivity/bond orders from the final coordinates.
An unspecified input stereocenter may acquire a specific conformer assignment;
explicit input stereochemistry must match. Identity changes are review failures,
not silent relabeling. Descriptors use accepted `optimized_smiles`; XYZ files retain
3D geometry for future descriptors. Current descriptors remain graph-based.

A common explicit basis replaces molecule-dependent auto selection. Protocol IDs
include the element-complete BSE basis data and scientific settings; acceptance
checks input SHA-256, supplied basis primitives, executed route and protocol marker.
Gaussian version and executed route enter the final protocol identity. Any mixed
protocols stop the training export. This verifies supplied-input provenance, not an
independent reconstruction of Gaussian's internal basis implementation. Missing
provenance is rejected. Legacy logs may be inspected for geometry, but cannot be
promoted to verified training labels by guessing settings or assigning an arbitrary
protocol ID. The earlier legacy-name convenience does not waive this requirement.

`accepted_candidates.csv` and `parse_rejections.csv` preserve audit results even
when a mixed-protocol export is refused. Keep each experiment's logs in its own
folder. No historical training dataset has been retroactively certified.

### Resume, acquisition and fixed evaluation

```bash
uv run python main.py --csv molecules.csv --id batch01 --sample 100 --seed 42 --resume
uv run python -m src.splits data/features.csv data/frozen_batch01
uv run python -m src.ml_analysis data/frozen_batch01/development.csv --name development --cv
uv run python -m src.audit_labels old_labels.csv rebuilt_labels.csv label_changes.csv
```

Resume requires the same settings, validates existing input hashes, and uses atomic
manifest replacement. Use one writer per experiment. PubChem queries are cached in
`data/pubchem_cache`; transient failures use bounded exponential retries and terminal
failures enter `failures.jsonl`. Acquisitions stop visibly after exhausted retries.

The split command never overwrites an existing split directory. By default it groups
ring molecules by Murcko scaffold and acyclic molecules by generic whole-molecule
topology; this is a documented heuristic, not a chemical-similarity guarantee.
`--random` requests a random split instead. Develop only on `development.csv`; keep
`final_test.csv` untouched. The analysis command still makes an internal holdout. For a frozen random-forest
choice, run `uv run python -m src.final_evaluate data/frozen_batch01/development.csv
data/frozen_batch01/final_test.csv results/final_batch01 --features conj --trees 200`
on one line. It fits on all development rows, refuses molecule/protocol overlap
errors and existing output directories, and reports the final scores. Do not use
those scores to choose features or hyperparameters. A fresh split cannot undo prior
inspection of the dataset. Ridge hyperparameter tuning now fits scaling inside each
inner validation fold.

Offline tests also run in GitHub Actions. CI does not execute Gaussian or SLURM.

## Separate QM9 benchmark

`src/qm9.py` uses the original published **133,885-molecule QM9** XYZ archive.
No Gaussian installation, cluster account or private calculation logs are needed.
QM9 supplies B3LYP/6-31G(2df,p) optimized structures and orbital energies; this is a
separate protocol from our generated Gaussian jobs. Do not pool their training rows.

```bash
# Download original archive and exclusion list; verify Figshare MD5 checksums.
uv run python -m src.qm9 download data/qm9/raw

# Start with a reproducible random sample. Omit --sample for all accepted records.
uv run python -m src.qm9 prepare data/qm9/raw/dsgdb9nsd.xyz.tar.bz2 data/qm9/raw/uncharacterized.txt data/qm9/prepared --sample 5000 --seed 42

# Freeze final test molecules before model development.
uv run python -m src.splits data/qm9/prepared/features.csv data/qm9/frozen
uv run python -m src.ml_analysis data/qm9/frozen/development.csv --name qm9 --feature-set conj --trees 200 --jobs 1

# Once model settings are fixed, evaluate the reserved final partition once.
uv run python -m src.final_evaluate data/qm9/frozen/development.csv data/qm9/frozen/final_test.csv results/qm9_final --features conj --trees 200
```

Preparation also accepts an already downloaded archive and exclusion list. Output
directories must be new. It reads archive members without extracting their paths,
excludes IDs parsed from the supplied uncharacterized file and the 11 convergence exceptions
listed in the original README (the legacy importer does not authenticate the supplied exclusion file), validates properties and graph/coordinate atom
counts, and excludes repeated canonical structures. Every skipped record appears
in `rejections.csv`. Consequently this is a filtered benchmark, not an exact
reproduction of every published QM9 split or sample count.

The importer uses **QM9's supplied optimized SMILES**, preserves original SMILES
and XYZ coordinates, and computes the same graph descriptors as our organic pipeline.
It does not claim to rerun Gaussian or verify the dataset's protocol from raw logs.
HOMO/LUMO/gap retain Hartree columns and are converted to eV exactly once using
27.211386245988 eV/Ha. Target quantities are not included as descriptor features.

Outputs: `features.csv`, `xyz/`, `rejections.csv`, and `metadata.json` with source
hashes, sampling seed, selection counts and unit conversion. Optional `--fingerprints`
adds 1,024 Morgan columns; leave it off initially to reduce memory and CSV size.
Sampling uses seeded reservoir sampling over accepted unique records, not the first
N archive entries. Full-data preparation retains records before writing and full-data
training can be substantially more expensive than a 5,000-molecule pilot.

Sources: [original collection](https://figshare.com/collections/978904),
[dataset](https://figshare.com/articles/dataset/1057646),
[format README](https://figshare.com/articles/dataset/1057641), and
[uncharacterized list](https://figshare.com/articles/dataset/1057644).
Cite Ramakrishnan, Dral, Rupp & von Lilienfeld, *Quantum chemistry structures and
properties of 134 kilo molecules*, Scientific Data 1, 140022 (2014).

## Choose QM9, tmQM, or GDB-13

Use the common entry point `python -m src.datasets`. Existing `src.qm9` commands
remain available. Each import writes its own directory and source hashes; do not
pool these datasets under a common protocol.

```bash
uv run python -m src.datasets qm9 download data/qm9/raw
uv run python -m src.datasets qm9 prepare data/qm9/raw/dsgdb9nsd.xyz.tar.bz2 data/qm9/raw/uncharacterized.txt data/qm9/prepared_v2 --sample 5000

# Obtain the publisher's 2024 tmQM/tmQM_y.csv, including its SMILES column.
uv run python -m src.datasets tmqm data/tmqm/tmQM_y.csv data/tmqm/prepared --sample 5000 --units hartree
uv run python -m src.splits data/tmqm/prepared/features.csv data/tmqm/frozen --random
uv run python -m src.ml_analysis data/tmqm/frozen/development.csv --name tmqm --feature-set metal
uv run python -m src.final_evaluate data/tmqm/frozen/development.csv data/tmqm/frozen/final_test.csv results/tmqm_final --features metal

# Supply a plain .smi file or gzip-compressed .smi.gz, not a tar archive.
uv run python -m src.datasets gdb13 data/gdb13/gdb13.rand1M.smi.gz data/gdb13/prepared --sample 5000
```

**tmQM:** The [2024 publisher CSV](https://github.com/uiocompcat/tmQM/tree/master/tmQM)
contains TPSSh-D3BJ/def2-SVP single-point labels on GFN2-xTB geometries. This external
benchmark is distinct from our organic Gaussian opt/freq calculations: selecting it
does not relax their SPE prohibition. Use supplied metal-complex SMILES; no metal
bond inference or organic-only filter is applied. Missing/unparseable graphs,
duplicate canonical structures and invalid labels enter `rejections.csv`.

`--units hartree` converts published orbital columns to eV; `--units ev` supports an
explicitly preconverted copy. Units are never inferred from magnitude. This adapter
targets the 2024 semicolon CSV schema; the 2020 CSV without SMILES is unsupported.
The `metal` feature set adds metal atomic number and total formal charge to the
usual graph/conjugation descriptors. This is an exploratory graph baseline; it does
not resolve the limitations of bond/valence representations for metal complexes.
Quantum energies, metal natural charge, polarizability and other calculated targets
are deliberately excluded from the input features. Random splits are the suggested
starting point; scaffold grouping may not be meaningful for every metal graph.

**GDB-13:** The [publisher downloads](https://gdb.unibe.ch/downloads/) contain molecular
structures without HOMO/LUMO labels. The adapter therefore exports **candidates.csv**,
not a supervised training dataset. It streams plain/gzip SMILES without retaining the
full database, samples valid rows reproducibly and deduplicates the sample. Sampling
is uniform over valid rows, not over unique structures if the supplied input has
repeats. Extra columns on a SMILES line are ignored. Unsupported records are logged.
The default 5,000 sample is bounded in memory; scanning a full GDB-13 file still takes
time. A publisher-provided million-molecule subset is a practical starting point.

Feed candidates into the existing Gaussian workflow to obtain labels. Its existing
10-heavy-atom filter still applies, even though GDB-13 itself contains up to 13:

```bash
uv run python main.py --csv data/gdb13/prepared/candidates.csv --id gdb13_pilot --sample 100
```

Prepare modes require local source files for tmQM/GDB-13. They do not automatically
download those large collections. The downloader subcommand remains QM9-specific.


## Active learning (separate versioned workflow)

See [implementation and reproduction guide](docs/active_learning.md) and
[gate-by-gate acceptance evidence](docs/active_learning_acceptance.md).

```bash
# Offline bounded acceptance; choose a new output directory.
uv run python -m src.active_learning.smoke --output data/al_smoke
uv run python -m src.active_learning --help
# Initialize only after replacing template paths/source hashes with verified files.
uv run python -m src.active_learning init config.json data/experiment
uv run python -m src.active_learning run data/experiment
uv run python -m src.active_learning finalize data/experiment
uv run python -m src.active_learning report data/experiment
```

The new package adds strict config/provenance, original-input QM9 adaptation, frozen
four-way random/group partitions, isolated oracles, four paired acquisition strategies,
shared persisted evaluators, transactional SQLite/resume, and a local prospective
`select/prepare/ingest/retry` bridge. It never submits jobs or automatically downloads
data. Prospective validation/final calculations use `__evaluation__` and are separate
overhead. Reports reload saved predictions, not training code. The AL finalizer loads
all frozen model checkpoints once; the legacy final evaluator above still refits RF.

Input-policy and observed-protocol IDs are now separate explicit columns; legacy
`protocol_id` exports remain compatible. v2 input features repair functional-center
counts while legacy `n_nitroso_noxide` retains frozen historical semantics. Existing
feature CSVs are not silently reinterpreted or certified. The common registry removes
duplicated analysis lists, with numeric fingerprint ordering.

The smoke experiment is software validation only. Real QM9 results, live Gaussian/
SLURM execution, historical-label certification and measured DFT cost savings remain
external checks. No superiority over random acquisition is assumed or claimed.
