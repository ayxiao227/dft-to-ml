# Input-only molecular active learning

This extension asks whether active learning reduces training labels or measured
calculation cost for **DFT HOMO–LUMO orbital-gap prediction in eV**, including on
unfamiliar structures. It does not optimize for small gaps or predict optical gaps.
The software does not assume an advantage over random sampling. See
[acceptance evidence](active_learning_acceptance.md) for actual tests and unrun checks.

## Local installation and bounded acceptance

Run on branch `fix/pipeline-integrity-audit`, Python >=3.12, in the locked environment:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m src.active_learning --help
uv run python -m src.active_learning.smoke --output data/al_smoke
```

The last command requires a new output directory. It creates **synthetic** graph
labels locally: 128 molecules, repeat seeds 42/43, initial 16, batch 8, two acquisition
rounds, committee 3×8 trees, downstream RF 8 trees, one CPU, both random/group splits.
It also compares paused/resumed execution and exercises prospective preparation,
accepted/failed synthetic logs and an explicit retry. It never downloads data or
runs Gaussian/SLURM/SSH. Outputs include expanded configs, SQLite, immutable model
checkpoints, per-ID predictions, acceptance JSON, and Markdown reports with PNG/SVG
figures and CSV tables. Models/datasets/generated reports must not be committed.
A tiny synthetic experiment demonstrates workflow correctness, not scientific gain.

## Configuration and sources

`examples/active_learning/qm9_research.json` and `gaussian_prospective.json` are
**templates**: replace paths and zero source hashes with verified local files before
initializing. No research experiment is launched automatically. Expanded configuration
is stored and SHA-256 hashed; missing fields use documented defaults in `config.py`.
Unknown/duplicate JSON keys, invalid types/enums/units, NaN/Inf, inconsistent budgets,
unsupported schema versions and drift are errors. Resume additionally checks pinned
NumPy/sklearn/RDKit versions, source/exclusion hashes, DB integrity and every referenced
artifact. Changed settings/source require a new directory/experiment.

Research defaults: initial 200, batch 100, maximum training labels/attempts 1200,
10 rounds, seeds 42–46, one CPU. The label budget includes initial labels. Validation
and final labels are separate fixed overhead, with actual sizes in `status`/reports.
There is no tuning pilot (0 labels), automatic retry or performance-driven stopping.
`retry.max_retries` bounds **explicit** retries, default 0. Initial and requested
calculations consume attempt budget even when rejected; no failed request is silently
replaced. Actual labels, attempts and measured cost must be compared separately.

For QM9, AL initialization reads the local **original XYZ archive**, not a legacy
`features.csv`. It reuses the original parser to revalidate numeric fields, atom
composition and once-only 27.211386245988 eV/Ha conversion, then independently validates
original/optimized graphs, specified stereochemistry and identity. Both raw SMILES
are preserved. The declared exclusion file hash must match; `exclusions_kind=custom`
is default. To declare `official`, supply its independently verified publisher hash
as `official_exclusions_sha256` matching the actual exclusions hash. This declaration
is not independent authentication of an arbitrary user-provided file. The supplied
uncharacterized list and 11 convergence exceptions have separate rejection reasons.
Archive members are streamed without extracting their paths; cohort records/dedup
maps are retained in memory, so curate a bounded local source for initial pilots.

Exact duplicate input/optimized components receive a deterministic lowest source-ID
representative with a mapping ledger, or are wholly quarantined for incompatible
mappings/protocols or labels differing by more than one printed gap unit (1e-4 Ha).
Labels are never averaged. This curation conditions the benchmark on successful,
identity-consistent records; it estimates neither raw-candidate failure probability
nor an exact published QM9 split. Dataset/source hashes and attrition are saved.

Prospective CSV: `source_record_id,input_smiles` (or GDB adapter `molecule_id,SMILES`),
with no labels or accepted flags. Enrollment reapplies the existing eligibility rule:
one connected neutral nonradical isotope-free graph, 1–10 heavy atoms and only
C/H/O/N/S/P/F/Cl. Carbon is not required. GDB's row-reservoir sample can have fewer
unique structures than requested; actual enrollment is reported. GDB is unlabeled.
QM9, custom Gaussian and tmQM labels are never pooled. tmQM remains a separate legacy
publisher benchmark, including its SPE-on-xTB provenance; it cannot enter this AL
experiment or relax the organic no-SPE contract.

## Input-only features and frozen splits

Features use only intended canonical isomeric **input** graphs. No tautomer collapsing,
stereo stripping, relaxed-only stereo, coordinates, HOMO/LUMO/labels, IDs, protocols
or costs enter X. Invalid graph/descriptor rows are rejected once, with reasons,
before shared partitions. Numeric features have one ordered semantic registry;
hydrogen includes implicit/explicit attached H, and all supported elements are counted.
Functional-center definitions and limitations are in `feature_registry.py`.
Nitroso, tertiary aliphatic/aromatic charge-separated amine N-oxide, and azo have
separate counts. Conjugated-carbonyl v2 counts a carbonyl center next to aromatic C
or an appropriate neighboring carbon pi system; amide/ester resonance alone does not
qualify. These are graph heuristics, not complete quantum-mechanical conjugation.

Default Morgan: radius 2, 2048 bits, input-specified chirality enabled. Bit and count
model representations are configurable; diversity always uses separate **binary**
fingerprints and Tanimoto distance. Cache keys include full canonical input, representation,
standardization/identity/schema and RDKit versions; arrays retain ordered IDs/names
with numeric fingerprint-bit order. Feature arrays are disk-backed and scoring is
chunked (4096 default). Large source curation still requires memory; chunk footprint
is validated against half the configured memory allowance, not a total OS RAM cap.

Four origins (`initial`, `pool`, `validation`, `final`) are immutable; acquisition status
evolves separately. Default fractions: final 20%, validation 20%, development 60%.
Group mode uses GroupShuffleSplit for final 20% of groups, then 25% of remaining
groups for validation. Actual row proportions differ. Initial molecules are sampled
uniformly from development; initial/pool may share groups. Validation/final groups
cannot overlap development or each other. The versioned
`murcko_plus_generic_acyclic_v1` uses ring Murcko scaffolds and generic full acyclic
topology (`CCC`, `CCO`, `COC` together, branched isobutane separate). It is not pure
scaffold splitting or a guaranteed OOD/similarity boundary. Too few groups/rows fail;
there is no silent random fallback. Canonical duplicates cross no partitions.

Oracle classes isolate training-pool labels, initial labels, validation evaluation,
and once-only final evaluation. Acquisition receives CandidateFeatures plus committed
training labels; unreserved, duplicate, acquired, validation/final and unknown requests
fail. The local file owner can still read files; this is a program-level boundary.

## Models and acquisition

Every strategy fits the same mean, StandardScaler+Ridge(alpha=1) and bounded RF,
plus an acquisition/diagnostic RF committee. Preprocessing fits on training rows
only. Default committee: 5 independently seeded outer molecule bootstraps, 100 trees
each, depth 20, leaf minimum 2, max_features 1.0, base `bootstrap=False`, one CPU.
The downstream RF uses the same structural settings. All refit from scratch by
round. The committee mean and sample SD (`ddof=1`, eV) are not calibrated intervals.
Diagnostic committee overhead on random/diversity is explicitly charged too.

Each round starts with sorted eligible unrequested unique IDs. RNG streams derive
from stable hashes of repeat seed, repeat, round, purpose and member; model seeds
are paired across strategies. Let b be the minimum of batch, remaining label/attempt
budgets and eligible pool size. Pending work blocks later rounds. Empty budgets end
cleanly; a partial last batch is valid.

| Strategy | Exact rule |
|---|---|
| random | Uniform without replacement over eligible unique IDs |
| diversity | Greedy largest minimum binary Tanimoto distance to current labeled set plus already selected batch; ties ascending source ID |
| uncertainty | Decreasing committee sample SD; ties ascending ID |
| uncertainty_diversity | Top L uncertain, L=min(N,max(b,min(20b,shortlist_cap))), then the same labeled-reference MaxMin |

`shortlist_cap` defaults to 5000 and cannot be smaller than configured batch.
Full-pool selection is default. Optional `candidate_cap` uses seeded hash-ID
priorities, applies equally to random and is reported as capped acquisition; later
strategy pools/subsets naturally differ. MaxMin retains an O(N) minimum-distance
vector and reads lazy fingerprint chunks without N×N or N×training matrices.
Runtime grows with reference/pool size. Nonfinite predictions fail before reservation;
equal/zero disagreement records a warning and keeps deterministic ties.

## CLI and manual Gaussian boundary

```bash
uv run python -m src.active_learning init config.json data/experiment
uv run python -m src.active_learning status data/experiment
uv run python -m src.active_learning run data/experiment
# To exercise retrospective pause/resume:
uv run python -m src.active_learning run data/experiment --stop-after 1
uv run python -m src.active_learning run data/experiment
uv run python -m src.active_learning finalize data/experiment
uv run python -m src.active_learning report data/experiment
```

`run` is retrospective only. For prospective experiments, round 0 already reserves
the initial set. Prepare and ingest the fixed heldout overhead separately using
strategy `__evaluation__` (never training labels), then initial and selected batches:

```bash
uv run python -m src.active_learning prepare data/experiment --strategy __evaluation__
uv run python -m src.active_learning ingest data/experiment --strategy __evaluation__ --outcomes heldout_results.json
uv run python -m src.active_learning prepare data/experiment --strategy random
uv run python -m src.active_learning ingest data/experiment --strategy random --outcomes initial_results.json
uv run python -m src.active_learning select data/experiment --strategy random
uv run python -m src.active_learning prepare data/experiment --strategy random --mark-pending
uv run python -m src.active_learning ingest data/experiment --strategy random --outcomes round_results.json
uv run python -m src.active_learning retry data/experiment --strategy random --id SOURCE_ID --reason 'Reviewed failed calculation'
```

Repeat `select/prepare/ingest` through the fixed caps; the final `select` saves the
last terminal checkpoint and returns complete. `--repeat` selects a configured repeat
index. Failed/missing heldout results block evaluation until a permitted explicit
retry succeeds; evaluation membership is never silently reduced. All training
strategies require their own selected attempts. Externally labeled Gaussian
initialization in retrospective mode must include original manifest columns, exact
input/log paths and expected policy/revision: all evidence is revalidated. There is
no manual accepted-CSV bypass or automatic legacy provenance assignment.

Each outcome is keyed to one registered attempt, not guessed recursive filenames:

```json
[
  {"source_record_id":"SOURCE_ID","attempt":1,"log_path":"/local/outputs/EXACT_REGISTERED_NAME.log",
   "resource_usage":{"job_id":"123","scope":"job","source":"supplied_scheduler_export",
      "cpu_seconds":120.0,"wall_seconds":90.0,"allocated_cpus":2,"allocated_cpu_hours":0.05}},
  {"source_record_id":"ANOTHER_ID","attempt":1,"outcome":"failed","reason":"Reviewed scheduler failure"}
]
```

Omit `resource_usage` or individual unknown measurements; unknown remains null,
never zero. CPU seconds are consumed time; allocated CPU-hours are separate supplied
allocation measurements. Job/step scope and source/job IDs prevent duplicate counting.
Missing files/outcomes remain outstanding; a whole rejected batch is still auditable.
Default retry cap 0 means no retry until a newly configured experiment permits it;
retry is always manual and counts a new attempt without another acquired-molecule count.

Prepare writes unique batch/attempt-safe conventional `molecules/gjf_files/<ID>` and
`scripts/txt_files/<ID>` layouts below `gaussian_attempts/<calculation_id>/`. Preserve
the exact manifests, snapshots, seeds and source/generated mappings. Transfer files
and submit only after human inspection using the existing Great Lakes conventions
in README; the program never submits. `--mark-pending` records manual dispatch.
Full prior manifests survive interrupted appends; unverified orphan files fail
visibly for review rather than being overwritten.

Acceptance requires exact retained input hash, BSE basis evidence/version, declared
neutral-singlet/gas/common-basis policy, allowed route, expected Gaussian revision,
protocol marker, final normal termination, no errors, optimization completion,
finite geometry/orbitals/frequencies, graph identity, and no mode below -10 cm^-1.
Small negative modes are retained and recorded, not certified as strict minima.
Only one exact settings-preserving `#n geom=allcheck guess=tcheck` automatic
opt/freq continuation is supported. Other links/restarts/continuation spellings and
SPE fail closed; more real Gaussian variants need reviewed fixtures and a versioned
policy. Supplied-input/log evidence does not independently prove Gaussian internals.
Attempt-scoped audit snapshots preserve reports/XYZ; conflicting second logs are
quarantined. One accepted training label per molecule; no last-write-wins resolution.

## State, recovery, finalization and report interpretation

SQLite is authoritative: configurations, records/mappings, partitions, selections,
rounds, attempts, labels, metrics, immutable artifact hashes and finalization status.
Foreign keys and uniqueness constraints apply. A nonblocking OS writer lock prevents
concurrent writers; do not delete locks by age. Files are staged/fsynced/renamed
before DB references commit. A crash leaves unreferenced garbage or the prior committed
state; resume verifies hashes and reuses only exact intended content. Do not manually
edit the ledger. Repeated ingestion of unchanged evidence is idempotent; conflicts
fail with evidence retained. Database seals protect against accidental drift, not an
owner deliberately rewriting both state and checksums. Copy the complete directory
and preserve declared source paths when moving work. Models are trusted project
pickle artifacts only, hash-checked before loading; never import arbitrary pickles.

Finalization freezes the complete checkpoint/config/partition set before final-label
access, reloads each saved pipeline and writes all outputs atomically without fitting.
An interrupted finalization resumes that same set. A completed finalization returns
its saved result and blocks acquisition/retraining/reconfiguration. There is no new
output-directory option that repeats evaluation; report regeneration is allowed.

Saved predictions include repeat/strategy/round/label and attempt budgets, source and
structure IDs, checkpoint hash, observed/predicted gap, predicted-minus-observed
residual, absolute error, uncertainty and eV units. MAE/RMSE/median/P90 use explicit
linear quantiles; R² is null with a reason for constant/fewer-than-two targets.
Constant uncertainty/errors give undefined Spearman. Validation membership is fixed.
Threshold results use the first observed validation checkpoint reaching a prespecified
threshold, or 'not reached'; no interpolation or test-driven acquisition.

Reports render saved artifacts only: learning/paired differences versus labels,
attempts and complete measured allocated CPU-hours, parity/residual/error distributions,
binned uncertainty and selective error, selected chemistry/diversity, input subgroups,
worst molecule depictions and full-SMILES tables, attrition/failure/retry/cost coverage.
Individual repeats and mean ± sample SD describe seeded-run variability conditional
on this cohort/split; shared molecules/repeats/rounds are not independent datasets.
Duplicate matched budgets use the latest checkpoint within a repeat. No significance
claim follows from a smoke run. No cost savings follow from simulated QM9 budgets.

## Schema mapping and compatibility

| Internal record | Legacy/source mapping |
|---|---|
| molecule | dataset + stable source_record_id; original canonical input/key; raw input/optimized strings and optional optimized key; source hash |
| calculation | unique calculation_id + attempt; source/input key; generator molecule_id (unprefixed hash), run_id, immutable input/log hashes, status/reason |
| requested protocol | input_protocol_id = existing manifest input-policy digest; no fabricated observed ID on candidates |
| observed protocol | observed_protocol_id = accepted policy + normalized executed route + Gaussian revision digest; legacy result protocol_id remains this observed digest |
| label | molecule/source/calculation/attempt association, gap_ev, eV, provenance type and acceptance version |
| experiment | canonical config/source/partition hashes, repeat/strategy, immutable origin, evolving state, finalization |

Legacy `src.acquisition.fetch` remains PubChem downloading. Legacy main/datasets/
QM9/tmQM/GDB/splits/analysis/final_evaluate entry points remain tested. Legacy
`build_descriptors` explicitly uses accepted optimized identities and v1 feature
semantics; `n_nitroso_noxide` keeps its old behavior rather than silently altering old
models. The v2 AL path is separate; historical feature CSVs/models are not migrated
or recertified. Analysis local/global feature lists now share a registry. Legacy
`final_evaluate` still refits an RF and is not used by AL finalization.

## Methodological references and limits

These sources motivate components and cautions, not an exact reproduction or promised gain:

- [MolPAL](https://github.com/coleygroup/molpal): modular pool/model/acquisition design; screening optimization differs from population error reduction.
- [modAL regression disagreement](https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html): regression committees, not classifier entropy.
- DeepChem [splitters](https://deepchem.readthedocs.io/en/stable/api_reference/splitters.html) and [models](https://deepchem.readthedocs.io/en/stable/api_reference/models.html): useful components; no dependency or turnkey Gaussian workflow is assumed.
- [Soleimany et al. 2021](https://doi.org/10.1021/acscentsci.1c00546): retrospective QM9 acquisition and random comparisons; different neural/multitask setting.
- [Heid et al. 2023](https://doi.org/10.1021/acs.jcim.3c00373): representation-dependent QM9 gap errors; agreement may share bias.
- [Yin et al. 2023](https://doi.org/10.1186/s13321-023-00753-5): uncertainty/AL under shift; targets differ and gains can be modest.
- [QM9 collection](https://figshare.com/collections/978904): publisher provenance distinct from verified custom Gaussian logs.
- [Connectivity-preserving QM9 work](https://doi.org/10.1039/D0SC05591C): preserve identity through relaxation.
- [Splitting study](https://doi.org/10.1186/s13321-021-00576-2): split design affects interpretation.
- [RDKit MaxMin](https://rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html): lazy diverse picking motivation; this project implements its own explicit tie/reference/chunk rules.

Real QM9 accuracy/AL gains, real Gaussian continuation variants, historical-label
certification, live Great Lakes submission/transfer, actual measured DFT costs and
cross-platform serialized byte equivalence remain externally unvalidated. Neural/3D
models are optional and absent. The bounded tests do not establish those outcomes.
