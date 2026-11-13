# Active learning acceptance evidence

Implementation specification: attached audit, Phase 4, Gates 0–8. The question is
label/cost efficiency for DFT orbital gaps in eV, not optical gaps or low-gap search.
No scientific advantage is assumed.

## Gate 0 — baseline: PASS

- Starting local and remote branch: `fix/pipeline-integrity-audit` at
  `0e2313b698e52cc182f3ff775e9bb0618d53ed5b` (tree
  `3f2a049ba62b0955372a896bb799c50d988826eb`). Clean fresh clone of the existing
  repository; an older dirty checkout in another scratch directory was not touched.
- Commands: `git status --short --branch`; `git rev-parse HEAD`;
  `git remote -v`; `git ls-remote origin refs/heads/fix/pipeline-integrity-audit`;
  `uv sync --locked`; `uv run python -m unittest discover -s tests -v`.
- Outcome: locked environment installed without lockfile edits; **18 tests passed
  in 5.257 seconds**. No applicable AGENTS.md in the repository/ancestor paths.
- Read all required source/test modules, README, AUDIT, project/CI configuration
  and lockfile package inventory. Existing PubChem downloader, two-way splitter,
  refitting RF final evaluator and absence of AL/model persistence revalidated.
- Inventory: legacy entry points in `main.py`, `src/main.py`, dataset adapters,
  parser, descriptors, splits, analysis and final evaluator; 4 test modules.
  No raw QM9 archive or Gaussian training logs in the repository.

## Requirements checklist

| Gate | Required scope | Evidence |
|---|---|---|
| 0 | branch, dirty state, inventory, locked baseline | above |
| 1 | strict config; molecular/calculation/label/experiment schemas; protocol separation; mappings | implemented; evidence below |
| 2 | original QM9 graphs, exclusion/numeric/identity/dedup audit; shared features, SMARTS, cache | implemented; evidence below |
| 3 | four immutable partitions, group/random, oracle isolation, drift/final locks | implemented; evidence below |
| 4 | persisted common evaluators, bootstrapped committee, exact four strategies, caps/chunks | implemented; evidence below |
| 5 | SQLite/locks/transactions, reservations/attempts, immutable artifacts, resume/crashes | implemented; evidence below |
| 6 | select/prepare/ingest/retry, strict Gaussian opt/freq, resource accounting | implemented; evidence below |
| 7 | saved predictions/metrics, once-only saved-model finalization, figures/tables/report | implemented; evidence below |
| 8 | full regression, bounded random/group smoke, resume, prospective cycle, real-data availability | implemented; evidence below |

Each later entry records commands and outcomes; unrun external checks stay unrun.

## Gate 1 — schemas/protocols: PASS (local)

Files: `config.py`, `records.py`, `datasets.py`, `gaussian_records.py`, contract tests.
Strict expanded JSON and SHA-256, explicit input/observed protocol columns, typed
record schemas and collision checks are implemented. Publisher QM9 is distinct
from verified Gaussian evidence; unlabeled candidates receive no observed protocol.
`protocol()` now rejects incompatible multiplicity/task/extra route settings.
Commands: `uv run python -m unittest discover -s tests -p test_al_contracts.py -v`.
Outcome: 3 tests passed (0.128s); ID round-trip is additionally exercised in Gate 6.
Legacy `protocol_id` exports retain their old meaning; the two new columns remove
ambiguity. No historical label certification is performed.

## Gate 2 — input chemistry/features: PASS (local)

Files: `feature_registry.py`, `descriptors.py`, `ml_analysis.py`, AL `datasets.py`,
`features.py`, `artifacts.py`, chemistry tests. The separate AL adapter reads raw
QM9 XYZ records; original/optimized graphs are validated and compared, exclusions
and raw source hashes retained. Custom exclusion sets are explicitly labeled.
Duplicate components are deterministically represented or wholly quarantined on
conflict; label tolerance is one printed 1e-4 Ha gap unit (0.0027211386245988 eV).
AL uses v2 features and 2048-bit chiral fingerprints; legacy optimized descriptors
and 1024-bit/frozen misnamed v1 SMARTS remain compatible. All old ML feature lists
now reference one registry; fingerprint columns use numeric ordering.
Command: `uv run python -m unittest discover -s tests -p 'test_al_*.py' -v`.
Outcome: 7 tests passed (0.128s), covering SMARTS controls, graph/stereo classes,
cache/representation/chirality/order, future-structure independence, dedup/conflicts.
The input cohort is conditional on successful identity-consistent QM9 records.
Official exclusion status depends on an independently verified declared source hash;
a matching user-provided hash alone is not publisher authentication.

## Gate 3 — frozen partitions/oracle: PASS (local)

Files: `partitioning.py`, `oracle.py`, partition tests. Four immutable origins,
shared seeded initial conditions, random/group modes and explicit insufficient-size
failures. Group mode reserves final groups then validation groups; achieved row
fractions are recorded. Oracle objects hold training-pool labels separately from
evaluation labels; unreserved/unknown/non-pool/duplicate query IDs fail.
Command: `uv run python -m unittest discover -s tests -p test_al_partitions.py -v`.
Outcome: 2 tests passed (0.135s). Config/partition artifact tampering and final locks
are exercised with the persistent runner at Gates 5/8. No tuning is implemented;
all comparisons use locked fixed settings and scheduled rounds.

## Gate 4 — models/acquisition: PASS (local)

Files: `models.py`, `acquisition.py`, algorithm tests. Common mean/Ridge/RF evaluators;
training-only scaling; independently seeded outer molecule bootstraps with base RF
bootstrap disabled; sample SD disagreement. Exact random, labeled-reference MaxMin,
uncertainty and uncertainty-shortlist MaxMin strategies; sorted-ID tie breaks,
recorded seed streams, explicit uniform hash-priority caps, chunked prediction and
Tanimoto updates with O(N) minimum-distance storage. Diagnostic committee overhead
is recorded even on random/diversity runs. All-zero scores retain the same algorithm.
Command: `uv run python -m unittest discover -s tests -p test_al_models_acquisition.py -v`.
Outcome: 3 tests passed (0.195s): hand calculations, ties/zero scores, seed/cap/order,
uniform sampling sanity, train-only scaler, chunk/reload equivalence and tampering.
Runtime still grows with reference count; no dataset-wide quadratic matrix is built.

## Gate 5 — transactional state/resume: PASS (local)

Files: `state.py`, `experiment.py`, `runner.py`, `artifacts.py`, `evaluation.py`,
state tests. SQLite foreign keys/unique active attempts and labels; OS single-writer
lock; same-filesystem fsync/rename before reference commit; content-addressed models;
immutable four-way origin, per-attempt status, config/source/environment/artifact
verification and database drift seal. Oracle delivery is atomic and idempotent.
Command: `uv run python -m unittest discover -s tests -p test_al_state.py -v`.
Outcome: 3 tests passed (3.450s), including six injected crash boundaries, repeated
resume, selections and numerical prediction equivalence, concurrent lock refusal,
pending-reservation reuse and partition tampering. Timings/model byte hashes may
differ between independent runs. Unreferenced crash artifacts are retained; conflicting
orphan bytes fail visibly. The integrity seal detects accidental edits, not a malicious
owner rewriting both database and seal. No age-based lock deletion.

## Gate 6 — prospective Gaussian/accounting: PASS (local)

Files: `gaussian.py`, `costs.py`, CLI, legacy generator/parser/provenance modules,
Gaussian tests. Explicit attempt mapping, source-to-generated ID round trip, seeded
conformers, complete prior-manifest snapshots, visible orphan refusal, isolated
parse audits, no `accepted=True` bypass, expected input/observed policy and revision.
Missing output stays outstanding; all-rejected batches commit evidence; explicit
retries retain failed attempts/costs. Evaluation calculations use `__evaluation__`
with separate heldout label access and disclosed overhead. No SPE, SSH or submission.
Commands: `uv run python -m unittest discover -s tests -p test_al_gaussian.py -v`;
`uv run python -m unittest discover -s tests -p test_records.py -v`.
Outcome: 4 new tests passed (1.181s), 4 legacy record tests passed (0.216s).
Real RDKit/BSE generation, `bash -n`, complete unmocked synthetic parser, protocol
columns, NaN/Inf/malformed orbitals, charge/spin/version, continuation, error/truncation,
small/large negative modes, input tampering, missing/all-rejected, retries/conflicts,
cost units/coverage and interrupted manifest append exercised.
Only an exact settings-preserving check-geometry/check-guess continuation is allowed;
other Gaussian continuation spellings fail closed pending reviewed real fixtures.
The -10 cm^-1 tolerance is a policy, not proof of a strict minimum. No live Gaussian,
real-log corpus, scheduler collection or measured scientific cost savings was tested.

## Gate 7 — artifact evaluation/reporting: PASS (local)

Files: `evaluation.py`, `finalization.py`, `reporting.py`, reporting tests. Checkpoints
save exact per-ID membership/predictions and defined residuals, MAE/RMSE/R²/median/P90,
committee rank diagnostics and overhead. Finalization freezes config/partitions/all
checkpoint hashes before accessing final labels, reloads models without fitting,
commits all final outputs atomically and resumes the same frozen evaluation after
interruption. Completed finalization returns the existing result without new access.
Reports save PNG/SVG, plot tables, paired budgets, sample-SD seeded-run curves,
errors/uncertainty/diversity/functional chemistry/subgroups/depictions/failure/cost
coverage and explicitly conditional interpretation. No threshold interpolation or
complete-cost claim from missing data. Legacy README/AUDIT reconciliation follows
at Gate 8 delivery.
Command: `uv run python -m unittest discover -s tests -p test_al_reporting.py -v`.
Outcome: 2 tests passed (8.631s): metric edges, residual sign, interrupted finalization,
no fitting in finalization/report, once-only final access, repeat render and frozen
acquisition refusal. Intermediate full regression: **39 tests passed in 17.657s**.


## Gate 8 — end-to-end delivery: PASS (local); real-data pilot UNRUN

Final commands, from the repository root:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m src.active_learning.smoke --output data/active_learning_acceptance_release
uv run python -m src.active_learning --help
uv run python -m src.active_learning status data/active_learning_acceptance_final/random
uv run python -m src.active_learning report data/active_learning_acceptance_final/random
uv run python -m src.active_learning finalize data/active_learning_acceptance_final/random
git diff --check
```

Use a fresh output name to reproduce the smoke command. Final regression outcome:
**46 tests passed in 21.448 seconds**, including all 18 pre-existing tests. An earlier
run passed 45 tests; the final suite adds the partial-last-batch/finalization check.
`uv.lock`, project dependencies, legacy dataset importers and main were not modified.
CLI status/report/completed-finalize were additionally invoked on the prior complete
bounded experiment; they succeeded without retraining or reopening final labels.
`test_al_adversarial.py` covers raw-QM9 archive curation, exclusions/unit/identity
checks, source/model/config drift, hidden-label permutation with final-access spy,
lazy bounded fingerprint chunks, atomic partial-delivery refusal, geometry/basis/
molecule/duplicate-filename evidence, duplicate job costs and ingestion crash/resume.

The final bounded smoke uses **128 accepted molecules, 0 cohort rejections**, two
repeat seeds, initial 16, batch 8, two rounds, committee 3×8 trees, downstream RF 8
trees, one CPU. All four strategies and all three evaluators plus committee were run.
For **each** random/group mode: 24 checkpoints, 192 validation/final metric records,
128 selected requests across runs. Group rows differ because groups are never split:

| Mode/repeat | Initial | Pool | Validation | Final |
|---|---:|---:|---:|---:|
| random/0 | 16 | 60 | 26 | 26 |
| random/1 | 16 | 60 | 26 | 26 |
| group/0 | 16 | 74 | 31 | 7 |
| group/1 | 16 | 92 | 13 | 7 |

Paused/resumed execution matched memberships/selections exactly and predictions to
absolute tolerance 1e-12. Prospective smoke prepared/ingested heldout and initial
calculations, retained one rejected synthetic calculation, explicitly retried it,
then selected/prepared/ingested two more rounds and finalized: **8 training labels,
9 training attempts, 1 retained rejected attempt**, plus separately recorded heldout
overhead. Actual RDKit/BSE generation was local; logs/orbital values were synthetic.
Final reports save 56 tables/figures each, including PNG/SVG. A representative group
learning-curve image was visually inspected: axes, legends, bands and labels readable.
No monotonicity correction was applied (RF curves can get worse after acquisition).

README/AUDIT contradictions were reconciled. Example expanded version-1 QM9 and
prospective Gaussian JSON templates, mappings, migration, CLI/state/failure/HPC
boundaries, references, limits and report interpretation are documented in
`docs/active_learning.md`. Template zero source hashes must be replaced with verified
local file hashes; research defaults were never run. Generated files/model binaries
and datasets remain untracked/ignored; only source, tests, docs and configs are committed.

### Remaining external acceptance

No verified raw QM9 archive/exclusion set was present in the repository or scratch
search, so the conditional <=500-molecule real-QM9 pilot is **UNRUN**, not passed.
No dataset was automatically downloaded. Real Gaussian logs/continuation variants,
live Great Lakes transfer/submission, historical label certification, measured DFT
cost savings and scientific superiority remain **UNVALIDATED**. The software deliberately
rejects unreviewed continuation spellings. Optional GNN/3D features are absent.
All required local implementation gates passed; external checks are not certified.

### Delivery integrity

Starting remote main: `4798d5a6c5594f3f51efc70bd97d62497a548ea3` (read-only check).
Starting target branch remained at the audited baseline during implementation.
The final diff was reviewed with `git diff --check` and only the intended source,
tests, documentation and example configs were staged. No PR was merged.
The final branch commit SHA is reported in the delivery response and Git history;
it is not embedded recursively into its own commit.


### Changed files (40)

- `AUDIT.md`
- `README.md`
- `docs/active_learning.md`
- `docs/active_learning_acceptance.md`
- `examples/active_learning/gaussian_prospective.json`
- `examples/active_learning/qm9_research.json`
- `src/active_learning/__init__.py`
- `src/active_learning/__main__.py`
- `src/active_learning/acquisition.py`
- `src/active_learning/artifacts.py`
- `src/active_learning/config.py`
- `src/active_learning/costs.py`
- `src/active_learning/datasets.py`
- `src/active_learning/evaluation.py`
- `src/active_learning/experiment.py`
- `src/active_learning/features.py`
- `src/active_learning/finalization.py`
- `src/active_learning/gaussian.py`
- `src/active_learning/models.py`
- `src/active_learning/oracle.py`
- `src/active_learning/partitioning.py`
- `src/active_learning/records.py`
- `src/active_learning/reporting.py`
- `src/active_learning/runner.py`
- `src/active_learning/smoke.py`
- `src/active_learning/state.py`
- `src/descriptors.py`
- `src/dft_parser.py`
- `src/feature_registry.py`
- `src/gaussian_records.py`
- `src/ml_analysis.py`
- `src/structure_automation.py`
- `tests/test_al_adversarial.py`
- `tests/test_al_chemistry.py`
- `tests/test_al_contracts.py`
- `tests/test_al_gaussian.py`
- `tests/test_al_models_acquisition.py`
- `tests/test_al_partitions.py`
- `tests/test_al_reporting.py`
- `tests/test_al_state.py`
