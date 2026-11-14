# Incremental finalization and bounded reporting

This update applies to `fix/pipeline-integrity-audit`. It reuses existing partitions,
labels, selections and saved models, and retains the original configuration and
frozen checkpoint-set hashes. It never fits, acquires or reinitializes during
`finalize` or `report`. No live experiment was used for development or verification.

## Artifact and ledger contract (version 2)

| Artifact | Contents and role |
|---|---|
| `final/checkpoints/<repeat>/<strategy>/<round>.predictions.jsonl` | One canonical JSON object per line. All original prediction fields, evaluators, molecule order, eV units, predicted-minus-observed residuals and uncertainty conventions are retained. |
| `final/checkpoints/<repeat>/<strategy>/<round>.result.json` | Version, completion state, repeat/strategy/round/model path, model SHA-256, frozen-contract hash, prediction path/content SHA-256/row count, and exact metrics for every evaluator. |
| `meta['final_checkpoint/'+SHA256(checkpoint identity)]` | Committed result-manifest path and SHA-256. This is the checkpoint completion record, committed with its artifact references, final metrics and one completion event. |
| `meta['finalization']` | Original frozen contract plus `format_version: 2`, committed/total checkpoint counts and overall status. On completion, includes overall manifest path/hash. |
| `final/manifest.json` | Versioned complete checkpoint index, original contract, checkpoint identities and committed result-manifest hashes. Published only after all checkpoints have valid committed results. |
| `reports/<input hash>/provenance.json` | Report version, input hash, finalization state, validation/final prediction artifact paths and hashes, checkpoint manifest identities/hashes and overall manifest identity/hash. |

There is **no required monolithic `final/predictions.json` or compatibility export**
for new results. Per-checkpoint manifests and SQLite retain all metrics; existing
CSV reports remain available. Completed legacy finalizations retain their original
`final/predictions.json`, `final/metrics.json` and ledger and are read without
reevaluation or migration. Validation arrays from older checkpoints are streamed.

Metrics retain the original NumPy calculations: exact linear median/P90, MAE, RMSE,
R² and exact SciPy Spearman ranking with the same undefined-value reasons. Numeric
predictions for one checkpoint remain resident for these calculations. No sampled
or approximate metrics are substituted.

## Migration and recovery

A legacy `"started"` experiment with no final outputs migrates on the next
`finalize`, inside the same database and with unchanged frozen contract hashes.
Already published legacy orphan outputs are retained; they do not establish any
checkpoint completion in the new format. Unexpected committed legacy partial
metrics cause an actionable error. There is no database deletion, seal reset,
configuration hash rewrite or label/model change to enable migration.

The old algorithm retried the **entire final evaluation loop** after interruption;
it did **not** resume from the last evaluated checkpoint. Version 2 skips only
verified, committed checkpoint results. Every restart still verifies the complete
frozen checkpoint/configuration/partition contract and all registered artifacts.

All publications use same-filesystem temporary files, file flush/fsync, atomic
rename and parent-directory fsync, followed by SQLite `synchronous=FULL` commits.
The existing OS single-writer lock remains mandatory for mutating operations and
full `status` validation.

| Interruption | Next `finalize` |
|---|---|
| Before prediction publication | Leaves the temporary file as evidence; recomputes this unfinished checkpoint. |
| After prediction publication, before result manifest | Recomputes the unfinished checkpoint into a new temporary file, requires an exact content-hash match before reusing the existing prediction file. |
| After result-manifest publication, before checkpoint DB commit | Validates frozen identities, model/prediction hashes, row count, full ordered molecule/evaluator membership, metadata, observed labels, residuals and exact recalculated metrics; commits that result without predicting it again. |
| After checkpoint DB commit | Verifies manifest/artifact hashes and matching metrics, skips prediction and creates no duplicate rows/events. |
| After all checkpoint commits, before overall completion | Verifies all committed results, reuses an identical overall manifest if published, and commits overall completion without prediction or final-label rereads. |
| After overall completion | Verifies and returns saved results without model loading, prediction or final-label access. |

Conflicting or corrupt published artifacts are preserved and cause an error naming
the affected path. Restore the verified original from a backup or investigate the
experiment before retrying; do not delete the database, edit hashes or remove
completion records to bypass verification. Interrupted hidden `*.tmp` files are
not results. Preserve them until their interruption/conflict has been investigated;
they are not automatically treated as valid predictions or overwritten.

Final metrics are visible in SQLite **per committed checkpoint** while status is
`started`. They are partial. Reports exclude **all final predictions and final
metrics** until overall finalization is `completed`, and state that policy in the
report. Report does not make predictions or fit models.

## Progress and monitoring

```bash
uv run python -m src.active_learning finalize data/qm9/al_pilot
# Separate terminal; safe while the finalizer owns the writer lock:
watch -n 5 'uv run python -m src.active_learning progress data/qm9/al_pilot'
# After the writer exits; full integrity/artifact verification:
uv run python -m src.active_learning status data/qm9/al_pilot
uv run python -m src.active_learning report data/qm9/al_pilot
```

Startup verification announces its start and completion. Finalization prints total,
verified previously committed and remaining checkpoints; the current
repeat/strategy/round/model path; each durable checkpoint commit with elapsed time
and **approximate ETA**; and overall success/output paths. ETA averages only work
processed during this invocation, excluding skipped checkpoints. Recovery of an
uncommitted published result counts as work processed in this invocation.

`status.finalization_progress` uses full integrity and committed artifact checks.
`progress` deliberately uses a read-only SQLite snapshot (`mode=ro`, `query_only`)
without acquiring the writer lock or reading models/predictions/labels. Its counts
reflect **database commits**, not files on disk. It explicitly labels itself a
lightweight snapshot, **not integrity verification**. It cannot authorize mutation
or prove artifact correctness. A briefly exclusive SQLite commit can delay its
read or cause a transient busy error; rerun the monitor.

## Memory and disk behavior

* Finalization retains one saved model bundle, one repeat's final-label table,
  one checkpoint's numeric predictions/metric arrays and a 256-molecule metadata
  lookup chunk. JSONL serialization writes records directly. Records and the model
  bundle are released after publication; the old repeat table is released before
  loading the next one.
* Ledger seals stream rows in the **identical schema-1 canonical order**, including
  large metadata TEXT via SQLite incremental BLOB reads. Frozen checkpoint hashes
  stream the original full checkpoint rows without retaining every validation-ID
  list or an experiment-sized JSON/string/bytes copy. Partition summary extraction
  streams JSON metadata. Source/artifact hashes use 1 MiB blocks. SQLite uses an
  8 MiB cache and disk temporary storage for ledger sorting.
* Reporting streams legacy arrays/new JSONL into a temporary SQLite prediction
  spool (4 MiB cache; disk sorting), streams `predictions.csv` and audit/attrition/round
  exports to disk (including large per-round candidate-ID metadata), and loads one exact diagnostic group at a time. RF plots process
  one partition/strategy across its latest repeats; uncertainty sorting processes
  one partition/strategy/repeat. Subgroups aggregate counts/error sums instead of
  retaining a row per tag. Worst-prediction metadata is bounded to 12 per RF group.
* The output mapping contains staged **paths**, not all output bytes. Figures write
  directly to files; completed outputs copy in 1 MiB blocks to durable artifacts.
  File/manifest hashes enter provenance and immutable ledger references.

Remaining limits: memory scales with one checkpoint's final membership, one model,
feature-ID/SMILES metadata, resident memory-mapped feature pages, one latest RF
partition/strategy group (all repeats), the relatively small metric/selection tables,
and plotting-library buffers. `resources.memory_mb` is not a hard OS RSS limit;
`chunk_size` bounds model scoring, not model size or exact rank/quantile arrays.
This update removes scaling with **total prediction records across checkpoints**;
it is not a constant-memory solution for arbitrarily large single checkpoints,
models or repeat counts. Feature validation may touch the full mapped feature files.
Selected-diversity reporting still performs the original exact comparisons and may
be slow on large acquisition histories. Full startup validation and ledger sealing
remain O(ledger + artifact bytes), so verification/commit time and I/O grow even
though they no longer build a full Python ledger. Reporting requires disk for a
prediction spool plus CSV/figures and immutable outputs; reserve several times the
saved prediction size. SIGKILL can leave disposable `.report.*` directories; remove
only identified stale benchmark/report staging directories after no writer is active.

## Applying to an existing local checkout without losing work

Stop any local finalizer before updating; a status of `started` is supported.
From the repository root:

```bash
git status --short --branch
git fetch origin fix/pipeline-integrity-audit
# If another branch is currently checked out, switch only after saving its changes.
# Save tracked and untracked code changes; ignored experiment data stays in place.
git stash push -u -m 'before incremental finalization update'
git switch fix/pipeline-integrity-audit
git merge --ff-only origin/fix/pipeline-integrity-audit
# Run only if the stash command above actually created that named stash:
git stash apply 'stash@{0}'
# Review conflicts and git diff. Keep the stash until the restored work is verified.
# Add only the new streaming parser; preserve the experiment's pinned numerical stack:
uv pip install --python .venv/bin/python 'ijson==3.6.0'
uv run --no-sync python -m src.active_learning status data/qm9/al_pilot
uv run --no-sync python -m src.active_learning finalize data/qm9/al_pilot
uv run --no-sync python -m src.active_learning report data/qm9/al_pilot
```

If there are no changes to stash, skip `stash apply`; never apply an unrelated older
stash. If fast-forward fails because of local commits, stop and reconcile those
commits on the requested branch without resetting, forcing, switching to `main` or
merging a PR. Resolve overlapping local edits before running the updated code.
Back up the complete experiment directory (including SQLite and its source data)
while no writer is active. Do not run `init`/`run` or delete/recreate the experiment.
For a fresh development checkout `uv sync --locked` is appropriate; an existing
experiment must retain its saved NumPy/scikit-learn/RDKit versions. The application
continues to enforce that environment check.

## Reproducible bounded verification

`tests/legacy_finalization_reference.py` retains the original finalization algorithm
from `a1a60b9` solely as a numerical/legacy compatibility reference. Tests compare
all prediction records and final metrics exactly; compare all populated CSV tables
and PNG figures across legacy/new formats; forbid fitting and repeated prediction;
exercise checkpoint and overall publication/commit boundaries, started migration,
legacy completed reads, corrupt/sealed-drift cases and lock-safe committed progress.

`python -m unittest discover -s tests -v` runs the complete suite.
`tests/benchmark_finalization_memory.py` constructs isolated synthetic ledgers
with 3,000 validation and 3,000 final molecules and 4 or 40 checkpoints referencing
one tiny trained bundle. This intentionally repeats feature rows/model references
to isolate checkpoint-count memory growth; it is not a scientific QM9 evaluation.
Preparation fits one tiny model only; measured finalize/verify/report forbid fitting.
Each stage runs in a **fresh Linux process**, reporting `resource.getrusage` peak
process RSS in KiB (includes native allocations and mapped resident pages, not only
Python allocations), and wall time. Measurements and comparison are recorded in
`docs/incremental_finalization_acceptance.md`.
