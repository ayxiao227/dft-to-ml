# Recoverable active-learning reporting (v3)

Implemented against remote `f0089d0` (which includes `29f10f0`), preserving the
pre-existing local `a0c8fbe` project-guidance commit. Work is confined to
`fix/pipeline-integrity-audit`. Main and PR merges are unaffected.

The latest WSL reporting crash has **not** been diagnosed. The earlier finalize
OOM does not establish the cause of this crash. No live QM9 experiment was used
or assumed available during implementation.

## What changed

- `CandidateFeatures` builds its molecule-ID index once. Reporting reads only
  binary feature arrays, avoiding faulting the unused model feature matrix into RAM.
- Selected diversity retrieves and packs fingerprints in chunks, then uses RDKit
  bulk Tanimoto over bounded reference chunks. A 4096-entry packed-fingerprint
  cache reuses references across batches. Each within-batch pair is evaluated once.
  There is no dataset-wide pairwise matrix or three-dimensional broadcast.
- Each batch's reference IDs still come from **accepted attempts in earlier
  rounds**, including initial labels. Every earlier selection in the current batch
  is included, regardless of its later outcome. Failed earlier-round attempts are
  not promoted into references; retries use the recorded accepted-attempt round.
  Selection order and duplicate-reference semantics are retained. Persisted
  acquisition scores are not substituted for this diagnostic. Empty/empty
  fingerprints retain the original zero-distance convention.
- Prediction JSON arrays and JSONL stream into per-source CSV chunks. Only latest
  diagnostic rows enter indexed SQLite storage; historical predictions remain in
  the full CSV. SQL cache size is 8 MiB and temporary sorts use disk. Groups expose
  row iterators instead of whole prediction DataFrames.
- RF subgroup counts/error sums stream over all rows and retain repeat membership.
  Worst rows use an indexed exact SQL order. Uncertainty diagnostics use exact
  `uncertainty_ev, source_record_id` ordering, the existing retained-count ceiling,
  and the original `array_split` bin sizes. Stored checkpoint metrics, including
  exact quantiles and tied-rank Spearman diagnostics, are exported unchanged.
- Chemistry is computed once per distinct input SMILES into a disk cache, with at
  most 4096 decoded summaries in memory. Cache values carry integrity hashes.
- CSVs, plots and publication are staged on disk. Figures are closed promptly;
  report outputs retain file paths rather than output-byte dictionaries.
- Completed legacy and incremental finalizations are supported. A started
  finalization contributes no final prediction/metric rows to a report.
- Report never fits a model, runs acquisition, reads an oracle, or generates
  predictions. Saved models, labels, partitions, predictions and finalization
  contracts remain reusable and unchanged.

The working-memory **target is 2 GiB** for the stated organic-molecule workload,
leaving headroom inside a 7.4 GiB WSL VM. It is not an enforced RSS limit or a
full-scale guarantee. Native libraries, mapped resident pages and plot rendering
still contribute to RSS; measurements below include them.

## Rendering change

Error panels use a deterministic reservoir of at most **20,000** rows per
partition/strategy, seed **271828**, preserving all repeats in the sampled
population. Groups below the limit use every row. The residual scatter is
rasterized in SVG. Observed/predicted hexbin and error histograms also describe
this plotting reservoir when the population exceeds the limit; histogram counts
are sample counts. `plot_sampling.json` records seed, limit and population for
each group, and `report.md` discloses sampling. All numeric tables, subgroup
statistics, uncertainty diagnostics and worst predictions use full membership.

## Recovery, invalidation and progress

Work is kept in `.report-work/<full-identity>/`, separate from immutable
`reports/<identity-prefix>/`. The identity includes the report version, relevant
implementation-file hashes, processing settings, config/environment/features,
all relevant source-table content hashes, non-report artifact hashes, and the
finalization contract/state. Updating inputs, config, source implementation or
processing settings selects new work and report directories. Report publication
itself does not invalidate the cache. v2 reports are retained.

Durable stages are: ingestion per prediction artifact; the combined latest-row
index/full CSV; RF diagnostics per partition/strategy; uncertainty diagnostics per
repeat/group; diversity/selected chemistry per acquisition batch; and the complete
output bundle. Source artifacts remain authoritative. Reusing a stage requires a
committed progress-ledger record plus verified file hashes and matching payload.
File existence alone never means completion.

Outputs are flushed/fsynced and atomically published before a synchronous SQLite
completion commit. On restart:

1. Before publication: rebuild the owned pending stage; no progress was committed.
2. After publication but before commit: recompute and accept only exact matching
   orphan outputs. Conflicting bytes are retained and cause an actionable error.
3. After commit: verify all stage output hashes and reuse the completed stage.
4. After final report file publication but before the experiment-ledger commit:
   replay immutable publication from the verified complete bundle and register
   exact matching files in one sealed ledger transaction.
5. After the ledger commit: repeated reporting returns the same report path.

Corrupt stages, chemistry values or progress ledgers produce errors; they are not
silently discarded. A partial anonymous copy left during a file-copy crash is
retained, and another copy is refused until that evidence is moved aside. This
bounds repeated interrupted-copy growth while preserving evidence. Recovery
checks do not bypass the experiment's writer lock, seals or artifact verification.

Flushed output covers startup ledger/source/artifact/final-manifest verification,
source identity, ingestion, metric aggregation, errors/subgroups, exact ranking,
chemistry/diversity, publication and completion. Counts distinguish committed work
from in-flight rows and cached work from new work. Ingestion's approximate ETA
uses only newly processed artifacts, assumes comparable remaining artifact sizes,
and is unavailable when all observed work was cached.

In another terminal, while report owns the writer lock:

```bash
uv run --no-sync python -m src.active_learning progress data/qm9/al_pilot
```

This reads committed finalization counts and reporting stage names without taking
the writer lock or hashing experiment files. It deliberately does not certify
integrity or count partially written output as completed work.

## Disk planning and cleanup

RAM savings and recovery trade off against disk. For a single identity, storage
is linear in input/prediction count: per-artifact CSV chunks, latest-row SQLite
shards, their combined index/full CSV, a complete output bundle, and one owned
pending directory per stage. There is no per-retry directory allocation. At peak,
allow roughly **four full CSV copies**, latest-row shards/indexes, SQLite journals
and temporary sort space, figures and small manifests. Actual bytes depend on
SMILES, IDs, CSV quoting and checkpoint membership; source JSON size is not a
reliable exact CSV-size estimator. Those copies include the completed report copy after publication.
Check `df -h data/qm9/al_pilot` before a large report. No automatic disk quota is
claimed; disk exhaustion can still interrupt a stage.

Invalidated identities are deliberately retained, so successive code/config
changes can accumulate work. To reclaim space, stop the writer and identify the
specific identity from progress/provenance. Move only its `.report-work/<identity>`
elsewhere for review, or manually remove that owned cache once its completed
report has passed normal integrity verification and resumption is unnecessary.
Moving a corrupt work directory aside permits a fresh rebuild without destroying
evidence. Do not delete the experiment database, writer lock, training/finalization
artifacts, immutable reports, or unrelated temporary directories. Interrupted
publication messages identify the precise temporary evidence requiring review.

## Validation and measurements

The detailed measured matrix is below; raw results are in
[`reporting_benchmark_results.json`](reporting_benchmark_results.json).
Benchmarks run in fresh Linux subprocesses, with one numerical thread, a 90-second
CPU limit, a 6 GiB address-space limit and a 120-second wall timeout. Peak RSS is
Linux `getrusage(RUSAGE_SELF).ru_maxrss`, including native/mapped allocations.
Disk is polled every 50 ms (20 ms for full tiny reports); short transients can be
missed. Wall time includes fixture generation where marked; diagnostic time
excludes generation. These are host observations, not WSL guarantees.

`benchmark_reporting_scaling.py` reads old implementation classes directly from
`f0089d0` using `git show`, without changing branches. It independently varies
checkpoint count, group size, repeats, selections, references and molecule-ID
index size. The prediction benchmark executes ingestion, exact subgroup and
ranking paths, but excludes figure rendering and final report publication.
Diversity workloads execute the real diagnostic, not zero-acquisition placeholders.
The full tiny integration workload includes actual acquisition, finalization,
startup verification, tables, figures and publication. Only preparation trains
bounded synthetic fixture models; report measurements prohibit fit/predict/oracle
access. No large dataset was downloaded and no research training or HPC job ran.

Tests include independent set-based Tanimoto references (including empty vectors,
duplicates and reference membership), a guard against rescanning feature IDs,
legacy/new prediction formats, exact uncertainty ties/bin sizes, quantile/rank
edges, full subgroup accounting, deterministic plotting samples, cache reuse and
invalidation, corrupt artifacts/ledgers, interruption at stage/publication/ledger
boundaries, committed progress under the writer lock, and no-fit/no-predict
reporting. Existing incremental-finalization tests also compare legacy/new report
CSV/PNG outputs and ensure partial final results remain hidden.

The full old/new integration comparison uses two repeats, four strategies,
24 checkpoints, 32 selections and 1,344 prediction records. All **13 nonempty CSV
tables** matched with `rtol=1e-12, atol=1e-13` (non-numeric values exact); two empty
CSV tables matched byte-for-byte. This comparison includes selections, diversity,
subgroups, uncertainty diagnostics, full predictions and stored checkpoint metrics.

### Measured matrix

Each pair is **old → optimized**. Prediction counts include RF and committee rows;
group size is latest RF membership across repeats. Diversity uses 2,048-bit vectors.
Temp work excludes synthetic input files and completed report output; retained cache
files count as work. Raw results also include sampled total disk with inputs.

| Workload | Diagnostic time (s) | Total wall (s) | Peak RSS (MiB) | Peak temp work (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Diversity: 1,000 IDs; 30 selected; 100 references | 0.200 → 0.003 | 0.249 → 0.063 | 130.1 → 130.3 | 0.0 → 0.0 |
| Diversity: 10,000 IDs; same selection/reference sizes | 1.809 → 0.002 | 2.018 → 0.188 | 221.0 → 222.0 | 0.0 → 0.0 |
| Diversity: 1,000 IDs; 100 selected; 100 references | 0.867 → 0.004 | 0.885 → 0.028 | 130.3 → 129.8 | 0.0 → 0.0 |
| Diversity: 2,000 IDs; 30 selected; 1,000 references | 2.927 → 0.016 | 3.000 → 0.081 | 155.9 → 155.6 | 0.0 → 0.0 |
| Diversity: 119,452 IDs; 10 selected; 50 references | 5.047 → 0.003 | 6.773 → 1.587 | 442.3 → 451.7 | 0.0 → 0.0 |
| Predictions: 8,000 rows; 2 checkpoints; 1 repeat | 0.731 → 0.160 | 0.766 → 0.192 | 117.7 → 112.0 | 3.7 → 5.0 |
| Predictions: 32,000 rows; 8 checkpoints; 1 repeat | 0.864 → 0.532 | 1.023 → 0.694 | 118.3 → 112.1 | 15.0 → 9.2 |
| Predictions: 40,000 rows; group 10,000; 1 repeat | 3.025 → 0.860 | 3.196 → 1.026 | 139.2 → 133.8 | 18.6 → 25.1 |
| Predictions: 40,000 rows; group 10,000; 5 repeats | 2.368 → 0.834 | 2.530 → 1.000 | 131.8 → 133.5 | 18.8 → 35.5 |
| Predictions: 200,000 rows; group 50,000; 5 repeats | 12.928 → 4.414 | 13.740 → 5.223 | 211.2 → 157.7 | 93.9 → 186.8 |
| Predictions: 500,000 rows; group 125,000; 5 repeats | 32.381 → 11.909 | 34.860 → 14.020 | 354.1 → 156.5 | 234.8 → 485.8 |

The optimized full diversity schedule performed **13,990,000 comparisons**
in **1.716 s**, with **3.269 s total wall**,
**451.6 MiB peak RSS**, and a **233.3 MiB**
synthetic fingerprint input. Its schedule is 20 runs × 10 batches × 100 selections,
starting with 200 references and accepting each batch for the next round.
The independent small-reference tests and manageable old/new benchmarks agree;
this full schedule was optimized-only.

Full tiny report, including figures/publication: **11.223 → 11.943 s**,
**334.7 → 331.1 MiB peak RSS**,
**5.13 → 6.27 MiB peak temp work**.
Cold tiny reports have staging overhead and are not uniformly faster. A fresh-process
verified repeat took **0.224 s**, **245.1 MiB peak RSS**,
and produced **0 new sampled disk bytes**, reusing the same report path.

Checks actually run: the complete unittest suite passed **60 tests in 30.952 s**;
after final identity/progress refinements, the focused report suite passed **7 tests
in 4.901 s**. CLI help and `git diff --check` passed. Every listed benchmark
completed within its limits; no final benchmark timed out. The first test attempt
used an unrelated older environment missing `ijson`; verification then used a
fresh `uv sync --locked` development environment, without altering an experiment
environment. No full QM9 report acceptance is claimed.

Reproduce the focused checks and bounded benchmarks from the repository root:

```bash
uv sync --locked  # fresh development checkout only; not the existing experiment environment
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m src.active_learning --help
git diff --check

timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_scaling.py diversity --impl old --ids 10000 --selected 30 --references 100
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_scaling.py diversity --impl new --ids 10000 --selected 30 --references 100
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_scaling.py diversity --impl new --ids 119452 --selected 100 --references 200 --rounds 10 --runs 20
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_scaling.py predictions --impl old --rows 25000 --repeats 5 --checkpoints 2
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_scaling.py predictions --impl new --rows 25000 --repeats 5 --checkpoints 2

# Use a new directory. Preparation trains tiny synthetic models only.
uv run --no-sync python tests/benchmark_reporting_integration.py prepare /tmp/report-integration-check
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_integration.py old /tmp/report-integration-check
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_integration.py new /tmp/report-integration-check
# Repeating new measures verified cache reuse.
timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python tests/benchmark_reporting_integration.py new /tmp/report-integration-check
```

### Remaining unverified properties

- The full 42-million-record, 220-checkpoint report was not run. Aggregate full-scale
  ingestion, SQL/index/CSV disk use, startup hashing duration, plot publication and
  end-to-end RSS/runtime on WSL remain unmeasured.
- The 119,452-ID diversity benchmark uses synthetic random binary vectors. It tests
  index size and the actual 13,990,000-comparison acquisition schedule, not the
  scientific chemistry or ledger histories of the unseen QM9 experiment. The old
  full comparison schedule was intentionally not run.
- Prediction scaling tests use representative repeat/group sizes and synthetic
  saved records with three distinct SMILES. They establish bounded group handling,
  not descriptor throughput on 119,452 unique QM9 structures.
- Recovery tests inject exceptions at publication/commit boundaries. They cover
  the resulting restart states but do not simulate every kernel kill, disk-full
  timing, storage failure or power-loss/filesystem combination. Partial anonymous
  copy files require explicit evidence review rather than silent removal.
- Disk samples can miss brief peaks; benchmark address-space/CPU/wall limits apply
  to benchmarks only. The report has no hard RSS or disk quota.
- The cause of the most recent WSL crash remains unknown.

## Update the existing local checkout, preserving changes

Run this in the existing WSL checkout. The subshell stops on a failed fetch,
branch switch, fast-forward or stash application. It saves tracked/untracked
changes only when present; ignored experiment data stays in place. The stash is
**retained** after application. If fast-forward is refused or conflicts occur,
keep the printed stash SHA, inspect the divergence/conflicts, and resolve them
before reporting; do not reset/clean away local work.

```bash
cd ~/dft_ml_project
(
    set -e
    git status --short --branch
    report_update_stash=
    if [ -n "$(git status --porcelain)" ]; then
        git stash push -u -m "Local work before reporting performance update"
        report_update_stash=$(git rev-parse refs/stash)
        printf 'Saved local work in stash %s\n' "$report_update_stash"
    fi
    git fetch origin fix/pipeline-integrity-audit
    git switch fix/pipeline-integrity-audit
    git merge --ff-only origin/fix/pipeline-integrity-audit
    if [ -n "$report_update_stash" ]; then
        git stash apply "$report_update_stash"
    fi
    git status --short --branch
)
```

Review restored code changes before the next command. Keep the existing numerical
Python environment; do not blindly run `uv sync`, upgrade dependencies or edit
recorded environment metadata. If startup reports an environment mismatch,
investigate the recorded versions rather than overriding its checks.

Rerun **only report**, using completed training and finalization results:

```bash
uv run --no-sync python -u -m src.active_learning report data/qm9/al_pilot
```

To retain a log and the real exit status:

```bash
set -o pipefail
uv run --no-sync python -u -m src.active_learning report data/qm9/al_pilot 2>&1 | tee al_pilot-report.log
```

The command prints the immutable report path. If interrupted, rerun that same
command to reuse verified completed stages. Do not rerun `init`, `run`, acquisition,
training, prediction generation or `finalize` for this reporting repair.

## Changed files

- `src/active_learning/features.py`: reusable molecule-ID index.
- `src/active_learning/report_storage.py`: per-source streaming and latest-only SQL index.
- `src/active_learning/report_diagnostics.py`: bounded exact diversity, chemistry and numeric diagnostics.
- `src/active_learning/report_cache.py`: stage ledger, verification, recovery and read-only progress.
- `src/active_learning/reporting.py`: v3 identity, stage orchestration and bounded plotting.
- `src/active_learning/state.py`, `finalization.py`, `__main__.py`: verification timing/counts and progress integration.
- `tests/test_al_report_scaling.py`, `test_al_reporting.py`: independent correctness and recovery coverage.
- `tests/benchmark_reporting_scaling.py`, `benchmark_reporting_integration.py`: bounded old/new workloads.
- `docs/reporting_performance.md`, `reporting_benchmark_results.json`, `active_learning.md`: acceptance evidence and usage.
