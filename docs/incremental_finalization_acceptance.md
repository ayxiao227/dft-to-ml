# Incremental finalization acceptance — 2026-11-14

Branch: `fix/pipeline-integrity-audit`; base: `a1a60b98b0782ad88d6bf10313b1c85cd74e9dff`.

## Verification actually run

* `uv run python -m unittest discover -s tests -v`: **55 tests passed** (25.507 seconds).
* The final focused rerun after the CSV newline compatibility adjustment passed **9 tests** (11.558 seconds).
* Nine new focused test methods cover exact legacy prediction/metric equivalence; report equivalence across formats; fitting/prediction prohibition; checkpoint freeze, prepublication, prediction rename, result publication, precommit and postcommit failures; overall manifest publication and completion commit failures; corrupt models/configuration/partitions/manifests/predictions; sealed contract drift; started migration; completed legacy reads; idempotence; partial-result hiding; bounded metadata lookups; and committed progress while the writer lock is held.
* Independent comparison with the original `a1a60b9` reporting module, on **2 repeats, all 4 strategies, 3 checkpoints/run**: **13 populated CSV tables** matched with rtol/atol 1e-14; **19 PNGs and 20 SVGs** matched byte-for-byte. Both report paths forbade fitting/prediction. This comparison was repeated after the streamed round export change.
* `uv lock --check` and `git diff --check` passed. Existing numerical dependency versions were retained; only `ijson` was added. The lock keeps revision 3 for the repository's existing uv CI version.

## Peak process RSS experiment

Linux, Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, RDKit 2026.3.6. Each stage ran in a **fresh subprocess**. `resource.getrusage(RUSAGE_SELF).ru_maxrss` reports the OS high-water resident set in KiB, including native allocations and resident mapped pages. This is not a tracemalloc measurement. New report measurements include startup validation of previously committed prediction/report artifacts and idempotent rerendering.

Each isolated synthetic fixture has 3,000 validation and 3,000 final molecules, one repeat/strategy, and 4 or 40 checkpoints referencing one tiny trained model bundle. Preparation repeats feature rows/model references to isolate checkpoint-count scaling; it is not a scientific result. Final prediction counts are **48,000 vs 480,000**; validation + final report inputs are **96,000 vs 960,000** rows. Preparation fits one tiny baseline bundle. Measured finalize/verify/report explicitly forbid fitting. No QM9 downloads, research training, HPC jobs or live experiment access occurred.

| Stage | 4 checkpoints peak KiB (MiB) | 40 checkpoints peak KiB (MiB) | Wall seconds, 4 → 40 |
|---|---:|---:|---:|
| Synthetic preparation | 233,616 (228.1) | 234,540 (229.0) | 0.656 → 4.006 |
| Full startup verification | 224,016 (218.8) | 225,040 (219.8) | 0.246 → 0.704 |
| Incremental finalization | 229,088 (223.7) | 229,396 (224.0) | 1.184 → 9.117 |
| Legacy finalization | 305,864 (298.7) | 1,007,320 (983.7) | 0.998 → 4.388 |
| Bounded reporting | 339,488 (331.5) | 341,256 (333.3) | 6.199 → 17.727 |
| Original reporting | 489,852 (478.4) | 1,934,652 (1889.3) | 5.494 → 12.996 |

Increasing total prediction records 10× added **0.3 MiB** to incremental finalization RSS. Bounded reporting stayed around **332 MiB**. Both legacy algorithms grew substantially with retained records. Small fresh-process differences include allocator/mapping/plotting variance. This is bounded scale evidence, not a universal hard-memory guarantee.

The legacy comparison uses the archived original finalization/reporting algorithms with current shared state/helpers. It therefore does not reintroduce the old full-ledger integrity hashing amplification, and is a conservative comparison of the retained-prediction paths. Legacy reports read array-format validation and final artifacts. Rerendering the new reports after the final round-export adjustment reused identical immutable outputs.

Reproduce each stage, using a fresh empty root for preparation:

```bash
uv run python tests/benchmark_finalization_memory.py prepare /tmp/al-rss-4 --checkpoints 4 --rows 3000
uv run python tests/benchmark_finalization_memory.py finalize /tmp/al-rss-4
uv run python tests/benchmark_finalization_memory.py report /tmp/al-rss-4
uv run python tests/benchmark_finalization_memory.py verify /tmp/al-rss-4
# Repeat in /tmp/al-rss-40 with --checkpoints 40.
# Prepare a separate root for the legacy comparison, then:
uv run python tests/benchmark_finalization_memory.py legacy-finalize /tmp/al-legacy-rss-40
git show a1a60b9:src/active_learning/reporting.py > /tmp/al-legacy-reporting.py
uv run python tests/benchmark_finalization_memory.py legacy-report /tmp/al-legacy-rss-40 --legacy-report-source /tmp/al-legacy-reporting.py
```

## Remaining limits

This workload does not reproduce 119,452 unique QM9 molecules, 220 real model bundles, five repeats or full-size feature arrays. Memory still includes one model/checkpoint's numeric arrays, one repeat's label table, mapped feature pages/metadata, one latest RF diagnostic group across repeats, plotting buffers and smaller metric/selection tables. There is no hard OS RSS cap. Exact selected-diversity diagnostics can remain slow; hashing and report disk spooling add I/O. See [formats, recovery, monitoring and local update instructions](incremental_finalization.md).

## Changed files

* `README.md`
* `docs/active_learning.md`
* `docs/active_learning_acceptance.md`
* `docs/incremental_finalization.md`
* `docs/incremental_finalization_acceptance.md`
* `pyproject.toml`
* `src/active_learning/__main__.py`
* `src/active_learning/artifacts.py`
* `src/active_learning/costs.py`
* `src/active_learning/evaluation.py`
* `src/active_learning/finalization.py`
* `src/active_learning/models.py`
* `src/active_learning/oracle.py`
* `src/active_learning/report_storage.py`
* `src/active_learning/reporting.py`
* `src/active_learning/state.py`
* `src/active_learning/streaming.py`
* `tests/benchmark_finalization_memory.py`
* `tests/legacy_finalization_reference.py`
* `tests/test_al_incremental.py`
* `tests/test_al_reporting.py`
* `uv.lock`
