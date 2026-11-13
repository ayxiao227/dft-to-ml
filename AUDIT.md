# Code quality audit and implementation plan

Baseline: `4798d5a6c5594f3f51efc70bd97d62497a548ea3`.
All changes belong to `fix/pipeline-integrity-audit`; main is unchanged.

## Critical (Fix Immediately)

| Finding | Impact | Branch status |
|---|---|---|
| Parser accepted failed/unconverged jobs | Invalid labels can contaminate training and testing | Fixed with conservative acceptance gate and rejection CSV; existing labels still need audit |
| Scratch removed after failed execution/copy | Loss of calculation outputs/checkpoints | Fixed: fail-fast scripts, checkpoints retained, deletion after transfer only |
| Outdated calls and incomplete CSV sampler | CLI crashes, omitted rows, duplicate samples | Fixed with explicit CLI and bounded sampling without replacement |
| Incompatible SMILES column names | Pipeline stages cannot connect directly | Fixed with one schema normalization boundary |
| Truncated names used as identity | Collisions, dropped molecules or wrong joins | Fixed with structure hash + manifest; ambiguous legacy data rejected |
| Downloader and CSV filters differ | Out-of-domain molecules enter dataset | Fixed with shared chemistry filter applied to every returned record |
| Scheduler text interpolated unchecked | Inputs can inject shell/scheduler content | Safe identifiers, validation and shell quoting added to generated scripts |
| Gaussian CPUs exceed allocation | Oversubscription | Shared CPU configuration; memory overhead and bounded conformer threads |
| Result tables overwritten without provenance | Irreproducible comparisons | Unique run directories, input hash, versions, settings and split memberships |

## Refactoring/Clean-up

Implemented: schema and identity module, explicit descriptor function, removed import
side effects, unified remote layout, removed stale interactive orchestration, reduced
commented-out parser/CSV code, named manifest fields and visible rejection reasons.

Remaining: split geometry/basis and SLURM rendering into separate modules; replace
remaining duplicated feature lists in ML analysis with one registry; separate model
training, diagnostics and plotting. Delete or archive the fully commented
`dft_calculations.py` and old notebooks after confirming they are not needed.

## Nice-to-Have Enhancements

1. Scaffold/group split and independent final test set. Canonical duplicate checks
   are implemented, but random splitting remains the documented evaluation design.
2. Parse method/basis/charge/spin into label provenance; compare original versus
   optimized connectivity, review imaginary modes and conformer sensitivity.
3. Persistent acquisition cache, bounded retry/backoff for transient PubChem errors,
   resumable generation and explicit download failure ledger. Network errors now
   fail visibly; automatic recovery is not implemented.
4. Cluster profiles, job arrays, submission/status tracking and remote transfer
   verification. Transfer is still manual; live HPC execution was not tested.
5. Nested model/feature selection; if tuning ridge alpha with internal CV, fit scaling
   within each inner fold rather than preprocessing all inner-training rows first.
6. CI regression workflow and fixture logs from real successful/failed Gaussian jobs
   with private data removed; benchmark larger batches before adding concurrency.

## Migration and validation

The CLI and remote layout intentionally change. Existing labels are not rewritten,
old scores are not endorsed, and raw logs are absent from the repository. Legacy
names require explicit opt-in and cannot repair historic collisions.

Run `uv run python -m unittest discover -s tests -v`. Tests cover failed-log rejection,
missing-log reporting, schema aliases, canonical identity, filtering, sampling,
actual RDKit/BSE Gaussian input generation, shell syntax, retained scratch after a
failed transfer, descriptor imports, ML run provenance and CV fold output. Gaussian
execution and actual SLURM scheduling require a separate pilot on the cluster.

## Prioritized next steps

- [ ] Review the branch diff, especially the CLI and remote-layout migration.
- [ ] Reparse existing raw logs with explicit legacy metadata; inspect all rejections.
- [ ] Audit electronic-structure protocols and optimized molecular identity.
- [ ] Run a three-molecule HPC pilot; confirm successful retrieval and failure recovery.
- [ ] Regenerate descriptors and evaluation artifacts from accepted, deduplicated labels.
- [ ] Freeze an untouched final test set and add scaffold-based evaluation.
- [ ] Add real Gaussian fixtures and CI; then implement resumability and cluster profiles.
