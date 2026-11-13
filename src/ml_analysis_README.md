# HOMO–LUMO analysis scripts

Keep `ml_analysis.py` and `ml_analysis_advanced.py` in the same directory. These scripts expect a CSV with `gap_ev` in eV and your existing molecular descriptors.

## Install and run

In your activated Ubuntu Python environment:

```bash
python -m pip install 'scikit-learn>=1.3' pandas numpy matplotlib joblib threadpoolctl
python ml_analysis.py path/to/molecules.csv --name molecules --feature-set base
python ml_analysis_advanced.py path/to/molecules.csv --name molecules --feature-set base
```

Start with `base`, then try `conj` if the six conjugation columns exist. Use `conj_fp` if the CSV also includes `morgan_` fingerprint columns. Fingerprint runs still need more time and memory than descriptor runs.

For optional boosting libraries:

```bash
python -m pip install xgboost catboost
python ml_analysis_advanced.py path/to/molecules.csv --feature-set conj --optional
```

`--optional` skips missing optional libraries with a message. It does not download or install them. Exact RBF SVR can become slow as the dataset grows; its kernel cache is bounded to 128 MB.

## Optimized original: `ml_analysis.py`

Existing imports and calls still work:

```python
from ml_analysis import analysis, cv_check
models, scores, predictions, importance, permutation = analysis(
    'path/to/molecules.csv', 'molecules', 'conj',
    n_jobs=1, n_estimators=200,
)
```

Changes:

- One worker by default, with native numerical threads also limited to one.
- Random forest defaults to 200 trees, maximum depth 20, and two samples per leaf. These are resource defaults, not guaranteed accuracy improvements. To compare the original settings, edit depth/leaf settings explicitly; `--trees 500` changes tree count only.
- Fingerprint columns are not loaded for descriptor-only runs. Numeric features use float32.
- Histogram boosting uses fewer bins and leaves.
- Saved figures use a headless backend, lower resolution, and close immediately.
- Permutation importance is disabled by default. Enable a bounded version using `permutation_repeats=3` or `--permutation-repeats 3`. It examines only the top 20 impurity-ranked RF features on at most 300 test molecules. Untested features are omitted, not scored zero; correlated features and preselection limit interpretation.
- CV evaluates MAE and R² from each fit together. It defaults to five folds and one repeat rather than three repeats. CV folds run sequentially.
- The normalized MAE column is now included in the saved comparison CSV.

For CV:

```bash
python ml_analysis.py path/to/molecules.csv --cv
```

This compares all available feature sets on the same complete rows; if additional feature sets have missing values, the CV population can differ from a standalone baseline run.

As before, the function returns fitted model objects. In a notebook, delete old returned models when finished; repeatedly retaining old results can still accumulate memory. The standalone command releases everything when it exits.

Baseline outputs remain in `results/`, `results/figures/`, and `data/model_predictions/`. Worst residuals still include the top 35 random-forest test predictions.

## Additional methods: `ml_analysis_advanced.py`

Methods: tuned Ridge, ExtraTrees, RandomForest, histogram gradient boosting, RBF support vector regression, and optional XGBoost/CatBoost. Defaults are four parameter trials per model and three CV folds. All fits run sequentially on one CPU. This deliberately takes longer than a baseline run.

- A reproducible 80/20 train/test split reserves the test set before tuning.
- Median imputation, removal of constant columns, and scaling (for Ridge/SVR) are fitted inside each CV training fold.
- Model and hyperparameter selection uses training-only CV MAE. CV scores are tuning scores and can be optimistic; the selected model's held-out test scores are the final evaluation.
- Only the CV-selected winner is evaluated on the test set, together with the training-mean baseline.
- Each completed model appends to the CV comparison; the best model so far is checkpointed. Restarting does not resume the previous search.
- The script writes test predictions and the 35 worst residuals, retaining molecule identifiers and the original zero-based CSV data-row number.

A lighter first run:

```bash
python ml_analysis_advanced.py path/to/molecules.csv --trials 2 --folds 3 --trees 100
```

Repeated molecules must stay together:

```bash
python ml_analysis_advanced.py path/to/molecules.csv --group-column CID
```

Use a complete canonical identity column if SMILES vary for the same molecule. If you already have scaffold IDs, passing that column evaluates generalization to unseen scaffolds; ordinary random splits evaluate a different task. The script does not compute scaffolds automatically.

Outputs are in `results/advanced/`: CV comparison, split assignments, final test metrics JSON, selected pipeline and feature names (`.joblib`), predictions, worst residuals, and diagnostic plot. The saved pipeline is fitted on the training partition only. Predict with columns in the saved `features` order. Only load joblib files from a trusted source.

Do not repeatedly choose feature sets using final test errors. Choose the representation using training validation or a predeclared experiment, and reserve a new untouched evaluation set if prior test results have influenced decisions.

## What the changes can establish

All-core forest prediction nested inside all-core permutation processing, unrestricted trees, unclosed figures, and duplicate CV fitting are clear resource problems in the original file. They are plausible contributors to the Ubuntu crash; no system log or real dataset was supplied to establish the crash's exact cause.

Lower memory use does not guarantee a lower prediction error. Conjugation information and nonlinear models may help, but improvement must be measured on your real DFT labels. Keep failed/unconverged calculations and mixed DFT methods out of the training target unless deliberately accounted for. No HOMO/LUMO energies are added as features, since their difference defines the target.
