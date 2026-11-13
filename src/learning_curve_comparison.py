"""Compare base/conjugation/fingerprint features with CV learning curves.

Example:
  python learning_curve_comparison.py molecules.csv --name homo_lumo

The script uses the same complete molecule rows, folds, sample sizes, and
random seed for each feature set. It reports training and validation MAE so
you can see whether additional data may help or whether the model/representation
is the larger limitation. All outputs are collected under results/<name>_learning_curves/.
"""
#python src/learning_curve_comparison.py data/descriptors/test_3_extra.csv.csv --name homo_lumo_3
import os
# Bound native numerical-library thread pools on shared/small machines.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    "MolWt", "HeavyAtomCount", "NumC", "NumN", "NumO", "RingCount",
    "AromaticRingCount", "RotatableBondCount", "HBD", "HBA", "TPSA",
    "FractionCSP3", "DoubleBondCount", "TripleBondCount",
]
CONJ_FEATURES = [
    "n_conj_bonds", "frac_conj_bonds", "n_carbonyl",
    "n_conj_carbonyl", "n_conj_diene", "n_nitroso_noxide",
]
TRAIN_FRACTIONS = np.array([0.20, 0.35, 0.50, 0.70, 0.85, 1.00])


def make_models(trees=200):
    """Fresh bounded baseline models for each feature-set learning curve."""
    return {
        "Ridge": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]),
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, max_bins=64, max_leaf_nodes=15, random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=trees, max_depth=20, min_samples_leaf=2,
            max_features="sqrt", random_state=42, n_jobs=1,
        ),
    }


@threadpool_limits.wrap(limits=1)
def run(data_path, name="homo_lumo", folds=5, trees=200, output_parent="results"):
    if folds < 2 or trees < 1:
        raise ValueError("folds must be at least 2 and trees must be positive")

    # Read only columns needed for the full comparison. For a fair comparison,
    # every feature set is evaluated on the complete rows required by conj_fp.
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    fp_features = [col for col in header if col.startswith("morgan_")]
    if not fp_features:
        raise ValueError("No morgan_ fingerprint columns found; conj_fp cannot be compared")
    feature_sets = {
        "base": BASE_FEATURES,
        "conj": BASE_FEATURES + CONJ_FEATURES,
        "conj_fp": BASE_FEATURES + CONJ_FEATURES + fp_features,
    }
    required = list(dict.fromkeys(sum(feature_sets.values(), []) + ["gap_ev"]))
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    # Fingerprints and descriptors are numeric. Float32 reduces dataframe RAM.
    df = pd.read_csv(data_path, usecols=required,
                     dtype={col: "float32" for col in required})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    if len(df) < folds * 2:
        raise ValueError(f"Need at least {folds * 2} complete molecules; found {len(df)}")

    X_all = df[required[:-1]]
    y = df["gap_ev"].to_numpy(dtype=np.float64)
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X_all, y))
    out_dir = Path(output_parent) / f"{Path(name).name}_learning_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Complete molecules shared by all feature sets: {len(df)}")
    print(f"CV folds: {folds}; models: Ridge, HistGradientBoosting, RandomForest")
    print(f"Output folder: {out_dir}")

    rows = []
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    total_runs = len(feature_sets) * len(make_models(trees))
    run_number = 0
    for feature_set, features in feature_sets.items():
        X = df[features]
        for model_name, model in make_models(trees).items():
            run_number += 1
            print(f"[{run_number}/{total_runs}] {feature_set}: {model_name}", flush=True)
            sizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
                estimator=model,
                X=X,
                y=y,
                train_sizes=TRAIN_FRACTIONS,
                cv=cv_splits,
                scoring=scorer,
                n_jobs=1,
                pre_dispatch=1,
                shuffle=False,  # folds were shuffled once and shared across comparisons
                return_times=True,
                error_score="raise",
            )
            # sklearn returns negated MAE because its scorers are maximized.
            for i, size in enumerate(sizes):
                rows.append({
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "TrainingMoleculesPerFold": int(size),
                    "TrainingFractionOfFullDataset": float(size / len(df)),
                    "TrainMAE_mean_eV": float(-train_scores[i].mean()),
                    "TrainMAE_std_eV": float(train_scores[i].std()),
                    "ValidationMAE_mean_eV": float(-validation_scores[i].mean()),
                    "ValidationMAE_std_eV": float(validation_scores[i].std()),
                    "FitSeconds_mean": float(fit_times[i].mean()),
                    "ScoreSeconds_mean": float(score_times[i].mean()),
                    "CompleteDatasetMolecules": len(df),
                    "CVFolds": folds,
                })
            del model, train_scores, validation_scores, fit_times, score_times
            gc.collect()

    results = pd.DataFrame(rows).sort_values(
        ["Model", "FeatureSet", "TrainingMoleculesPerFold"]
    )
    results.to_csv(out_dir / "learning_curve_data.csv", index=False)

    # At 100% learning-curve size, each fold trains on the largest available
    # CV training partition (typically (folds-1)/folds of all rows).
    comparison = results.loc[
        results["TrainingMoleculesPerFold"] == results["TrainingMoleculesPerFold"].max(),
        ["FeatureSet", "Model", "TrainingMoleculesPerFold",
         "ValidationMAE_mean_eV", "ValidationMAE_std_eV", "TrainMAE_mean_eV"]
    ].rename(columns={"ValidationMAE_mean_eV": "CV_MAE_eV",
                      "ValidationMAE_std_eV": "CV_MAE_std_eV"})
    comparison = comparison.sort_values("CV_MAE_eV")
    comparison.to_csv(out_dir / "base_conj_conj_fp_comparison.csv", index=False)

    for model_name in results["Model"].unique():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        subset = results[results["Model"] == model_name]
        for feature_set in feature_sets:
            curve = subset[subset["FeatureSet"] == feature_set].sort_values(
                "TrainingMoleculesPerFold"
            )
            x = curve["TrainingMoleculesPerFold"].to_numpy()
            mae = curve["ValidationMAE_mean_eV"].to_numpy()
            spread = curve["ValidationMAE_std_eV"].to_numpy()
            line, = ax.plot(x, mae, marker="o", label=f"{feature_set} validation")
            ax.fill_between(x, mae - spread, mae + spread, alpha=0.14,
                            color=line.get_color())
            ax.plot(x, curve["TrainMAE_mean_eV"], linestyle="--", alpha=0.65,
                    color=line.get_color(), label=f"{feature_set} training")
        ax.set_xlabel("Training molecules per CV fold")
        ax.set_ylabel("Mean absolute error (eV)")
        ax.set_title(f"{model_name}: learning curves by feature set")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(out_dir / f"{safe_name}_learning_curve.png", dpi=160)
        plt.close(fig)

    print("\nFeature-set comparison (lower CV MAE is better at the largest training size):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved plots and CSV files to {out_dir}")
    return results, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="CSV containing gap_ev and base/conj/fingerprint features")
    parser.add_argument("--name", default="homo_lumo", help="Run name used for results subfolder")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=200,
                        help="Random forest tree count (bounded CPU: n_jobs=1)")
    parser.add_argument("--output-parent", default="results")
    args = parser.parse_args()
    run(args.data_path, name=args.name, folds=args.folds, trees=args.trees,
        output_parent=args.output_parent)
