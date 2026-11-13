"""Bounded model tuning for HOMO-LUMO gaps (all errors in eV).

Place alongside optimized ml_analysis.py. Core dependencies: numpy, pandas,
scikit-learn, matplotlib, joblib, threadpoolctl. Optional: xgboost, catboost.
Example: python ml_analysis_advanced.py data.csv --feature-set conj --trials 4

Models/parameters are chosen by training-only CV MAE; only the selected model
is evaluated on the untouched 20% test partition. Missing descriptors are
imputed inside each fold; no HOMO/LUMO energies are added as predictors.
Use --group-column CID to keep repeated molecule IDs in the same partition
and CV fold. Equivalent SMILES need canonicalization upstream, or a canonical
identity column. Default row splits assume independent, distinct molecules.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ml_analysis import BASE_FEATURES_GLOBAL, CONJ_FEATURES_GLOBAL


def pipeline(estimator, scale=False):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("constant", VarianceThreshold())]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def candidates(trees, optional):
    # Deliberately bounded tree sizes and searches; no nested all-core execution.
    yield "Ridge", pipeline(Ridge(solver="lsqr"), True), {
        "model__alpha": [.01, .1, 1., 10., 100., 1000., 10000.]}
    yield "ExtraTrees", pipeline(ExtraTreesRegressor(n_estimators=trees, n_jobs=1, random_state=42)), {
        "model__max_features": [.5, 1.], "model__min_samples_leaf": [1, 2, 4],
        "model__max_depth": [12, 20]}
    yield "RandomForest", pipeline(RandomForestRegressor(n_estimators=trees, n_jobs=1, random_state=42)), {
        "model__max_features": ["sqrt", .5, 1.], "model__min_samples_leaf": [1, 2, 4],
        "model__max_depth": [12, 20]}
    yield "HistGradientBoosting", pipeline(HistGradientBoostingRegressor(
        max_iter=300, max_bins=64, early_stopping=True, random_state=42)), {
        "model__learning_rate": [.03, .06, .1], "model__max_leaf_nodes": [7, 15, 31],
        "model__min_samples_leaf": [10, 20, 30], "model__l2_regularization": [0., 1., 10.]}
    # Kernel cache is capped; exact SVR may still take longer on large datasets.
    yield "RBF_SVR", pipeline(SVR(kernel="rbf", cache_size=128), True), {
        "model__C": [1., 10., 100.], "model__epsilon": [.05, .1, .2],
        "model__gamma": ["scale", .001, .01, .1]}
    if optional:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("Skipping XGBoost: install xgboost to enable it.", flush=True)
        else:
            yield "XGBoost", pipeline(XGBRegressor(n_estimators=300, tree_method="hist",
                objective="reg:squarederror", n_jobs=1, random_state=42)), {
                "model__max_depth": [3, 5, 7], "model__learning_rate": [.03, .06, .1],
                "model__subsample": [.8, 1.], "model__colsample_bytree": [.7, 1.],
                "model__reg_lambda": [1., 10.]}
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            print("Skipping CatBoost: install catboost to enable it.", flush=True)
        else:
            yield "CatBoost", pipeline(CatBoostRegressor(iterations=300,
                loss_function="RMSE", thread_count=1, random_seed=42,
                verbose=False, allow_writing_files=False)), {
                "model__depth": [4, 6, 8], "model__learning_rate": [.03, .06, .1],
                "model__l2_leaf_reg": [1., 3., 10.]}


@threadpool_limits.wrap(limits=1)
def analysis_advanced(data_path, name="advanced", feature_set="base", *, trials=4,
                      folds=3, trees=200, optional=False, group_column=None,
                      output_dir="results/advanced"):
    if trials < 1 or folds < 2 or trees < 1:
        raise ValueError("Need trials >= 1, folds >= 2, trees >= 1")
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    features = list(BASE_FEATURES_GLOBAL)
    if feature_set not in ("base", "conj", "conj_fp"):
        raise ValueError("Unknown feature set")
    if feature_set != "base":
        features += CONJ_FEATURES_GLOBAL
    if feature_set == "conj_fp":
        fp = [c for c in header if c.startswith("morgan_")]
        if not fp:
            raise ValueError("conj_fp requires morgan_ fingerprint columns")
        features += fp
    missing = sorted(set(features + ["gap_ev"]) - set(header))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if group_column and group_column not in header:
        raise ValueError(f"Missing group column: {group_column}")
    identifiers = [c for c in ("CID", "Name", "SMILES", "cid", "name", "smiles", group_column)
                   if c and c in header]
    columns = list(dict.fromkeys(identifiers + features + ["gap_ev"]))
    df = pd.read_csv(data_path, usecols=columns)
    # Original zero-based CSV data-row index is retained for identifying outliers.
    df.insert(0, "SourceRow", np.arange(len(df)))
    df[features + ["gap_ev"]] = df[features + ["gap_ev"]].apply(pd.to_numeric, errors="coerce")
    df[features + ["gap_ev"]] = df[features + ["gap_ev"]].replace([np.inf, -np.inf], np.nan)
    initial = len(df)
    df = df.dropna(subset=["gap_ev"]).copy()
    if len(df) < max(20, folds * 3):
        raise ValueError("Too few finite targets for holdout/CV")
    if group_column and df[group_column].isna().any():
        raise ValueError("Group IDs cannot be missing; provide a complete identity column")
    X = df[features].astype("float32")
    y = df["gap_ev"].to_numpy(dtype="float64")
    idx = np.arange(len(df))
    if group_column:
        groups = df[group_column].astype(str).to_numpy()
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
        if len(np.unique(groups[train])) < folds or len(test) < 2:
            raise ValueError("Too few independent groups for requested folds/holdout")
        cv = list(GroupKFold(n_splits=folds).split(X.iloc[train], y[train], groups[train]))
        if any(len(v) < 2 for _, v in cv):
            raise ValueError("Each validation fold needs at least two rows for R2")
    else:
        train, test = train_test_split(idx, test_size=.2, random_state=42)
        cv = list(KFold(n_splits=folds, shuffle=True, random_state=42).split(train))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"{Path(name).name}_{feature_set}"
    split_table = df[["SourceRow"] + list(dict.fromkeys(identifiers))].copy()
    split_table["Partition"] = "train"
    split_table.iloc[test, split_table.columns.get_loc("Partition")] = "test"
    split_table.to_csv(f"{prefix}_split.csv", index=False)
    print(f"Retained {len(df)}/{initial} targets. Training: {len(train)}, test: {len(test)}. "
          f"{len(features)} features; {trials} trials/model, {folds} folds; one CPU.", flush=True)
    rows = []
    best_score, best_model, best_name = -np.inf, None, None
    for model_name, estimator, params in candidates(trees, optional):
        print(f"Tuning {model_name}...", flush=True)
        start = time.perf_counter()
        search = RandomizedSearchCV(estimator, params, n_iter=trials,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            refit="mae", cv=cv, n_jobs=1, pre_dispatch=1, random_state=42,
            error_score="raise", return_train_score=False)
        search.fit(X.iloc[train], y[train])
        i = search.best_index_
        row = {"Model": model_name, "CV_MAE_eV": -search.best_score_,
               "CV_MAE_std": search.cv_results_["std_test_mae"][i],
               "CV_RMSE_eV": -search.cv_results_["mean_test_rmse"][i],
               "CV_R2": search.cv_results_["mean_test_r2"][i],
               "Seconds": time.perf_counter() - start, "Parameters": json.dumps(search.best_params_)}
        rows.append(row)
        pd.DataFrame(rows).sort_values("CV_MAE_eV").to_csv(f"{prefix}_cv_comparison.csv", index=False)
        if search.best_score_ > best_score:
            best_score, best_model, best_name = search.best_score_, search.best_estimator_, model_name
            # Checkpoint is trained on the training partition, never the test set.
            joblib.dump({"model": best_model, "features": features, "target": "gap_ev"},
                        f"{prefix}_model.joblib", compress=3)
        print(f"  CV MAE {row['CV_MAE_eV']:.4f} eV ({row['Seconds']:.1f}s)", flush=True)
        del search, estimator
        gc.collect()
    pred = best_model.predict(X.iloc[test])
    mean_pred = DummyRegressor().fit(X.iloc[train], y[train]).predict(X.iloc[test])
    metrics = {"SelectedModel": best_name, "SelectionCV_MAE_eV": -best_score,
        "Test_MAE_eV": mean_absolute_error(y[test], pred),
        "Test_RMSE_eV": float(np.sqrt(mean_squared_error(y[test], pred))),
        "Test_R2": r2_score(y[test], pred),
        "MeanBaseline_Test_MAE_eV": mean_absolute_error(y[test], mean_pred),
        "TrainingRows": len(train), "TestRows": len(test), "FeatureSet": feature_set,
        "GroupColumn": group_column, "Seed": 42}
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    residuals = df.iloc[test][["SourceRow"] + list(dict.fromkeys(identifiers))].copy()
    residuals["Actual"] = y[test]
    residuals["Predicted"] = pred
    residuals["Residual"] = y[test] - pred
    residuals["AbsoluteResidual"] = np.abs(y[test] - pred)
    residuals = residuals.sort_values("AbsoluteResidual", ascending=False)
    residuals.to_csv(f"{prefix}_predictions.csv", index=False)
    residuals.head(35).to_csv(f"{prefix}_worst_residuals.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y[test], pred, s=12, alpha=.6)
    lo, hi = min(y[test].min(), pred.min()), max(y[test].max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], "--")
    axes[0].set(xlabel="DFT gap (eV)", ylabel="Predicted gap (eV)", title=best_name)
    axes[1].scatter(pred, y[test] - pred, s=12, alpha=.6)
    axes[1].axhline(0, linestyle="--")
    axes[1].set(xlabel="Predicted gap (eV)", ylabel="DFT − predicted (eV)", title="Test residuals")
    fig.tight_layout()
    fig.savefig(f"{prefix}_diagnostics.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    return best_model, pd.DataFrame(rows).sort_values("CV_MAE_eV"), residuals, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path")
    parser.add_argument("--name", default="advanced")
    parser.add_argument("--feature-set", choices=["base", "conj", "conj_fp"], default="base")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--optional", action="store_true", help="Try installed XGBoost and CatBoost")
    parser.add_argument("--group-column", help="Keep repeated molecule IDs together")
    parser.add_argument("--output-dir", default="results/advanced")
    args = parser.parse_args()
    analysis_advanced(args.data_path, args.name, args.feature_set, trials=args.trials,
        folds=args.folds, trees=args.trees, optional=args.optional,
        group_column=args.group_column, output_dir=args.output_dir)
