"""Resource-conscious HOMO-LUMO baseline analysis.
Existing analysis(path, name, feature_set) calls and return tuple are preserved.
Defaults: one CPU, 200 bounded trees, permutation importance OFF.
Set permutation_repeats=3 to inspect the top 20 RF features (conditional subset).
The RF defaults change predictions; compare error before increasing tree limits.
"""
import os
# Set before importing numpy/sklearn. Runtime limits below also cover notebooks.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
from threadpoolctl import threadpool_limits
import gc
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


@threadpool_limits.wrap(limits=1)
def analysis(DATA_PATH, CSV_NAME, FEATURE_SET="base", *, n_jobs=1,
             n_estimators=200, permutation_repeats=0, permutation_top_k=20,
             permutation_max_samples=300):
    """Run baselines; n_jobs must be a positive, explicitly bounded worker count.

    Permutation ranking is limited to RF's top-k impurity-ranked features;
    untested features are not assigned zero importance. Correlated descriptors
    and this preselection can hide useful features. Keep it diagnostic only.
    """
    if n_jobs < 1 or n_estimators < 1:
        raise ValueError("n_jobs and n_estimators must be positive")
    if permutation_repeats < 0 or permutation_top_k < 1 or permutation_max_samples < 1:
        raise ValueError("Invalid permutation limits")

    CSV_NAME = f"{CSV_NAME}_{FEATURE_SET}"
    # --------------------------------------------------
    # 1. Output directories
    # --------------------------------------------------

    PREDICTION_DIR = "data/model_predictions"
    FIGURE_DIR = "results/figures"

    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------

    # Compact numeric features, and avoid fingerprint columns for descriptor runs.
    header = pd.read_csv(DATA_PATH, nrows=0).columns
    usecols = [c for c in header if FEATURE_SET == "conj_fp" or not c.startswith("morgan_")]
    numeric = [c for c in usecols if c.startswith("morgan_") or c in
               BASE_FEATURES_GLOBAL + CONJ_FEATURES_GLOBAL + ["gap_ev"]]
    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype={c: "float32" for c in numeric})

    print(f"Dataset contains {len(df)} molecules")


    # --------------------------------------------------
    # 3. Features / target
    # --------------------------------------------------

    TARGET = "gap_ev"

    # FEATURES = [
    #     "MolWt",
    #     "HeavyAtomCount",
    #     "NumC",
    #     "NumN",
    #     "NumO",
    #     "RingCount",
    #     "AromaticRingCount",
    #     "RotatableBondCount",
    #     "HBD",
    #     "HBA",
    #     "TPSA",
    #     "FractionCSP3",
    #     "DoubleBondCount",
    #     "TripleBondCount",
    # ]
    # df = df.dropna(subset=FEATURES + [TARGET])
    TARGET = "gap_ev"

    BASE_FEATURES = [
        "MolWt", "HeavyAtomCount", "NumC", "NumN", "NumO", "RingCount",
        "AromaticRingCount", "RotatableBondCount", "HBD", "HBA", "TPSA",
        "FractionCSP3", "DoubleBondCount", "TripleBondCount",
    ]
    CONJ_FEATURES = [
        "n_conj_bonds", "frac_conj_bonds", "n_carbonyl",
        "n_conj_carbonyl", "n_conj_diene", "n_nitroso_noxide",
    ]
    FP_FEATURES = [c for c in df.columns if c.startswith("morgan_")]

    FEATURE_SETS = {
        "base": BASE_FEATURES,
        "conj": BASE_FEATURES + CONJ_FEATURES,
        "conj_fp": BASE_FEATURES + CONJ_FEATURES + FP_FEATURES,
    }
    if FEATURE_SET not in FEATURE_SETS:
        raise ValueError("FEATURE_SET must be base, conj, or conj_fp")
    FEATURES = FEATURE_SETS[FEATURE_SET]
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if FEATURE_SET == "conj_fp" and not FP_FEATURES:
        raise ValueError("conj_fp requires morgan_ columns")
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Usable molecules: {len(df)}")
    if len(df) < 10:
        raise ValueError("At least 10 complete molecules are required")

    X = df[FEATURES]
    y = df[TARGET]

    # X = df[FEATURES]
    # y = df[TARGET]
    


    # --------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training molecules: {len(X_train)}")
    print(f"Testing molecules:  {len(X_test)}")

    print(f"\nTarget mean: {y.mean():.3f} eV")
    print(f"Target std:  {y.std():.3f} eV")


    # --------------------------------------------------
    # 5. Define baseline models
    # --------------------------------------------------

    models = {

        # Always predicts training-set average
        "Mean Predictor": DummyRegressor(
            strategy="mean"
        ),

        # Linear baseline
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            RidgeCV(
                alphas=[
                    0.001,
                    0.01,
                    0.1,
                    1,
                    10,
                    100,
                    1000
                ]
            )
        ),

        # Strong nonlinear tabular baseline
        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=300,
            max_bins=64, max_leaf_nodes=15,
            random_state=42
        ),

        # Your current model
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=n_jobs
        ),
    }


    # --------------------------------------------------
    # 6. Train and evaluate all models
    # --------------------------------------------------

    model_results = []
    predictions = {}

    for name, model in models.items():

        print(f"Fitting {name}...", flush=True)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        print(f"  finished in {time.perf_counter() - started:.1f}s", flush=True)

        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        model_results.append({
            "Model": name,
            "MAE_eV": mae,
            "RMSE_eV": rmse,
            "R2": r2
        })


    # --------------------------------------------------
    # 7. Optional one-descriptor baseline
    # --------------------------------------------------

    size_feature = ["HeavyAtomCount"]

    X_train_size = X_train[size_feature]
    X_test_size = X_test[size_feature]

    size_model = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=[
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000
            ]
        )
    )

    size_model.fit(
        X_train_size,
        y_train
    )

    y_pred_size = size_model.predict(
        X_test_size
    )

    predictions["Heavy Atom Only"] = y_pred_size

    model_results.append({
        "Model": "Heavy Atom Only",
        "MAE_eV": mean_absolute_error(
            y_test,
            y_pred_size
        ),
        "RMSE_eV": np.sqrt(
            mean_squared_error(
                y_test,
                y_pred_size
            )
        ),
        "R2": r2_score(
            y_test,
            y_pred_size
        )
    })


    # --------------------------------------------------
    # 8. Results dataframe
    # --------------------------------------------------

    results_df = pd.DataFrame(
        model_results
    )

    results_df = results_df.sort_values(
        "MAE_eV"
    )

    print("\nModel Comparison")
    print("----------------")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        f"results/{CSV_NAME}_model_comparison.csv",
        index=False
    )


    # --------------------------------------------------
    # 9. Compare MAE to target spread
    # --------------------------------------------------

    target_std = y_test.std()

    results_df["MAE_over_target_std"] = (
        results_df["MAE_eV"] / target_std
    )

    print("\nMAE relative to test-set target standard deviation")
    print("--------------------------------------------------")

    print(
        results_df[
            [
                "Model",
                "MAE_eV",
                "MAE_over_target_std"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # Include the normalized metric in the saved table as well.
    results_df.to_csv(f"results/{CSV_NAME}_model_comparison.csv", index=False)

    # --------------------------------------------------
    # 10. Model comparison plot
    # --------------------------------------------------

    plot_results = results_df.sort_values(
        "MAE_eV",
        ascending=True
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        plot_results["Model"],
        plot_results["MAE_eV"]
    )

    plt.xlabel("Mean Absolute Error (eV)")
    plt.ylabel("Model")
    plt.title("HOMO-LUMO Gap Model Comparison")

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/{CSV_NAME}_model_comparison.png",
        dpi=150
    )

    #plt.show()
    plt.close()


    # --------------------------------------------------
    # 11. Random Forest detailed analysis
    # --------------------------------------------------

    rf = models["Random Forest"]

    rf_pred = predictions[
        "Random Forest"
    ]

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_pred
        )
    )

    rf_r2 = r2_score(
        y_test,
        rf_pred
    )


    # --------------------------------------------------
    # 12. Save Random Forest predictions + worst residuals
    # --------------------------------------------------

    residuals = y_test.values - rf_pred
    absolute_residuals = np.abs(residuals)

    # Pull the original molecule information using the test-set indices
    prediction_df = df.loc[X_test.index].copy()
    prediction_df = prediction_df.drop(columns=FP_FEATURES, errors="ignore")

    # Add model results
    prediction_df["Actual"] = y_test.values
    prediction_df["Predicted"] = rf_pred
    prediction_df["Residual"] = residuals
    prediction_df["AbsoluteResidual"] = absolute_residuals

    # Sort so the worst predictions are first
    prediction_df = prediction_df.sort_values(
        "AbsoluteResidual",
        ascending=False
    )

    prediction_path = (
        f"{PREDICTION_DIR}/{CSV_NAME}.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"\nPredictions saved to: {prediction_path}"
    )

    # --------------------------------------------------
    # 13. Worst residuals
    # --------------------------------------------------

    N_WORST = 35

    worst_predictions = prediction_df.head(N_WORST)

    print(
        f"\nTop {N_WORST} Worst Predictions"
    )

    print(
        "------------------------------"
    )

    # Change these depending on which identifier columns exist
    display_columns = []

    for column in [
        "CID",
        "Name",
        "SMILES",
        "Actual",
        "Predicted",
        "Residual",
        "AbsoluteResidual"
    ]:
        if column in worst_predictions.columns:
            display_columns.append(column)

    print(
        worst_predictions[
            display_columns
        ].to_string(index=False)
    )

    worst_path = (
        f"{PREDICTION_DIR}/"
        f"{CSV_NAME}_worst_residuals.csv"
    )

    worst_predictions.to_csv(
        worst_path,
        index=False
    )

    print(
        f"\nWorst residuals saved to: "
        f"{worst_path}"
    )
    # --------------------------------------------------
    # 13. Predicted vs actual
    # --------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_test,
        rf_pred,
        alpha=0.6
    )

    minimum = min(
        y_test.min(),
        rf_pred.min()
    )

    maximum = max(
        y_test.max(),
        rf_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "DFT HOMO-LUMO Gap (eV)"
    )

    plt.ylabel(
        "Predicted HOMO-LUMO Gap (eV)"
    )

    plt.title(
        "Random Forest: Predicted vs. DFT Gap"
    )

    plt.text(
        0.05,
        0.95,
        (
            f"MAE = {rf_mae:.3f} eV\n"
            f"RMSE = {rf_rmse:.3f} eV\n"
            f"$R^2$ = {rf_r2:.3f}"
        ),
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/{CSV_NAME}_predicted_vs_actual.png",
        dpi=150
    )

    #plt.show()
    plt.close()



    # --------------------------------------------------
    # 14. Residual plot
    # --------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        rf_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted HOMO-LUMO Gap (eV)"
    )

    plt.ylabel(
        "Residual (DFT - Predicted) (eV)"
    )

    plt.title(
        "Random Forest Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/{CSV_NAME}_residuals.png",
        dpi=150
    )

    #plt.show()
    plt.close()
    # --------------------------------------------------
    # Plot worst residuals
    # --------------------------------------------------

    worst_plot = worst_predictions.sort_values(
        "AbsoluteResidual",
        ascending=True
    )

    if "SMILES" in worst_plot.columns:
        labels = worst_plot["SMILES"]
    elif "Name" in worst_plot.columns:
        labels = worst_plot["Name"]
    elif "CID" in worst_plot.columns:
        labels = worst_plot["CID"].astype(str)
    else:
        labels = worst_plot.index.astype(str)

    plt.figure(figsize=(10, 7))

    plt.barh(
        labels,
        worst_plot["AbsoluteResidual"]
    )

    plt.xlabel("Absolute Residual (eV)")
    plt.ylabel("Molecule")
    plt.title(
        f"Top {N_WORST} Worst Random Forest Predictions"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/"
        f"{CSV_NAME}_worst_residuals.png",
        dpi=150
    )

    #plt.show()
    plt.close()

    # --------------------------------------------------
    # 15. Random Forest feature importance
    # --------------------------------------------------

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)

    plot_imp = importance_df.tail(20)          # top 20 only

    plt.figure(figsize=(8, 7))
    plt.barh(plot_imp["Feature"], plot_imp["Importance"])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{CSV_NAME}_feature_importance.png", dpi=150)
    #plt.show()
    plt.close()

    # --------------------------------------------------
    # 16. Permutation importance
    # --------------------------------------------------

    permutation_df = bounded_permutation_importance(
        rf, X_test, y_test, repeats=permutation_repeats,
        top_k=permutation_top_k, max_samples=permutation_max_samples)
    permutation_df.to_csv(f"results/{CSV_NAME}_permutation_importance.csv", index=False)
    if not permutation_df.empty:
        plt.figure(figsize=(8, 7))
        plt.barh(permutation_df["Feature"], permutation_df["Importance"],
                 xerr=permutation_df["Std"])
        plt.xlabel("Increase in MAE After Permutation (eV)")
        plt.title("Permutation Importance (preselected RF features)")
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/{CSV_NAME}_permutation_importance.png", dpi=150)
        plt.close()
    else:
        print("Permutation importance skipped; enable with permutation_repeats=3.")

    # --------------------------------------------------
    # 17. Return useful objects
    # --------------------------------------------------
    print(FEATURE_SET + " complete!")
    return (
        models,
        results_df,
        prediction_df,
        importance_df,
        permutation_df
    )



BASE_FEATURES_GLOBAL = [
    "MolWt", "HeavyAtomCount", "NumC", "NumN", "NumO", "RingCount",
    "AromaticRingCount", "RotatableBondCount", "HBD", "HBA", "TPSA",
    "FractionCSP3", "DoubleBondCount", "TripleBondCount",
]
CONJ_FEATURES_GLOBAL = [
    "n_conj_bonds", "frac_conj_bonds", "n_carbonyl", "n_conj_carbonyl",
    "n_conj_diene", "n_nitroso_noxide",
]


def bounded_permutation_importance(model, X, y, *, repeats, top_k, max_samples):
    columns = ["Feature", "Importance", "Std"]
    if not repeats:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(42)
    chosen_rows = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
    sample = X.iloc[chosen_rows].copy()
    target = np.asarray(y)[chosen_rows]
    baseline = mean_absolute_error(target, model.predict(sample))
    selected = np.argsort(model.feature_importances_)[-min(top_k, X.shape[1]):]
    rows = []
    for j in selected:
        col = sample.columns[j]
        original = sample[col].to_numpy(copy=True)
        scores = []
        for _ in range(repeats):
            sample[col] = rng.permutation(original)
            scores.append(mean_absolute_error(target, model.predict(sample)) - baseline)
        sample[col] = original
        rows.append({"Feature": col, "Importance": np.mean(scores), "Std": np.std(scores)})
    return pd.DataFrame(rows, columns=columns).sort_values("Importance")


@threadpool_limits.wrap(limits=1)
def cv_check(DATA_PATH, TARGET="gap_ev", *, n_splits=5, n_repeats=1,
             n_estimators=200, n_jobs=1):
    """Descriptive CV comparison; not an independent final test estimate.

    All feature sets use the same complete rows and folds. Each fit supplies
    MAE and R2 together. Folds run serially to avoid duplicating forests.
    """
    from sklearn.model_selection import RepeatedKFold, cross_validate
    if n_jobs < 1 or n_splits < 2 or n_repeats < 1 or n_estimators < 1:
        raise ValueError("Invalid CV/resource limits")
    header = pd.read_csv(DATA_PATH, nrows=0).columns
    fp = [c for c in header if c.startswith("morgan_")]
    sets = {"base": BASE_FEATURES_GLOBAL}
    if all(c in header for c in CONJ_FEATURES_GLOBAL):
        sets["conj"] = BASE_FEATURES_GLOBAL + CONJ_FEATURES_GLOBAL
        if fp:
            sets["conj_fp"] = sets["conj"] + fp
    columns = list(dict.fromkeys(sum(sets.values(), []) + [TARGET]))
    df = pd.read_csv(DATA_PATH, usecols=columns, dtype="float32")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < n_splits * 2:
        raise ValueError("Need at least two validation molecules per fold")
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rows = []
    for set_name, features in sets.items():
        models = {
            "Ridge Regression": make_pipeline(StandardScaler(), RidgeCV(alphas=[.01, .1, 1, 10, 100, 1000])),
            "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, max_bins=64, max_leaf_nodes=15, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators,
                max_depth=20, min_samples_leaf=2, max_features="sqrt",
                random_state=42, n_jobs=n_jobs),
        }
        for name, model in models.items():
            print(f"CV {set_name}: {name}", flush=True)
            scores = cross_validate(model, df[features], df[TARGET], cv=cv,
                scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
                n_jobs=1, pre_dispatch=1, error_score="raise")
            rows.append({"FeatureSet": set_name, "Model": name,
                "MAE_mean": -scores["test_MAE"].mean(),
                "MAE_std": scores["test_MAE"].std(),
                "R2_mean": scores["test_R2"].mean(), "R2_std": scores["test_R2"].std()})
        del models
        gc.collect()
    result = pd.DataFrame(rows).sort_values(["Model", "FeatureSet"])
    os.makedirs("results", exist_ok=True)
    result.to_csv("results/cv_check_summary.csv", index=False)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path")
    parser.add_argument("--name", default="homo_lumo")
    parser.add_argument("--feature-set", choices=["base", "conj", "conj_fp"], default="base")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--permutation-repeats", type=int, default=0)
    parser.add_argument("--cv", action="store_true", help="Run CV instead of holdout analysis")
    args = parser.parse_args()
    if args.cv:
        cv_check(args.data_path, n_estimators=args.trees, n_jobs=args.jobs)
    else:
        analysis(args.data_path, args.name, args.feature_set, n_jobs=args.jobs,
                 n_estimators=args.trees, permutation_repeats=args.permutation_repeats)
