import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/descriptors/test_1.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset contains {len(df)} molecules")
print(df.head())


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

TARGET = "gap_ev"

FEATURES = [
    "MolWt",
    "HeavyAtomCount",
    "NumC",
    "NumN",
    "NumO",
    # "NumS",
    # "NumF",
    # "NumCl",
    # "NumBr",
    # "NumI",
    "RingCount",
    "AromaticRingCount",
    "RotatableBondCount",
    "HBD",
    "HBA",
    "TPSA",
    "FractionCSP3",
    "DoubleBondCount",
    "TripleBondCount",
]

# Remove rows with missing data
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]


# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training molecules: {len(X_train)}")
print(f"Testing molecules:  {len(X_test)}")


# --------------------------------------------------
# 4. Random Forest
# --------------------------------------------------

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

y_pred = rf.predict(X_test)


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")
print("--------------------")
print(f"MAE:  {mae:.4f} eV")
print(f"RMSE: {rmse:.4f} eV")
print(f"R²:   {r2:.4f}")


# --------------------------------------------------
# 7. Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance")
print("------------------")
print(importance.to_string(index=False))


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_pred - y_test.values
})

results.to_csv(
    "data/model_predictions/data_1.csv",
    index=False
)

result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": result.importances_mean,
    "Std": result.importances_std
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print(importance_df)

print("\nPredictions saved to data/model_predictions.csv")