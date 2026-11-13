import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RotatableBondCount": Lipinski.NumRotatableBonds(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumC": sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()),
        "NumN": sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()),
        "NumO": sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()),
        "NumF": sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms()),
        "DoubleBondCount": sum(
            b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds()
        ),
        "TripleBondCount": sum(
            b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds()
        ),
    }
_PATTERNS = {
    "n_carbonyl": "[CX3]=[OX1]",
    "n_conj_carbonyl": "[OX1]=[CX3]-[#6,#7]=,#[#6,#7,#8]",  # C=O adjacent to another pi bond
    "n_conj_diene": "[#6]=[#6]-[#6]=[#6]",
    "n_ene_yne": "[#6]=[#6]-[#6]#[#6]",
    "n_nitroso_noxide": "[#7]=[#8,#7]",
}

_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _PATTERNS.items()}
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def extra_features(smi, include_fp=True):
    m = Chem.MolFromSmiles(smi)
    n_bonds = max(m.GetNumBonds(), 1)
    n_conj = sum(b.GetIsConjugated() for b in m.GetBonds())
    feats = {"n_conj_bonds": n_conj, "frac_conj_bonds": n_conj / n_bonds}
    for name, patt in _PATTERNS.items():
        feats[name] = len(m.GetSubstructMatches(patt))
    if include_fp:
        fp = _gen.GetFingerprintAsNumPy(m)
        feats.update({f"morgan_{i}": int(v) for i, v in enumerate(fp)})
    return feats

def get_all_features(smi):
    desc = get_descriptors(smi)
    extra = extra_features(smi, include_fp=True)
    return {**desc, **extra}


def main():
    url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/"
        "datasets/qm9.csv"
    )
    df = pd.read_csv(url)
    df = df.dropna(subset=["smiles", "gap"]).copy()

    # Start small while checking the pipeline.
    df = df.sample(n=min(10_000, len(df)), random_state=42)

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        features = get_all_features(row["smiles"])
        if features is None:
            skipped += 1
            continue

        rows.append({
            "smiles": row["smiles"],
            **features,
            "gap_ev": float(row["gap"]) * 27.211386245988,
        })

    data = pd.DataFrame(rows)

    X = data.drop(columns=["smiles", "gap_ev"])
    y = data["gap_ev"]

    train_idx, test_idx = train_test_split(
        data.index, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        scoring="neg_mean_absolute_error",
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )

    parameter_space = {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_space,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1,
        random_state=42,
        verbose=1,
        refit=True,
    )

    # Tune using only the training portion.
    search.fit(X.loc[train_idx], y.loc[train_idx])

    print("Best parameters:", search.best_params_)
    print(f"Cross-validation MAE: {-search.best_score_:.4f} eV")

    # Evaluate the selected model on the untouched test portion.
    best_model = search.best_estimator_
    pred = best_model.predict(X.loc[test_idx])

    print(f"Test MAE: {mean_absolute_error(y.loc[test_idx], pred):.4f} eV")
    print(f"Test MSE: {mean_squared_error(y.loc[test_idx], pred):.4f} eV²")
    print(f"Test R²:  {r2_score(y.loc[test_idx], pred):.4f}")
    print("Boosting rounds used:", best_model.n_iter_) 


if __name__ == "__main__":
    main()