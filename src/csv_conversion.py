"""Validate and sample molecules consistently across ingestion routes."""
from pathlib import Path
import pandas as pd
from rdkit import Chem
from src.schema import normalize_molecules, canonical_smiles

ALLOWED = {"C", "H", "O", "N", "S", "P", "F", "Cl"}
MAX_HEAVY = 10


def passes_filters(smi):
    if not isinstance(smi, str):
        return False
    m = Chem.MolFromSmiles(smi)
    return bool(m is not None
        and {a.GetSymbol() for a in m.GetAtoms()}.issubset(ALLOWED)
        and len(Chem.GetMolFrags(m)) == 1
        and Chem.GetFormalCharge(m) == 0
        and not any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in m.GetAtoms())
        and 0 < m.GetNumHeavyAtoms() <= MAX_HEAVY)


def validated_molecules(df):
    df = normalize_molecules(df)
    df = df.loc[df.SMILES.map(passes_filters)].copy()
    df["SMILES"] = df.SMILES.map(canonical_smiles)
    return df.drop_duplicates("SMILES").reset_index(drop=True)


def sample_valid_molecules(df, n_requested, seed=42, smiles_col="SMILES"):
    df = validated_molecules(df.rename(columns={smiles_col: "SMILES"}))
    if not isinstance(n_requested, int) or not 0 < n_requested <= len(df):
        raise ValueError(f"Requested count must be between 1 and {len(df)}")
    return df.sample(n=n_requested, random_state=seed).reset_index(drop=True)


def csv_validator(filename, iteration=None, output="validated", indexes=None,
                  output_dir="molecules/csv_files/processed"):
    df = normalize_molecules(pd.read_csv(filename))
    if indexes is not None:
        df = df.iloc[list(indexes)]
    valid = validated_molecules(df)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    from src.schema import safe_component
    name = safe_component(f"{output}{iteration if iteration is not None else ''}")
    valid.to_csv(destination / f"{name}.csv", index=False)
    return valid
