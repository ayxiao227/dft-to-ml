import pandas as pd
import regex as re
from pathlib import Path

def csv_validator(filename, iteration):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.replace('_', '')

    constraints = {
        "max_molecular_weight": 150,
        "max_heavy_atoms": 10,
        "allowed_elements": {"C", "H", "O", "N"},
        "neutral_only": True
    }
    properties = ["IUPACName", "MolecularFormula", "ConnectivitySMILES", "MolecularWeight", "Charge", "HeavyAtomCount"]

    valid_df = df[
        (df["Charge"] == 0)
        & (df["HeavyAtomCount"] < 10)
        & (df["MolecularWeight"] < 150)
    ]

    valid_df = valid_df[
        valid_df.apply(lambda x: x.issubset(constraints["allowed_elements"]))
    ]

    df.to_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/processed_molecules_" + str(iteration) +".csv", index=False)
    return valid_df

