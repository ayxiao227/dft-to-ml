import pandas as pd
import regex as re
import random
import numpy as np
from pathlib import Path
from src.formula_to_atoms import get_elements_from_smiles as fta
from rdkit import Chem


ALLOWED = {"C", "H", "O", "N", "S", "P", "F", "Cl"}
MAX_HEAVY = 10


def sample_valid_molecules(df, n_requested, seed=42, smiles_col="SMILES"):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))              # every row exactly once, random order

    kept = []
    for pos in order:
        smi = df.iloc[pos][smiles_col]
        if passes_filters(smi):                   # your filter function from before
            kept.append(pos)
            if len(kept) == n_requested:
                break

    if len(kept) < n_requested:
        print(f"Warning: only {len(kept)} molecules passed the filters (requested {n_requested})")

    return df.iloc[kept].copy()
def passes_filters(smi):
    # 1. Element check (your existing function)
    elements = fta(smi)
    if elements is None or not elements.issubset(ALLOWED):
        return False

    # 2. Structure checks
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    if len(Chem.GetMolFrags(m)) != 1:                                # single molecule, no salts/mixtures
        return False
    if Chem.GetFormalCharge(m) != 0:                                 # neutral only
        return False
    if any(a.GetNumRadicalElectrons() > 0 for a in m.GetAtoms()):    # closed-shell only
        return False
    if any(a.GetIsotope() != 0 for a in m.GetAtoms()):               # no isotopically labeled variants
        return False

    # 3. Size check
    n_heavy = m.GetNumHeavyAtoms()
    return 0 < n_heavy <= MAX_HEAVY

def csv_validator(filename, iteration, output, indexes = None):
    """
    filename = csv filename of ALL the molecules
    iteration can be any number, will be appended at the end of the filename
    output is the name of the output file
    indexes is a list of row indices to select

    if you want an iteration, make sure to add a _ at the end of output
    if you don't want an iteration don't add a _ and set iteration to None
    """
    df = pd.read_csv(filename)
    df.columns = df.columns.str.replace('_', '')        # note: renames all columns

    new_df = df.iloc[indexes] if indexes else df

    n_before = len(new_df)
    valid_df = new_df[new_df["SMILES"].apply(passes_filters)]
    print(f"Kept {len(valid_df)} of {n_before} molecules")

    # Top up with random valid molecules only when a specific count was requested
    if indexes:
        shortfall = len(indexes) - len(valid_df)
        if shortfall > 0:
            pool = df.drop(index=new_df.index)           # exclude every row already tried
            extra = sample_valid_molecules(pool, n_requested=shortfall, seed=43)
            valid_df = pd.concat([valid_df, extra])

    valid_df = valid_df.drop_duplicates(subset="SMILES")  # no repeats from the top-up
    out_name = f"{output}{iteration}.csv" if iteration is not None else f"{output}.csv"


    print(valid_df.head())
    #rename the column names of the valid_df to match the key names in the properties list
    #from: Compound_CID,IUPAC_Name,SMILES,Heavy_Atom_Count,Molecular_Weight,Molecular_Formula,Charge,Name
    #to: CID,MolecularFormula,MolecularWeight,ConnectivitySMILES,IUPACName,Charge,HeavyAtomCount
    valid_df = valid_df.rename(columns={
        "CompoundCID": "CID",
        "Molecular_Formula": "MolecularFormula",
        "SMILES": "ConnectivitySMILES",
        "Molecular_Weight": "MolecularWeight",
        "Charge": "Charge",
        "Heavy_Atom_Count": "HeavyAtomCount",
        "IUPAC_Name" : "IUPACName"
    })
    #CID,MolecularFormula,MolecularWeight,ConnectivitySMILES,IUPACName,Charge,HeavyAtomCount
    valid_df.to_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/processed/" + out_name, index=False)
    print(len(valid_df), "valid molecules saved to " + out_name)
    return valid_df

# if __name__ == "__main__":
#     filename = input("Enter the path to the CSV file: ").strip()
#     print(type(filename))
#     iteration = 0
#     output = input("Enter the output file name (without extension): ").strip()
#     valid_df = csv_validator(filename, iteration, output)
#     print("Valid molecules saved to " + output + str(iteration) + ".csv")