import pandas as pd
import regex as re
from pathlib import Path
from src.formula_to_atoms import get_elements_from_smiles as fta

def csv_validator(filename, iteration, output):
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
        valid_df["SMILES"].apply(
            lambda f: (res := fta(f)) is not None and res.issubset(constraints["allowed_elements"])
        )
    ]

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
    valid_df.to_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/" + output + str(iteration) +".csv", index=False)
    return valid_df

if __name__ == "__main__":
    filename = input("Enter the path to the CSV file: ").strip()
    print(type(filename))
    iteration = 0
    output = input("Enter the output file name (without extension): ").strip()
    valid_df = csv_validator(filename, iteration, output)
    print("Valid molecules saved to " + output + str(iteration) + ".csv")