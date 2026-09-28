import pandas as pd
import regex as re
from pathlib import Path

def get_elements_from_smiles_regex(smiles):
    # bracket atoms: [ ... ]  |  two-letter organic subset: Br, Cl  |  single-letter (incl. aromatic)
    token_re = re.compile(r'\[([^\]]+)\]|(Br|Cl|[BCNOPSFI]|[bcnops])')

    elements = set()
    for m in token_re.finditer(smiles):
        bracket_content, plain = m.group(1), m.group(2)
        if bracket_content is not None:
            # inside brackets: optional isotope digits, then the element symbol
            mm = re.match(r'^\d*([A-Za-z][a-z]?)', bracket_content)
            if mm:
                sym = mm.group(1)
                elements.add(sym.capitalize())
        else:
            elements.add(plain.capitalize())
    return elements

def csv_validator(filename, iteration):
    df = pd.read_csv(filename)

    constraints = {
        "max_molecular_weight": 150,
        "max_heavy_atoms": 10,
        "allowed_elements": {"C", "H", "O", "N"},
        "neutral_only": True
    }
    properties = ["IUPACName", "MolecularFormula", "CanonicalSMILES", "MolecularWeight", "Charge", "HeavyAtomCount"]

    valid_df = df[
        (df["Charge"] == 0)
        & (df["Heavy_Atom_Count"] < 10)
        & (df["Molecular_Weight"] < 150)
    ]

    valid_df = valid_df[
        valid_df["SMILES"].apply(
            lambda smiles:
                get_elements_from_smiles_regex(smiles)
                .issubset(constraints["allowed_elements"])
        )
    ]

    df.to_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/processed_molecules_" + str(iteration) +".csv", index=False)
    return valid_df
