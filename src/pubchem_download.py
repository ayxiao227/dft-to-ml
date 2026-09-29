import pubchempy as pcp
import pandas as pd
from src.formula_to_atoms import get_elements_from_smiles as fta
import time

#Constraints:
#Elements: C, H, O, N only
#Molecular weight: < 150 g/mol
#Heavy atoms: ≤ 10
#No metals
#No salts
#No radicals
#Neutral molecules
#One connected component

REQUEST_DELAY = 0.5

constraints = {
    "max_molecular_weight": 150,
    "max_heavy_atoms": 10,
    "allowed_elements": {"C", "H", "O", "N"},
    "neutral_only": True
}

properties = ["IUPACName", "MolecularFormula", "ConnectivitySMILES", "MolecularWeight", "Charge", "HeavyAtomCount"]

def download_pubchem_data(molecules, iteration, validate):
    data = []
    valid_data = []
    #get properties for each molecule
    for name in molecules:
        print("Molecule: ", name, end=" ")
        result = pcp.get_properties(properties, name, "name")
        try:
            print("Retrieved: ", result[0]["IUPACName"], result[0]["MolecularFormula"])
        except (IndexError, KeyError):
            print(f"Error: Could not retrieve properties for '{name}'.")
            continue
        result[0]["MoleculeName"] = name  # Add the original molecule name to the result
        # if result:
        data.append(result)
        if validate:
            if (fta(result[0]["ConnectivitySMILES"])).issubset(constraints["allowed_elements"]) and result[0]["Charge"] == 0 and result[0]["HeavyAtomCount"] < 10 and float(result[0]["MolecularWeight"]) < 150:
                valid_data.append(result)
        else:
            valid_data.append(result)
            # if get_elements_from_smiles_regex(result[0]["ConnectivitySMILES"]).issubset(constraints["allowed_elements"]) and result[0]["Charge"] == 0 and result[0]["HeavyAtomCount"] < 10 and result[0]["MolecularWeight"] < 150:
            #     valid_data.append(result)
        # else:
        #     print(f"Warning: Could not find '{name}' in PubChem.")
        time.sleep(REQUEST_DELAY)

    #flatten list so it becomes list of dictionaries
    flat_list = [item for sublist in data for item in sublist]
    valid_flat_list = [item for sublist in valid_data for item in sublist]

    df = pd.DataFrame(flat_list)
    valid_df = pd.DataFrame(valid_flat_list)

    df.to_csv("molecules/csv_files/pubchem_molecules_" + str(iteration) + ".csv", index=False)
    valid_df.to_csv("molecules/csv_files/pubchem_molecules_valid_" + str(iteration) + ".csv", index=False)
    return valid_df

# def main():

#     molecules = ["2-ethoxyethanol", "hydrogen gas"]        
#     download_pubchem_data(molecules)

# if __name__ == "__main__":
#     main()
