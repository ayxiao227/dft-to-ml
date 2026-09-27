import pubchempy as pcp
import pandas as pd
import re

#Constraints:
#Elements: C, H, O, N only
#Molecular weight: < 150 g/mol
#Heavy atoms: ≤ 10
#No metals
#No salts
#No radicals
#Neutral molecules
#One connected component


#claude generated
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

constraints = {
    "max_molecular_weight": 150,
    "max_heavy_atoms": 10,
    "allowed_elements": {"C", "H", "O", "N"},
    "neutral_only": True
}

properties = ["IUPACName", "MolecularFormula", "CanonicalSMILES", "MolecularWeight", "Charge", "HeavyAtomCount"]
molecules = ["methane", "ethanol", "acetone", "ammonia"]

data = []
valid_data = []
#get properties for each molecule
for name in molecules:
    result = pcp.get_properties(properties, name, "name")
    if result:
        data.append(result)
        if get_elements_from_smiles_regex(result[0]["ConnectivitySMILES"]).issubset(constraints["allowed_elements"]) and result[0]["Charge"] == 0 and result[0]["HeavyAtomCount"] < 10 and result[0]["MolecularWeight"] < 150:
            valid_data.append(result)
    else:
        print(f"Warning: Could not find '{name}' in PubChem.")

#flatten list so it becomes list of dictionaries
flat_list = [item for sublist in data for item in sublist]
valid_flat_list = [item for sublist in valid_data for item in sublist]

df = pd.DataFrame(flat_list)
valid_df = pd.DataFrame(valid_flat_list)