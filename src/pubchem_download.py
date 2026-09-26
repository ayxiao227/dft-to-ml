import pubchempy as pcp
import pandas as pd

#Constraints:
#Elements: C, H, O, N only
#Molecular weight: < 150 g/mol
#Heavy atoms: ≤ 10
#No metals
#No salts
#No radicals
#Neutral molecules
#One connected component

constraints = {
    "max_molecular_weight": 150,
    "max_heavy_atoms": 10,
    "allowed_elements": {"C", "H", "O", "N"},
    "neutral_only": True
}

# 1. Fetch only CanonicalSMILES (CID is always included by default)
properties = pcp.get_properties('CanonicalSMILES', ['aspirin', 'ibuprofen', 'caffeine'], 'name')

# 2. Convert to a DataFrame and save
df = pd.DataFrame(properties)
df.set_index('CID', inplace=True) # Makes CID the first column
df.to_csv('smiles_only.csv')