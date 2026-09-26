from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CCO"

mol = Chem.MolFromSmiles(smiles)

mol = Chem.AddHs(mol)

AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)