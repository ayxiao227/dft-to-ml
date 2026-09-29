from rdkit import Chem

def get_elements_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # makes implicit hydrogens explicit atoms
    return {atom.GetSymbol() for atom in mol.GetAtoms()}
