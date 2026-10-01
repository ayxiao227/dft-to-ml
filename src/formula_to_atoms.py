from rdkit import Chem

def get_elements_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    mol = Chem.AddHs(mol)  # makes implicit hydrogens explicit atoms
    return {atom.GetSymbol() for atom in mol.GetAtoms()}
