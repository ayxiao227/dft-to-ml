"""Freeze development/final partitions before feature or model selection."""
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.run_metadata import check_molecular_identity


def chemical_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms():
        return 'ring:' + Chem.MolToSmiles(scaffold)
    # Empty Murcko scaffolds are not a useful common group. Group acyclic
    # molecules by generic whole-molecule topology instead (documented heuristic).
    return 'acyclic:' + Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol))


def freeze(input_path, output_dir, grouped=True, seed=42):
    df = pd.read_csv(input_path)
    check_molecular_identity(df)
    if len(df) < 10:
        raise ValueError('At least 10 molecules required')
    groups = df.SMILES.map(chemical_group)
    if grouped:
        if groups.nunique() < 2:
            raise ValueError('Not enough distinct chemical groups')
        development, final = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,groups=groups))
    else:
        development, final = train_test_split(range(len(df)),test_size=.2,random_state=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    df.iloc[development].to_csv(output/'development.csv',index=False)
    df.iloc[final].to_csv(output/'final_test.csv',index=False)
    mapping = pd.DataFrame({'SMILES':df.SMILES,'group':groups,'partition':'development'})
    mapping.loc[final,'partition']='final'
    mapping.to_csv(output/'partitions.csv',index=False)
    from src.gaussian_records import digest
    (output/'source.sha256').write_text(digest(Path(input_path).read_bytes()))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('output');p.add_argument('--random',action='store_true')
    a=p.parse_args();freeze(a.input,a.output,grouped=not a.random)
