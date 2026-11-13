import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator

from rdkit import Chem

def has_isolated_h(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms())


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}  # becomes NaN row after apply(pd.Series)

    return {
        "MolWt": Descriptors.MolWt(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumC": sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()),
        "NumN": sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()),
        "NumO": sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RotatableBondCount": Lipinski.NumRotatableBonds(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "DoubleBondCount": sum(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds()),
        "TripleBondCount": sum(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds()),
    }

_PATTERNS = {
    "n_carbonyl": "[CX3]=[OX1]",
    "n_conj_carbonyl": "[OX1]=[CX3]-[#6,#7]=,#[#6,#7,#8]",  # C=O adjacent to another pi bond
    "n_conj_diene": "[#6]=[#6]-[#6]=[#6]",
    "n_ene_yne": "[#6]=[#6]-[#6]#[#6]",
    "n_nitroso_noxide": "[#7]=[#8,#7]",
}
_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _PATTERNS.items()}
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def extra_features(smi, include_fp=True):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {}
    n_bonds = max(m.GetNumBonds(), 1)
    n_conj = sum(b.GetIsConjugated() for b in m.GetBonds())
    feats = {"n_conj_bonds": n_conj, "frac_conj_bonds": n_conj / n_bonds}
    for name, patt in _PATTERNS.items():
        feats[name] = len(m.GetSubstructMatches(patt))
    if include_fp:
        fp = _gen.GetFingerprintAsNumPy(m)
        feats.update({f"morgan_{i}": int(v) for i, v in enumerate(fp)})
    return feats

def get_all_features(smi):
    desc = calculate_descriptors(smi)
    extra = extra_features(smi, include_fp=True)
    return {**desc, **extra}

def build_descriptors(input_path, output_path):
    from pathlib import Path
    from src.schema import normalize_molecules
    df = normalize_molecules(pd.read_csv(input_path))
    if 'optimized_smiles' not in df or 'protocol_id' not in df or df.protocol_id.nunique() != 1 or df.protocol_id.isna().any():
        raise ValueError('Descriptors require accepted optimized identities and one verified protocol')
    if 'accepted' not in df or not df.accepted.eq(True).all():
        raise ValueError('Unaccepted records cannot become descriptors')
    df['SMILES'] = df['optimized_smiles']
    features = df["SMILES"].apply(get_all_features).apply(pd.Series)
    if features.empty or "MolWt" not in features or features["MolWt"].isna().any():
        raise ValueError("Invalid or empty molecular dataset")
    overlap = set(df.columns) & set(features.columns)
    df = pd.concat([df.drop(columns=list(overlap)), features], axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build graph-based molecular descriptors")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    build_descriptors(args.input, args.output)
