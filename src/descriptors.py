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

df = pd.read_csv("data/log_data/data_3.csv")

desc_df = df["SMILES"].apply(get_all_features).apply(pd.Series)
df = pd.concat([df, desc_df], axis=1)

print(df[["SMILES", "homo_ev", "lumo_ev", "gap_ev"]].head())
print(desc_df.head())

# Optional: drop rows where SMILES failed to parse
n_bad = desc_df["MolWt"].isna().sum()
print(f"{n_bad} invalid SMILES")
df = df.dropna(subset=["MolWt"])

print(desc_df.head())
print(len(df))

df.to_csv("data/descriptors/test_3_extra.csv", index=False)