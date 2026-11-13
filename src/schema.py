"""Shared molecule schema and deterministic record identity."""
import hashlib
import re
import pandas as pd
from rdkit import Chem


def normalize_molecules(df):
    df = df.copy()
    aliases = {"CompoundCID": "CID", "ConnectivitySMILES": "SMILES",
               "CanonicalSMILES": "SMILES"}
    known = {"CID", "SMILES", "Name", "IUPACName", "Charge", "MolecularFormula", "MolecularWeight", "HeavyAtomCount"}
    names = []
    for column in df.columns:
        compact = str(column).replace("_", "")
        names.append(aliases.get(compact, compact if compact in known else column))
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous duplicate columns after schema normalization")
    df.columns = names
    if "SMILES" not in df:
        raise ValueError("Molecule table requires SMILES or ConnectivitySMILES")
    return df


def canonical_smiles(value):
    if not isinstance(value, str):
        raise ValueError("SMILES must be a string")
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {value!r}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def molecule_key(smiles):
    return hashlib.sha256(canonical_smiles(smiles).encode()).hexdigest()[:20]


def safe_component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe path or scheduler identifier: {value!r}")
    return value
