"""Organic opt/freq records. Unsupported or ambiguous jobs fail closed."""
import hashlib
import json
import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from src.schema import canonical_smiles


def digest(value):
    return hashlib.sha256(value).hexdigest()


def routes(text):
    return [''.join(block.split()).lower() for block in re.findall(
        r'^\s*(#[^\n]*(?:\n(?!\s*-{5,})[^\n]*)*)\n\s*-{5,}', text, re.M)]


def protocol(options):
    """Fingerprint a common element-complete basis policy, not molecule composition."""
    import basis_set_exchange as bse
    basis = options.get('basis', 'def2-TZVP')
    if basis == 'auto' or options.get('diffuse') is not None:
        raise ValueError('Dataset requires an explicit common basis; no per-molecule auto policy')
    settings = {k: options.get(k, default) for k, default in
                [('method','b3lyp'), ('dispersion','gd3bj'), ('basis',basis)]}
    settings.update(route_policy='opt=calcfc freq 5D 7F scf=tight int=ultrafine',
                    charge=0, multiplicity=1, solvent='gas', basis_data=bse.get_basis(
                        basis, elements=[1,6,7,8,9,15,16,17]))
    return digest(json.dumps(settings, sort_keys=True).encode())


def reconstruct(text):
    """Infer a graph from the last printed orientation; retain XYZ separately."""
    blocks = list(re.finditer(r'(?:Standard|Input) orientation:', text))
    if not blocks:
        raise ValueError('No Cartesian orientation found')
    lines = text[blocks[-1].end():].splitlines()
    separators = [i for i,l in enumerate(lines) if re.fullmatch(r'\s*-{5,}\s*',l)]
    atoms = []
    for line in lines[separators[1]+1:separators[2]]:
        p = line.split()
        z = int(p[1])
        if z not in {1,6,7,8,9,15,16,17}:
            raise ValueError('Non-organic-dataset element in geometry')
        atoms.append(f'{Chem.GetPeriodicTable().GetElementSymbol(z)} {p[3]} {p[4]} {p[5]}')
    xyz = f'{len(atoms)}\noptimized geometry\n' + '\n'.join(atoms) + '\n'
    states = set(re.findall(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)',text))
    if states != {('0','1')}:
        raise ValueError('Dataset requires neutral singlets throughout the job')
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(mol, charge=0, maxIterations=10000)
    mol = Chem.RemoveHs(mol)
    from src.csv_conversion import passes_filters
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    if not passes_filters(smiles):
        raise ValueError('Reconstructed molecule fails organic dataset filters')
    return smiles, xyz


def identity_matches(original, optimized):
    # Compare graph identity ignoring unspecified stereochemistry. Explicit input
    # stereochemistry must still agree with the reconstructed conformer.
    a, b = Chem.MolFromSmiles(original), Chem.MolFromSmiles(optimized)
    if Chem.MolToSmiles(a, isomericSmiles=False) != Chem.MolToSmiles(b, isomericSmiles=False):
        return False
    return b.HasSubstructMatch(a, useChirality=True) and a.GetNumAtoms() == b.GetNumAtoms()


def inspect_record(path, row, manifest_dir):
    text = Path(path).read_text(errors='replace')
    route_list = routes(text)
    if not route_list or 'opt' not in route_list[0] or 'freq' not in route_list[0]:
        raise ValueError('Only opt/freq inputs accepted; SPE excluded')
    if len(route_list) > 2 or any(not r.startswith('#ngeom=allcheckguess=tcheck') or 'freq' not in r for r in route_list[1:]):
        raise ValueError('Unsupported linked/restarted calculation; manual review required')
    input_path = Path(row['input_path'])
    if not input_path.is_absolute():
        input_path = Path(manifest_dir) / input_path
    content = input_path.read_bytes()
    if digest(content) != row['input_sha256']:
        raise ValueError('Input snapshot hash mismatch')
    source = content.decode()
    options = json.loads(row['options'])
    if protocol(options) != row['protocol_id']:
        raise ValueError('Protocol declaration does not match settings')
    # Verify the exact supplied orbital functions against the pinned policy.
    from src.structure_automation import _gaussian_basis_data
    original_mol = Chem.AddHs(Chem.MolFromSmiles(row.get('input_smiles', row['SMILES'])))
    elements = tuple(sorted({a.GetAtomicNum() for a in original_mol.GetAtoms()}))
    basis_text, _, _ = _gaussian_basis_data(options['basis'], elements)
    if basis_text not in source:
        raise ValueError('Input basis primitives do not match declared policy')
    expected = re.search(r'^#.*$', source, re.M)
    if expected is None or ''.join(expected.group().split()).lower() != route_list[0]:
        raise ValueError('Executed route differs from input snapshot')
    marker = 'Protocol: ' + row['protocol_id']
    if marker not in source or marker not in text:
        raise ValueError('Missing protocol provenance marker; legacy logs require explicit migration')
    optimized, xyz = reconstruct(text)
    if not identity_matches(row.get('input_smiles',row['SMILES']), optimized):
        raise ValueError('Optimized identity differs from input; review required')
    versions = re.findall(r'Gaussian 16:\s*([^\n]+)', text)
    if not versions:
        raise ValueError('Gaussian version not identified')
    return {'optimized_smiles':optimized, 'SMILES':optimized, 'identity_match':True,
            'protocol_id':digest((row['protocol_id'] + route_list[0] + versions[0].strip()).encode()),
            'gaussian_version':versions[0].strip(), 'xyz':xyz,
            'log_sha256':digest(Path(path).read_bytes())}
