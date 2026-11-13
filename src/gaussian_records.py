"""Organic opt/freq records. Unsupported or ambiguous jobs fail closed."""
import hashlib
import json
import re
import math
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from src.schema import canonical_smiles


def digest(value):
    return hashlib.sha256(value).hexdigest()


def routes(text):
    return [''.join(block.split()).lower() for block in re.findall(
        r'^\s*(#[^\n]*(?:\n(?!\s*-{5,})[^\n]*)*)\n\s*-{5,}', text, re.M)]


def validate_options(options):
    """Fail early: this policy is neutral-singlet gas-phase opt/freq only."""
    allowed = {'basis', 'method', 'dispersion', 'diffuse', 'basis_task',
               'multiplicity', 'seed', 'nproc', 'mem', 'protocol_id'}
    if set(options) - allowed:
        raise ValueError('Unsupported scientific/generation options')
    multiplicity=options.get('multiplicity',1)
    if (multiplicity is not None and type(multiplicity) is not int) or multiplicity not in (None,1):
        raise ValueError('Organic dataset requires neutral singlets')
    if options.get('basis_task', 'general') not in ('general', 'geometry'):
        raise ValueError('Only opt/freq geometry tasks are supported')
    for k, default in [('method', 'b3lyp'), ('dispersion', 'gd3bj')]:
        if not isinstance(options.get(k, default), str) or not re.fullmatch(r'[A-Za-z0-9-]+', options.get(k, default)):
            raise ValueError('Invalid scientific setting')
    if type(options.get('seed', 42)) is not int or not 0 <= options.get('seed', 42) < 2**31:
        raise ValueError('Invalid conformer seed')
    if type(options.get('nproc', 12)) is not int or options.get('nproc', 12) < 1:
        raise ValueError('Invalid CPU count')
    if not isinstance(options.get('mem', '8GB'), str) or not re.fullmatch(r'[1-9][0-9]*GB', options.get('mem', '8GB')):
        raise ValueError('Memory must be positive integer GB')


def protocol(options):
    """Fingerprint a common element-complete basis policy, not molecule composition."""
    import basis_set_exchange as bse
    validate_options(options)
    basis = options.get('basis', 'def2-TZVP')
    if not isinstance(basis,str) or not basis.strip():raise ValueError('Explicit common basis name required')
    if basis.lower() == 'auto' or options.get('diffuse') is not None:
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
    if len(separators)<3:raise ValueError('Truncated final orientation')
    atoms = []
    for line in lines[separators[1]+1:separators[2]]:
        p = line.split()
        if len(p)!=6 or not all(math.isfinite(float(v.replace('D','e'))) for v in p[3:]):raise ValueError('Malformed/nonfinite Cartesian geometry')
        if int(p[0]) != len(atoms)+1:raise ValueError('Invalid geometry atom indices')
        z = int(p[1])
        if z not in {1,6,7,8,9,15,16,17}:
            raise ValueError('Non-organic-dataset element in geometry')
        atoms.append(f'{Chem.GetPeriodicTable().GetElementSymbol(z)} {p[3]} {p[4]} {p[5]}')
    if not atoms:raise ValueError('Empty final geometry')
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
    if len(route_list) > 2:
        raise ValueError('Unsupported linked/restarted calculation; manual review required')
    input_path = Path(row['input_path'])
    if not input_path.is_absolute():
        input_path = Path(manifest_dir) / input_path
    content = input_path.read_bytes()
    if digest(content) != row['input_sha256']:
        raise ValueError('Input snapshot hash mismatch')
    source = content.decode()
    input_states=re.findall(r'^\s*(-?\d+)\s+(\d+)\s*$',source,re.M)
    if not input_states or input_states[0]!=('0','1'):raise ValueError('Input snapshot is not a neutral singlet')
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
    expected_source = f"# opt=calcfc freq {options.get('method','b3lyp')}/gen 5D 7F" + (' geom=connectivity' if 'geom=connectivity' in source else '') + f" empiricaldispersion={options.get('dispersion','gd3bj')} scf=tight int=ultrafine"
    expected = re.search(r'^#.*$', source, re.M)
    if expected is None or ''.join(expected.group().split()).lower() != ''.join(expected_source.split()).lower():
        raise ValueError('Input route violates declared opt/freq policy')
    if expected is None or ''.join(expected.group().split()).lower() != route_list[0]:
        raise ValueError('Executed route differs from input snapshot')
    continuation = '#ngeom=allcheckguess=tcheck' + route_list[0].lstrip('#').replace('opt=calcfc','').replace('geom=connectivity','')
    if route_list[1:] and route_list[1:] != [continuation]:
        raise ValueError('Unsupported or scientifically changed opt/freq continuation')
    if '--link1--' in source.lower():raise ValueError('Linked input unsupported')
    marker = 'Protocol: ' + row['protocol_id']
    if marker not in source or marker not in text:
        raise ValueError('Missing protocol provenance marker; legacy logs require explicit migration')
    optimized, xyz = reconstruct(text)
    if not identity_matches(row.get('input_smiles',row['SMILES']), optimized):
        raise ValueError('Optimized identity differs from input; review required')
    versions = re.findall(r'Gaussian 16:\s*([^\n]+)', text)
    if not versions or len(set(v.strip() for v in versions))!=1:
        raise ValueError('Gaussian version missing or inconsistent')
    return {'optimized_smiles':optimized, 'SMILES':optimized, 'identity_match':True,
            'input_protocol_id': row['protocol_id'],
            'observed_protocol_id': digest((row['protocol_id'] + route_list[0] + versions[0].strip()).encode()),
            'protocol_id':digest((row['protocol_id'] + route_list[0] + versions[0].strip()).encode()),
            'gaussian_version':versions[0].strip(), 'xyz':xyz,
            'log_sha256':digest(Path(path).read_bytes())}
