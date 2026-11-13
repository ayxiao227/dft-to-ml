"""Gaussian input generation with molecule-aware, paper-informed basis sets.

Reference: Bursch et al., Angew. Chem. Int. Ed. 2022, 61, e202205735,
https://doi.org/10.1002/anie.202205735, sections 2.6 and 3.2-3.4.
need to install basis-set-exchange
bse stores basis data locally, don't need wifi

Basis set selection informed by paper recommendations:
Main-group molecules: def2-TZVP.
Transition-metal complexes: def2-TZVPP (extra polarization).
Anions / any formally negative site: add diffuse functions (TZVPD/TZVPPD).
Explicit dipole/polarizability/noncovalent tasks also use diffuse functions.
Reaction energies/barriers: def2-TZVPP or TZVPPD; check convergence to QZ.
ECPs are written from the selected basis data, never guessed by element.

The paper recommends TZ quality, augmentation when needed and matching ECPs;
it does not prescribe the transition-metal selection rule above. No automatic
switch to smaller bases for large molecules is made. Geometry and frequency
calculations use the same basis. Functional and dispersion settings are retained.
For consistent ML labels, use an explicit common basis policy for comparable
molecules and record the selected basis; do not silently mix old/new labels.
"""
from time import strftime
from pathlib import Path
from functools import lru_cache
from numbers import Integral
import random
import logging
import json
import pandas as pd
from src.schema import normalize_molecules, molecule_key, safe_component
import re
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

log = logging.getLogger(__name__)

BOND_ORDER = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}


def _has_transition_metal(mol):
    return any(21 <= a.GetAtomicNum() <= 30 or
               39 <= a.GetAtomicNum() <= 48 or
               72 <= a.GetAtomicNum() <= 80 for a in mol.GetAtoms())


def select_reference_basis(mol, charge, basis="auto", diffuse=None,
                           basis_task="general"):
    """Select a def2 basis from charge, elements and the requested task.

    basis overrides auto selection. diffuse=True/False overrides augmentation
    only for auto mode. Supported tasks: general, geometry, energy,
    reaction_energy, barrier, noncovalent, dipole, polarizability.
    Molecule size alone does not justify decreasing basis quality.
    """
    tasks = {"general", "geometry", "energy", "reaction_energy", "barrier",
             "noncovalent", "dipole", "polarizability"}
    if basis_task not in tasks:
        raise ValueError(f"Unknown basis_task: {basis_task!r}; choose {sorted(tasks)}")
    if diffuse is not None and not isinstance(diffuse, bool):
        raise ValueError("diffuse must be None, True or False")
    if basis is not None and not isinstance(basis, str):
        raise ValueError("basis must be a basis-set name or 'auto'")
    if isinstance(basis, str) and basis.lower() != "auto":
        if not basis.strip():
            raise ValueError("basis must not be empty")
        return basis.strip()
    zs = {a.GetAtomicNum() for a in mol.GetAtoms()}
    if any(z < 1 or z > 86 or 57 <= z <= 71 for z in zs):
        raise ValueError("Automatic def2 policy does not cover dummy atoms, "
                         "lanthanides or elements beyond Rn; provide a "
                         "validated explicit basis and spin state")
    augmented = (charge < 0 or any(a.GetFormalCharge() < 0 for a in mol.GetAtoms())
                 or basis_task in {"noncovalent", "dipole", "polarizability"})
    if diffuse is not None:
        augmented = diffuse
    extra_polarization = (_has_transition_metal(mol) or
                          basis_task in {"reaction_energy", "barrier", "noncovalent"})
    name = "def2-TZVPP" if extra_polarization else "def2-TZVP"
    return name + ("D" if augmented else "")


@lru_cache(maxsize=256)
def _gaussian_basis_data(basis, atomic_numbers):
    """Return (Gaussian basis/ECP text, has_ecp, BSE data version).

    Use BSE's actual primitives and matching ECPs for every element, so diffuse
    sets do not depend on Gaussian recognizing an undocumented basis keyword.
    Cache by basis and sorted elements for batch generation.
    """
    try:
        import basis_set_exchange as bse
    except ImportError as exc:
        raise ImportError("Install basis data with: python -m pip install "
                          "basis-set-exchange") from exc
    data = bse.get_basis(basis, elements=list(atomic_numbers))
    if any(not data["elements"].get(str(z), {}).get("electron_shells")
           for z in atomic_numbers):
        raise ValueError(f"{basis} does not supply orbital functions for every element")
    has_ecp = any(e.get("ecp_potentials") for e in data["elements"].values())
    text = bse.write_formatted_basis_str(data, "gaussian94")
    return text.rstrip(), has_ecp, data.get("version", "unknown")


def _build_best_conformer(mol, seed=42):
    """Embed several conformers, optimize with MMFF94 (UFF fallback),
    and return (conformer_id, force_field_name) for the lowest-energy one."""
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_confs = 1 if n_rot == 0 else (10 if n_rot <= 3 else 30)
 
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
 
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:  # retry with random starting coordinates
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        return None, None
    
    ff, results = None, None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=2000)
            ff = "MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(mol):
            results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=2000)
            ff = "UFF"
    except RuntimeError as e:
        log.warning("force field optimization failed (%s); using embedded geometry", e)
        ff, results = None, None

    if results is None:
        # No usable force field: keep the ETKDG geometry, take the first conformer
        return cids[0], "ETKDG only (no force field)"

    scored = list(zip(cids, results))
    converged = [(cid, e) for cid, (flag, e) in scored if flag == 0]
    pool = converged if converged else [(cid, e) for cid, (flag, e) in scored]
    best_cid = min(pool, key=lambda x: x[1])[0]
    return best_cid, ff


def safe_name(cid, name, ID, max_len=40):
    #cid, name, ID
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # commas, brackets, spaces, slashes, parentheses -> _
    s = s.strip("_")[:max_len].rstrip("_")
    if ID is not None:
        return f"{ID}_{s}"
    else:
        return f"{cid}_{s}"


def smiles_to_gjf(
    smiles, safename,
    charge, geo_correction, ID, output_dir,
    mem="8GB",
    nproc=12,
    method="b3lyp",
    basis="auto",
    dispersion="gd3bj",
    seed=42,
    overwrite=False,
    multiplicity=None,
    diffuse=None,
    basis_task="general",
    protocol_id=None,
):
    """Write a Gaussian opt/freq input; return True on success.

    basis='auto' selects the reference policy; explicit BSE names override it.
    diffuse and basis_task control auto mode (see select_reference_basis).
    Transition-metal complexes require an explicit multiplicity: SMILES radical
    counts do not establish metal spin states. Electron parity is checked but
    cannot prove that the chosen spin state is physically appropriate.
    Existing positional arguments and overwrite behavior are preserved.
    """
    # 1. SMILES -> RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.error("%s: invalid SMILES: %s", safename, smiles)
        return False
    mol = Chem.AddHs(mol)
 
    # 2. Element, charge and multiplicity checks
    formal_charge = Chem.GetFormalCharge(mol)
    if charge is None:
        charge = formal_charge
    elif charge != formal_charge:
        log.error("%s: charge=%s passed in, but SMILES formal charge is %s",
                  safename, charge, formal_charge)
        return False
 
    if not isinstance(charge, Integral):
        # pandas often supplies an integral charge as a float.
        try:
            if not float(charge).is_integer():
                raise ValueError("non-integral charge")
            charge = int(charge)
        except (ValueError, TypeError, OverflowError):
            log.error("%s: charge must be an integer", safename)
            return False
    if multiplicity is None:
        if _has_transition_metal(mol):
            log.error("%s: specify multiplicity for a transition-metal complex", safename)
            return False
        multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1
    if not isinstance(multiplicity, Integral) or multiplicity < 1:
        log.error("%s: multiplicity must be a positive integer", safename)
        return False
    n_electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if n_electrons < 1 or multiplicity > n_electrons + 1 or (n_electrons - (multiplicity - 1)) % 2 != 0:
        log.error("%s: charge %s and multiplicity %s are inconsistent (%s electrons)",
                  safename, charge, multiplicity, n_electrons)
        return False
 
    try:
        selected_basis = select_reference_basis(mol, charge, basis, diffuse, basis_task)
        atomic_numbers = tuple(sorted({a.GetAtomicNum() for a in mol.GetAtoms()}))
        basis_text, has_ecp, basis_version = _gaussian_basis_data(
            selected_basis, atomic_numbers)
    except ImportError:
        raise  # missing dependency needs action, rather than a silent batch skip
    except (KeyError, ValueError, RuntimeError) as exc:
        log.error("%s: cannot prepare basis/ECP data: %s", safename, exc)
        return False
    log.info("%s: basis=%s, BSE version=%s, ECP=%s", safename,
             selected_basis, basis_version, has_ecp)
 
    # 3. 3D geometry
    best_cid, ff = _build_best_conformer(mol, seed=seed)
    if best_cid is None:
        log.error("%s: could not generate 3D geometry for %s", safename, smiles)
        return False
    conf = mol.GetConformer(best_cid)
 
    # 4. File paths (sanitize the name so it is safe for files and SLURM)
    safename = re.sub(r"[^A-Za-z0-9_.-]", "_", str(safename))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{safename}.gjf"
    if filename.exists() and not overwrite:
        log.warning("%s: %s already exists, skipping (pass overwrite=True to replace)",
                    safename, filename)
        return False
    chk_name = filename.with_suffix(".chk").name
 
    # 5. Build the Gaussian input
    route = f"# opt=calcfc freq {method}/{'genecp' if has_ecp else 'gen'} 5D 7F"
    if geo_correction:
        route += " geom=connectivity"
    route += f" empiricaldispersion={dispersion} scf=tight int=ultrafine"
 
    lines = [
        f"%mem={mem}",
        f"%nprocshared={nproc}",
        f"%chk={chk_name}",
        route,
        "",
        f"ID {ID}: {smiles} (start geometry: {ff})",
        f"Basis: {selected_basis}; BSE version {basis_version}; task: {basis_task}",
        f"Protocol: {protocol_id}",
        "",
        f"{charge} {multiplicity}",
    ]
 
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2}{p.x:16.8f}{p.y:16.8f}{p.z:16.8f}")
 
    if geo_correction:
        lines.append("")
        for atom in mol.GetAtoms():
            i = atom.GetIdx() + 1
            conns = []
            for bond in atom.GetBonds():
                j = bond.GetOtherAtom(atom).GetIdx() + 1
                if j > i:  # list each bond once
                    conns.append(f"{j} {BOND_ORDER.get(bond.GetBondType(), 1.0):.1f}")
            lines.append(f"{i} " + " ".join(conns) if conns else f"{i}")
 
    lines += ["", basis_text, "", ""]
 
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return True


from src.scheduler import create_txt_file


def generate_slurm_scripts(df, correction=False, ID=None, son=False,
                           user="xaustin", dft_folder="dft_ml", gjf_options=None, resume=False):
    """Generate inputs, matching scheduler resources, and an identity manifest.

    All modes use gjf_files/<ID> and log_files/<ID> on the remote host.
    The legacy son argument is accepted only for API compatibility.
    """
    ID = safe_component(ID or strftime("%H%M%S_%d%m%Y"))
    df = normalize_molecules(df)
    options = dict(gjf_options or {})
    from src.gaussian_records import protocol, digest
    from src.csv_conversion import passes_filters
    options.setdefault("basis", "def2-TZVP")
    options["protocol_id"] = protocol(options)
    cpus = options.setdefault("nproc", 12)
    mem = options.setdefault("mem", "8GB")
    if not isinstance(cpus, int) or cpus < 1 or not re.fullmatch(r"[1-9][0-9]*GB", mem):
        raise ValueError("Use positive integer nproc and memory in integer GB")
    gjf_dir = Path("molecules/gjf_files") / ID
    txt_dir = Path("scripts/txt_files") / ID
    old = None
    if (gjf_dir / "manifest.csv").exists():
        if not resume:
            raise FileExistsError("Experiment already exists; use resume with unchanged inputs")
        old = pd.read_csv(gjf_dir / "manifest.csv").set_index('molecule_id')
        if not old['options'].eq(json.dumps(options, sort_keys=True)).all():
            raise ValueError('Resume configuration changed')
        if set(old.index) - set(df.SMILES.map(molecule_key)):
            raise ValueError('Resume cannot remove previously requested molecules')
    records, seen = [], set()
    for _, row in df.iterrows():
        if not passes_filters(row['SMILES']):
            raise ValueError('Only validated organic molecules accepted')
        key = molecule_key(row["SMILES"])
        if key in seen:
            raise ValueError(f"Duplicate molecular identity: {key}")
        seen.add(key)
        name = f"{ID}_{key}"
        charge = row.get("Charge")
        if pd.isna(charge):
            charge = None
        if old is not None and key in old.index and bool(old.loc[key, 'generated']):
            previous = old.loc[key].to_dict()
            if previous['options'] != json.dumps(options, sort_keys=True):
                raise ValueError('Resume configuration changed')
            if digest((gjf_dir / previous['input_path']).read_bytes()) != previous['input_sha256']:
                raise ValueError('Existing input was modified')
            if not (txt_dir / (name + '.txt')).exists():
                create_txt_file(name, txt_dir, ID, son, user, dft_folder, cpus=cpus, memory=f'{int(mem[:-2]) + 2}G')
            records.append({**previous, 'molecule_id':key})
            continue
        ok = smiles_to_gjf(row["SMILES"], name, charge, correction, ID, gjf_dir, **options)
        records.append({**row.to_dict(), "file": name + ".log", "molecule_key": key,
                        "generated": ok, "options": json.dumps(options, sort_keys=True),
                        "molecule_id":key, "run_id":ID + '_' + key, "input_smiles":row['SMILES'],
                        "input_path":name + '.gjf', "protocol_id":options['protocol_id'],
                        "input_sha256":digest((gjf_dir / (name + '.gjf')).read_bytes()) if ok else ''})
        # Save progress so failed records remain visible rather than disappearing.
        gjf_dir.mkdir(parents=True, exist_ok=True)
        temporary = gjf_dir / 'manifest.csv.tmp'
        pd.DataFrame(records).to_csv(temporary, index=False)
        temporary.replace(gjf_dir / 'manifest.csv')
        if ok:
            create_txt_file(name, txt_dir, ID, son, user, dft_folder,
                            cpus=cpus, memory=f"{int(mem[:-2]) + 2}G")
    return pd.DataFrame(records)


def generate_slurm_scripts_from_csv(df, correction=False, ID=None, son=False,
                                    randomize=None, gjf_options=None,
                                    user="xaustin", dft_folder="dft_ml", seed=42, resume=False):
    df = normalize_molecules(df)
    if randomize not in (None, ""):
        count = int(randomize)
        if not 0 < count <= len(df):
            raise ValueError(f"Sample count must be between 1 and {len(df)}")
        df = df.sample(n=count, replace=False, random_state=seed)
    return generate_slurm_scripts(df, correction, ID, son, user, dft_folder, gjf_options, resume=resume)
