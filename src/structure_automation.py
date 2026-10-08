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
    params.numThreads = 0
 
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:  # retry with random starting coordinates
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        return None, None
    
    ff, results = None, None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
            ff = "MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(mol):
            results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
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
    route += f" empiricaldispersion={dispersion}"
 
    lines = [
        f"%mem={mem}",
        f"%nprocshared={nproc}",
        f"%chk={chk_name}",
        route,
        "",
        f"ID {ID}: {smiles} (start geometry: {ff})",
        f"Basis: {selected_basis}; BSE version {basis_version}; task: {basis_task}",
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


def create_txt_file(
    molecule_name,
    output_dir, ID, son, user, dft_folder,
    account="nszym1",
    scratch_dir="/scratch/nszym_root/nszym1/",
    memory="10g",
    time="02-00:00",
    cpus=8,
):
    """
    Create a Slurm script (.txt) for running a Gaussian calculation
    on the Great Lakes HPC.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule / Gaussian input file, without .gjf
        Example: "AX_2_2_ethoxyethanol"

    output_dir : str or Path
        Local directory where the .txt file should be created.

    account : str
        Slurm account to charge.

    scratch_dir : str
        Scratch directory on Great Lakes.

    memory : str
        Memory requested from Slurm.

    time : str
        Maximum job runtime.

    cpus : int
        Number of CPUs requested.
    """

    # Convert output directory to a Path object
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the Slurm script
    if not son:
        script = f"""#!/bin/bash
#SBATCH --job-name={molecule_name}
#SBATCH --mail-user={user}@umich.edu
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --account={account}

FLDR_NAME="{molecule_name}"

SCRATCH_FLDR="{scratch_dir}{user}"

# Making temporary directories
mkdir -p $SCRATCH_FLDR/${{SLURM_JOB_ID}}
cd $SCRATCH_FLDR/${{SLURM_JOB_ID}}

# Copying input file to temporary directory and running Gaussian
cp /home/{user}/{dft_folder}/{ID}/$FLDR_NAME/${{SLURM_JOB_NAME}}.gjf .

module purge
module load Chemistry
module load gaussian/16-revC01-avx2

g16 ${{SLURM_JOB_NAME}}.gjf
formchk ${{SLURM_JOB_NAME}}.chk

# Copying files back to home
mkdir /home/{user}/{dft_folder}/{ID}/$FLDR_NAME/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}
cd /home/{user}/{dft_folder}/{ID}/$FLDR_NAME/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}

cp $SCRATCH_FLDR/${{SLURM_JOB_ID}}/*.log .
cp $SCRATCH_FLDR/${{SLURM_JOB_ID}}/${{SLURM_JOB_NAME}}.fchk .

rm -r $SCRATCH_FLDR/${{SLURM_JOB_ID}}
"""
    else:
        script = f"""#!/bin/bash
#SBATCH --job-name={molecule_name}
#SBATCH --mail-user=xaustin@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --account={account}

FLDR_NAME="{molecule_name}"

SCRATCH_FLDR="{scratch_dir}{user}"

# Making temporary directories
mkdir -p $SCRATCH_FLDR/${{SLURM_JOB_ID}}
cd $SCRATCH_FLDR/${{SLURM_JOB_ID}}

# Copying input file to temporary directory and running Gaussian
cp /home/xaustin/dft_ml/gjf_files/{ID}/${{SLURM_JOB_NAME}}.gjf .

module purge
module load Chemistry
module load gaussian/16-revC01-avx2

g16 ${{SLURM_JOB_NAME}}.gjf
formchk ${{SLURM_JOB_NAME}}.chk

# Copying files back to home
cd /home/xaustin/dft_ml/log_files/{ID}/

cp $SCRATCH_FLDR/${{SLURM_JOB_ID}}/*.log .
cp $SCRATCH_FLDR/${{SLURM_JOB_ID}}/${{SLURM_JOB_NAME}}.fchk .

rm -r $SCRATCH_FLDR/${{SLURM_JOB_ID}}
"""

    # Name of the output .txt file
    output_file = output_dir / f"{molecule_name}.txt"

    # Write the script to disk
    output_file.write_text(script)

    return output_file


def generate_slurm_scripts(df, correction, ID, son, user, dft_folder, gjf_options=None):
    """
    Generate Slurm scripts for a DataFrame of molecules. \n
    gjf_options forwards basis, diffuse, basis_task and multiplicity to the writer.
    df = dataframe of molecules \n
    correction is the geometric correction \n
    ID is the experiment ID \n
    son = default for AX \n
    dft_folder not necessary for son \n
    """
    # df = pd.read_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/pubchem_molecules_valid.csv")
    # output_dir = "molecules/"
    if ID is None:
        ID = strftime("%H%M_%d%m%Y")
    gjf_options = {} if gjf_options is None else dict(gjf_options)
    gjf_dir = Path("molecules/gjf_files/" + ID)
    txt_dir = Path("scripts/txt_files/" + ID)
    iteration = 0
    failed = []
    for _, row in df.iterrows():
        iteration += 1
        safe_molecule_name = safe_name(row["CID"], row["Name"], ID)
        
        test = smiles_to_gjf(
            row["ConnectivitySMILES"],
            safe_molecule_name,
            row["Charge"], correction, ID, gjf_dir, **gjf_options
            )
        if test:
            create_txt_file(safe_molecule_name, txt_dir, ID, son, user, dft_folder)
        else:
            failed.append(safe_molecule_name)
        if iteration % 500 == 0:
            print(iteration)


def generate_slurm_scripts_from_csv(df, correction, ID, son, randomize, gjf_options=None):
    gjf_options = {} if gjf_options is None else dict(gjf_options)
    iteration = 0
    if ID is None:
        ID = strftime("%H%M_%d%m%Y")
    gjf_dir = Path("molecules/gjf_files/" + ID)
    txt_dir = Path("scripts/txt_files/" + ID)
    if randomize is not None:
        while iteration < randomize:
            index = random.randint(0,86400)
            iteration += 1
            row = df.iloc[index]
            safe_molecule_name = safe_name(row["CID"], row["Name"], ID)
            
            test = smiles_to_gjf(
                row["ConnectivitySMILES"],
                safe_molecule_name,
                row["Charge"], correction, ID, gjf_dir, **gjf_options
                )
            if test:
                create_txt_file(safe_molecule_name, txt_dir, ID, son)
            if iteration % 100 == 0:
                print(iteration)
