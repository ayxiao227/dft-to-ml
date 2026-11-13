from time import strftime

from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
# from src.formula_to_atoms import get_elements_from_smiles as fta
from src.formula_to_atoms import get_elements_from_smiles as fta
import random
import logging

import re
log = logging.getLogger(__name__)
 
MAX_ATOMIC_NUMBER = 30  # 6-31G** is defined through Zn
 
BOND_ORDER = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}
 
REFERENCE_BASIS = {
    "Ru": "LANL2DZ",
    "P": "6-31+g**",
    "C": "6-31g**",
    "H": "6-31g**",
    "N": "6-31g**",
    "O" : "6-31g**",
}


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

# def smiles_to_gjf(
#     smiles, safename,
#     charge, geo_correction, ID, output_dir,
#     mem="8GB",
#     nproc=12,
#     method="b3lyp",
#     basis="6-31g**",
#     dispersion="gd3bj",
# ):
#     # -------------------------
#     # 1. SMILES → RDKit molecule
#     # -------------------------
#     mol = Chem.MolFromSmiles(smiles)

#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")

#     # Add hydrogens
#     mol = Chem.AddHs(mol)
#     # Calculate multiplicity (number of unpaired electrons + 1)
#     multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1
#     # -------------------------
#     # 2. Generate 3D geometry
#     # -------------------------
#     status = AllChem.EmbedMolecule(
#         mol,
#         randomSeed=42
#     )

#     if status != 0:
#         # raise ValueError(
#         #     f"Could not generate 3D geometry for {smiles}"
#         # )
#         return False

#     # Pre-optimize geometry
#     AllChem.MMFFOptimizeMolecule(mol)

#     # -------------------------
#     # 3. File information
#     # -------------------------
#     output_dir.mkdir(parents=True, exist_ok=True)
#     filename = output_dir / f"{safename}.gjf"

#     chk_name = filename.with_suffix(".chk").name
#     # -------------------------
#     # 4. Write Gaussian file
#     # -------------------------
#     with open(filename, "w") as f:

#         # Resource specifications
#         f.write(f"%mem={mem}\n")
#         f.write(f"%nprocshared={nproc}\n")
#         f.write(f"%chk={chk_name}\n")

#         # Route section
#         f.write(
#             f"# opt=calcfc freq "
#             f"{method}/gen ")
#         if geo_correction:
#             f.write(f"geom=connectivity ")
#         f.write(f"empiricaldispersion={dispersion}\n")

#         # Title
#         f.write("\n")
#         f.write(f"Generated from SMILES: {smiles}\n")

#         # Charge / multiplicity
#         f.write("\n")
#         f.write(f"{charge} {multiplicity}\n")

#         # -------------------------
#         # 5. Cartesian coordinates
#         # -------------------------
#         conf = mol.GetConformer()

#         for atom in mol.GetAtoms():

#             pos = conf.GetAtomPosition(atom.GetIdx())

#             f.write(
#                 f"{atom.GetSymbol():<2}"
#                 f"{pos.x:16.8f}"
#                 f"{pos.y:16.8f}"
#                 f"{pos.z:16.8f}\n"
#             )

#         # -------------------------
#         # 6. Connectivity
#         # -------------------------
#         if geo_correction:
#             f.write("\n")
#             for atom in mol.GetAtoms():

#                 atom_idx = atom.GetIdx() + 1

#                 connections = []

#                 for bond in atom.GetBonds():

#                     other_atom = bond.GetOtherAtom(atom)
#                     other_idx = other_atom.GetIdx() + 1

#                     # Only list atoms with higher index
#                     # to avoid duplicating bonds
#                     if other_idx > atom_idx:

#                         bond_type = bond.GetBondType()

#                         if bond_type == Chem.BondType.SINGLE:
#                             order = 1.0
#                         elif bond_type == Chem.BondType.DOUBLE:
#                             order = 2.0
#                         elif bond_type == Chem.BondType.TRIPLE:
#                             order = 3.0
#                         elif bond_type == Chem.BondType.AROMATIC:
#                             order = 1.5
#                         else:
#                             order = 1.0

#                         connections.append(
#                             f"{other_idx} {order:.1f}"
#                         )

#                 if connections:
#                     f.write(
#                         f"{atom_idx} "
#                         + " ".join(connections)
#                         + "\n"
#                     )
#                 else:
#                     f.write(f"{atom_idx}\n")

#         # -------------------------
#         # 7. Basis set
#         # -------------------------
#         f.write("\n")
#         atoms = fta(smiles)
#         for a in atoms:
#             f.write(f"{a} ")
#         f.write("0\n")
#         #f.write("C H O N " + "0\n") some molecules do not have all these atoms, so including them all will cause p
#         f.write(f"{basis}\n")
#         f.write("****\n")
#         f.write("\n")
#     return True

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

log = logging.getLogger(__name__)

BOND_ORDER = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}


def _build_best_conformer(mol, seed=42):
    """Embed several conformers, optimize with MMFF94 (UFF fallback), return the lowest-energy one."""
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_confs = 1 if n_rot == 0 else (10 if n_rot <= 3 else 30)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 0

    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:                                  # retry with random starting coordinates
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        return None, None

    if AllChem.MMFFHasAllMoleculeParams(mol):
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
        ff = "MMFF94"
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
        ff = "UFF"

    # results[i] = (converged_flag, energy); flag 0 means converged
    scored = list(zip(cids, results))
    converged = [(cid, e) for cid, (flag, e) in scored if flag == 0]
    pool = converged if converged else [(cid, e) for cid, (flag, e) in scored]
    best_cid = min(pool, key=lambda x: x[1])[0]
    return best_cid, ff


# def smiles_to_gjf(
#     smiles, safename,
#     charge, geo_correction, ID, output_dir,
#     mem="8GB",
#     nproc=12,
#     method="b3lyp",
#     basis="6-31g**",
#     dispersion="gd3bj",
#     seed=42,
# ):
#     # -------------------------
#     # 1. SMILES -> RDKit molecule
#     # -------------------------
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")
#     mol = Chem.AddHs(mol)

#     # -------------------------
#     # 2. Charge / multiplicity checks
#     # -------------------------
#     formal_charge = Chem.GetFormalCharge(mol)
#     if charge is None:
#         charge = formal_charge
#     elif charge != formal_charge:
#         log.warning("%s: charge=%s passed in, but SMILES formal charge is %s", safename, charge, formal_charge)

#     multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1
#     n_electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
#     if (n_electrons - (multiplicity - 1)) % 2 != 0:
#         log.error("%s: charge %s and multiplicity %s are inconsistent (%s electrons)",
#                   safename, charge, multiplicity, n_electrons)
#         return False

#     symbols = sorted({a.GetSymbol() for a in mol.GetAtoms()})   # replaces fta(smiles)
#     if any(a.GetAtomicNum() > 18 for a in mol.GetAtoms()):
#         log.warning("%s: contains elements beyond Ar; check that %s is appropriate", safename, basis)

#     # -------------------------
#     # 3. 3D geometry
#     # -------------------------
#     best_cid, ff = _build_best_conformer(mol, seed=seed)
#     if best_cid is None:
#         log.error("%s: could not generate 3D geometry for %s", safename, smiles)
#         return False
#     conf = mol.GetConformer(best_cid)

#     # -------------------------
#     # 4. Build the Gaussian file
#     # -------------------------
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     filename = output_dir / f"{safename}.gjf"
#     chk_name = filename.with_suffix(".chk").name

#     route = f"# opt=calcfc freq {method}/gen"
#     if geo_correction:
#         route += " geom=connectivity"
#     route += f" empiricaldispersion={dispersion}"

#     lines = [
#         f"%mem={mem}",
#         f"%nprocshared={nproc}",
#         f"%chk={chk_name}",
#         route,
#         "",
#         f"Generated from SMILES: {smiles} (start geometry: {ff})",
#         "",
#         f"{charge} {multiplicity}",
#     ]

#     for atom in mol.GetAtoms():
#         p = conf.GetAtomPosition(atom.GetIdx())
#         lines.append(f"{atom.GetSymbol():<2}{p.x:16.8f}{p.y:16.8f}{p.z:16.8f}")

#     if geo_correction:
#         lines.append("")
#         for atom in mol.GetAtoms():
#             i = atom.GetIdx() + 1
#             conns = []
#             for bond in atom.GetBonds():
#                 j = bond.GetOtherAtom(atom).GetIdx() + 1
#                 if j > i:                                    # list each bond once
#                     conns.append(f"{j} {BOND_ORDER.get(bond.GetBondType(), 1.0):.1f}")
#             lines.append(f"{i} " + " ".join(conns) if conns else f"{i}")

#     lines += ["", " ".join(symbols) + " 0", basis, "****", "", ""]

#     with open(filename, "w") as f:
#         f.write("\n".join(lines))
#     return True

def smiles_to_gjf(
    smiles, safename,
    charge, geo_correction, ID, output_dir,
    mem="8GB",
    nproc=12,
    method="b3lyp",
    basis="6-31g**",
    dispersion="gd3bj",
    seed=42,
    overwrite=False,
):
    """Write one .gjf for `smiles` (lowest-energy conformer). Returns True on success."""
    # 1. SMILES -> RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.error("%s: invalid SMILES: %s", safename, smiles)
        return False
    mol = Chem.AddHs(mol)
 
    # 2. Element, charge and multiplicity checks
    if any(a.GetAtomicNum() > MAX_ATOMIC_NUMBER for a in mol.GetAtoms()):
        log.error("%s: element beyond Zn; %s is not defined", safename, basis)
        return False
 
    formal_charge = Chem.GetFormalCharge(mol)
    if charge is None:
        charge = formal_charge
    elif charge != formal_charge:
        log.error("%s: charge=%s passed in, but SMILES formal charge is %s",
                  safename, charge, formal_charge)
        return False
 
    multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1
    n_electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if (n_electrons - (multiplicity - 1)) % 2 != 0:
        log.error("%s: charge %s and multiplicity %s are inconsistent (%s electrons)",
                  safename, charge, multiplicity, n_electrons)
        return False
 
    symbols = sorted({a.GetSymbol() for a in mol.GetAtoms()})
 
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
    route = f"# opt=calcfc freq {method}/gen"
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
 
    lines += ["", " ".join(symbols) + " 0", basis, "****", "", ""]
 
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


def generate_slurm_scripts(df, correction, ID, son, user, dft_folder):
    """
    Generate Slurm scripts for a DataFrame of molecules. \n
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
            row["Charge"], correction, ID, gjf_dir
            )
        if test:
            create_txt_file(safe_molecule_name, txt_dir, ID, son, user, dft_folder)
        else:
            failed.append(safe_molecule_name)
        if iteration % 500 == 0:
            print(iteration)
    # with open("scripts/logs/slurm_script_generation.log", "w") as log_file:
    #     log_file.write(f"Generated Slurm scripts for {iteration} molecules in experiment {ID}\n")
    #     log_file.write(f"Failed to generate Slurm scripts for {len(failed)} molecules:\n")
    #     for molecule in failed:
    #         log_file.write(f" - {molecule}\n")

def generate_slurm_scripts_from_csv(df, correction, ID, son, randomize):
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
                row["Charge"], correction, ID, gjf_dir
                )
            if test:
                create_txt_file(safe_molecule_name, txt_dir, ID, son)
            if iteration % 100 == 0:
                print(iteration)

    #     create_txt_file(row["Compound_CID"], "molecules/txt_files")
    # smiles_to_gjf()
    # create_txt_file(safe_molecule_name, "./molecules/txt_files")

# if __name__ == "__main__":
#     main()
