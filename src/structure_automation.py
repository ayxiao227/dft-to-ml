from time import strftime

from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
# from src.formula_to_atoms import get_elements_from_smiles as fta
from src.formula_to_atoms import get_elements_from_smiles as fta
import random

import re

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
    basis="6-31g**",
    dispersion="gd3bj",
):
    # -------------------------
    # 1. SMILES → RDKit molecule
    # -------------------------
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)
    # Calculate multiplicity (number of unpaired electrons + 1)
    multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1
    # -------------------------
    # 2. Generate 3D geometry
    # -------------------------
    status = AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    if status != 0:
        # raise ValueError(
        #     f"Could not generate 3D geometry for {smiles}"
        # )
        return False

    # Pre-optimize geometry
    AllChem.MMFFOptimizeMolecule(mol)

    # -------------------------
    # 3. File information
    # -------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{safename}.gjf"

    chk_name = filename.with_suffix(".chk").name
    # -------------------------
    # 4. Write Gaussian file
    # -------------------------
    with open(filename, "w") as f:

        # Resource specifications
        f.write(f"%mem={mem}\n")
        f.write(f"%nprocshared={nproc}\n")
        f.write(f"%chk={chk_name}\n")

        # Route section
        f.write(
            f"# opt=calcfc freq "
            f"{method}/gen ")
        if geo_correction:
            f.write(f"geom=connectivity ")
        f.write(f"empiricaldispersion={dispersion}\n")

        # Title
        f.write("\n")
        f.write(f"Generated from SMILES: {smiles}\n")

        # Charge / multiplicity
        f.write("\n")
        f.write(f"{charge} {multiplicity}\n")

        # -------------------------
        # 5. Cartesian coordinates
        # -------------------------
        conf = mol.GetConformer()

        for atom in mol.GetAtoms():

            pos = conf.GetAtomPosition(atom.GetIdx())

            f.write(
                f"{atom.GetSymbol():<2}"
                f"{pos.x:16.8f}"
                f"{pos.y:16.8f}"
                f"{pos.z:16.8f}\n"
            )

        # -------------------------
        # 6. Connectivity
        # -------------------------
        if geo_correction:
            f.write("\n")
            for atom in mol.GetAtoms():

                atom_idx = atom.GetIdx() + 1

                connections = []

                for bond in atom.GetBonds():

                    other_atom = bond.GetOtherAtom(atom)
                    other_idx = other_atom.GetIdx() + 1

                    # Only list atoms with higher index
                    # to avoid duplicating bonds
                    if other_idx > atom_idx:

                        bond_type = bond.GetBondType()

                        if bond_type == Chem.BondType.SINGLE:
                            order = 1.0
                        elif bond_type == Chem.BondType.DOUBLE:
                            order = 2.0
                        elif bond_type == Chem.BondType.TRIPLE:
                            order = 3.0
                        elif bond_type == Chem.BondType.AROMATIC:
                            order = 1.5
                        else:
                            order = 1.0

                        connections.append(
                            f"{other_idx} {order:.1f}"
                        )

                if connections:
                    f.write(
                        f"{atom_idx} "
                        + " ".join(connections)
                        + "\n"
                    )
                else:
                    f.write(f"{atom_idx}\n")

        # -------------------------
        # 7. Basis set
        # -------------------------
        f.write("\n")
        atoms = fta(smiles)
        for a in atoms:
            f.write(f"{a} ")
        f.write("0\n")
        #f.write("C H O N " + "0\n") some molecules do not have all these atoms, so including them all will cause p
        f.write(f"{basis}\n")
        f.write("****\n")
        f.write("\n")
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

SCRATCH_FLDR="{scratch_dir}/{user}"

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

SCRATCH_FLDR="{scratch_dir}"

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
    # df = pd.read_csv("/home/ayxiao227/dft_ml_project/molecules/csv_files/pubchem_molecules_valid.csv")
    # output_dir = "molecules/"
    if ID is None:
        ID = strftime("%H%M_%d%m%Y")
    gjf_dir = Path("molecules/gjf_files/" + ID)
    txt_dir = Path("scripts/txt_files/" + ID)
    iteration = 0
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
        if iteration % 100 == 0:
            print(iteration)
        if iteration == 5000:
            return True

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
