"""SLURM rendering, independent of molecular geometry generation."""
import re
from pathlib import Path
from src.schema import safe_component

def create_txt_file(molecule_name, output_dir, ID, son=False, user="xaustin",
                    dft_folder="dft_ml", account="nszym1",
                    scratch_dir="/scratch/nszym_root/nszym1", memory="10G",
                    time="02-00:00", cpus=12):
    """Fail closed and retain scratch whenever execution or transfer fails."""
    import shlex
    for value in (molecule_name, ID, user, dft_folder, account):
        safe_component(value)
    if not re.fullmatch(r"[1-9][0-9]*[GMgm]", memory):
        raise ValueError("SLURM memory must be a positive integer with G or M suffix")
    if not re.fullmatch(r"[0-9:-]+", time) or not isinstance(cpus, int) or cpus < 1:
        raise ValueError("Invalid runtime or CPU count")
    if not Path(scratch_dir).is_absolute():
        raise ValueError("scratch_dir must be absolute")
    base = Path(f"/home/{user}/{dft_folder}")
    source = base / "gjf_files" / ID / f"{molecule_name}.gjf"
    destination = base / "log_files" / ID
    quote = shlex.quote
    script = f"""#!/bin/bash
#SBATCH --job-name={molecule_name}
#SBATCH --nodes=1
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --account={account}
set -euo pipefail
scratch_root={quote(str(Path(scratch_dir) / user))}
mkdir -p "$scratch_root"
work=$(mktemp -d "$scratch_root/${{SLURM_JOB_ID:?}}.XXXXXX")
trap 'echo "Job failed; retained scratch: $work" >&2' ERR
cd "$work"
cp -- {quote(str(source))} input.gjf
module purge
module load Chemistry
module load gaussian/16-revC01-avx2
g16 input.gjf
# Gaussian can leave an error log even if a wrapper reports success.
grep -q 'Normal termination of Gaussian' input.log
if grep -q 'Error termination' input.log; then exit 1; fi
formchk {quote(molecule_name + '.chk')} output.fchk
mkdir -p {quote(str(destination))}
# Each job gets its own output directory; reruns never overwrite prior results.
result_dir={quote(str(destination))}/"${{SLURM_JOB_ID}}"
mkdir "$result_dir"
cp -- input.gjf "$result_dir/{molecule_name}.gjf"
cp -- input.log "$result_dir/{molecule_name}.log"
cp -- {quote(molecule_name + '.chk')} "$result_dir/{molecule_name}.chk"
cp -- output.fchk "$result_dir/{molecule_name}.fchk"
rm -rf -- "$work"
"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{molecule_name}.txt"
    path.write_text(script)
    return path


