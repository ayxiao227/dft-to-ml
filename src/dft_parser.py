#!/usr/bin/env python3
"""Extract HOMO/LUMO energies from a Gaussian .log file.

Usage: python -m src.dft_parser LOGS MANIFEST OUTPUT
"""
import re
import math
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from src.gaussian_records import inspect_record
from src.structure_automation import safe_name 

HARTREE_TO_EV = 27.211386245988

# Gaussian prints fixed-width columns, so negative numbers can run together
# (e.g. "-19.14435-19.14248"). This regex handles that.
NUM = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?|nan|inf(?:inity)?)", re.I)


def numbers(payload):
    tokens=NUM.findall(payload)
    if not tokens or NUM.sub('',payload).strip():raise ValueError('Malformed numeric fields')
    values=[float(t.replace('D','e').replace('d','e')) for t in tokens]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite numeric fields')
    return values


def parse_orbital_blocks(path):
    """Return a list of dicts, one per orbital-energy block in the file.

    Each dict has 'occ' and 'virt' lists (Hartree). For open-shell
    calculations there are also 'occ_beta' and 'virt_beta'.
    A geometry optimization produces one block per step; the last is final.
    """
    blocks = []
    current = None
    last_kind = None  # tracks the previous line's label to detect a new block

    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"\s*(Alpha|Beta)\s+(occ\.|virt\.)\s+eigenvalues --(.*)", line)
            if not m:
                if re.search(r'\b(?:Alpha|Beta)\s+.*eigenvalues',line):raise ValueError('Malformed orbital channel header')
                continue
            spin, kind, rest = m.groups()
            key = ("occ" if kind == "occ." else "virt") + ("_beta" if spin == "Beta" else "")
            vals = numbers(rest)

            # A new block starts when we see Alpha occ. after anything else
            # that isn't Alpha occ. (or at the very start).
            if spin == "Alpha" and kind == "occ." and last_kind != "Alpha occ.":
                current = {"occ": [], "virt": [], "occ_beta": [], "virt_beta": []}
                blocks.append(current)
            last_kind = f"{spin} {kind}"
            if current is None:raise ValueError('Orbital block lacks occupied orbitals')
            current[key].extend(vals)
    return blocks


def homo_lumo(block):
    """Compute HOMO, LUMO and gap (Hartree and eV) from a block."""
    if not block['occ'] or not block['virt'] or bool(block['occ_beta']) != bool(block['virt_beta']):
        raise ValueError('Incomplete orbital block/spin channel')
    if not all(math.isfinite(v) for values in block.values() for v in values):raise ValueError('Nonfinite orbital block')
    if block["occ_beta"]:  # open-shell: HOMO is the higher of the two spins
        homo = max(block["occ"][-1], block["occ_beta"][-1])
        lumo = min(block["virt"][0], block["virt_beta"][0])
    else:
        homo, lumo = block["occ"][-1], block["virt"][0]
    result = {
        "homo_hartree": homo,
        "lumo_hartree": lumo,
        "gap_hartree": lumo - homo,
        "homo_ev": homo * HARTREE_TO_EV,
        "lumo_ev": lumo * HARTREE_TO_EV,
        "gap_ev": (lumo - homo) * HARTREE_TO_EV,
    }
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite derived orbital fields')
    return result


def validate_log(path, require_minimum=True):
    """Conservative gate for a single opt/freq Gaussian calculation."""
    text = Path(path).read_text(errors="replace")
    if "Error termination" in text or not text.strip() or "Normal termination of Gaussian" not in text.strip().splitlines()[-1]:
        raise ValueError("Calculation failed or lacks normal termination")
    if "Optimization completed" not in text:
        raise ValueError("Optimization completion not found")
    if require_minimum:
        frequencies = [v for line in text.splitlines() if "Frequencies --" in line
                       for v in numbers(line.split("--", 1)[1])]
        if not frequencies:
            raise ValueError("No vibrational frequencies; minimum not verified")
        if any(f < -10.0 for f in frequencies):
            raise ValueError("Imaginary frequency below -10 cm^-1")


def parse_log(folder_path, ID, processed_path, *, require_minimum=True, legacy_names=False):
    from src.schema import normalize_molecules
    raw = normalize_molecules(pd.read_csv(processed_path))
    if "file" not in raw:
        if not legacy_names:
            raise ValueError("Use generated manifest.csv, or explicitly enable legacy_names")
        raw["file"] = raw["Name"].map(lambda x: safe_name(None, x, ID) + ".log")
    if raw["file"].duplicated().any():
        raise ValueError("Ambiguous duplicate filenames in manifest")
    required = {'molecule_id','run_id','input_path','input_sha256','protocol_id'}
    if not required.issubset(raw.columns):
        raise ValueError('Manifest lacks verified input/protocol metadata; migrate legacy records first')
    if raw[['molecule_id','run_id']].duplicated().any():
        raise ValueError('Duplicate molecule/run metadata')
    metadata = raw.set_index("file")
    rows, rejected, seen = [], [], set()
    logs = sorted(Path(folder_path).rglob("*.log"))
    names = [p.name for p in logs]
    if len(names) != len(set(names)):
        raise ValueError("Multiple runs for one molecule; select an explicit run directory")
    for path in logs:
        seen.add(path.name)
        try:
            if path.name not in metadata.index:
                raise ValueError("Log has no matching manifest record")
            validate_log(path, require_minimum)
            details = inspect_record(path, metadata.loc[path.name], Path(processed_path).parent)
            xyz_dir = Path(folder_path) / 'optimized_xyz'
            xyz_dir.mkdir(exist_ok=True)
            xyz_path = xyz_dir / (details['log_sha256'] + '.xyz')
            xyz_path.write_text(details.pop('xyz'))
            blocks = parse_orbital_blocks(path)
            if not blocks:
                raise ValueError("No orbital energies")
            energies = homo_lumo(blocks[-1])
            if energies["gap_ev"] < 0:
                raise ValueError("Negative orbital gap requires review")
            rows.append({**metadata.loc[path.name].to_dict(), "file": path.name,
                         "log_path": str(path), **energies, **details,
                         "xyz_path":str(xyz_path), "accepted":True, "rejection_reason":"",
                         "optimization_converged":True,
                         "minimum_frequency_cm1":min(v for l in path.read_text().splitlines()
                         if 'Frequencies --' in l for v in numbers(l.split('--',1)[1]))})
        except (ValueError, IndexError, KeyError, RuntimeError, OSError) as exc:
            rejected.append({**(metadata.loc[path.name].to_dict() if path.name in metadata.index else {}),
                             "file": path.name, "log_path":str(path), "accepted":False,
                             "rejection_reason":str(exc), "reason": str(exc)})
    rejected.extend({**metadata.loc[name].to_dict(), "file": name, "reason": "Missing log"} for name in metadata.index if name not in seen)
    report = Path(folder_path) / "parse_rejections.csv"
    pd.DataFrame(rejected).to_csv(report, index=False)
    if not rows:
        raise ValueError(f"No valid labels; inspect {report}")
    result = pd.DataFrame(rows)
    result.to_csv(Path(folder_path) / 'accepted_candidates.csv', index=False)
    if result.protocol_id.nunique() != 1:
        raise ValueError('Mixed protocols: candidates saved for review, no training export produced')
    if result.molecule_id.duplicated().any():
        raise ValueError('Select exactly one accepted run per molecule')
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--id", default="legacy")
    parser.add_argument("--legacy-names", action="store_true")
    args = parser.parse_args()
    result = parse_log(args.logs, args.id, args.manifest, legacy_names=args.legacy_names)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
