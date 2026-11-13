#!/usr/bin/env python3
"""Extract HOMO/LUMO energies from a Gaussian .log file.

Usage: python -m src.dft_parser LOGS MANIFEST OUTPUT
"""
import re
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from src.structure_automation import safe_name 

HARTREE_TO_EV = 27.211386245988

# Gaussian prints fixed-width columns, so negative numbers can run together
# (e.g. "-19.14435-19.14248"). This regex handles that.
NUM = re.compile(r"-?\d+\.\d+")


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
                continue
            spin, kind, rest = m.groups()
            key = ("occ" if kind == "occ." else "virt") + ("_beta" if spin == "Beta" else "")
            vals = [float(x) for x in NUM.findall(rest)]

            # A new block starts when we see Alpha occ. after anything else
            # that isn't Alpha occ. (or at the very start).
            if spin == "Alpha" and kind == "occ." and last_kind != "Alpha occ.":
                current = {"occ": [], "virt": [], "occ_beta": [], "virt_beta": []}
                blocks.append(current)
            last_kind = f"{spin} {kind}"
            current[key].extend(vals)
    return blocks


def homo_lumo(block):
    """Compute HOMO, LUMO and gap (Hartree and eV) from a block."""
    if block["occ_beta"]:  # open-shell: HOMO is the higher of the two spins
        homo = max(block["occ"][-1], block["occ_beta"][-1])
        lumo = min(block["virt"][0], block["virt_beta"][0])
    else:
        homo, lumo = block["occ"][-1], block["virt"][0]
    return {
        "homo_hartree": homo,
        "lumo_hartree": lumo,
        "gap_hartree": lumo - homo,
        "homo_ev": homo * HARTREE_TO_EV,
        "lumo_ev": lumo * HARTREE_TO_EV,
        "gap_ev": (lumo - homo) * HARTREE_TO_EV,
    }


def validate_log(path, require_minimum=True):
    """Conservative gate for a single opt/freq Gaussian calculation."""
    text = Path(path).read_text(errors="replace")
    if "Error termination" in text or not text.strip() or "Normal termination of Gaussian" not in text.strip().splitlines()[-1]:
        raise ValueError("Calculation failed or lacks normal termination")
    if "Optimization completed" not in text:
        raise ValueError("Optimization completion not found")
    if require_minimum:
        frequencies = [float(v) for line in text.splitlines() if "Frequencies --" in line
                       for v in NUM.findall(line.split("--", 1)[1])]
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
            blocks = parse_orbital_blocks(path)
            if not blocks:
                raise ValueError("No orbital energies")
            energies = homo_lumo(blocks[-1])
            if energies["gap_ev"] < 0:
                raise ValueError("Negative orbital gap requires review")
            rows.append({**metadata.loc[path.name].to_dict(), "file": path.name,
                         "log_path": str(path), **energies})
        except (ValueError, IndexError, KeyError) as exc:
            rejected.append({"file": path.name, "reason": str(exc)})
    rejected.extend({"file": name, "reason": "Missing log"} for name in metadata.index if name not in seen)
    report = Path(folder_path) / "parse_rejections.csv"
    pd.DataFrame(rejected, columns=["file", "reason"]).to_csv(report, index=False)
    if not rows:
        raise ValueError(f"No valid labels; inspect {report}")
    return pd.DataFrame(rows)


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
