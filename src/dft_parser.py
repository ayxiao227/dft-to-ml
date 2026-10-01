#!/usr/bin/env python3
"""Extract HOMO/LUMO energies from a Gaussian .log file.

Usage:  python parse_homo_lumo.py file.log
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


def parse_log(folder_path, ID, processed_path):
    # iteration = 0
    # rows = []
    # print(folder_path)
    # folder = Path(folder_path)
    # for log in sorted(folder.glob("*.log")):
    #     iteration += 1
    #     try:
    #         blocks = parse_orbital_blocks(log)
    #         if not blocks:
    #             print(f"Skipping {log.name}: no orbital data")
    #             continue
    #         r = homo_lumo(blocks[-1])            # final geometry
    #         rows.append({"file": log.name, **r})
    #     except Exception as e:                   # one bad file won't stop the run
    #         print(f"Failed on {log.name}: {e}")
    # print(iteration)
    # df = pd.DataFrame(rows)
    # # df['SafeName'] = df['file'].apply(lambda x: x.replace(".log", ""))
    # #match IUPACNames from the processed files after csv conversion and from the Pubchem download
    # #this way can have the SMILES which is much easier to match than names
    # raw_df = pd.read_csv(processed_path, usecols = ["IUPACName", "SMILES","Name"])
    # raw_df["file"] = raw_df["IUPACName"].apply(
    # lambda x: safe_name(None, x, ID) + ".log")
    # # print(raw_df.head(5))
    # print(df.head(5))
    # print(raw_df.head(5))
    
    # df = pd.merge(df, raw_df, how = "inner", on = "file")
    #     # 1. Merge based on the first column
    # # merge_col1 = pd.merge(raw_df, df, left_on='IUPACName', right_on='IUPACName', how='inner')

    # # # 2. Merge based on the second column
    # # merge_col2 = pd.merge(raw_df, df, left_on='Name', right_on='IUPACName', how='inner')

    # # # 3. Combine both results and remove identical rows
    # # final_df = pd.concat([merge_col1, merge_col2]).drop_duplicates()
    # # print(final_df.head(5))
    # # print(len(final_df))
    # df.to_csv("data/log_data/" + ID + ".csv", index = False)

    # return df
    folder = Path(folder_path)
    logs = sorted(folder.glob("*.log"))
    rows, skipped = [], []

    for log in logs:
        try:
            blocks = parse_orbital_blocks(log)
            if not blocks:
                skipped.append({"file": log.name, "reason": "no orbital data"})
                continue
            rows.append({"file": log.name, **homo_lumo(blocks[-1])})
        except Exception as e:
            skipped.append({"file": log.name, "reason": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    raw_df = pd.read_csv(processed_path, usecols=["IUPACName", "SMILES", "Name"])
    raw_df["file"] = raw_df["Name"].apply(lambda x: safe_name(None, x, ID) + ".log")
    raw_df = raw_df.drop_duplicates("file")          # the 1 duplicate key

    df = pd.merge(df, raw_df, how="inner", on="file", validate="one_to_one")
    print(len(df))   # expect ~670
    still_unmatched = {l.name for l in logs} - set(raw_df["file"])
    print(len(still_unmatched), sorted(still_unmatched))
    
    return df

def main():
    df = parse_log("data/log_files", "data_1", "molecules/csv_files/test_1.csv")
    df.to_csv("data/log_data/data_1.csv")

if __name__ == "__main__":
    main()
