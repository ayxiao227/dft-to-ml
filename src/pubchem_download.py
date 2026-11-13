import pubchempy as pcp
import pandas as pd
import time
from pathlib import Path
from src.schema import normalize_molecules
from src.csv_conversion import validated_molecules

REQUEST_DELAY = 0.5

properties = ["IUPACName", "MolecularFormula", "ConnectivitySMILES", "MolecularWeight", "Charge", "HeavyAtomCount"]

def download_pubchem_data(molecules, iteration, validate, classifier):
    records = []
    for query in molecules:
        # Fail visibly on network/API errors; do not silently claim a complete download.
        results = pcp.get_properties(properties, query, classifier)
        for record in results or []:
            records.append({**record, "Name": str(query)})
        time.sleep(REQUEST_DELAY)
    df = normalize_molecules(pd.DataFrame(records, columns=["CID", *properties, "Name"]))
    valid = validated_molecules(df) if validate else df.copy()
    destination = Path("molecules/csv_files")
    destination.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination / f"pubchem_molecules_{iteration}.csv", index=False)
    valid.to_csv(destination / f"pubchem_molecules_valid_{iteration}.csv", index=False)
    return valid
