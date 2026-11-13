"""Immutable run directories and dataset identity checks."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from src.schema import canonical_smiles, safe_component


def check_molecular_identity(df):
    if "SMILES" not in df:
        raise ValueError("Evaluation requires SMILES to audit molecular duplicates")
    keys = df.SMILES.map(canonical_smiles)
    if keys.duplicated().any():
        raise ValueError("Duplicate canonical molecules: resolve repeated labels before evaluation")


def create_run(data_path, label, settings):
    safe_component(label)
    root = Path("results/runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label + "_" + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    import sklearn, rdkit, numpy
    metadata = {"dataset_path": str(data_path),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "commit": commit, "working_tree_dirty": dirty, "python": platform.python_version(),
        "sklearn": sklearn.__version__, "rdkit": rdkit.__version__, "numpy": numpy.__version__,
        "settings": settings, "split_seed": 42,
        "interpretation": "Random molecular split; not scaffold generalization or an untouched final test."}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return root
