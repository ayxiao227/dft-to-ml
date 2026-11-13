"""Internal schemas are distinct from legacy CSV columns; identity is not tautomerized."""
from dataclasses import dataclass, asdict
import math
from src.schema import canonical_smiles, molecule_key

IDENTITY_VERSION = 'rdkit_canonical_isomeric_sha256_80_v1'
ACCEPTANCE_VERSION = 'organic_optfreq_minfreq_minus10_v2'

@dataclass(frozen=True)
class MolecularRecord:
    dataset: str
    source_record_id: str
    input_smiles: str
    input_structure_key: str
    source_hash: str
    raw_input_smiles: str
    raw_optimized_smiles: str | None = None
    optimized_structure_key: str | None = None
    identity_policy: str = IDENTITY_VERSION

    @classmethod
    def create(cls, dataset, source_id, smiles, source_hash, optimized=None):
        canonical = canonical_smiles(smiles)
        return cls(dataset, str(source_id), canonical, molecule_key(canonical),
                   source_hash, smiles, optimized,
                   molecule_key(optimized) if optimized is not None else None)

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class CalculationRecord:
    calculation_id: str
    attempt_id: int
    source_record_id: str
    input_structure_key: str
    molecule_id: str
    run_id: str
    input_protocol_id: str
    input_path: str | None = None
    input_sha256: str | None = None
    log_path: str | None = None
    log_sha256: str | None = None
    observed_protocol_id: str | None = None
    status: str = 'reserved'
    reason: str | None = None

@dataclass(frozen=True)
class LabelRecord:
    source_record_id: str
    input_structure_key: str
    calculation_id: str
    attempt_id: int
    gap_ev: float
    provenance_type: str
    observed_protocol_id: str
    units: str = 'eV'
    acceptance_policy: str = ACCEPTANCE_VERSION

    def __post_init__(self):
        if self.units != 'eV' or not math.isfinite(self.gap_ev) or self.gap_ev < 0:
            raise ValueError('Invalid orbital-gap label/units')
        if self.provenance_type not in ('publisher_qm9', 'verified_gaussian', 'synthetic_fixture') or not self.observed_protocol_id:
            raise ValueError('Explicit verified provenance required')

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_hash: str
    config_hash: str
    source_hash: str
    partition_hash: str
    repeat: int
    strategy: str
    finalized: bool = False


def check_collisions(records):
    seen, ids = {}, set()
    for r in records:
        if r.source_record_id in ids:
            raise ValueError('Duplicate source record ID')
        ids.add(r.source_record_id)
        previous = seen.setdefault(r.input_structure_key, r.input_smiles)
        if previous != r.input_smiles:
            raise ValueError('Truncated identity hash collision')
