"""Allowlisted pre-calculation features; IDs, labels, relaxed data never enter X."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
from src.descriptors import input_features, FeatureError
from src.feature_registry import INPUT_FEATURES, registry
from src.active_learning.config import canonical, hashed

@dataclass(frozen=True)
class CandidateFeatures:
    ids: tuple[str, ...]
    smiles: tuple[str, ...]
    names: tuple[str, ...]
    x: np.ndarray
    binary: np.ndarray
    settings_hash: str

    def __post_init__(self):
        if len(self.smiles)!=len(self.ids) or len(set(self.names))!=len(self.names) or self.binary.ndim!=2 or len(set(self.ids)) != len(self.ids) or self.x.shape != (len(self.ids),len(self.names)) or self.binary.shape[0] != len(self.ids):
            raise FeatureError('Feature ID/order/shape mismatch')
        for start in range(0,len(self.ids),4096):
            if not np.isfinite(self.x[start:start+4096]).all() or not np.isin(self.binary[start:start+4096], [0,1]).all():
                raise FeatureError('Nonfinite/nonbinary features')
        allowed = set(INPUT_FEATURES) | {f'morgan_{i}' for i in range(self.binary.shape[1])}
        if set(self.names) - allowed:
            raise FeatureError('Feature allowlist violated')

    def fingerprints(self,ids):
        positions={ident:i for i,ident in enumerate(self.ids)}
        return FingerprintRows(self.binary,np.asarray([positions[i] for i in ids]))

    def rows(self, ids, binary=False):
        positions = {ident:i for i,ident in enumerate(self.ids)}
        return (self.binary if binary else self.x)[[positions[i] for i in ids]]


class FingerprintRows:
    """Lazy row view: diversity reads only a fingerprint chunk, never all N×bits."""
    def __init__(self,matrix,indices):self.matrix=matrix;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,key):return self.matrix[self.indices[key]]
    def reordered(self,order):return FingerprintRows(self.matrix,self.indices[order])


def metadata(settings):
    return {'settings': settings, 'schema': 'input_2d_v2', 'standardization': 'none_v1',
            'identity': 'canonical_isomeric_v1', 'rdkit': rdBase.rdkitVersion,
            'registry': registry(INPUT_FEATURES, 'input_2d_v2'),
            'fingerprint': 'Morgan; numeric bit order; diversity always binary Tanimoto'}


def featurize(records, settings, cache=None, array_dir=None):
    meta = metadata(settings); mh = hashed(meta)
    names = tuple(INPUT_FEATURES + [f'morgan_{i}' for i in range(settings['bits'])])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=settings['radius'], fpSize=settings['bits'], includeChirality=settings['chirality'])
    values, fps, accepted, rejected = [], [], [], []
    if array_dir:
        directory=Path(array_dir);directory.mkdir(parents=True,exist_ok=True)
        values=np.lib.format.open_memmap(directory/'features_work.npy',mode='w+',dtype=np.float32,shape=(len(records),len(names)))
        fps=np.lib.format.open_memmap(directory/'binary_work.npy',mode='w+',dtype=np.uint8,shape=(len(records),settings['bits']))
    for record in sorted(records, key=lambda r: r.source_record_id):
        try:
            key = hashed([record.input_smiles, meta])
            path = Path(cache)/f'{key}.json' if cache else None
            if path and path.exists():
                entry=json.loads(path.read_text())
                if entry['key'] != key or entry['hash'] != hashed([entry['values'],entry['binary']]):
                    raise FeatureError('Feature cache corrupted')
                x, binary = entry['values'], entry['binary']
            else:
                desc, mol = input_features(record.input_smiles)
                binary = generator.GetFingerprintAsNumPy(mol).tolist()
                fp = binary if settings['kind']=='bit' else generator.GetCountFingerprintAsNumPy(mol).tolist()
                x = list(desc.values()) + fp
                if path:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    # Cache is advisory; authoritative experiment arrays are immutable artifacts.
                    from src.active_learning.artifacts import atomic_write
                    atomic_write(path, canonical({'key':key,'values':x,'binary':binary,'hash':hashed([x,binary])}).encode())
            if len(x) != len(names) or len(binary) != settings['bits'] or not np.isfinite(x).all():
                raise FeatureError('Malformed cached/descriptor shape')
            if array_dir:
                values[len(accepted)]=x;fps[len(accepted)]=binary
            else:values.append(x);fps.append(binary)
            accepted.append(record)
        except (FeatureError, ValueError, RuntimeError) as exc:
            rejected.append({'source_record_id':record.source_record_id,'reason':str(exc),'stage':'input_features'})
    if not accepted:
        raise FeatureError('No usable input features')
    if array_dir:values.flush();fps.flush()
    result=CandidateFeatures(tuple(r.source_record_id for r in accepted), tuple(r.input_smiles for r in accepted), names,
        values[:len(accepted)] if array_dir else np.asarray(values,dtype=np.float32),
        fps[:len(accepted)] if array_dir else np.asarray(fps,dtype=np.uint8), mh)
    return result, accepted, rejected
