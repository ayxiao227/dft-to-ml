"""Strict, versioned configuration; canonical expanded settings are the contract."""
import copy
import hashlib
import json
import math
from pathlib import Path

STRATEGIES = ('random', 'diversity', 'uncertainty', 'uncertainty_diversity')
DEFAULTS = {
    'schema_version': 1, 'mode': 'retrospective', 'dataset': 'qm9',
    'task': 'dft_homo_lumo_gap', 'units': 'eV',
    'source': {'path': '', 'sha256': '', 'exclusions_path': None,
               'exclusions_sha256': None, 'exclusions_kind': 'custom',
               'official_exclusions_sha256': None},
    'representation': {'radius': 2, 'bits': 2048, 'chirality': True, 'kind': 'bit',
                       'version': 'input_2d_v2'},
    'split': {'mode': 'group', 'final_fraction': .2, 'validation_fraction': .2,
              'policy': 'murcko_plus_generic_acyclic_v1'},
    'budgets': {'initial': 200, 'batch': 100, 'max_labels': 1200,
                'max_attempts': 1200, 'rounds': 10},
    'repeat_seeds': [42, 43, 44, 45, 46],
    'models': {'members': 5, 'trees': 100, 'downstream_trees': 100,
               'max_depth': 20, 'min_samples_leaf': 2, 'max_features': 1.0,
               'ridge_alpha': 1.0},
    'acquisition': {'strategies': list(STRATEGIES), 'shortlist_cap': 5000,
                    'candidate_cap': None, 'chunk_size': 4096},
    'resources': {'threads': 1, 'memory_mb': 2048},
    'retry': {'max_retries': 0}, 'target_mae_ev': None,
    'protocol': {'input_policy_id': None, 'observed_protocol_id': None,
                 'expected_revision': None,
                 'options': {'basis': 'def2-TZVP', 'method': 'b3lyp',
                             'dispersion': 'gd3bj', 'multiplicity': 1,
                             'seed': 42, 'nproc': 1, 'mem': '2GB'}},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(seed, repeat, round_number, purpose, member=0):
    return int(hashed([seed, repeat, round_number, purpose, member])[:8], 16) % (2**31 - 1)


def _merge(default, supplied, path='config'):
    if not isinstance(supplied, dict):
        raise ValueError(f'{path}: expected object')
    extra = set(supplied) - set(default)
    if extra:
        raise ValueError(f'{path}: unknown keys {sorted(extra)}')
    out = copy.deepcopy(default)
    for k, v in supplied.items():
        if isinstance(default[k], dict):
            out[k] = _merge(default[k], v, path + '.' + k)
        else:
            out[k] = v
    return out


def validate(supplied):
    c = _merge(DEFAULTS, supplied)
    canonical(c)  # rejects NaN/Inf anywhere, including unused settings
    def enum(value, allowed, key):
        if value not in allowed:
            raise ValueError(f'{key}: expected one of {allowed}')
    def integer(value, minimum, key):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key}: expected integer >= {minimum}')
    enum(c['schema_version'], [1], 'schema_version')
    if type(c['schema_version']) is not int:
        raise ValueError('schema_version must be integer')
    enum(c['mode'], ['retrospective', 'prospective'], 'mode')
    enum(c['dataset'], ['qm9', 'gaussian', 'gdb13', 'synthetic'], 'dataset')
    enum(c['task'], ['dft_homo_lumo_gap'], 'task')
    enum(c['units'], ['eV'], 'units')
    enum(c['split']['mode'], ['random', 'group'], 'split.mode')
    enum(c['split']['policy'], ['murcko_plus_generic_acyclic_v1'], 'split.policy')
    enum(c['source']['exclusions_kind'], ['custom', 'official'], 'exclusions_kind')
    enum(c['representation']['version'], ['input_2d_v2'], 'representation.version')
    enum(c['representation']['kind'], ['bit', 'count'], 'representation.kind')
    for key in ['final_fraction', 'validation_fraction']:
        v = c['split'][key]
        if type(v) not in (int, float) or not 0 < v < 1:
            raise ValueError(f'{key}: fraction must be strictly between 0 and 1')
    if sum(c['split'][k] for k in ['final_fraction', 'validation_fraction']) >= 1:
        raise ValueError('No development fraction remains')
    for k, minimum in [('initial', 1), ('batch', 1), ('max_labels', 1), ('max_attempts', 1), ('rounds', 0)]:
        integer(c['budgets'][k], minimum, k)
    b = c['budgets']
    if b['initial'] > b['max_labels'] or b['max_attempts'] < b['max_labels']:
        raise ValueError('Require initial <= max_labels <= max_attempts')
    for k in ['members', 'trees', 'downstream_trees', 'max_depth', 'min_samples_leaf']:
        integer(c['models'][k], 1, k)
    for k in ['max_features', 'ridge_alpha']:
        v = c['models'][k]
        if type(v) not in (int, float) or v <= 0:
            raise ValueError(f'{k}: positive finite number required')
    if c['models']['max_features'] > 1:
        raise ValueError('max_features must be <= 1')
    integer(c['representation']['radius'], 0, 'radius')
    integer(c['representation']['bits'], 1, 'bits')
    if type(c['representation']['chirality']) is not bool:
        raise ValueError('chirality must be boolean')
    a = c['acquisition']
    if not isinstance(a['strategies'], list) or not a['strategies'] or len(set(a['strategies'])) != len(a['strategies']):
        raise ValueError('Require unique strategies')
    for s in a['strategies']:
        enum(s, STRATEGIES, 'strategy')
    if c['models']['members'] < 2:
        raise ValueError('Committee uncertainty requires at least two members')
    integer(a['shortlist_cap'], b['batch'], 'shortlist_cap')
    if a['candidate_cap'] is not None:
        integer(a['candidate_cap'], b['batch'], 'candidate_cap')
    integer(a['chunk_size'], 1, 'chunk_size')
    integer(c['resources']['threads'], 1, 'threads')
    integer(c['resources']['memory_mb'], 16, 'memory_mb')
    integer(c['retry']['max_retries'], 0, 'max_retries')
    if not isinstance(c['repeat_seeds'], list) or not c['repeat_seeds'] or len(set(c['repeat_seeds'])) != len(c['repeat_seeds']):
        raise ValueError('Require unique repeat seeds')
    for seed in c['repeat_seeds']:
        integer(seed, 0, 'repeat seed')
    if c['target_mae_ev'] is not None and (type(c['target_mae_ev']) not in (int, float) or c['target_mae_ev'] < 0):
        raise ValueError('target_mae_ev must be nonnegative or null')
    import re
    for k in ['sha256', 'exclusions_sha256', 'official_exclusions_sha256']:
        v = c['source'][k]
        if (v is None and k != 'sha256'):
            continue
        if not isinstance(v, str) or re.fullmatch('[a-f0-9]{64}', v) is None:
            raise ValueError(f'source.{k}: SHA-256 required')
    if not isinstance(c['source']['path'], str) or not c['source']['path']:
        raise ValueError('source.path required')
    if c['source']['exclusions_path'] is not None and not isinstance(c['source']['exclusions_path'], str):
        raise ValueError('exclusions_path must be string or null')
    if c['dataset'] == 'qm9' and (not c['source']['exclusions_path'] or not c['source']['exclusions_sha256']):
        raise ValueError('QM9 requires an explicitly hashed exclusion set')
    if c['source']['exclusions_kind'] == 'official' and c['source']['official_exclusions_sha256'] != c['source']['exclusions_sha256']:
        raise ValueError('Official exclusions require matching independently verified declared hash')
    from src.gaussian_records import protocol
    policy = protocol(c['protocol']['options'])
    for k in ['input_policy_id', 'observed_protocol_id', 'expected_revision']:
        v = c['protocol'][k]
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError(f'protocol.{k}: nonempty string or null required')
    if c['protocol']['input_policy_id'] not in (None, policy):
        raise ValueError('Declared input policy differs from options')
    if c['dataset']=='gaussian' or c['mode']=='prospective':
        if not all(c['protocol'][k] for k in ['input_policy_id','observed_protocol_id','expected_revision']):
            raise ValueError('Gaussian initialization requires exact expected protocol/revision')
    if c['mode'] == 'prospective':
        if c['dataset'] not in ('gaussian', 'gdb13'):
            raise ValueError('Prospective requires separate organic Gaussian candidates')
        if not all(c['protocol'][k] for k in ['input_policy_id', 'observed_protocol_id', 'expected_revision']):
            raise ValueError('Prospective initialization requires explicit expected protocol/revision')
    elif c['dataset'] not in ('qm9', 'synthetic', 'gaussian'):
        raise ValueError('Unlabeled candidates cannot be retrospective labels')
    return c


def load(path):
    def no_duplicates(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f'Duplicate JSON key {k}')
            out[k] = v
        return out
    return validate(json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates))
