"""Shared fixed evaluators and molecule-bootstrap RF committee, refit per round."""
import pickle
import platform
import time
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from .config import seed_for, hashed

MODEL_VERSION='sklearn_fixed_v1'


def forest(config, seed, acquisition=False):
    m=config['models']
    return RandomForestRegressor(n_estimators=m['trees'] if acquisition else m['downstream_trees'],
        max_depth=m['max_depth'],min_samples_leaf=m['min_samples_leaf'],max_features=m['max_features'],
        bootstrap=False,random_state=seed,n_jobs=config['resources']['threads'])


def fit(features, labels, config, repeat, round_number):
    ids=sorted(labels)
    if not ids:raise ValueError('Cannot fit without committed labels')
    x=features.rows(ids);y=np.asarray([labels[i] for i in ids],dtype=float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite training target')
    seed=config['repeat_seeds'][repeat]
    evaluators={'mean':DummyRegressor(strategy='mean'),
        'ridge':make_pipeline(StandardScaler(),Ridge(alpha=config['models']['ridge_alpha'])),
        'rf':forest(config,seed_for(seed,repeat,round_number,'downstream_rf'))}
    started=time.perf_counter()
    with threadpool_limits(limits=config['resources']['threads']):
        for model in evaluators.values():model.fit(x,y)
    evaluator_seconds=time.perf_counter()-started
    committee=[];member_seeds=[];bootstrap_hashes=[]
    started=time.perf_counter()
    with threadpool_limits(limits=config['resources']['threads']):
        for member in range(config['models']['members']):
            bs=seed_for(seed,repeat,round_number,'outer_bootstrap',member)
            fs=seed_for(seed,repeat,round_number,'committee_forest',member)
            rows=np.random.default_rng(bs).integers(len(ids),size=len(ids))
            model=forest(config,fs,True);model.fit(x[rows],y[rows]);committee.append(model)
            member_seeds.append({'bootstrap':bs,'forest':fs})
            bootstrap_hashes.append(hashed([ids[j] for j in rows]))
    meta={'version':MODEL_VERSION,'config_hash':hashed(config),'feature_hash':features.settings_hash,
        'feature_names':list(features.names),'training_features_sha256':__import__('hashlib').sha256(x.tobytes()).hexdigest(),'label_ids_hash':hashed(ids),'labels_hash':hashed([[i,labels[i]] for i in ids]),
        'repeat':repeat,'round':round_number,'member_seeds':member_seeds,'bootstrap_hashes':bootstrap_hashes,
        'threads':config['resources']['threads'],'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,
        'evaluator_seconds':evaluator_seconds,'committee_seconds':time.perf_counter()-started,
        'committee_role':'acquisition; diagnostic overhead also charged to random/diversity'}
    return {'metadata':meta,'evaluators':evaluators,'committee':committee}


def predict(bundle, features, ids, chunk_size=4096):
    output={k:[] for k in bundle['evaluators']};means=[];stds=[]
    with threadpool_limits(limits=bundle['metadata'].get('threads',1)):
        for start in range(0,len(ids),chunk_size):
            x=features.rows(ids[start:start+chunk_size])
            for name,model in bundle['evaluators'].items():output[name].extend(model.predict(x).tolist())
            p=np.asarray([model.predict(x) for model in bundle['committee']])
            if p.shape[0]<2 or not np.isfinite(p).all():raise ValueError('Nonfinite/invalid committee predictions')
            means.extend(p.mean(axis=0).tolist());stds.extend(p.std(axis=0,ddof=1).tolist())
    output['committee']=means;output['uncertainty']=stds
    if not all(np.isfinite(v).all() for v in output.values()):raise ValueError('Nonfinite predictions')
    return output


def serialize(bundle):
    return pickle.dumps(bundle,protocol=5)


def load(path, expected_hash, config, feature_hash):
    # Never load user-supplied arbitrary pickle paths: caller must use registered artifacts.
    from .artifacts import file_hash
    if file_hash(path)!=expected_hash:raise ValueError('Model artifact hash mismatch')
    bundle=pickle.loads(path.read_bytes())
    meta=bundle['metadata']
    if meta['version']!=MODEL_VERSION or meta['config_hash']!=hashed(config) or meta['feature_hash']!=feature_hash or meta['sklearn']!=sklearn.__version__ or meta['numpy']!=np.__version__:
        raise ValueError('Stale/incompatible model artifact')
    return bundle
