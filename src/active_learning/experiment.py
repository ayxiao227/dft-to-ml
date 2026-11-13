"""Initialization and immutable input artifacts; no fit or oracle query on import."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import sklearn
from rdkit import rdBase
from .config import validate,canonical,hashed
from .datasets import load_source
from .features import featurize,CandidateFeatures,metadata
from .partitioning import freeze
from .state import Store


def initialize(config,root):
    config=validate(config);root=Path(root).resolve()
    if root.exists():
        with Store(root) as s:
            if s.meta('config_hash')!=hashed(config):raise ValueError('Existing experiment config differs')
        return root
    records,labels,ledger=load_source(config)
    root.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+root.name+'.init.',dir=root.parent))
    features,records,rejected=featurize(records,config['representation'],cache=root.parent/'_al_feature_cache',array_dir=temp/'feature_work');ledger.extend(rejected)
    labels={i:labels[i] for i in features.ids if i in labels}
    partitions=[freeze(records,config,repeat) for repeat in range(len(config['repeat_seeds']))]
    # Matrices are memory mapped; fail if even one scoring chunk exceeds the limit.
    chunk_bytes=min(len(records),config['acquisition']['chunk_size'])*features.x.shape[1]*8
    if chunk_bytes>config['resources']['memory_mb']*1024**2//2:
        raise ValueError('Scoring chunk exceeds half configured memory; reduce chunk_size')
    with Store(temp,create=True) as s, s.transaction():
        s.set_meta('config',config);s.set_meta('config_hash',hashed(config))
        s.set_meta('source_hash',config['source']['sha256']);s.set_meta('experiment_hash',hashed([config,[p['hash'] for p in partitions]]))
        s.set_meta('environment',{'numpy':np.__version__,'sklearn':sklearn.__version__,'rdkit':rdBase.rdkitVersion})
        s.set_meta('features',{'ids':list(features.ids),'smiles':list(features.smiles),'names':list(features.names),'settings_hash':features.settings_hash})
        s.set_meta('partitions',partitions);s.set_meta('attrition',{'accepted':len(records),'rejected':len(ledger)})
        s.artifact('config.json',canonical(config).encode(),'config')
        s.artifact('molecules.json',canonical([r.to_dict() for r in records]).encode(),'molecular_records')
        s.artifact('curation.json',canonical(ledger).encode(),'attrition')
        s.artifact('feature_registry.json',canonical(metadata(config['representation'])).encode(),'registry')
        for name,array,kind in [('features.npy',features.x,'input_features'),('binary.npy',features.binary,'binary_diversity')]:
            staged=temp/(name+'.staged')
            mapped=np.lib.format.open_memmap(staged,mode='w+',dtype=array.dtype,shape=array.shape)
            for start in range(0,len(array),4096):mapped[start:start+4096]=array[start:start+4096]
            mapped.flush();del mapped
            with staged.open('rb') as stream:os.fsync(stream.fileno())
            os.rename(staged,temp/name)
            s.artifact_file(name,kind)
        for r in records:s.db.execute('INSERT INTO molecules VALUES (?,?)',(r.source_record_id,canonical(r.to_dict())))
        for repeat,p in enumerate(partitions):
            s.artifact(f'repeat_{repeat}/partitions.json',canonical(p).encode(),'partitions')
            for r in p['rows']:s.db.execute('INSERT INTO partitions VALUES (?,?,?,?)',(repeat,r['source_record_id'],r['origin'],r['group_key']))
            for role,origins in [('pool',('initial','pool')),('validation',('validation',)),('final',('final',))]:
                subset={r['source_record_id']:labels[r['source_record_id']] for r in p['rows'] if r['origin'] in origins and r['source_record_id'] in labels}
                if config['mode']=='prospective' and role!='pool':continue
                s.artifact(f'repeat_{repeat}/oracle_{role}.json',canonical(subset).encode(),'isolated_oracle_'+role)
            if config['mode']=='prospective':
                strategy='__evaluation__'
                s.db.execute('INSERT INTO runs VALUES (?,?,0)',(repeat,strategy))
                s.db.execute('INSERT INTO rounds VALUES (?,?,0,?,?)',(repeat,strategy,'reserved',canonical({'heldout':True})))
                for r in p['rows']:
                    if r['origin'] not in ('validation','final'):continue
                    ident=r['source_record_id'];calc=hashed([s.meta('experiment_hash'),repeat,strategy,ident,1])
                    s.db.execute('INSERT INTO attempts(repeat,strategy,id,attempt,round,calculation_id,status) VALUES (?,?,?,1,0,?,?)',(repeat,strategy,ident,calc,'reserved'))
            for strategy in config['acquisition']['strategies']:
                s.db.execute('INSERT INTO runs VALUES (?,?,0)',(repeat,strategy))
                s.db.execute('INSERT INTO rounds VALUES (?,?,0,?,?)',(repeat,strategy,'reserved',canonical({'initial':True})))
                for r in p['rows']:
                    if r['origin']!='initial':continue
                    ident=r['source_record_id'];calc=hashed([s.meta('experiment_hash'),repeat,strategy,ident,1])
                    s.db.execute('INSERT INTO attempts(repeat,strategy,id,attempt,round,calculation_id,status) VALUES (?,?,?,1,0,?,?)',(repeat,strategy,ident,calc,'reserved'))
                if config['mode']=='retrospective':
                    deliveries=[{'id':r['source_record_id'],'attempt':1,'status':'accepted','value':labels[r['source_record_id']]['gap_ev'],'log_hash':None,'evidence':labels[r['source_record_id']]} for r in p['rows'] if r['origin']=='initial']
                    # Initial delivery stays inside the initialization transaction.
                    for d in deliveries:
                        s.db.execute('INSERT INTO labels VALUES (?,?,?,1,?,?)',(repeat,strategy,d['id'],d['value'],canonical(d['evidence'])))
                        s.db.execute("UPDATE attempts SET status='accepted',evidence=? WHERE repeat=? AND strategy=? AND id=?",(canonical(d['evidence']),repeat,strategy,d['id']))
                    s.db.execute("UPDATE rounds SET status='terminal' WHERE repeat=? AND strategy=? AND round=0",(repeat,strategy))
        s.event('initialized',{'config_hash':hashed(config),'rows':len(records)})
    import shutil
    del features
    shutil.rmtree(temp/'feature_work')
    # No existing experiment is replaced, including during concurrent initialization.
    if root.exists():raise FileExistsError('Experiment appeared during initialization; staged copy retained')
    os.rename(temp,root)
    return root


def read_features(s):
    m=s.meta('features')
    return CandidateFeatures(tuple(m['ids']),tuple(m['smiles']),tuple(m['names']),
        np.load(s.root/'features.npy',mmap_mode='r',allow_pickle=False),
        np.load(s.root/'binary.npy',mmap_mode='r',allow_pickle=False),m['settings_hash'])
