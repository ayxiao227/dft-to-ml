"""Synchronous orchestration. Selection sees X and committed labels only."""
import json
import time
from .config import canonical,hashed
from .state import crash
from .experiment import read_features
from .oracle import RetrospectiveOracle,EvaluationOracle
from . import models
from .acquisition import select as choose
from .evaluation import metrics,prediction_rows


def checkpoint(s,repeat,strategy,round_number,failpoint=None):
    old=s.rows('SELECT * FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(repeat,strategy,round_number))
    if old:return old[0]
    s.ensure_mutable();config=s.meta('config');features=read_features(s);labels=s.labels(repeat,strategy)
    if not labels:raise ValueError('No accepted labels available for checkpoint')
    bundle=models.fit(features,labels,config,repeat,round_number)
    members=s.rows("SELECT id FROM partitions WHERE repeat=? AND origin='validation' ORDER BY id",(repeat,))
    ids=[r['id'] for r in members]
    oracle=EvaluationOracle(s.root/f'repeat_{repeat}/oracle_validation.json',ids)
    # Prospective evaluation labels must themselves have verified evidence; no surrogate labels.
    table=oracle.read(ids)
    if set(table)!=set(ids):raise ValueError('Validation labels are not available')
    started=time.perf_counter();p=models.predict(bundle,features,ids,config['acquisition']['chunk_size'])
    prediction_seconds=time.perf_counter()-started
    attempts=s.db.execute('SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
    base=f'checkpoints/{repeat}/{strategy}/{round_number}'
    # Content-address paths avoid timing/pickle orphan collisions after a precommit crash.
    content=models.serialize(bundle);sha=__import__('hashlib').sha256(content).hexdigest()
    model_path=base+'/'+sha+'.pkl';pred_path=base+'/'+sha+'.validation.json'
    rows=prediction_rows(s,repeat,strategy,round_number,sha,ids,table,p,len(labels),attempts,'validation')
    metadata={**bundle['metadata'],'label_count':len(labels),'attempt_count':attempts,'validation_ids':ids,'prediction_seconds':prediction_seconds}
    with s.transaction(failpoint):
        s.artifact(model_path,content,'model',failpoint)
        s.artifact(pred_path,canonical(rows).encode(),'validation_predictions',failpoint)
        s.db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?)',(repeat,strategy,round_number,model_path,pred_path,canonical(metadata)))
        for evaluator in bundle['evaluators']|{'committee':None}:
            result=metrics([table[i]['gap_ev'] for i in ids],p[evaluator],p['uncertainty'] if evaluator=='committee' else None)
            s.db.execute('INSERT INTO metrics VALUES (?,?,?,?,?,?)',(repeat,strategy,round_number,evaluator,'validation',canonical(result)))
        s.event('checkpoint',{'repeat':repeat,'strategy':strategy,'round':round_number,'model_hash':sha})
    return s.rows('SELECT * FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(repeat,strategy,round_number))[0]


def select(s,repeat,strategy,failpoint=None):
    s.ensure_mutable();config=s.meta('config')
    if not s.rows('SELECT * FROM runs WHERE repeat=? AND strategy=?',(repeat,strategy)):raise ValueError('Unknown run')
    previous=s.rows('SELECT * FROM rounds WHERE repeat=? AND strategy=? ORDER BY round DESC LIMIT 1',(repeat,strategy))[0]
    if previous['status']!='terminal':return {'status':'pending','round':previous['round']}
    cp=checkpoint(s,repeat,strategy,previous['round'])
    round_number=previous['round']+1
    if round_number>config['budgets']['rounds']:return {'status':'complete','reason':'round cap'}
    labels=s.labels(repeat,strategy)
    attempts=s.db.execute('SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
    eligible=[r['id'] for r in s.rows("SELECT p.id FROM partitions p WHERE p.repeat=? AND p.origin='pool' AND NOT EXISTS (SELECT 1 FROM selections q WHERE q.repeat=p.repeat AND q.strategy=? AND q.id=p.id) ORDER BY p.id",(repeat,strategy))]
    b=min(config['budgets']['batch'],config['budgets']['max_labels']-len(labels),config['budgets']['max_attempts']-attempts,len(eligible))
    if b<=0:return {'status':'complete','reason':'label/attempt/pool cap'}
    feature=read_features(s);sha=s.rows('SELECT sha256 FROM artifacts WHERE path=?',(cp['path'],))[0]['sha256']
    bundle=models.load(s.root/cp['path'],sha,config,feature.settings_hash)
    started=time.perf_counter()
    rows,metadata=choose(strategy,eligible,sorted(labels),feature,b,config,repeat,round_number,
        lambda ids:models.predict(bundle,feature,ids,config['acquisition']['chunk_size'])['uncertainty'])
    metadata['acquisition_seconds']=time.perf_counter()-started
    crash('before_reservation',failpoint)
    with s.transaction(failpoint):
        s.db.execute('INSERT INTO rounds VALUES (?,?,?,?,?)',(repeat,strategy,round_number,'reserved',canonical(metadata)))
        for row in rows:
            ident=row['source_record_id'];calc=hashed([s.meta('experiment_hash'),repeat,strategy,ident,1])
            s.db.execute('INSERT INTO selections VALUES (?,?,?,?,?)',(repeat,strategy,round_number,ident,canonical(row)))
            s.db.execute('INSERT INTO attempts(repeat,strategy,id,attempt,round,calculation_id,status) VALUES (?,?,?,1,?,?,?)',(repeat,strategy,ident,round_number,calc,'reserved'))
        s.event('reserved',{'repeat':repeat,'strategy':strategy,'round':round_number,'ids':[r['source_record_id'] for r in rows]})
    crash('after_reservation',failpoint)
    return {'status':'reserved','round':round_number,'ids':[r['source_record_id'] for r in rows]}


def run(s,stop_after=None,failpoint=None):
    config=s.meta('config')
    if config['mode']!='retrospective':raise ValueError('run is retrospective-only; use select/prepare/ingest')
    s.ensure_mutable();completed=0
    for r in s.rows('SELECT * FROM runs ORDER BY repeat,strategy'):
        repeat,strategy=r['repeat'],r['strategy'];partitions=s.meta('partitions')[repeat]['rows']
        oracle=RetrospectiveOracle(s.root/f'repeat_{repeat}/oracle_pool.json',partitions,s.reservation_ids)
        while True:
            latest=s.rows('SELECT * FROM rounds WHERE repeat=? AND strategy=? ORDER BY round DESC LIMIT 1',(repeat,strategy))[0]
            if latest['status']=='reserved':
                key=canonical([repeat,strategy,latest['round']]);ids=s.reservation_ids(key);table=oracle.query(key,ids)
                deliveries=[{'id':i,'attempt':1,'status':'accepted','value':table[i]['gap_ev'],'log_hash':None,'evidence':table[i]} for i in ids]
                s.deliver(repeat,strategy,latest['round'],deliveries,failpoint)
                completed+=1
                if stop_after is not None and completed>=stop_after:return
                continue
            checkpoint(s,repeat,strategy,latest['round'],failpoint)
            result=select(s,repeat,strategy,failpoint)
            if result['status']=='complete':break
            if result['status']=='pending':raise ValueError('Outstanding non-retrospective attempt')
