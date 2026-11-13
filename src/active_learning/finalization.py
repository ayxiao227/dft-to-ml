"""One final evaluation of frozen saved checkpoints; retry resumes the same work."""
import json
from .config import canonical,hashed
from .experiment import read_features
from .oracle import EvaluationOracle
from .state import crash
from . import models
from .evaluation import metrics,prediction_rows


def finalize(s,failpoint=None):
    config=s.meta('config');existing=s.meta('finalization')
    checkpoints=s.rows('SELECT * FROM checkpoints ORDER BY repeat,strategy,round')
    if not checkpoints:raise ValueError('No checkpoints available')
    snapshot=hashed(checkpoints)
    contract={'experiment_hash':s.meta('experiment_hash'),'config_hash':s.meta('config_hash'),
        'partition_hashes':[p['hash'] for p in s.meta('partitions')],'checkpoint_set_hash':snapshot}
    if existing:
        if any(existing.get(k)!=v for k,v in contract.items()):raise ValueError('Frozen finalization contract changed')
        if existing['status']=='completed':return existing
    else:
        if s.rows("SELECT * FROM attempts WHERE status IN ('reserved','prepared','pending')"):
            raise ValueError('Outstanding calculations prevent finalization')
        for run in s.rows("SELECT * FROM runs WHERE strategy!='__evaluation__'"):
            repeat,strategy=run['repeat'],run['strategy']
            latest=s.rows('SELECT * FROM rounds WHERE repeat=? AND strategy=? ORDER BY round DESC LIMIT 1',(repeat,strategy))[0]
            cp=s.rows('SELECT * FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(repeat,strategy,latest['round']))
            if not cp:raise ValueError('Latest terminal round has no saved checkpoint; run/select first')
            count=len(s.labels(repeat,strategy))
            attempts=s.db.execute('SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
            pool=s.db.execute("SELECT COUNT(*) FROM partitions p WHERE p.repeat=? AND p.origin='pool' AND NOT EXISTS (SELECT 1 FROM selections q WHERE q.repeat=p.repeat AND q.strategy=? AND q.id=p.id)",(repeat,strategy)).fetchone()[0]
            if latest['round']<config['budgets']['rounds'] and count<config['budgets']['max_labels'] and attempts<config['budgets']['max_attempts'] and pool:
                raise ValueError('Scheduled acquisitions incomplete; final-test values cannot stop acquisition')
        with s.transaction():s.set_meta('finalization',{**contract,'status':'started'})
    crash('after_finalization_freeze',failpoint)
    features=read_features(s);tables={};all_rows=[];metric_rows=[]
    for cp in checkpoints:
        repeat,strategy,round_number=cp['repeat'],cp['strategy'],cp['round']
        if repeat not in tables:
            ids=[r['id'] for r in s.rows("SELECT id FROM partitions WHERE repeat=? AND origin='final' ORDER BY id",(repeat,))]
            tables[repeat]=(ids,EvaluationOracle(s.root/f'repeat_{repeat}/oracle_final.json',ids).read(ids))
        ids,table=tables[repeat]
        sha=s.rows('SELECT sha256 FROM artifacts WHERE path=?',(cp['path'],))[0]['sha256']
        bundle=models.load(s.root/cp['path'],sha,config,features.settings_hash)
        p=models.predict(bundle,features,ids,config['acquisition']['chunk_size']);meta=json.loads(cp['metadata'])
        all_rows.extend(prediction_rows(s,repeat,strategy,round_number,sha,ids,table,p,meta['label_count'],meta['attempt_count'],'final'))
        for evaluator in list(bundle['evaluators'])+['committee']:
            metric_rows.append((repeat,strategy,round_number,evaluator,'final',canonical(metrics([table[i]['gap_ev'] for i in ids],p[evaluator],p['uncertainty'] if evaluator=='committee' else None))))
    with s.transaction(failpoint):
        sha=s.artifact('final/predictions.json',canonical(all_rows).encode(),'final_predictions',failpoint)
        s.artifact('final/metrics.json',canonical([{'repeat':r,'strategy':st,'round':rd,'evaluator':e,'partition':p,**json.loads(m)} for r,st,rd,e,p,m in metric_rows]).encode(),'final_metrics',failpoint)
        for row in metric_rows:s.db.execute('INSERT INTO metrics VALUES (?,?,?,?,?,?)',row)
        result={**contract,'status':'completed','predictions_sha256':sha}
        s.set_meta('finalization',result);s.db.execute('UPDATE runs SET finalized=1')
        s.event('finalized',result)
    return result
