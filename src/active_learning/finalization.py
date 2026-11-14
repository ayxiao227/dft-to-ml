"""Checkpoint-durable final evaluation of the unchanged frozen scientific contract."""
import json
import time
import numpy as np
from .config import canonical,hashed
from .experiment import read_features
from .oracle import EvaluationOracle
from .state import crash
from .artifacts import file_hash,publish_stream
from .streaming import meta_values,prediction_records,rows_hash
from . import models
from .evaluation import metrics,iter_prediction_rows

VERSION=2
EVALUATORS=('mean','ridge','rf','committee')
PREFIX='final_checkpoint/'


def identity(cp):return {k:cp[k] for k in ('repeat','strategy','round','path')}


def paths(cp):
    base=f"final/checkpoints/{cp['repeat']}/{cp['strategy']}/{cp['round']}"
    return base+'.predictions.jsonl',base+'.result.json'


def checkpoint_refs(s):
    # metadata embeds every validation ID: keep only small descriptors across checkpoints.
    return s.rows('SELECT repeat,strategy,round,path,predictions FROM checkpoints ORDER BY repeat,strategy,round')


def checkpoint_counts(s,cp):
    row=s.db.execute("SELECT json_extract(metadata,'$.label_count'),json_extract(metadata,'$.attempt_count') FROM checkpoints WHERE repeat=? AND strategy=? AND round=?",
                     (cp['repeat'],cp['strategy'],cp['round'])).fetchone()
    return {'label_count':row[0],'attempt_count':row[1]}


def contract_for(s,checkpoints):
    return {'experiment_hash':s.meta('experiment_hash'),'config_hash':s.meta('config_hash'),
        'partition_hashes':list(meta_values(s,'partitions','item.hash')),'checkpoint_set_hash':rows_hash(dict(r) for r in s.db.execute('SELECT * FROM checkpoints ORDER BY repeat,strategy,round'))}


def completion_key(cp):return PREFIX+hashed(identity(cp))


def safe_path(s,relative):
    path=(s.root/relative).resolve()
    if not path.is_relative_to(s.root):raise ValueError('Unsafe final artifact path: '+relative)
    return path


def read_manifest(path):
    try:
        doc=json.loads(path.read_text())
        if not isinstance(doc,dict):raise ValueError('Expected manifest object')
        return doc
    except (ValueError,OSError) as exc:
        raise ValueError(f'Cannot read final manifest; evidence retained at {path}. Restore a verified original or investigate before retrying: {exc}') from exc


def load_result(s,cp,contract,reference=None):
    """Verify identities and bytes. A file alone never establishes completion."""
    pred_path,result_path=paths(cp)
    path=safe_path(s,result_path)
    if reference is not None:
        if reference.get('path')!=result_path or file_hash(path)!=reference.get('sha256'):
            raise ValueError('Final manifest hash/identity changed; retain artifacts and restore verified originals: '+result_path)
    doc=read_manifest(path)
    sha=s.db.execute('SELECT sha256 FROM artifacts WHERE path=?',(cp['path'],)).fetchone()[0]
    expected={'version':VERSION,'status':'completed','checkpoint':identity(cp),'model_sha256':sha,
              'contract_hash':hashed(contract)}
    if any(doc.get(k)!=v for k,v in expected.items()):raise ValueError('Final checkpoint manifest contract/identity changed: '+result_path)
    if (not isinstance(doc.get('predictions'),dict) or not isinstance(doc.get('metrics'),dict)
            or any(not isinstance(m,dict) for m in doc['metrics'].values())):
        raise ValueError('Corrupt final manifest fields; evidence retained: '+result_path)
    artifact=doc['predictions'];n=s.db.execute("SELECT COUNT(*) FROM partitions WHERE repeat=? AND origin='final'",(cp['repeat'],)).fetchone()[0]
    if (artifact.get('path')!=pred_path or artifact.get('row_count')!=n*len(EVALUATORS)
            or set(doc['metrics'])!=set(EVALUATORS) or any(m.get('n')!=n for m in doc['metrics'].values())
            or file_hash(safe_path(s,pred_path))!=artifact.get('sha256')):
        raise ValueError('Final prediction artifact/count/hash changed; retain evidence: '+pred_path)
    if reference is not None:
        for relative,digest in [(result_path,reference['sha256']),(pred_path,artifact['sha256'])]:
            row=s.db.execute('SELECT sha256 FROM artifacts WHERE path=?',(relative,)).fetchone()
            if row is None or row[0]!=digest:raise ValueError('Missing/conflicting committed final artifact: '+relative)
        saved={r['evaluator']:json.loads(r['data']) for r in s.db.execute(
            "SELECT evaluator,data FROM metrics WHERE repeat=? AND strategy=? AND round=? AND partition='final'",
            (cp['repeat'],cp['strategy'],cp['round']))}
        if saved!=doc['metrics']:raise ValueError('Final checkpoint metrics/manifest mismatch')
    return doc


def verified_results(s,checkpoints=None,contract=None):
    checkpoints=checkpoints if checkpoints is not None else checkpoint_refs(s)
    contract=contract if contract is not None else contract_for(s,checkpoints)
    existing=s.meta('finalization',{})
    if existing and any(existing.get(k)!=v for k,v in contract.items()):raise ValueError('Frozen finalization contract changed')
    results={}
    verify_started=time.monotonic()
    for index,cp in enumerate(checkpoints,1):
        reference=s.meta(completion_key(cp))
        if reference is not None:results[completion_key(cp)]=load_result(s,cp,contract,reference)
        if index%25==0:print(f'Final manifest verification {index}/{len(checkpoints)} elapsed={time.monotonic()-verify_started:.1f}s',flush=True)
    stored=s.db.execute('SELECT COUNT(*) FROM meta WHERE key LIKE ?',(PREFIX+'%',)).fetchone()[0]
    if stored!=len(results):raise ValueError('Final completion record outside frozen checkpoint set')
    if existing and existing.get('status') not in ('started','completed'):raise ValueError('Invalid finalization completion state')
    if existing and existing.get('format_version') not in (None,VERSION):raise ValueError('Unsupported finalization manifest version')
    if existing.get('format_version')==VERSION:
        if existing.get('total_checkpoints')!=len(checkpoints):raise ValueError('Frozen final checkpoint count mismatch')
        if existing.get('completed_checkpoints')!=len(results):raise ValueError('Final committed progress count mismatch')
        final_metrics=s.db.execute("SELECT COUNT(*) FROM metrics WHERE partition='final'").fetchone()[0]
        if final_metrics!=len(results)*len(EVALUATORS):raise ValueError('Uncommitted/duplicate final metrics')
        if existing['status']=='completed':
            if len(results)!=len(checkpoints):raise ValueError('Incomplete finalization marked completed')
            expected=overall_manifest(s,checkpoints,contract)
            manifest=safe_path(s,existing['manifest_path'])
            if file_hash(manifest)!=existing['manifest_sha256'] or read_manifest(manifest)!=expected:
                raise ValueError('Final overall manifest changed')
    return results


def progress(s):
    state=s.meta('finalization');total=s.db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]
    if state and state.get('format_version')==VERSION:done=len(verified_results(s))
    else:done=total if state and state['status']=='completed' else 0
    return {'total':total,'completed':done,'remaining':total-done,'verification':'full integrity and committed artifact verification'}


def overall_manifest(s,checkpoints,contract):
    return {'version':VERSION,'status':'completed','contract':contract,
            'checkpoints':[{'checkpoint':identity(cp),**s.meta(completion_key(cp))} for cp in checkpoints]}


def verify_orphan_rows(s,cp,doc,ids,table):
    """Recover a published but uncommitted result without trusting its metrics."""
    meta=checkpoint_counts(s,cp);observed=np.asarray([table[i]['gap_ev'] for i in ids]);it=iter(prediction_records(safe_path(s,doc['predictions']['path'])))
    for evaluator in EVALUATORS:
        values=np.empty(len(ids));unc=np.empty(len(ids)) if evaluator=='committee' else None
        for i,ident in enumerate(ids):
            row=next(it,None)
            if row is None:raise ValueError('Orphan final predictions truncated')
            record=json.loads(s.db.execute('SELECT record FROM molecules WHERE id=?',(ident,)).fetchone()[0])
            expected={'repeat':cp['repeat'],'strategy':cp['strategy'],'round':cp['round'],'partition':'final',
                'label_count':meta['label_count'],'attempt_count':meta['attempt_count'],'source_record_id':ident,
                'checkpoint_hash':doc['model_sha256'],'evaluator':evaluator,'observed_gap_ev':table[ident]['gap_ev'],
                'units':'eV','input_smiles':record['input_smiles'],'input_structure_key':record['input_structure_key']}
            if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Orphan final prediction membership/metadata conflict')
            p=row['predicted_gap_ev'];residual=p-table[ident]['gap_ev']
            if row['residual_ev']!=residual or row['absolute_error_ev']!=abs(residual):raise ValueError('Orphan final residual conflict')
            values[i]=p
            if unc is not None:unc[i]=row['uncertainty_ev']
            elif row['uncertainty_ev'] is not None:raise ValueError('Unexpected evaluator uncertainty')
        if metrics(observed,values,unc)!=doc['metrics'][evaluator]:raise ValueError('Orphan final metrics conflict')
    if next(it,None) is not None:raise ValueError('Extra orphan final prediction records')


def finalize(s,failpoint=None):
    config=s.meta('config');existing=s.meta('finalization')
    checkpoints=checkpoint_refs(s)
    if not checkpoints:raise ValueError('No checkpoints available')
    print('Verifying frozen finalization contract and committed checkpoints',flush=True)
    contract=contract_for(s,checkpoints)
    if existing:
        if any(existing.get(k)!=v for k,v in contract.items()):raise ValueError('Frozen finalization contract changed')
        if existing['status']=='completed':
            verified_results(s,checkpoints,contract)
            print('Finalization already completed; verified saved results; no prediction required',flush=True)
            return existing
    else:
        if s.db.execute("SELECT 1 FROM attempts WHERE status IN ('reserved','prepared','pending') LIMIT 1").fetchone():
            raise ValueError('Outstanding calculations prevent finalization')
        for run in s.rows("SELECT * FROM runs WHERE strategy!='__evaluation__'"):
            repeat,strategy=run['repeat'],run['strategy']
            latest=s.rows('SELECT * FROM rounds WHERE repeat=? AND strategy=? ORDER BY round DESC LIMIT 1',(repeat,strategy))[0]
            cp=s.rows('SELECT 1 FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(repeat,strategy,latest['round']))
            if not cp:raise ValueError('Latest terminal round has no saved checkpoint; run/select first')
            count=s.db.execute('SELECT COUNT(*) FROM labels WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
            attempts=s.db.execute('SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
            pool=s.db.execute("SELECT COUNT(*) FROM partitions p WHERE p.repeat=? AND p.origin='pool' AND NOT EXISTS (SELECT 1 FROM selections q WHERE q.repeat=p.repeat AND q.strategy=? AND q.id=p.id)",(repeat,strategy)).fetchone()[0]
            if latest['round']<config['budgets']['rounds'] and count<config['budgets']['max_labels'] and attempts<config['budgets']['max_attempts'] and pool:
                raise ValueError('Scheduled acquisitions incomplete; final-test values cannot stop acquisition')
    if not existing or 'format_version' not in existing:
        if s.db.execute("SELECT 1 FROM metrics WHERE partition='final' LIMIT 1").fetchone():raise ValueError('Legacy started finalization has unexpected committed final metrics; retain database for investigation')
        with s.transaction():s.set_meta('finalization',{**contract,'status':'started','format_version':VERSION,'completed_checkpoints':0,'total_checkpoints':len(checkpoints)})
    crash('after_finalization_freeze',failpoint)
    completed=verified_results(s,checkpoints,contract);done=len(completed)
    print(f'Verification completed. Checkpoints: total={len(checkpoints)}, already completed={done}, remaining={len(checkpoints)-done}',flush=True)
    features=None;table=None;ids=None;current_repeat=None;elapsed=0.;timed=0
    for cp in checkpoints:
        if completion_key(cp) in completed:continue
        started=time.perf_counter();repeat,strategy,round_number=cp['repeat'],cp['strategy'],cp['round']
        print(f'Finalizing repeat={repeat} strategy={strategy} round={round_number} checkpoint={cp["path"]}',flush=True)
        if current_repeat!=repeat:
            table=None  # release the previous repeat before reading the next oracle
            ids=[r[0] for r in s.db.execute("SELECT id FROM partitions WHERE repeat=? AND origin='final' ORDER BY id",(repeat,))]
            table=EvaluationOracle(s.root/f'repeat_{repeat}/oracle_final.json',ids).read(ids);current_repeat=repeat
        pred_path,result_path=paths(cp)
        if safe_path(s,result_path).exists():
            doc=load_result(s,cp,contract)
            verify_orphan_rows(s,cp,doc,ids,table)
        else:
            if features is None:features=read_features(s)
            sha=s.db.execute('SELECT sha256 FROM artifacts WHERE path=?',(cp['path'],)).fetchone()[0]
            bundle=models.load(s.root/cp['path'],sha,config,features.settings_hash)
            p=models.predict(bundle,features,ids,config['acquisition']['chunk_size']);meta=checkpoint_counts(s,cp)
            observed=np.asarray([table[i]['gap_ev'] for i in ids])
            metric_data={e:metrics(observed,p[e],p['uncertainty'] if e=='committee' else None) for e in EVALUATORS}
            def write(out):
                for row in iter_prediction_rows(s,repeat,strategy,round_number,sha,ids,table,p,meta['label_count'],meta['attempt_count'],'final'):
                    out.write(canonical(row));out.write('\n')
            pred_sha=publish_stream(safe_path(s,pred_path),write,failpoint)
            del p,bundle,observed
            doc={'version':VERSION,'status':'completed','checkpoint':identity(cp),'model_sha256':sha,'contract_hash':hashed(contract),
                 'predictions':{'path':pred_path,'sha256':pred_sha,'row_count':len(ids)*len(EVALUATORS)},'metrics':metric_data}
            publish_stream(safe_path(s,result_path),lambda out:json.dump(doc,out,sort_keys=True,separators=(',',':'),allow_nan=False))
        crash('after_final_result_publication',failpoint)
        with s.transaction(failpoint):
            s.artifact_file(pred_path,'final_predictions')
            result_sha=s.artifact_file(result_path,'final_checkpoint_manifest')
            for evaluator,data in doc['metrics'].items():
                s.db.execute('INSERT INTO metrics VALUES (?,?,?,?,?,?)',(repeat,strategy,round_number,evaluator,'final',canonical(data)))
            s.set_meta(completion_key(cp),{'path':result_path,'sha256':result_sha})
            done+=1
            s.set_meta('finalization',{**contract,'status':'started','format_version':VERSION,'completed_checkpoints':done,'total_checkpoints':len(checkpoints)})
            s.event('final_checkpoint_completed',{'checkpoint':identity(cp),'manifest_sha256':result_sha})
        crash('after_final_checkpoint_commit',failpoint)
        duration=time.perf_counter()-started;elapsed+=duration;timed+=1
        print(f'Durable checkpoint completed {done}/{len(checkpoints)}; elapsed={duration:.1f}s; approximate ETA={(len(checkpoints)-done)*elapsed/timed:.1f}s',flush=True)
    verified_results(s,checkpoints,contract)
    manifest=overall_manifest(s,checkpoints,contract);relative='final/manifest.json'
    manifest_sha=publish_stream(safe_path(s,relative),lambda out:json.dump(manifest,out,sort_keys=True,separators=(',',':'),allow_nan=False))
    crash('after_final_manifest_publication',failpoint)
    with s.transaction():
        s.artifact_file(relative,'final_manifest')
        result={**contract,'status':'completed','format_version':VERSION,'completed_checkpoints':done,'total_checkpoints':len(checkpoints),
                'manifest_path':relative,'manifest_sha256':manifest_sha}
        s.set_meta('finalization',result);s.db.execute('UPDATE runs SET finalized=1');s.event('finalized',result)
        crash('before_finalization_commit',failpoint)
    crash('after_finalization_commit',failpoint)
    print(f'Finalization completed: {done} checkpoints. Manifest: {s.root/relative}; predictions: {s.root/"final/checkpoints"}',flush=True)
    return result
