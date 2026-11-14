"""Isolated synthetic memory workload, not a scientific experiment.

Run with the repository environment, e.g.:
  uv run python tests/benchmark_finalization_memory.py prepare /tmp/al-memory-4 --checkpoints 4 --rows 3000
  uv run python tests/benchmark_finalization_memory.py finalize /tmp/al-memory-4
  uv run python tests/benchmark_finalization_memory.py verify /tmp/al-memory-4
  uv run python tests/benchmark_finalization_memory.py report /tmp/al-memory-4

Each invocation is a fresh process and emits Linux peak process RSS via getrusage.
Preparation trains ONE tiny fixture model. It constructs a separate ledger containing
repeated feature rows and repeated references to that model, never research training.
The measured finalize/verify/report paths are real, with fitting prohibited.
"""
import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_al_state import fixture
from src.active_learning.state import Store
from src.active_learning.config import canonical,hashed
from src.active_learning.experiment import initialize,read_features
from src.active_learning.runner import run
from src.active_learning.evaluation import metrics,iter_prediction_rows
from src.active_learning.artifacts import publish_stream
from src.active_learning.features import CandidateFeatures
from src.active_learning import models


def prepare(root,count,n):
    import tempfile
    if root.exists():raise ValueError('Use a new empty benchmark directory')
    root.mkdir(parents=True)
    config=fixture(root);config['acquisition']['strategies']=['random']
    config['budgets'].update(max_labels=4,max_attempts=4,rounds=0)
    baseline=initialize(config,root/'tiny_training_fixture')
    with Store(baseline) as original:
        run(original)
        cp=original.rows('SELECT * FROM checkpoints')[0]
        old_features=read_features(original)
        records=[json.loads(r['record']) for r in original.rows('SELECT * FROM molecules ORDER BY id')]
        extra=[];extra_rows=[];table={};membership=[]
        for role in ['validation','final']:
            for i in range(n):
                template=records[i%len(records)];record=dict(template)
                ident=f'memory_{role}_{i:07}'
                record['source_record_id']=ident;extra.append(record)
                extra_rows.append(i%len(records));membership.append({'source_record_id':ident,'origin':role,'group_key':'synthetic_memory_only'})
                table[ident]={'gap_ev':1.+(i%37)*.03}
        ids=tuple(old_features.ids)+tuple(r['source_record_id'] for r in extra)
        smiles=tuple(old_features.smiles)+tuple(r['input_smiles'] for r in extra)
        target=root/'experiment';target.mkdir()
        with Store(target,create=True) as s,s.transaction():
            for row in original.rows("SELECT * FROM meta WHERE key!='integrity'"):s.set_meta(row['key'],json.loads(row['value']))
            # New artificial frozen partitions. No data from a live experiment is used.
            partition={'rows':membership,'hash':hashed(membership),'sizes':{'initial':0,'pool':0,'validation':n,'final':n},
                       'achieved_fractions':{'validation':.5,'final':.5}}
            s.set_meta('partitions',[partition]);s.set_meta('experiment_hash',hashed([config,[partition['hash']]]))
            s.set_meta('attrition',{'accepted':len(ids),'rejected':0})
            s.set_meta('features',{'ids':ids,'smiles':smiles,'names':old_features.names,'settings_hash':old_features.settings_hash})
            for name,array in [('features.npy',old_features.x),('binary.npy',old_features.binary)]:
                dest=np.lib.format.open_memmap(target/name,mode='w+',dtype=array.dtype,shape=(len(ids),array.shape[1]))
                dest[:len(array)]=array
                for start in range(0,len(extra_rows),1024):dest[len(array)+start:len(array)+start+1024]=array[extra_rows[start:start+1024]]
                dest.flush();del dest;s.artifact_file(name,'input_features' if name=='features.npy' else 'binary_diversity')
            for record in records+extra:s.db.execute('INSERT INTO molecules VALUES (?,?)',(record['source_record_id'],canonical(record)))
            for r in membership:s.db.execute('INSERT INTO partitions VALUES (0,?,?,?)',(r['source_record_id'],r['origin'],r['group_key']))
            s.db.execute("INSERT INTO runs VALUES (0,'random',0)")
            s.artifact('curation.json',b'[]','attrition')
            s.artifact('repeat_0/partitions.json',canonical(partition).encode(),'partitions')
            for role in ['validation','final']:
                subset={r['source_record_id']:table[r['source_record_id']] for r in membership if r['origin']==role}
                s.artifact(f'repeat_0/oracle_{role}.json',canonical(subset).encode(),'isolated_oracle_'+role)
            model_sha=s.artifact(cp['path'],(original.root/cp['path']).read_bytes(),'model')
            features=read_features(s);validation_ids=sorted(i for i in table if i.startswith('memory_validation'))
            bundle=models.load(s.root/cp['path'],model_sha,config,features.settings_hash)
            predictions=models.predict(bundle,features,validation_ids,config['acquisition']['chunk_size'])
            for rd in range(count):
                s.db.execute('INSERT INTO rounds VALUES (0,?,?,?,?)',('random',rd,'terminal',canonical({'memory_fixture':True})))
                relative=f'validation/{rd}.json'
                def write(out):
                    out.write('[');first=True
                    for row in iter_prediction_rows(s,0,'random',rd,model_sha,validation_ids,table,predictions,4,4,'validation'):
                        if not first:out.write(',')
                        out.write(canonical(row));first=False
                    out.write(']')
                publish_stream(s.root/relative,write);s.artifact_file(relative,'validation_predictions')
                s.db.execute('INSERT INTO checkpoints VALUES (0,?,?,?,?,?)',('random',rd,cp['path'],relative,cp['metadata']))
                for evaluator in ['mean','ridge','rf','committee']:
                    m=metrics([table[i]['gap_ev'] for i in validation_ids],predictions[evaluator],predictions['uncertainty'] if evaluator=='committee' else None)
                    s.db.execute('INSERT INTO metrics VALUES (0,?,?,?,?,?)',('random',rd,evaluator,'validation',canonical(m)))
            s.event('synthetic_memory_fixture',{'checkpoints':count,'rows_per_partition':n,'model':'one repeated tiny trained bundle'})
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','finalize','verify','report','legacy-finalize','legacy-report'])
    p.add_argument('root',type=Path);p.add_argument('--checkpoints',type=int,default=4);p.add_argument('--rows',type=int,default=3000);p.add_argument('--legacy-report-source',type=Path)
    a=p.parse_args();start=time.perf_counter()
    if a.stage=='prepare':prepare(a.root,a.checkpoints,a.rows)
    else:
        with Store(a.root/'experiment') as s,patch.object(models,'fit',side_effect=AssertionError('Benchmark must not fit')):
            if a.stage=='finalize':
                from src.active_learning.finalization import finalize
                finalize(s)
            elif a.stage=='report':
                from src.active_learning.reporting import report
                report(s)
            elif a.stage=='legacy-finalize':
                from legacy_finalization_reference import finalize
                finalize(s)
            elif a.stage=='legacy-report':
                if a.legacy_report_source is None:raise ValueError('Pass --legacy-report-source extracted from a1a60b9')
                namespace={'__package__':'src.active_learning'}
                exec(compile(a.legacy_report_source.read_text(),str(a.legacy_report_source),'exec'),namespace)
                namespace['report'](s)

    print(json.dumps({'stage':a.stage,'wall_seconds':round(time.perf_counter()-start,3),
                      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},sort_keys=True),flush=True)

if __name__=='__main__':main()
