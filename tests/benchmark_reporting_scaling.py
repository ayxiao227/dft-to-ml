"""Bounded synthetic report workloads. Each invocation must be a fresh process.

Examples (Linux):
  timeout 120s uv run --no-sync python tests/benchmark_reporting_scaling.py diversity --impl new --ids 119452 --selected 100 --references 1100
  timeout 120s uv run --no-sync python tests/benchmark_reporting_scaling.py predictions --impl new --rows 5000 --repeats 5 --checkpoints 4

Old path sources are read with `git show f0089d0:...`; no checkout/branch mutation.
No research data, training, or prediction generation is used. Synthetic saved
prediction rows and random binary vectors exercise the actual diagnostic functions.
CPU limit 90 seconds, address-space limit 6 GiB, wall timeout supplied by caller.
Disk is sampled every 50 ms; small transient files may fall between samples.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import resource
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.active_learning.features import CandidateFeatures
from src.active_learning.acquisition import distances
from src.active_learning.report_diagnostics import batch_diversity, Chemistry, ranking, rf_summary, FingerprintLookup
from src.active_learning.report_cache import StageCache
from src.active_learning.report_storage import Predictions


def old_class(path,name):
    namespace={'__package__':'src.active_learning'}
    source=subprocess.check_output(['git','show','f0089d0:'+path],cwd=ROOT,text=True)
    exec(compile(source,path,'exec'),namespace)
    return namespace[name]


def diversity(args,root):
    path=root/'fingerprints.npy'
    fp=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(args.ids,args.bits))
    rng=np.random.default_rng(17)
    for start in range(0,args.ids,1024):fp[start:start+1024]=rng.integers(0,2,fp[start:start+1024].shape,dtype=np.uint8)
    fp.flush()
    cls=CandidateFeatures if args.impl=='new' else old_class('src/active_learning/features.py','CandidateFeatures')
    f=cls(tuple(map(str,range(args.ids))),('CC',)*args.ids,(),np.empty((args.ids,0)),fp,'fixture')
    selected=list(map(str,range(args.references,args.references+args.selected)));refs=list(map(str,range(args.references)))
    start=time.perf_counter()
    if args.rounds>1 or args.runs>1:
        if args.impl!='new':raise ValueError('Large schedule is new-only; bound old comparisons separately')
        out=[];lookup=FingerprintLookup(f)
        for run in range(args.runs):
            sequence=rng.permutation(args.ids)[:args.references+args.rounds*args.selected].astype(str).tolist()
            refs=sequence[:args.references]
            for rd in range(args.rounds):
                selected=sequence[args.references+rd*args.selected:args.references+(rd+1)*args.selected]
                out.extend(batch_diversity(f,selected,refs,lookup=lookup));refs.extend(selected)
    elif args.impl=='new':out=batch_diversity(f,selected,refs)
    else:
        out=[]
        for ident in selected:
            one=f.rows([ident],True)[0]
            out.append(min((float(distances(f.rows([ref],True),one,1024)[0]) for ref in refs),default=1.));refs.append(ident)
    return {'diagnostic_seconds':time.perf_counter()-start,'result_sha':hashlib.sha256(np.asarray(out).tobytes()).hexdigest(),'comparisons':args.runs*sum(args.selected*(args.references+rd*args.selected)+args.selected*(args.selected-1)//2 for rd in range(args.rounds))}


def predictions(args,root):
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE checkpoints(repeat INTEGER,strategy TEXT,round INTEGER,predictions TEXT)')
    # Preparation is measured separately, and writes bounded chunks only.
    for repeat in range(args.repeats):
        for rd in range(args.checkpoints):
            path=root/f'{repeat}_{rd}.jsonl'
            with path.open('w') as out:
                for evaluator in ['rf','committee']:
                    for i in range(args.rows):
                        y=float(i%13);p=y+float(i%5)*.1
                        row={'repeat':repeat,'strategy':'random','round':rd,'partition':'validation','source_record_id':f'mol{i:07}',
                             'input_smiles':['CC','CCC','C=O'][i%3],'evaluator':evaluator,'observed_gap_ev':y,'predicted_gap_ev':p,
                             'residual_ev':p-y,'absolute_error_ev':abs(p-y),'uncertainty_ev':float(i%17) if evaluator=='committee' else None,'units':'eV'}
                        out.write(json.dumps(row)+'\n')
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?)',(repeat,'random',rd,path.name))
    s=SimpleNamespace(root=root,db=db,meta=lambda key,default=None:default)
    work=root/'work';work.mkdir();start=time.perf_counter()
    if args.impl=='new':
        cache=StageCache(root,'benchmark');p=Predictions(s,work,cache)
        chem=Chemistry(work/'chemistry.sqlite3')
        try:
            summary=rf_summary(p,('validation','random'),chem,20000,271828)
            totals=[(r[0],r[2],r[3]) for r in summary['subgroups'] if r[1]=='all']
            ranked=[ranking(p,key) for key in p.keys('committee',True)]
        finally:chem.close();p.close();cache.close();db.close()
    else:
        import pandas as pd
        from src.descriptors import input_features
        old=old_class('src/active_learning/report_storage.py','Predictions');p=old(s,work)
        totals=[];ranked=[]
        try:
            for _,group in p.groups('rf'):
                for repeat,g in group.groupby('repeat'):
                    total=0.
                    for row in g.itertuples():input_features(row.input_smiles);total+=row.absolute_error_ev
                    totals.append((int(repeat),len(g),total))
                # Same allocation/copies as old diagnostic sorting; worst is full-population.
                group.sort_values(['absolute_error_ev','source_record_id'],ascending=[False,True]).head(12)
            for _,group in p.groups('committee',True):
                ordered=group.sort_values(['uncertainty_ev','source_record_id'])
                ranked.append([float(ordered.head(max(1,int(np.ceil(len(ordered)*f)))).absolute_error_ev.mean()) for f in [.1,.25,.5,.75,1.]])
        finally:p.close();db.close()
    return {'diagnostic_seconds':time.perf_counter()-start,'totals':totals,'prediction_rows':args.rows*args.repeats*args.checkpoints*2}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['diversity','predictions'])
    p.add_argument('--impl',choices=['old','new'],required=True)
    for name,default in [('ids',10000),('bits',2048),('selected',40),('references',200),('rows',3000),('repeats',1),('checkpoints',2),('rounds',1),('runs',1)]:p.add_argument('--'+name,type=int,default=default)
    args=p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(90,90));resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='report-benchmark-') as tmp:
        root=Path(tmp);done=threading.Event();peak=[0];work_peak=[0]
        def sample():
            while not done.wait(.05):
                try:
                    sizes=[(p,p.stat().st_size) for p in root.rglob('*') if p.is_file()]
                    peak[0]=max(peak[0],sum(size for _,size in sizes))
                    work_peak[0]=max(work_peak[0],sum(size for p,size in sizes if p.relative_to(root).parts[0] in ['work','.report-work']))
                except FileNotFoundError:pass
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        with contextlib.redirect_stdout(io.StringIO()):result=diversity(args,root) if args.mode=='diversity' else predictions(args,root)
        done.set();thread.join();final=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());peak[0]=max(peak[0],final)
        print(json.dumps({'settings':vars(args),'wall_seconds':time.perf_counter()-start,
                          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'sampled_peak_disk_bytes':peak[0],'sampled_peak_work_bytes':work_peak[0],'final_disk_bytes':final,**result},sort_keys=True),flush=True)
if __name__=='__main__':main()
