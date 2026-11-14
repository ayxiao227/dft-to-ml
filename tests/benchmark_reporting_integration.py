"""Full tiny report comparison with acquisition rounds and completed finalization.

Run `prepare /tmp/report-integration` once, then old/new in separate subprocesses:
  timeout 120s uv run --no-sync python tests/benchmark_reporting_integration.py old /tmp/report-integration
  timeout 120s uv run --no-sync python tests/benchmark_reporting_integration.py new /tmp/report-integration
Preparation trains only bounded synthetic fixture models. Reporting forbids fit,
predict and oracle access. This is not a large-group or research benchmark.
"""
import argparse
import contextlib
import io
import json
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_al_state import fixture
from benchmark_reporting_scaling import old_class
from src.active_learning.experiment import initialize
from src.active_learning.state import Store
from src.active_learning.runner import run
from src.active_learning.finalization import finalize
from src.active_learning.reporting import report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','old','new']);p.add_argument('root',type=Path)
    args=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(90,90));resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    if args.mode=='prepare':
        args.root.mkdir(parents=True,exist_ok=False);config=fixture(args.root);config['repeat_seeds']=[42,43]
        path=initialize(config,args.root/'base')
        with Store(path) as s:run(s);finalize(s)
        for name in ['old','new']:shutil.copytree(path,args.root/name)
        return
    path=args.root/args.mode;done=threading.Event();peak=[0];work_peak=[0]
    baseline=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    def sample():
        while not done.wait(.02):
            try:
                peak[0]=max(peak[0],sum(p.stat().st_size for p in path.rglob('*') if p.is_file())-baseline)
                work_peak[0]=max(work_peak[0],sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and (p.relative_to(path).parts[0]=='.report-work' or p.relative_to(path).parts[0].startswith('.report.'))))
            except FileNotFoundError:pass
    thread=threading.Thread(target=sample,daemon=True);thread.start();start=time.perf_counter()
    old_report=old_class('src/active_learning/reporting.py','report') if args.mode=='old' else None
    if old_report:old_report.__globals__['Predictions']=old_class('src/active_learning/report_storage.py','Predictions')
    with contextlib.redirect_stdout(io.StringIO()),Store(path) as s,patch('src.active_learning.models.fit',side_effect=AssertionError('fit forbidden')),patch('src.active_learning.models.predict',side_effect=AssertionError('predict forbidden')),patch('src.active_learning.oracle.EvaluationOracle.read',side_effect=AssertionError('oracle forbidden')):
        if old_report:
            old_features=old_class('src/active_learning/features.py','CandidateFeatures')
            with patch('src.active_learning.experiment.CandidateFeatures',old_features):result=old_report(s)
        else:result=report(s)
        counts={table:s.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['checkpoints','selections']}
    done.set();thread.join()
    print(json.dumps({'mode':args.mode,'wall_seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'sampled_peak_extra_disk_bytes':peak[0],'sampled_peak_work_bytes':work_peak[0],**counts,**result}),flush=True)
if __name__=='__main__':main()
