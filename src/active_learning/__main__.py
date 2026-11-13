"""Reproducible input-only molecular active learning; local preparation only."""
import argparse
import json
from pathlib import Path
from .config import load,canonical,hashed
from .experiment import initialize
from .state import Store
from .runner import run,select
from .gaussian import prepare,ingest,retry
from .costs import coverage


def status(s):
    return {'experiment_hash':s.meta('experiment_hash'),'config_hash':s.meta('config_hash'),
        'mode':s.meta('config')['mode'],'attrition':s.meta('attrition'),
        'partitions':[{'sizes':p['sizes'],'achieved_fractions':p['achieved_fractions'],'hash':p['hash']} for p in s.meta('partitions')],
        'available_pool':s.rows("SELECT r.repeat,r.strategy,COUNT(p.id) AS available FROM runs r LEFT JOIN partitions p ON p.repeat=r.repeat AND p.origin='pool' AND NOT EXISTS (SELECT 1 FROM selections q WHERE q.repeat=r.repeat AND q.strategy=r.strategy AND q.id=p.id) WHERE r.strategy!='__evaluation__' GROUP BY r.repeat,r.strategy ORDER BY r.repeat,r.strategy"),
        'attempt_statuses':s.rows('SELECT repeat,strategy,status,COUNT(*) AS count FROM attempts GROUP BY repeat,strategy,status ORDER BY repeat,strategy,status'),
        'cost_coverage':coverage(s.rows('SELECT * FROM attempts')),'finalization':s.meta('finalization')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init',help='Curate local source and freeze experiment; never downloads')
    p.add_argument('config');p.add_argument('experiment')
    for command in ['run','select','prepare','ingest','status','retry','finalize','report']:
        p=sub.add_parser(command,help={'run':'Retrospective synchronous rounds only','select':'Commit a pool reservation; pending rounds block new selection',
            'prepare':'Write opt/freq inputs/scripts only; no submission','ingest':'Validate explicit attempt/log associations',
            'status':'Read states, overhead and cost coverage','retry':'Reserve explicit new attempt with audited reason',
            'finalize':'Freeze checkpoint set and evaluate saved models once','report':'Render saved artifacts without fitting'}[command])
        p.add_argument('experiment');p.add_argument('--config',help='Optional expected config; drift is refused')
        if command in ['select','prepare','ingest','retry']:
            p.add_argument('--repeat',type=int,default=0);p.add_argument('--strategy',required=True)
        if command=='run':p.add_argument('--stop-after',type=int,help='Pause after this many deliveries for local resume checks')
        if command=='prepare':p.add_argument('--mark-pending',action='store_true',help='Record that prepared calculations were manually dispatched')
        if command=='ingest':p.add_argument('--outcomes',required=True,help='JSON list: source_record_id, attempt, log_path or explicit outcome/reason, optional resource_usage')
        if command=='retry':p.add_argument('--id',required=True);p.add_argument('--reason',required=True)
    args=parser.parse_args()
    try:
        if args.command=='init':
            print(initialize(load(args.config),args.experiment));return
        with Store(args.experiment) as s:
            if args.config and hashed(load(args.config))!=s.meta('config_hash'):raise ValueError('Config drift; create a new experiment')
            result=None
            if args.command=='run':run(s,args.stop_after);result=status(s)
            elif args.command=='select':result=select(s,args.repeat,args.strategy)
            elif args.command=='prepare':result=prepare(s,args.repeat,args.strategy,args.mark_pending)
            elif args.command=='ingest':result=ingest(s,args.repeat,args.strategy,json.loads(Path(args.outcomes).read_text()))
            elif args.command=='status':result=status(s)
            elif args.command=='retry':result=retry(s,args.repeat,args.strategy,args.id,args.reason)
            elif args.command=='finalize':
                from .finalization import finalize
                result=finalize(s)
            elif args.command=='report':
                from .reporting import report
                result=report(s)
            print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        parser.exit(2,f'error: {exc}\n')

if __name__=='__main__':main()
