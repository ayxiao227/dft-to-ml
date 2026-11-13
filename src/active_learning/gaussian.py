"""Attempt-scoped local opt/freq bridge. Never invokes Gaussian, SSH or sbatch."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pandas as pd
from src.structure_automation import generate_slurm_scripts
from src.dft_parser import parse_log,validate_log,parse_orbital_blocks,homo_lumo
from src.gaussian_records import inspect_record
from .config import canonical,hashed,seed_for
from .artifacts import atomic_write,file_hash
from .costs import validate_cost

@contextmanager
def working_directory(path):
    previous=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(previous)


def _require_prospective(s):
    s.ensure_mutable()
    if s.meta('config')['mode']!='prospective':raise ValueError('Gaussian bridge requires prospective mode')


def _require_run(s,repeat,strategy):
    if not s.rows('SELECT * FROM runs WHERE repeat=? AND strategy=?',(repeat,strategy)):
        raise ValueError('Unknown repeat/strategy')


def prepare(s,repeat,strategy,mark_pending=False):
    _require_prospective(s);_require_run(s,repeat,strategy);config=s.meta('config')
    attempts=s.rows("SELECT * FROM attempts WHERE repeat=? AND strategy=? AND status IN ('reserved','prepared','pending') ORDER BY round,id,attempt",(repeat,strategy))
    result=[]
    for attempt in attempts:
        if attempt['input_record']:
            if mark_pending and attempt['status']=='prepared':
                with s.transaction():s.db.execute("UPDATE attempts SET status='pending' WHERE calculation_id=?",(attempt['calculation_id'],))
            result.append(json.loads(attempt['input_record']));continue
        molecular=json.loads(s.rows('SELECT record FROM molecules WHERE id=?',(attempt['id'],))[0]['record'])
        generation_id='al_'+attempt['calculation_id'][:24]+'_a'+str(attempt['attempt'])
        work=s.root/'gaussian_attempts'/attempt['calculation_id'];work.mkdir(parents=True,exist_ok=True)
        seed=seed_for(config['repeat_seeds'][repeat],repeat,attempt['round'],'conformer',attempt['attempt'])
        options={**config['protocol']['options'],'seed':seed}
        with working_directory(work):
            manifest_path=Path('molecules/gjf_files')/generation_id/'manifest.csv'
            df=generate_slurm_scripts(pd.DataFrame([{'SMILES':molecular['input_smiles'],'source_record_id':attempt['id'],'molecule_id':attempt['id']}]),
                ID=generation_id,gjf_options=options,resume=manifest_path.exists())
        row=df.iloc[0].to_dict()
        # Convert numpy scalars without inventing missing metadata.
        row=json.loads(df.to_json(orient='records'))[0]
        if not row['generated']:
            with s.transaction():
                s.db.execute("UPDATE attempts SET status='failed',reason='Local input generation failed' WHERE calculation_id=?",(attempt['calculation_id'],))
                s.complete_round(repeat,strategy,attempt['round'])
            continue
        if row['source_record_id']!=attempt['id'] or row['molecule_id']!=molecular['input_structure_key'] or row['protocol_id']!=config['protocol']['input_policy_id']:
            raise ValueError('Source-to-generator mapping or requested policy mismatch')
        manifest_path=work/'molecules/gjf_files'/generation_id/'manifest.csv'
        record={'generation_id':generation_id,'conformer_seed':seed,'manifest_path':str(manifest_path.relative_to(s.root)),
            'row':row,'source_record_id':attempt['id'],'attempt':attempt['attempt'],'calculation_id':attempt['calculation_id']}
        with s.transaction():
            for path in sorted(work.rglob('*')):
                if path.is_file():s.artifact(str(path.relative_to(s.root)),path.read_bytes(),'gaussian_input_or_script')
            s.db.execute('UPDATE attempts SET status=?,input_record=? WHERE calculation_id=?',('pending' if mark_pending else 'prepared',canonical(record),attempt['calculation_id']))
            s.event('prepared',{'calculation_id':attempt['calculation_id'],'source_record_id':attempt['id'],'conformer_seed':seed})
        result.append(record)
    return result


def verify_external_label(row,base,config):
    required={'log_path','input_path','input_sha256','options','protocol_id','molecule_id','run_id','SMILES'}
    if not required.issubset(row):raise ValueError('External label lacks retained input/log evidence')
    path=Path(row['log_path']);path=path if path.is_absolute() else base/path
    validate_log(path);details=inspect_record(path,row,base)
    blocks=parse_orbital_blocks(path)
    if not blocks or any(blocks[-1][k] for k in ('occ_beta','virt_beta')):raise ValueError('Neutral singlet orbital evidence required')
    details.update(homo_lumo(blocks[-1]));_expected(details,config)
    details.pop('xyz',None)
    return details


def _expected(details,config):
    p=config['protocol']
    if details['gaussian_version']!=p['expected_revision'] or details['observed_protocol_id']!=p['observed_protocol_id'] or details['input_protocol_id']!=p['input_policy_id']:
        raise ValueError('Unexpected Gaussian revision/observed route/input policy')


def ingest(s,repeat,strategy,outcomes,failpoint=None):
    _require_prospective(s);_require_run(s,repeat,strategy)
    if not isinstance(outcomes,list):raise ValueError('Outcomes must be a JSON list with explicit source_record_id and attempt')
    config=s.meta('config');deliveries=[];artifacts=[];costs=[];rounds=set();seen=set()
    for outcome in outcomes:
        allowed={'source_record_id','attempt','log_path','outcome','reason','resource_usage'}
        if set(outcome)-allowed:raise ValueError('Unknown outcome keys')
        ident=outcome['source_record_id'];number=outcome['attempt']
        if type(number) is not int or (ident,number) in seen:raise ValueError('Duplicate/invalid attempt')
        seen.add((ident,number))
        rows=s.rows('SELECT * FROM attempts WHERE repeat=? AND strategy=? AND id=? AND attempt=?',(repeat,strategy,ident,number))
        if not rows:raise ValueError('Unregistered attempt')
        a=rows[0];rounds.add(a['round']);record=json.loads(a['input_record']) if a['input_record'] else None
        if outcome.get('resource_usage') is not None:
            cost=validate_cost(outcome['resource_usage'])
            for other in s.rows('SELECT calculation_id,cost FROM attempts WHERE cost IS NOT NULL'):
                old=json.loads(other['cost'])
                if (old['job_id'],old['source'])==(cost['job_id'],cost['source']) and other['calculation_id']!=a['calculation_id']:
                    raise ValueError('Duplicate job/step cost accounting')
            for prior_calc,prior_cost in costs:
                if prior_calc!=a['calculation_id'] and (prior_cost['job_id'],prior_cost['source'])==(cost['job_id'],cost['source']):
                    raise ValueError('Duplicate cost within incoming batch')
            if a['cost'] is not None and a['cost']!=canonical(cost):raise ValueError('Conflicting attempt cost')
            costs.append((a['calculation_id'],cost))
        if not outcome.get('log_path'):
            terminal=outcome.get('outcome')
            if terminal is None:continue # absent output is not failure
            if terminal not in ('failed','excluded') or not outcome.get('reason'):raise ValueError('Explicit terminal outcome needs reason')
            deliveries.append({'id':ident,'attempt':number,'status':terminal,'reason':outcome['reason'],'log_hash':None,
                'evidence':{'explicit_terminal':terminal,'reason':outcome['reason']}})
            continue
        if record is None:raise ValueError('Attempt has no registered input')
        log=Path(outcome['log_path'])
        if not log.exists():continue
        if log.name!=record['row']['file']:raise ValueError('Log basename differs from exact registered manifest association')
        sha=file_hash(log)
        if a['log_hash'] and a['log_hash']!=sha:
            with s.transaction():
                relative=f'quarantine/{a["calculation_id"]}/{sha}.log'
                s.artifact(relative,log.read_bytes(),'conflicting_log')
                s.event('conflicting_result_quarantined',{'calculation_id':a['calculation_id'],'log_sha256':sha})
            raise ValueError('Conflicting result quarantined; original result retained')
        audit=s.root/'gaussian_audits'/a['calculation_id']/sha;audit.mkdir(parents=True,exist_ok=True)
        snapshot=audit/log.name
        if snapshot.exists() and file_hash(snapshot)!=sha:raise ValueError('Conflicting log snapshot')
        atomic_write(snapshot,log.read_bytes())
        manifest=s.root/record['manifest_path']
        try:
            parsed=parse_log(audit,record['generation_id'],manifest)
            if len(parsed)!=1:raise ValueError('Attempt-scoped parse must return exactly one result')
            details=json.loads(parsed.to_json(orient='records'))[0]
            blocks=parse_orbital_blocks(snapshot)
            if blocks[-1]['occ_beta'] or blocks[-1]['virt_beta']:raise ValueError('Neutral singlet orbital block required')
            _expected(details,config)
            # pandas JSON defaults round to ten places; label comes directly from parser floats.
            value=float(parsed.gap_ev.iloc[0])
            status='accepted';reason=None
        except (ValueError,IndexError,KeyError,RuntimeError) as exc:
            details={'input_protocol_id':record['row']['protocol_id'],'input_sha256':record['row']['input_sha256'],
                'log_sha256':sha,'reason':str(exc),'acceptance_policy':'organic_optfreq_minfreq_minus10_v2'}
            value=None;status='rejected';reason=str(exc)
        details.update(source_record_id=ident,calculation_id=a['calculation_id'],attempt_id=number,
            provenance_type='verified_gaussian',units='eV',acceptance_policy='organic_optfreq_minfreq_minus10_v2')
        deliveries.append({'id':ident,'attempt':number,'status':status,'reason':reason,'value':value,'log_hash':sha,'evidence':details})
        artifacts.extend((str(p.relative_to(s.root)),p.read_bytes(),'gaussian_parse_evidence') for p in sorted(audit.rglob('*')) if p.is_file())
    if len(rounds)>1:raise ValueError('Ingest one registered round at a time')
    if not rounds:return {'delivered':0,'pending':True}
    def commit_artifacts():
        for path,content,kind in artifacts:s.artifact(path,content,kind)
        for calc,cost in costs:s.db.execute('UPDATE attempts SET cost=? WHERE calculation_id=?',(canonical(cost),calc))
        if strategy=='__evaluation__':
            for role in ['validation','final']:
                ids=[r['id'] for r in s.rows('SELECT id FROM partitions WHERE repeat=? AND origin=? ORDER BY id',(repeat,role))]
                accepted=s.labels(repeat,strategy)
                if all(i in accepted for i in ids):
                    table={i:{'gap_ev':accepted[i],'units':'eV','provenance_type':'verified_gaussian','observed_protocol_id':config['protocol']['observed_protocol_id']} for i in ids}
                    s.artifact(f'repeat_{repeat}/oracle_{role}.json',canonical(table).encode(),'isolated_oracle_'+role)
    s.deliver(repeat,strategy,next(iter(rounds)),deliveries,failpoint,before_commit=commit_artifacts)
    return {'delivered':len(deliveries),'pending':len(deliveries)<len(outcomes)}


def retry(s,repeat,strategy,ident,reason):
    _require_prospective(s);_require_run(s,repeat,strategy)
    if not reason:raise ValueError('Audited retry reason required')
    config=s.meta('config');rows=s.rows('SELECT * FROM attempts WHERE repeat=? AND strategy=? AND id=? ORDER BY attempt DESC LIMIT 1',(repeat,strategy,ident))
    if not rows or rows[0]['status'] not in ('failed','rejected'):raise ValueError('Only a failed/rejected terminal molecule may be retried')
    previous=rows[0]
    latest=s.db.execute('SELECT MAX(round) FROM rounds WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
    if latest!=previous['round']:raise ValueError('Cannot reopen an older round after later acquisition')
    if previous['attempt']>config['retry']['max_retries']:raise ValueError('Configured explicit retry cap reached; new experiment required')
    attempts=s.db.execute('SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=?',(repeat,strategy)).fetchone()[0]
    # Evaluation calculation overhead is counted separately, but retries still bounded.
    if strategy!='__evaluation__' and attempts>=config['budgets']['max_attempts']:raise ValueError('Attempt budget exhausted')
    number=previous['attempt']+1;calc=hashed([s.meta('experiment_hash'),repeat,strategy,ident,number])
    with s.transaction():
        s.db.execute('INSERT INTO attempts(repeat,strategy,id,attempt,round,calculation_id,status,reason) VALUES (?,?,?,?,?,?,?,?)',
            (repeat,strategy,ident,number,previous['round'],calc,'reserved',reason))
        s.db.execute("UPDATE rounds SET status='reserved' WHERE repeat=? AND strategy=? AND round=?",(repeat,strategy,previous['round']))
        s.event('explicit_retry',{'calculation_id':calc,'id':ident,'attempt':number,'reason':reason})
    return {'calculation_id':calc,'attempt':number}
