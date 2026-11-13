"""Bounded synthetic acceptance, explicitly unrelated to real scientific accuracy."""
import argparse
import csv
import json
from itertools import product
from pathlib import Path
import numpy as np
from rdkit import Chem
from src.schema import canonical_smiles
from src.csv_conversion import passes_filters
from src.splits import chemical_group
from src.gaussian_records import protocol,digest
from .config import validate,DEFAULTS,canonical,hashed
from .artifacts import atomic_write,file_hash
from .experiment import initialize
from .state import Store
from .runner import run,select
from .finalization import finalize
from .reporting import report
from .gaussian import prepare,ingest,retry

REVISION='ES64L-G16RevC.01'
ROUTE='# opt=calcfc freq b3lyp/gen 5D 7F empiricaldispersion=gd3bj scf=tight int=ultrafine'


def molecules(count=128):
    by_group={}
    candidates=set()
    for n in range(1,10):
        candidates.add('C'*n)
        for position in range(n):
            for element in ['N','O','S']:
                candidates.add('C'*position+element+'C'*(n-position-1))
        if n>=3:
            candidates.add('C1'+'C'*(n-1)+'1')
            candidates.add('C1'+'C'*(n-2)+'O1')
        if n>=4:
            for position in range(1,n-1):
                candidates.add('C'*position+'(C)'+'C'*(n-position-1))
    for length in range(1,7):
        candidates.update(''.join(atoms) for atoms in product('CNO',repeat=length))
    for smi in sorted(candidates):
        if not passes_filters(smi):continue
        smi=canonical_smiles(smi);by_group.setdefault(chemical_group(smi),set()).add(smi)
    lists=[sorted(v) for _,v in sorted(by_group.items())];selected=[]
    while len(selected)<count and any(lists):
        for group in lists:
            if group and len(selected)<count:selected.append(group.pop(0))
    if len(selected)<count:raise ValueError('Synthetic generator has insufficient unique graphs')
    return sorted(selected)


def source_file(path,smiles,labeled=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as out:
        writer=csv.writer(out);writer.writerow(['source_record_id','input_smiles']+(['gap_ev'] if labeled else []))
        for i,smi in enumerate(smiles):
            mol=Chem.MolFromSmiles(smi)
            gap=2.+.12*mol.GetNumHeavyAtoms()+.07*mol.GetRingInfo().NumRings()+.03*sum(a.GetAtomicNum()==7 for a in mol.GetAtoms())
            writer.writerow([f'synth_{i:04}',smi]+([gap] if labeled else []))


def config_for(source,split):
    return validate({'dataset':'synthetic','source':{'path':str(source.resolve()),'sha256':file_hash(source)},
        'repeat_seeds':[42,43],'split':{'mode':split},'budgets':{'initial':16,'batch':8,'max_labels':32,'max_attempts':32,'rounds':2},
        'models':{'members':3,'trees':8,'downstream_trees':8},'resources':{'threads':1,'memory_mb':512}})


def synthetic_log(experiment,record,directory,failed=False):
    """Wrap actual generated coordinates in a complete *synthetic* Gaussian log."""
    manifest=experiment/record['manifest_path'];row=record['row'];source=(manifest.parent/row['input_path']).read_text()
    lines=source.splitlines();coords=[]
    for line in lines[lines.index('0 1')+1:]:
        if not line.strip():break
        symbol,*xyz=line.split();coords.append((Chem.GetPeriodicTable().GetAtomicNumber(symbol),xyz))
    route=next(l for l in lines if l.startswith('#'))
    text=' Gaussian 16: '+REVISION+'\n'+route+'\n --------\nProtocol: '+row['protocol_id']+'\n Charge = 0 Multiplicity = 1\n Standard orientation:\n ----------------\n Center Atomic Atomic Coordinates\n Number Number Type X Y Z\n ----------------\n'
    text+=''.join(f' {i} {z} 0 '+ ' '.join(xyz)+'\n' for i,(z,xyz) in enumerate(coords,1))
    text+=' ----------------\n Optimization completed.\n Alpha  occ. eigenvalues -- -0.50000\n Alpha virt. eigenvalues -- 0.10000\n Frequencies -- -5.000 100.000 200.000\n Normal termination of Gaussian\n'
    if failed:text+=' Error termination (synthetic failure)\n'
    directory.mkdir(parents=True,exist_ok=True);path=directory/row['file'];atomic_write(path,text.encode());return path


def prospective(root):
    # Restrict to simple acyclic molecules with enough topology groups for optional tests.
    source=root/'prospective_candidates.csv';source_file(source,molecules(40),False)
    c=config_for(source,'random');options=DEFAULTS['protocol']['options'].copy();pid=protocol(options)
    c.update(mode='prospective',dataset='gaussian',repeat_seeds=[42]);c['acquisition']['strategies']=['random']
    c['budgets'].update(initial=4,batch=2,max_labels=8,max_attempts=10,rounds=2);c['retry']['max_retries']=1
    c['protocol'].update(input_policy_id=pid,observed_protocol_id=digest((pid+''.join(ROUTE.split()).lower()+REVISION).encode()),expected_revision=REVISION)
    c=validate(c);atomic_write(root/'prospective_config.json',canonical(c).encode());experiment=initialize(c,root/'prospective')
    with Store(experiment) as s:
        # Heldout evaluation data is independently requested and never enters training.
        for strategy in ['__evaluation__','random']:
            items=prepare(s,0,strategy,True);outcomes=[]
            for index,item in enumerate(items):
                failed=strategy=='random' and index==0
                log=synthetic_log(experiment,item,root/'prospective_logs'/item['calculation_id'],failed)
                outcomes.append({'source_record_id':item['source_record_id'],'attempt':item['attempt'],'log_path':str(log)})
            ingest(s,0,strategy,outcomes)
        failed=items[0]['source_record_id'];retry(s,0,'random',failed,'Bounded synthetic resume acceptance')
        item=prepare(s,0,'random',True)[0];log=synthetic_log(experiment,item,root/'prospective_logs'/item['calculation_id'])
        ingest(s,0,'random',[{'source_record_id':failed,'attempt':2,'log_path':str(log)}])
    with Store(experiment) as s:
        for _ in range(2):
            selected=select(s,0,'random')
            if selected['status']!='reserved':raise AssertionError(selected)
            items=prepare(s,0,'random',True)
            outcomes=[]
            for item in items:
                log=synthetic_log(experiment,item,root/'prospective_logs'/item['calculation_id'])
                outcomes.append({'source_record_id':item['source_record_id'],'attempt':item['attempt'],'log_path':str(log)})
            ingest(s,0,'random',outcomes)
        if select(s,0,'random')['status']!='complete':raise AssertionError('Expected round cap')
        finalize(s)
        result={'training_labels':len(s.labels(0,'random')),'training_attempts':s.db.execute("SELECT COUNT(*) FROM attempts WHERE strategy='random'").fetchone()[0],
            'failed_attempts':s.db.execute("SELECT COUNT(*) FROM attempts WHERE status='rejected'").fetchone()[0]}
    return result


def acceptance(output,render=True):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'synthetic.csv';source_file(source,molecules())
    results={}
    for split in ['random','group']:
        c=config_for(source,split);atomic_write(root/f'{split}_config.json',canonical(c).encode())
        experiment=initialize(c,root/split)
        with Store(experiment) as s:
            run(s);finalize(s)
            results[split]={'cohort':s.meta('attrition'),'partitions':[p['sizes'] for p in s.meta('partitions')],
                'checkpoints':len(s.rows('SELECT * FROM checkpoints')),'metrics':len(s.rows('SELECT * FROM metrics')),
                'selections':len(s.rows('SELECT * FROM selections'))}
            if render:results[split]['report']=report(s)
    c=config_for(source,'random');experiment=initialize(c,root/'resumed')
    with Store(experiment) as s:run(s,stop_after=1)
    with Store(experiment) as s:run(s)
    with Store(root/'random') as a,Store(experiment) as b:
        selection_sql='SELECT repeat,strategy,round,id FROM selections ORDER BY repeat,strategy,round,id'
        if a.rows(selection_sql)!=b.rows(selection_sql):raise AssertionError('Resume selections differ')
        for cp in a.rows('SELECT * FROM checkpoints'):
            other=b.rows('SELECT * FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(cp['repeat'],cp['strategy'],cp['round']))[0]
            x=json.loads((a.root/cp['predictions']).read_text());y=json.loads((b.root/other['predictions']).read_text())
            np.testing.assert_allclose([r['predicted_gap_ev'] for r in x],[r['predicted_gap_ev'] for r in y],rtol=0,atol=1e-12)
    results['resume']='identical memberships/selections and numerically equivalent predictions'
    results['prospective']=prospective(root)
    results['real_qm9']='unrun: verified local raw archive/exclusion set unavailable; no automatic download'
    atomic_write(root/'acceptance.json',canonical(results).encode());return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--no-report',action='store_true')
    args=p.parse_args();print(json.dumps(acceptance(args.output,not args.no_report),indent=2))

if __name__=='__main__':main()
