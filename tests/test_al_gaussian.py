import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from rdkit import Chem
from src.active_learning.config import validate,DEFAULTS
from src.active_learning.artifacts import file_hash
from src.active_learning.experiment import initialize
from src.active_learning.state import Store
from src.active_learning.gaussian import prepare,ingest,retry
from src.active_learning.costs import validate_cost,coverage
from src.gaussian_records import protocol,digest,inspect_record
from src.structure_automation import generate_slurm_scripts
from src.dft_parser import parse_log,parse_orbital_blocks,homo_lumo,validate_log
from test_al_partitions import records

REVISION='ES64L-G16RevC.01'
ROUTE='# opt=calcfc freq b3lyp/gen 5D 7F empiricaldispersion=gd3bj scf=tight int=ultrafine'


def prospective_config(root):
    source=root/'candidates.csv'
    with source.open('w') as out:
        writer=csv.writer(out);writer.writerow(['source_record_id','input_smiles'])
        for r in records():writer.writerow([r.source_record_id,r.input_smiles])
    options=DEFAULTS['protocol']['options'].copy();pid=protocol(options)
    observed=digest((pid+''.join(ROUTE.split()).lower()+REVISION).encode())
    return validate({'mode':'prospective','dataset':'gaussian','source':{'path':str(source),'sha256':file_hash(source)},
        'split':{'mode':'random'},'repeat_seeds':[42],'representation':{'bits':32},
        'budgets':{'initial':2,'batch':2,'max_labels':6,'max_attempts':8,'rounds':2},
        'models':{'members':3,'trees':2,'downstream_trees':2},'acquisition':{'strategies':['random']},
        'retry':{'max_retries':1},'protocol':{'input_policy_id':pid,'observed_protocol_id':observed,'expected_revision':REVISION,'options':options}})


def synthetic_log(experiment,record,directory):
    """Unmocked parser fixture: real RDKit/BSE input geometry, synthetic log wrapper."""
    manifest=experiment/record['manifest_path'];row=record['row'];source=(manifest.parent/row['input_path']).read_text()
    lines=source.splitlines();start=lines.index('0 1')+1;coords=[]
    for line in lines[start:]:
        if not line.strip():break
        symbol,*xyz=line.split();coords.append((Chem.GetPeriodicTable().GetAtomicNumber(symbol),xyz))
    route=next(l for l in lines if l.startswith('#'))
    text=' Gaussian 16: '+REVISION+'\n'+route+'\n --------\nProtocol: '+row['protocol_id']+'\n Charge = 0 Multiplicity = 1\n Standard orientation:\n ----------------\n Center Atomic Atomic Coordinates\n Number Number Type X Y Z\n ----------------\n'
    text+=''.join(f' {i} {z} 0 '+ ' '.join(xyz)+'\n' for i,(z,xyz) in enumerate(coords,1))
    text+=' ----------------\n Optimization completed.\n Alpha  occ. eigenvalues -- -0.50000\n Alpha virt. eigenvalues -- 0.10000\n Frequencies -- -5.000 100.000 200.000\n Normal termination of Gaussian\n'
    directory.mkdir(parents=True,exist_ok=True);path=directory/row['file'];path.write_text(text);return path

class GaussianBridge(unittest.TestCase):
    def test_actual_prepare_ingest_resume_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=prospective_config(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:
                items=prepare(s,0,'random',True)
                self.assertEqual(prepare(s,0,'random',True),items)
                self.assertEqual({r['source_record_id'] for r in items},set(r['id'] for r in s.rows("SELECT id FROM attempts WHERE strategy='random'")))
                for script in experiment.rglob('*.txt'):subprocess.run(['bash','-n',str(script)],check=True)
                outcomes=[]
                for i,item in enumerate(items):
                    log=synthetic_log(experiment,item,root/'outputs'/str(i))
                    if i:log.write_text(log.read_text()+' Error termination\n')
                    outcomes.append({'source_record_id':item['source_record_id'],'attempt':1,'log_path':str(log)})
                ingest(s,0,'random',outcomes)
                self.assertEqual(len(s.labels(0,'random')),1)
                self.assertEqual(ingest(s,0,'random',outcomes)['delivered'],2)
                failed=items[1]['source_record_id'];retry(s,0,'random',failed,'Synthetic parser recovery test')
                item=next(r for r in prepare(s,0,'random') if r['source_record_id']==failed)
                self.assertEqual(item['attempt'],2)
                log=synthetic_log(experiment,item,root/'retry')
                ingest(s,0,'random',[{'source_record_id':failed,'attempt':2,'log_path':str(log)}])
                self.assertEqual(len(s.labels(0,'random')),2)
                log.write_text(log.read_text().replace('0.10000','0.20000'))
                with self.assertRaises(ValueError):ingest(s,0,'random',[{'source_record_id':failed,'attempt':2,'log_path':str(log)}])
            with Store(experiment) as s:
                self.assertEqual(len(s.labels(0,'random')),2)
                self.assertTrue(list((experiment/'quarantine').rglob('*.log')))

    def test_strict_synthetic_parser_adversaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=prospective_config(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:item=prepare(s,0,'random')[0]
            log=synthetic_log(experiment,item,root/'audit');text=log.read_text();manifest=experiment/item['manifest_path']
            accepted=parse_log(log.parent,'x',manifest)
            self.assertEqual(len(accepted),1);self.assertEqual(accepted.minimum_frequency_cm1.iloc[0],-5)
            self.assertNotEqual(accepted.input_protocol_id.iloc[0],accepted.observed_protocol_id.iloc[0])
            changes=[('0.10000','NaN'),('0.10000','Inf'),('Frequencies -- -5.000','Frequencies -- -50.000'),
                ('Gaussian 16: '+REVISION,'Gaussian 09: unknown'),('Multiplicity = 1','Multiplicity = 3'),
                ('Charge = 0','Charge = 1'),('opt=calcfc freq','sp'),('Alpha virt.','Beta virt.'),
                ('Alpha  occ. eigenvalues -- -0.50000','Alpha  occ. eigenvalues -- ****'),
                ('Normal termination of Gaussian','Error termination of Gaussian')]
            for old,new in changes:
                with self.subTest(change=(old,new)):
                    log.write_text(text.replace(old,new))
                    with self.assertRaises(ValueError):parse_log(log.parent,'x',manifest)
            continuation='#ngeom=allcheckguess=tcheck'+''.join(ROUTE.split()).lower().lstrip('#').replace('opt=calcfc','')
            log.write_text(text.replace(' Optimization completed.',continuation+'\n --------\n Optimization completed.'))
            self.assertEqual(len(parse_log(log.parent,'x',manifest)),1)
            log.write_text(log.read_text().replace('guess=tcheck','guess=read'))
            with self.assertRaises(ValueError):parse_log(log.parent,'x',manifest)
            log.write_text(text+' Error termination\n')
            with self.assertRaises(ValueError):validate_log(log)
            input_path=manifest.parent/item['row']['input_path'];original=input_path.read_text()
            log.write_text(text);input_path.write_text(original+'tamper')
            with self.assertRaises(ValueError):parse_log(log.parent,'x',manifest)

    def test_missing_all_rejected_cost_units_and_duplicate_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=prospective_config(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:
                items=prepare(s,0,'random')
                self.assertEqual(ingest(s,0,'random',[{'source_record_id':items[0]['source_record_id'],'attempt':1}])['delivered'],0)
                cost={'source':'local_scheduler_export','job_id':'1','scope':'job','cpu_seconds':5.,'wall_seconds':10.,'allocated_cpus':2,'allocated_cpu_hours':20/3600}
                outcomes=[]
                for item in items:
                    log=synthetic_log(experiment,item,root/item['source_record_id']);log.write_text(log.read_text().replace('-5.000','-100.000'))
                    outcomes.append({'source_record_id':item['source_record_id'],'attempt':1,'log_path':str(log)})
                outcomes[0]['resource_usage']=cost
                ingest(s,0,'random',outcomes)
                self.assertEqual(s.labels(0,'random'),{})
                cov=coverage(s.rows("SELECT * FROM attempts WHERE strategy='random'"))
                self.assertFalse(cov['cpu_seconds']['complete']);self.assertEqual(cov['cpu_seconds']['measured_sum'],5.)
                self.assertEqual(cov['allocated_cpu_hours']['measured_sum'],20/3600)
            with self.assertRaises(ValueError):validate_cost({**cost,'cpu_seconds':-1})

    def test_resume_preserves_prior_manifest_on_interrupted_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous=os.getcwd();os.chdir(tmp)
            try:
                generate_slurm_scripts(pd.DataFrame({'SMILES':['C','CC']}),ID='test')
                real=__import__('src.structure_automation',fromlist=['smiles_to_gjf']).smiles_to_gjf
                def interrupt(smiles,*args,**kwargs):
                    if smiles=='CCC':raise RuntimeError('interrupted')
                    return real(smiles,*args,**kwargs)
                with patch('src.structure_automation.smiles_to_gjf',side_effect=interrupt):
                    with self.assertRaises(RuntimeError):generate_slurm_scripts(pd.DataFrame({'SMILES':['C','CCCC','CCC','CC']}),ID='test',resume=True)
                old=pd.read_csv('molecules/gjf_files/test/manifest.csv');self.assertEqual(len(old),3)
                self.assertIn('CC',old.SMILES.tolist())
                self.assertTrue(list(Path('molecules/gjf_files/test').glob('manifest.resume.*.csv')))
            finally:os.chdir(previous)
