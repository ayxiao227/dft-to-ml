import csv
import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rdkit import Chem
from src.active_learning.config import validate,canonical,hashed
from src.active_learning.datasets import load_source
from src.active_learning.features import featurize,FingerprintRows
from src.active_learning.artifacts import file_hash
from src.active_learning.experiment import initialize
from src.active_learning.state import Store
from src.active_learning.runner import run,select
from src.active_learning.gaussian import prepare,ingest
from src.active_learning.acquisition import maxmin
from src.gaussian_records import inspect_record,digest,reconstruct
from src.dft_parser import parse_log
from test_al_state import fixture
from test_al_gaussian import prospective_config,synthetic_log


def qm9_xyz(ident,original,optimized,gap=.5):
    mol=Chem.AddHs(Chem.MolFromSmiles(optimized));n=mol.GetNumAtoms()
    header=f'{n}\ngdb {ident} 1 2 3 0 1 -0.4 0.1 {gap} 1 0.1 -40 -40 -40 -40 1\n'
    return header+''.join(a.GetSymbol()+' 0 0 0 0\n' for a in mol.GetAtoms())+'100 200\n'+original+' '+optimized+'\nInChI fixture\n'

class Adversarial(unittest.TestCase):
    def test_raw_qm9_identity_exclusions_conflicts_and_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'raw.tar.bz2';exclusions=root/'custom.txt';exclusions.write_text('999 C C\n')
            rows=[qm9_xyz(2,'C','C'),qm9_xyz(1,'C','C'),qm9_xyz(3,'CCO','COC'),
                  qm9_xyz(6,'OCC','CCO'),qm9_xyz(7,'CCO','CCO'),qm9_xyz(21725,'N','N'),qm9_xyz(999,'O','O')]
            with tarfile.open(archive,'w:bz2') as tar:
                for i,text in enumerate(rows):
                    b=text.encode();info=tarfile.TarInfo(f'{i}.xyz');info.size=len(b);tar.addfile(info,io.BytesIO(b))
            c=validate({'dataset':'qm9','source':{'path':str(archive),'sha256':file_hash(archive),'exclusions_path':str(exclusions),'exclusions_sha256':file_hash(exclusions)}})
            records,labels,ledger=load_source(c)
            self.assertEqual([r.source_record_id for r in records],['qm9_1','qm9_6'])
            self.assertAlmostEqual(labels['qm9_1']['gap_ev'],.5*27.211386245988)
            reasons=[r['reason'] for r in ledger]
            self.assertTrue(any('graph' in r for r in reasons));self.assertIn('Additional convergence exception',reasons);self.assertIn('custom uncharacterized exclusion',reasons)
            f,_,_=featurize(records,c['representation']);self.assertEqual(f.smiles,('C','CCO'))
            archive.write_bytes(archive.read_bytes()+b'tamper')
            with self.assertRaises(ValueError):load_source(c)

    def test_no_final_access_and_hidden_label_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);a=initialize(c,root/'a');b=initialize(c,root/'b')
            with Store(b) as s:
                path=b/'repeat_0/oracle_pool.json';table=json.loads(path.read_text());roles=s.meta('partitions')[0]['rows']
                pool=[r['source_record_id'] for r in roles if r['origin']=='pool'];values=[table[i]['gap_ev'] for i in pool][::-1]
                for ident,value in zip(pool,values):table[ident]['gap_ev']=value
                # Build an independent source fixture via a new experiment, not an in-place config edit.
            source_b=root/'permuted.csv'
            with Path(c['source']['path']).open() as inp,source_b.open('w') as out:
                reader=csv.DictReader(inp);writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader()
                for row in reader:
                    if row['source_record_id'] in table and row['source_record_id'] in pool:row['gap_ev']=table[row['source_record_id']]['gap_ev']
                    writer.writerow(row)
            c2=json.loads(canonical(c));c2['source'].update(path=str(source_b),sha256=file_hash(source_b));d=initialize(c2,root/'d')
            original=__import__('src.active_learning.oracle',fromlist=['EvaluationOracle']).EvaluationOracle.read
            def spy(oracle,ids):
                self.assertNotEqual(oracle._EvaluationOracle__path.name,'oracle_final.json')
                return original(oracle,ids)
            with patch('src.active_learning.oracle.EvaluationOracle.read',spy),Store(a) as s,Store(d) as t:
                for strategy in c['acquisition']['strategies']:
                    x=select(s,0,strategy);y=select(t,0,strategy)
                    self.assertEqual(x['ids'],y['ids'])

    def test_feature_disk_view_bounded_chunks(self):
        array=np.random.default_rng(2).integers(0,2,size=(1000,64),dtype=np.uint8)
        class Guarded(FingerprintRows):
            def reordered(self,order):return Guarded(self.matrix,self.indices[order])
            def __array__(self,*args):raise AssertionError('Full fingerprint materialization forbidden')
            def __getitem__(self,key):
                result=super().__getitem__(key)
                if result.ndim==2:self_check(result.shape[0])
                return result
        def self_check(n):self.assertLessEqual(n,17)
        picked=maxmin([str(i).zfill(4) for i in range(1000)],Guarded(array,np.arange(1000)),[array[0]],4,17)
        self.assertEqual(len(picked),4)

    def test_atomic_partial_batch_refused_and_source_config_model_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:
                selected=select(s,0,'random');ids=selected['ids'];before=len(s.labels(0,'random'))
                deliveries=[{'id':ids[0],'attempt':1,'status':'accepted','value':2.,'log_hash':None,'evidence':{'synthetic':True}},
                            {'id':'unknown','attempt':1,'status':'accepted','value':2.,'log_hash':None,'evidence':{}}]
                with self.assertRaises(ValueError):s.deliver(0,'random',1,deliveries)
                self.assertEqual(len(s.labels(0,'random')),before)
                cp=s.rows('SELECT * FROM checkpoints')[0];path=experiment/cp['path'];path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                with Store(experiment):pass
            changed=json.loads(canonical(c));changed['models']['trees']=9
            with self.assertRaises(ValueError):initialize(changed,experiment)

    def test_partial_last_batch_and_early_finalization_refusal(self):
        from src.active_learning.finalization import finalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);c['budgets'].update(max_labels=7,max_attempts=7)
            experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:
                select(s,0,'random')
                with self.assertRaises(ValueError):finalize(s)
                run(s)
                for strategy in c['acquisition']['strategies']:
                    self.assertEqual(len(s.labels(0,strategy)),7)
                    count=s.db.execute('SELECT COUNT(*) FROM selections WHERE strategy=? AND round=2',(strategy,)).fetchone()[0]
                    self.assertEqual(count,1)

    def test_geometry_basis_molecule_and_duplicate_basename_adversaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=prospective_config(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:item=prepare(s,0,'random')[0]
            log=synthetic_log(experiment,item,root/'logs');text=log.read_text();manifest=experiment/item['manifest_path'];row=item['row'].copy()
            with self.assertRaises(ValueError):reconstruct(text+' Standard orientation:\n -------\n truncated\n')
            coords=text.split(' ----------------\n')[2].splitlines()
            line=next(l for l in coords if len(l.split())==6 and l.split()[0].isdigit())
            bad=line.split();bad[3]='nan'
            with self.assertRaises(ValueError):reconstruct(text.replace(line,' '.join(bad)))
            row['input_smiles']='CCCCCCCCCC' if row['input_smiles']!='CCCCCCCCCC' else 'C'
            with self.assertRaises(ValueError):inspect_record(log,row,manifest.parent)
            row=item['row'].copy();input_path=manifest.parent/row['input_path'];content=input_path.read_text();input_path.write_text(content.split('****')[0])
            row['input_sha256']=digest(input_path.read_bytes())
            with self.assertRaises(ValueError):inspect_record(log,row,manifest.parent)
            nested=log.parent/'nested';nested.mkdir();(nested/log.name).write_text(text)
            with self.assertRaises(ValueError):parse_log(log.parent,'x',manifest)

    def test_duplicate_cost_batch_and_prospective_delivery_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=prospective_config(root);experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:
                items=prepare(s,0,'random');outcomes=[]
                cost={'job_id':'same','source':'scheduler','scope':'job','cpu_seconds':5}
                for item in items:
                    log=synthetic_log(experiment,item,root/item['source_record_id'])
                    outcomes.append({'source_record_id':item['source_record_id'],'attempt':1,'log_path':str(log),'resource_usage':cost})
                with self.assertRaises(ValueError):ingest(s,0,'random',outcomes)
                self.assertEqual(s.labels(0,'random'),{})
                for o in outcomes:o.pop('resource_usage')
                with self.assertRaises(RuntimeError):ingest(s,0,'random',outcomes,failpoint='after_label_delivery')
                self.assertEqual(s.labels(0,'random'),{})
            with Store(experiment) as s:
                ingest(s,0,'random',outcomes);self.assertEqual(len(s.labels(0,'random')),2)
