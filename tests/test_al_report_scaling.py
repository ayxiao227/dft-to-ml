"""Independent small references for report math, streaming, and recovery."""
import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.active_learning.features import CandidateFeatures
from src.active_learning.report_diagnostics import batch_diversity, FingerprintLookup, Chemistry, ranking, rf_summary
from src.active_learning.report_cache import StageCache, progress
from src.active_learning.report_storage import Predictions
from src.active_learning.reporting import report_identity


def features(n=25,bits=32):
    rng=np.random.default_rng(12)
    fp=rng.integers(0,2,(n,bits),dtype=np.uint8);fp[0]=0;fp[1]=0
    return CandidateFeatures(tuple(map(str,range(n))),('CC',)*n,(),np.empty((n,0)),fp,'fixture')


def source(root,checkpoints=2,repeats=2,n=11,final='legacy'):
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE checkpoints(repeat INTEGER,strategy TEXT,round INTEGER,predictions TEXT)')
    saved=[]
    def rows(rep,rd,partition):
        for evaluator in ['rf','committee']:
            for i in range(n):
                p=(i%7)*.1;y=(i%3)*.2
                yield {'repeat':rep,'strategy':'random','round':rd,'partition':partition,'source_record_id':f'mol{i:04}',
                       'input_smiles':['CC','CCC','C=O'][i%3],'evaluator':evaluator,'observed_gap_ev':y,
                       'predicted_gap_ev':p,'residual_ev':p-y,'absolute_error_ev':abs(p-y),'uncertainty_ev':float(i%3) if evaluator=='committee' else None,'units':'eV'}
    for rep in range(repeats):
        for rd in range(checkpoints):
            path=root/f'{rep}_{rd}.json';data=list(rows(rep,rd,'validation'));path.write_text(json.dumps(data));saved.extend(data)
            db.execute('INSERT INTO checkpoints VALUES (?,?,?,?)',(rep,'random',rd,path.name))
    final_rows=[r for rep in range(repeats) for rd in range(checkpoints) for r in rows(rep,rd,'final')]
    (root/'final').mkdir();(root/'final/predictions.json').write_text(json.dumps(final_rows))
    (root/'final/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in final_rows))
    state={'status':'completed'} if final=='legacy' else {'status':'completed','format_version':2} if final=='new' else {'status':'started','format_version':2}
    return SimpleNamespace(root=root,db=db,meta=lambda key,default=None:state if key=='finalization' else default),saved,final_rows


class ReportScaling(unittest.TestCase):
    def test_batched_diversity_independent_reference_and_index(self):
        f=features();index=f._positions
        class NoScan(tuple):
            def __iter__(self):raise AssertionError('Rebuilt the full molecule-ID index')
        object.__setattr__(f,'ids',NoScan(f.ids))
        selected=['0','1','4','9','7'];refs=['2','3','2']
        for ids in [selected,refs]*5:
            f.rows(ids);f.fingerprints(ids)
            self.assertIs(f._positions,index)
        expected=[];history=list(refs)
        for ident in selected:
            a=set(np.flatnonzero(f.binary[int(ident)]));dist=[]
            for ref in history:
                b=set(np.flatnonzero(f.binary[int(ref)]));dist.append(1-len(a&b)/len(a|b) if a|b else 0.)
            expected.append(min(dist,default=1.));history.append(ident)
        lookup=FingerprintLookup(f,limit=8)
        for chunk in [1,3,1024]:np.testing.assert_allclose(batch_diversity(f,selected,refs,chunk,lookup),expected,rtol=0,atol=1e-15)
        self.assertEqual(batch_diversity(f,['0','1'],[]),[1.,0.])
        self.assertLessEqual(len(lookup.cache),8)
        # A failed previous selection is included within its batch, but only
        # explicitly accepted earlier-round IDs are passed to the next batch.
        self.assertEqual(batch_diversity(f,['1'],['0']),[0.])
        self.assertEqual(batch_diversity(f,['1'],[]),[1.])

    def test_streamed_legacy_new_partial_and_numeric_reference(self):
        outputs=[]
        for fmt in ['legacy','new','partial']:
            with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);s,saved,final=source(root,final=fmt);cache=StageCache(root,'fixture')
                docs={'one':{'predictions':{'path':'final/predictions.jsonl'}}}
                with patch('src.active_learning.finalization.verified_results',return_value=docs):
                    p=Predictions(s,root,cache)
                _,row_iterator=next(p.groups('rf'))
                self.assertFalse(isinstance(row_iterator,pd.DataFrame))
                self.assertIsInstance(next(row_iterator),dict)
                self.assertEqual(p.partitions(),['validation'] if fmt=='partial' else ['final','validation'])
                expected=saved+([] if fmt=='partial' else final)
                with p.csv.open() as inp:self.assertEqual(len(list(csv.DictReader(inp))),len(expected))
                with ChemistryContext(root) as chem:
                    keys=('validation','random');summary=rf_summary(p,keys,chem,5,77)
                    rows=[r for r in saved if r['round']==1 and r['evaluator']=='rf']
                    for rep,tag,count,total in summary['subgroups']:
                        subset=[r for r in rows if r['repeat']==rep and tag in chem.get(r['input_smiles'])['tags']]
                        self.assertEqual(count,len(subset));self.assertAlmostEqual(total,sum(r['absolute_error_ev'] for r in subset),places=14)
                    self.assertEqual(summary['population'],len(rows));self.assertEqual(len(summary['sample']),5)
                    self.assertEqual(summary,rf_summary(p,keys,chem,5,77))
                d,sel=ranking(p,('validation','random',0))
                frame=pd.DataFrame(r for r in saved if r['round']==1 and r['repeat']==0 and r['evaluator']=='committee').sort_values(['uncertainty_ev','source_record_id'])
                for row in sel:self.assertAlmostEqual(row['MAE_eV'],frame.head(row['count']).absolute_error_ev.mean(),places=14)
                for actual,indices in zip(d,np.array_split(np.arange(len(frame)),5)):
                    self.assertEqual(actual['count'],len(indices));self.assertAlmostEqual(actual['mean_uncertainty_ev'],frame.iloc[indices].uncertainty_ev.mean(),places=14)
                # Original exact quantile values remain untouched in saved metrics;
                # ordering and ties here match pandas' source-ID tie break exactly.
                self.assertEqual([r['source_record_id'] for r in p.rows('committee',('validation','random',0),order='u,ident')],frame.source_record_id.tolist())
                outputs.append((d,sel));p.close()
                with patch('src.active_learning.report_storage.prediction_records',side_effect=AssertionError('Reingestion')),patch('src.active_learning.finalization.verified_results',return_value=docs):
                    p=Predictions(s,root,cache);p.close()
                cache.close();s.db.close()
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[1],outputs[2])

    def test_stage_crash_boundaries_corruption_and_committed_progress(self):
        for boundary in ['before_report_publication','after_report_publication','after_report_commit']:
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);calls=[]
                def build(work):
                    calls.append(1);path=work/'table.csv';path.write_text('a\n1\n');return {'n':1},{'table.csv':path}
                cache=StageCache(root,'v1',boundary)
                with self.assertRaisesRegex(RuntimeError,'Injected crash'):cache.run('table',build)
                done=progress(root)[0]['committed_stages'];self.assertEqual(done,['table'] if boundary=='after_report_commit' else [])
                cache.close();cache=StageCache(root,'v1');value,files=cache.run('table',build)
                self.assertEqual(value,{'n':1});self.assertEqual(len(calls),1 if boundary=='after_report_commit' else 2)
                files['table.csv'].write_text('tampered')
                with self.assertRaisesRegex(ValueError,'Corrupt reporting stage'):cache.run('table',build)
                self.assertEqual(files['table.csv'].read_text(),'tampered');cache.close()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=StageCache(root,'old','after_report_publication')
            def build(work):
                path=work/'a';path.write_text('a');return {},{'a':path}
            with self.assertRaises(RuntimeError):cache.run('x',build)
            target=next((cache.root/'objects').glob('*/a'));target.write_text('conflict');cache.failpoint=None
            with self.assertRaisesRegex(ValueError,'conflict retained'):cache.run('x',build)
            cache.close()
            cache=StageCache(root,'new');cache.run('x',build);cache.close()

    def test_corrupt_progress_and_chemistry_are_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=StageCache(root,'fixture');path=cache.root/'progress.sqlite3';cache.close()
            path.write_bytes(b'corrupt ledger')
            with self.assertRaisesRegex(ValueError,'Corrupt report progress ledger retained'):StageCache(root,'fixture')
            self.assertEqual(path.read_bytes(),b'corrupt ledger')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'chemistry.sqlite3';chem=Chemistry(path);chem.get('CC');chem.close()
            with sqlite3.connect(path) as db:db.execute("UPDATE chemistry SET data='broken'")
            chem=Chemistry(path)
            with self.assertRaisesRegex(ValueError,'Corrupt chemistry cache retained'):chem.get('CC')
            chem.close()

    def test_interrupted_copy_evidence_does_not_accumulate(self):
        from src.active_learning.config import hashed
        with tempfile.TemporaryDirectory() as tmp:
            cache=StageCache(Path(tmp),'fixture');target=cache.root/'objects'/hashed('stage');target.mkdir(parents=True)
            evidence=target/'.table.csv.interrupted';evidence.write_bytes(b'partial')
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'Interrupted report publication evidence retained'):
                    cache.run('stage',lambda work:(_ for _ in ()).throw(AssertionError('must not generate more output')))
            self.assertEqual(list(target.iterdir()),[evidence]);self.assertEqual(progress(tmp)[0]['committed_stages'],[]);cache.close()


class ChemistryContext:
    def __init__(self,root):self.chem=Chemistry(root/'chemistry.sqlite3')
    def __enter__(self):return self.chem
    def __exit__(self,*args):self.chem.close()
