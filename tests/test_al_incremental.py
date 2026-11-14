"""Deterministic numerical, migration, publication and integrity regression tests."""
import contextlib
import io
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal
from src.active_learning.config import canonical,hashed
from src.active_learning.experiment import initialize
from src.active_learning.state import Store,TABLES
from src.active_learning.runner import run
from src.active_learning.finalization import finalize,verified_results,paths,progress,contract_for,completion_key
from src.active_learning.reporting import report
from src.active_learning.streaming import prediction_records,meta_values
from src.active_learning.__main__ import lightweight_progress,status
from src.active_learning import models
from test_al_state import fixture
from legacy_finalization_reference import finalize as legacy_finalize


class Incremental(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);c=fixture(self.root)
        c['acquisition']['strategies']=['random'];self.path=initialize(c,self.root/'experiment')
        with Store(self.path) as s:run(s)

    def test_numeric_and_report_equivalence_to_legacy(self):
        legacy=self.root/'legacy';shutil.copytree(self.path,legacy)
        with Store(legacy) as s,patch.object(models,'fit',side_effect=AssertionError('no fit')):
            legacy_finalize(s)
            expected=json.loads((legacy/'final/predictions.json').read_text())
            metric_rows=s.rows("SELECT * FROM metrics WHERE partition='final' ORDER BY repeat,strategy,round,evaluator")
            with patch.object(models,'predict',side_effect=AssertionError('report must not predict')):
                legacy_report=Path(report(s)['report']).parent
            with patch.object(models,'predict',side_effect=AssertionError('no reevaluation')):
                legacy_finalize(s);finalize(s)
        with Store(self.path) as s,patch.object(models,'fit',side_effect=AssertionError('no fit')):
            result=finalize(s);docs=verified_results(s)
            actual=[r for doc in docs.values() for r in prediction_records(s.root/doc['predictions']['path'])]
            self.assertEqual(actual,expected)
            self.assertEqual(s.rows("SELECT * FROM metrics WHERE partition='final' ORDER BY repeat,strategy,round,evaluator"),metric_rows)
            with patch.object(models,'predict',side_effect=AssertionError('no reevaluation')):
                finalize(s);out=report(s);self.assertEqual(out,report(s))
            current=Path(out['report']).parent
            for path in legacy_report.glob('*.csv'):
                if path.stat().st_size<2:continue
                try:a=pd.read_csv(path);b=pd.read_csv(current/path.name)
                except pd.errors.EmptyDataError:continue
                assert_frame_equal(a,b,check_exact=False,rtol=1e-14,atol=1e-14,obj=path.name)
            for path in legacy_report.glob('*.png'):self.assertEqual(path.read_bytes(),(current/path.name).read_bytes(),path.name)
            self.assertEqual(result['format_version'],2)
            provenance=json.loads((current/'provenance.json').read_text())
            self.assertTrue(any(a['kind']=='final_checkpoint_manifest' for a in provenance['prediction_artifacts']))

    def test_all_checkpoint_boundaries(self):
        for boundary,committed,published in [
            ('after_finalization_freeze',0,False),('before_final_publication',0,False),
            ('after_artifact_rename',0,False),('after_final_result_publication',0,True),
            ('before_database_commit',0,True),('after_database_commit',1,True),('after_final_checkpoint_commit',1,True)]:
            with self.subTest(boundary=boundary):
                root=self.root/boundary;shutil.copytree(self.path,root)
                with Store(root) as s:
                    count=s.db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]
                    with self.assertRaisesRegex(RuntimeError,'Injected crash'):finalize(s,boundary)
                    self.assertEqual(progress(s)['completed'],committed)
                    self.assertEqual(lightweight_progress(root)['completed'],committed)
                    with self.assertRaises(RuntimeError):
                        with Store(root):pass
                with Store(root) as s,patch.object(models,'fit',side_effect=AssertionError('no fit')),patch.object(models,'predict',wraps=models.predict) as predict:
                    result=finalize(s)
                    self.assertEqual(predict.call_count,count-(1 if published else 0))
                    events=s.rows("SELECT * FROM events WHERE kind='final_checkpoint_completed'")
                    self.assertEqual(len(events),count)
                    self.assertEqual(s.db.execute("SELECT COUNT(*) FROM metrics WHERE partition='final'").fetchone()[0],count*4)
                    with patch.object(models,'predict',side_effect=AssertionError('no repeated prediction')):self.assertEqual(finalize(s),result)
                    self.assertEqual(len(s.rows("SELECT * FROM events WHERE kind='final_checkpoint_completed'")),count)
                    self.assertEqual(status(s)['finalization_progress']['completed'],count)

    def test_overall_manifest_and_completion_boundaries(self):
        for boundary in ['after_final_manifest_publication','before_finalization_commit','after_finalization_commit']:
            with self.subTest(boundary=boundary):
                root=self.root/boundary;shutil.copytree(self.path,root)
                with Store(root) as s:
                    with self.assertRaisesRegex(RuntimeError,'Injected crash'):finalize(s,boundary)
                    self.assertEqual(progress(s)['remaining'],0)
                with Store(root) as s,patch.object(models,'predict',side_effect=AssertionError('no repeated prediction')),patch('src.active_learning.oracle.EvaluationOracle.read',side_effect=AssertionError('no final label reread')):
                    self.assertEqual(finalize(s)['status'],'completed')
                    self.assertEqual(len(s.rows("SELECT * FROM events WHERE kind='finalized'")),1)

    def test_sealed_configuration_and_partition_contract_drift(self):
        for change in ['config','partition','checkpoint']:
            with self.subTest(change=change):
                root=self.root/change;shutil.copytree(self.path,root)
                with Store(root) as s:
                    with self.assertRaises(RuntimeError):finalize(s,'after_finalization_freeze')
                    with s.transaction():
                        if change=='config':
                            config=s.meta('config');config['models']['trees']+=1
                            s.set_meta('config',config);s.set_meta('config_hash',hashed(config))
                        elif change=='partition':
                            parts=s.meta('partitions');parts[0]['hash']='a'*64;s.set_meta('partitions',parts)
                        else:s.db.execute("UPDATE checkpoints SET metadata='{}' WHERE round=0")
                with self.assertRaisesRegex(ValueError,'Frozen finalization contract changed'):
                    with Store(root):pass

    def test_started_migration_and_digest_compatibility(self):
        with Store(self.path) as s:
            cp=s.rows('SELECT * FROM checkpoints ORDER BY repeat,strategy,round');contract=contract_for(s,cp)
            with s.transaction():
                s.set_meta('finalization',{**contract,'status':'started'})
                s.set_meta('escaped',{'text':('a\\\"\n\u00e9'*20000)})
            old={table:s.rows(f'SELECT * FROM {table} ORDER BY '+','.join(str(i+1) for i in range(len(s.db.execute(f'SELECT * FROM {table} LIMIT 0').description)))) for table in TABLES}
            old['meta']=s.rows("SELECT * FROM meta WHERE key != 'integrity' ORDER BY key")
            self.assertEqual(s._digest(),hashed(old))
            self.assertEqual(list(meta_values(s,'partitions','item.hash')),contract['partition_hashes'])
        with Store(self.path) as s:
            finalize(s);self.assertEqual(s.meta('finalization')['config_hash'],contract['config_hash'])
            self.assertEqual(s.meta('finalization')['checkpoint_set_hash'],contract['checkpoint_set_hash'])

    def test_corruption_rejected_and_evidence_preserved(self):
        with Store(self.path) as s:finalize(s)
        for kind in ['model','config','partition','manifest','prediction','completion']:
            with self.subTest(kind=kind):
                root=self.root/kind;shutil.copytree(self.path,root)
                if kind in ('config','partition','completion'):
                    with sqlite3.connect(root/'state.sqlite3') as db:
                        if kind=='config':db.execute("UPDATE meta SET value='{}' WHERE key='config'")
                        elif kind=='completion':db.execute("DELETE FROM meta WHERE key LIKE 'final_checkpoint/%'")
                        else:db.execute("UPDATE partitions SET group_key='corrupt'")
                else:
                    with sqlite3.connect(root/'state.sqlite3') as db:
                        artifact_kind={'model':'model','manifest':'final_checkpoint_manifest','prediction':'final_predictions'}[kind]
                        relative=db.execute('SELECT path FROM artifacts WHERE kind=? LIMIT 1',(artifact_kind,)).fetchone()[0]
                    path=root/relative;path.write_bytes(path.read_bytes()+b'corrupt');before=path.read_bytes()
                with self.assertRaises(ValueError):
                    with Store(root):pass
                if kind not in ('config','partition','completion'):self.assertEqual(path.read_bytes(),before)

    def test_uncommitted_corrupt_artifact_and_manifest_rejected(self):
        for kind in ['prediction','manifest','metric','membership','malformed']:
            with self.subTest(kind=kind):
                root=self.root/kind;shutil.copytree(self.path,root)
                with Store(root) as s:
                    with self.assertRaises(RuntimeError):finalize(s,'after_final_result_publication')
                    cp=s.rows('SELECT * FROM checkpoints ORDER BY repeat,strategy,round')[0];pred,manifest=paths(cp)
                    if kind=='prediction':
                        path=root/pred;path.write_text('corrupt')
                    elif kind=='malformed':
                        path=root/manifest;path.write_text('corrupt')
                    else:
                        path=root/manifest;doc=json.loads(path.read_text())
                        if kind=='manifest':doc['checkpoint']['round']+=1
                        elif kind=='metric':doc['metrics']['rf']['MAE_eV']+=1
                        else:
                            from src.active_learning.artifacts import file_hash
                            pp=root/pred;rows=list(prediction_records(pp));rows[0]['source_record_id']='wrong'
                            pp.write_text(''.join(canonical(r)+'\n' for r in rows));doc['predictions']['sha256']=file_hash(pp)
                        path.write_text(canonical(doc))
                    before=path.read_bytes()
                with Store(root) as s,patch.object(models,'predict',side_effect=AssertionError('do not overwrite evidence')):
                    with self.assertRaises(ValueError):finalize(s)
                self.assertEqual(path.read_bytes(),before)

    def test_partial_results_hidden_and_progress_under_lock(self):
        with Store(self.path) as s:
            with self.assertRaises(RuntimeError):finalize(s,'after_final_checkpoint_commit')
            self.assertEqual(lightweight_progress(self.path)['completed'],1)
            self.assertFalse(lightweight_progress(self.path)['final_metrics_complete'])
            with patch.object(models,'fit',side_effect=AssertionError('no fit')),patch.object(models,'predict',side_effect=AssertionError('no prediction')):
                out=Path(report(s)['report']).parent
            self.assertEqual(set(pd.read_csv(out/'predictions.csv').partition),{'validation'})
            self.assertEqual(set(pd.read_csv(out/'metrics.csv').partition),{'validation'})

    def test_metadata_lookup_is_bounded(self):
        from src.active_learning.evaluation import prediction_rows
        with Store(self.path) as s:
            queries=[];s.db.set_trace_callback(queries.append)
            ident=s.db.execute('SELECT id FROM molecules LIMIT 1').fetchone()[0]
            rows=prediction_rows(s,0,'random',0,'hash',[ident],{ident:{'gap_ev':1.}}, {'mean':[2.],'uncertainty':[0.]},4,4,'final')
            self.assertEqual(rows[0]['residual_ev'],1.)
            self.assertFalse(any(q=='SELECT * FROM molecules' for q in queries))
            self.assertTrue(any('WHERE id IN (' in q for q in queries))
