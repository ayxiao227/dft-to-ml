import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.active_learning.evaluation import metrics
from src.active_learning.experiment import initialize
from src.active_learning.state import Store
from src.active_learning.runner import run,select
from src.active_learning.finalization import finalize
from src.active_learning.reporting import report
from test_al_state import fixture

class Reporting(unittest.TestCase):
    def test_metric_edges_and_sign(self):
        a=metrics([1],[3],[0]);self.assertEqual(a['MAE_eV'],2);self.assertIsNone(a['R2']);self.assertIsNone(a['uncertainty_error_spearman'])
        a=metrics([1,1],[2,3],[0,0]);self.assertIsNone(a['R2']);self.assertEqual(a['P90_AE_eV'],1.9)
        tied=metrics([0,0,0,0],[0,1,1,2],[0,0,1,1])
        self.assertEqual(tied['median_AE_eV'],1.)
        self.assertAlmostEqual(tied['P90_AE_eV'],1.7,places=14)
        self.assertAlmostEqual(tied['uncertainty_error_spearman'],1/np.sqrt(2),places=14)
        with self.assertRaises(ValueError):metrics([1],[float('nan')])

    def test_finalization_retry_no_fit_and_no_second_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);c['acquisition']['strategies']=['random'];experiment=initialize(c,root/'experiment')
            with Store(experiment) as s:run(s)
            for failpoint in ['after_finalization_freeze','after_artifact_rename','before_database_commit']:
                with Store(experiment) as s:
                    with self.assertRaises(RuntimeError):finalize(s,failpoint)
            with Store(experiment) as s,patch('src.active_learning.models.fit',side_effect=AssertionError('No fitting allowed')):
                result=finalize(s);self.assertEqual(result['status'],'completed')
                with patch('src.active_learning.oracle.EvaluationOracle.read',side_effect=AssertionError('No new final access')),patch('src.active_learning.models.predict',side_effect=AssertionError('No new predictions')):
                    self.assertEqual(finalize(s),result)
                    with self.assertRaisesRegex(RuntimeError,'Injected crash'):report(s,'before_report_ledger_commit')
                    self.assertEqual(s.db.execute("SELECT COUNT(*) FROM artifacts WHERE kind='report'").fetchone()[0],0)
                    # Progress is readable while this Store owns the single-writer lock.
                    from src.active_learning.__main__ import lightweight_progress
                    self.assertIn('complete-report',lightweight_progress(experiment)['reporting'][0]['committed_stages'])
                    output=report(s);self.assertTrue(Path(output['report']).exists())
                    with self.assertRaisesRegex(RuntimeError,'Injected crash'):report(s,'after_report_ledger_commit')
                    self.assertEqual(report(s),output)
                    evidence=Path(output['report']).parent/'.metrics.csv.interrupted'
                    evidence.write_bytes(b'partial')
                    with self.assertRaisesRegex(ValueError,'Interrupted report publication evidence retained'):report(s)
                    self.assertEqual(evidence.read_bytes(),b'partial');evidence.unlink()
                    import pandas as pd
                    table=pd.read_csv(Path(output['report']).parent/'metrics.csv')
                    for saved in s.rows('SELECT * FROM metrics'):
                        actual=table[(table['repeat']==saved['repeat'])&(table.strategy==saved['strategy'])&(table['round']==saved['round'])&(table.partition==saved['partition'])&(table.evaluator==saved['evaluator'])].iloc[0]
                        for key,value in json.loads(saved['data']).items():
                            if isinstance(value,(int,float)):self.assertAlmostEqual(actual[key],value,places=13)
                    from src.active_learning.reporting import report_identity
                    from src.active_learning.report_cache import SETTINGS
                    old_identity=report_identity(s)
                    with patch.dict(SETTINGS,plot_limit=7):self.assertNotEqual(report_identity(s),old_identity)
                    with s.transaction():s.db.execute("UPDATE attempts SET reason='audited reason' WHERE id=(SELECT MIN(id) FROM attempts)")
                    self.assertNotEqual(report_identity(s),old_identity)
                with self.assertRaises(ValueError):select(s,0,'random')
                self.assertGreater(len(list(Path(output['report']).parent.glob('*.svg'))),0)
                from src.active_learning.finalization import verified_results
                from src.active_learning.streaming import prediction_records
                doc=next(iter(verified_results(s).values()))
                predictions=list(prediction_records(experiment/doc['predictions']['path']))
                self.assertAlmostEqual(predictions[0]['residual_ev'],predictions[0]['predicted_gap_ev']-predictions[0]['observed_gap_ev'])
