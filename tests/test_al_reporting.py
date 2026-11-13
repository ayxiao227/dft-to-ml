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
                with patch('src.active_learning.oracle.EvaluationOracle.read',side_effect=AssertionError('No new final access')):
                    self.assertEqual(finalize(s),result)
                    output=report(s);self.assertTrue(Path(output['report']).exists())
                    self.assertEqual(report(s),output)
                with self.assertRaises(ValueError):select(s,0,'random')
                self.assertGreater(len(list(Path(output['report']).parent.glob('*.svg'))),0)
                predictions=json.loads((experiment/'final/predictions.json').read_text())
                self.assertAlmostEqual(predictions[0]['residual_ev'],predictions[0]['predicted_gap_ev']-predictions[0]['observed_gap_ev'])
