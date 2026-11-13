import csv
import json
import tempfile
import unittest
from pathlib import Path
from src.active_learning.config import validate
from src.active_learning.artifacts import file_hash
from src.active_learning.experiment import initialize
from src.active_learning.state import Store
from src.active_learning.runner import run,select


def fixture(root,mode='random'):
    source=root/'source.csv'
    from test_al_partitions import records
    with source.open('w') as f:
        w=csv.writer(f);w.writerow(['source_record_id','input_smiles','gap_ev'])
        for i,r in enumerate(records()):w.writerow([r.source_record_id,r.input_smiles,1+i*.03])
    return validate({'dataset':'synthetic','source':{'path':str(source),'sha256':file_hash(source)},
        'repeat_seeds':[42], 'split':{'mode':mode},'representation':{'bits':32},
        'budgets':{'initial':4,'batch':2,'max_labels':8,'max_attempts':8,'rounds':2},
        'models':{'members':3,'trees':2,'downstream_trees':2}})

class State(unittest.TestCase):
    def test_resume_membership_predictions_and_single_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);a=initialize(c,root/'a');b=initialize(c,root/'b')
            with Store(a) as s:run(s)
            with Store(b) as s:
                run(s,stop_after=1)
                with self.assertRaises(RuntimeError):
                    with Store(b):pass
            with Store(b) as s:run(s)
            with Store(a) as s,Store(b) as t:
                self.assertEqual(s.rows('SELECT repeat,strategy,round,id FROM selections ORDER BY repeat,strategy,round,id'),t.rows('SELECT repeat,strategy,round,id FROM selections ORDER BY repeat,strategy,round,id'))
                for cp in s.rows('SELECT * FROM checkpoints'):
                    other=t.rows('SELECT * FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(cp['repeat'],cp['strategy'],cp['round']))[0]
                    x=json.loads((a/cp['predictions']).read_text());y=json.loads((b/other['predictions']).read_text())
                    self.assertEqual([r['predicted_gap_ev'] for r in x],[r['predicted_gap_ev'] for r in y])

    def test_crashes_reservations_and_atomic_delivery(self):
        for point in ['before_reservation','after_reservation','after_artifact_rename','after_label_delivery','before_database_commit','after_database_commit']:
            with self.subTest(point=point),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);c=fixture(root);path=initialize(c,root/'experiment')
                with Store(path) as s:
                    with self.assertRaises(RuntimeError):run(s,failpoint=point)
                with Store(path) as s:
                    run(s)
                    self.assertEqual(len(s.labels(0,'random')),8)
                    self.assertEqual(s.db.execute('SELECT COUNT(*) FROM selections WHERE strategy=?',('random',)).fetchone()[0],4)

    def test_drift_tamper_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root);path=initialize(c,root/'experiment')
            with Store(path) as s:
                self.assertEqual(select(s,0,'random')['status'],'reserved')
                self.assertEqual(select(s,0,'random')['status'],'pending')
            with Store(path) as s:
                s.db.execute("UPDATE partitions SET origin='final' WHERE id=(SELECT MIN(id) FROM partitions)")
            with self.assertRaises(ValueError):
                with Store(path):pass
