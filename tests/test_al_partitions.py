import json
import tempfile
import unittest
from pathlib import Path
from src.active_learning.records import MolecularRecord
from src.active_learning.partitioning import freeze
from src.active_learning.config import validate
from src.active_learning.oracle import RetrospectiveOracle
from src.splits import chemical_group


def records():
    # Many topology groups, all within the 10-heavy-atom eligibility rule.
    smiles=['C'*n for n in range(1,11)]+['O'+'C'*n for n in range(1,10)]+['N'+'C'*n for n in range(1,10)]+['C1'+'C'*n+'1' for n in range(2,9)]
    return [MolecularRecord.create('synthetic',f'id{i:03}',s,'a'*64,s) for i,s in enumerate(smiles)]

class Partitions(unittest.TestCase):
    def test_frozen_group_and_identity(self):
        c=validate({'dataset':'synthetic','source':{'path':'x','sha256':'a'*64},'budgets':{'initial':4,'batch':2,'max_labels':8,'max_attempts':8}})
        for mode in ['random','group']:
            c['split']['mode']=mode
            a=freeze(records(),c,0);b=freeze(records()[::-1],c,0)
            self.assertEqual(a,b)
            if mode=='group':
                groups={}
                for row in a['rows']:
                    role='development' if row['origin'] in ('initial','pool') else row['origin']
                    groups.setdefault(row['group_key'],set()).add(role)
                self.assertTrue(all(len(v)==1 for v in groups.values()))
            self.assertEqual(a['sizes']['initial'],4)
        self.assertEqual(chemical_group('CCC'),chemical_group('CCO'))
        self.assertEqual(chemical_group('COC'),chemical_group('CCO'))
        self.assertNotEqual(chemical_group('CCC'),chemical_group('CC(C)C'))
        with self.assertRaises(ValueError):freeze(records()[:4],c,0)

    def test_oracle_access_and_spy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pool.json';p.write_text(json.dumps({'i':1.,'p':2.}))
            roles=[{'source_record_id':i,'origin':role} for i,role in [('i','initial'),('p','pool'),('v','validation'),('f','final')]]
            queried=[]
            def reservation(key):
                queried.append(key)
                if key!='reserved':raise ValueError('No reservation')
                return ['p']
            o=RetrospectiveOracle(p,roles,reservation)
            self.assertEqual(o.initial(['i']),{'i':1.})
            self.assertEqual(o.query('reserved',['p']),{'p':2.})
            for ids in [['f'],['v'],['i'],['unknown'],['p','p']]:
                with self.assertRaises(ValueError):o.query('reserved',ids)
            with self.assertRaises(ValueError):o.initial(['p'])
            with self.assertRaises(ValueError):o.query('absent',['p'])
