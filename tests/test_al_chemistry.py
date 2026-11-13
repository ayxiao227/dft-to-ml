import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from src.descriptors import input_features, FeatureError, extra_features
from src.active_learning.config import validate
from src.active_learning.records import MolecularRecord
from src.active_learning.features import featurize
from src.active_learning.datasets import deduplicate, validate_labeled
from src.gaussian_records import identity_matches

class Chemistry(unittest.TestCase):
    def test_functional_centers(self):
        cases=[('CN=O',1,0,0,0),('CN=NC',0,0,1,0),('C[N+](C)(C)[O-]',0,1,0,0),
            ('[O-][n+]1ccccc1',0,1,0,0),('C[N+](=O)[O-]',0,0,0,0),
            ('CC(=O)c1ccccc1',0,0,0,1),('C=CC(=O)C',0,0,0,1),('CC(=O)C',0,0,0,0),
            ('C=CC(=O)C=C',0,0,0,1),('CC(=O)N',0,0,0,0),('CC(=O)OC',0,0,0,0),('O=NCCN=O',2,0,0,0)]
        for smi,nitroso,noxide,azo,conj in cases:
            f,_=input_features(smi)
            self.assertEqual(tuple(f[k] for k in ['n_nitroso','n_amine_noxide','n_azo','n_conj_carbonyl_v2']), (nitroso,noxide,azo,conj),smi)
        self.assertEqual(extra_features('CN=NC',False)['n_nitroso_noxide'],1) # frozen v1
        f,_=input_features('C');self.assertEqual(f['NumH'],4)
        for smi in ['broken','',None]:
            with self.assertRaises(FeatureError):input_features(smi)

    def test_identity_classes(self):
        for a,b,expected in [('OCC','CCO',True),('CCO','COC',False),('CC=O','C=CO',False),
            ('FC(Cl)Br','F[C@H](Cl)Br',True),('F[C@H](Cl)Br','F[C@@H](Cl)Br',False)]:
            self.assertEqual(identity_matches(a,b),expected)

    def test_cache_representation_and_no_future_features(self):
        settings=validate({'dataset':'synthetic','source':{'path':'x','sha256':'a'*64}})['representation']
        recs=[MolecularRecord.create('x','id','CCCCC','a'*64)]
        with tempfile.TemporaryDirectory() as tmp:
            a,_,_=featurize(recs,settings,tmp);b,_,_=featurize(recs,settings,tmp)
            np.testing.assert_array_equal(a.x,b.x)
            changed=[replace(recs[0],raw_optimized_smiles='N',optimized_structure_key='future')]
            c,_,_=featurize(changed,settings,tmp);np.testing.assert_array_equal(a.x,c.x)
            count,_,_=featurize(recs,{**settings,'kind':'count'},tmp)
            self.assertFalse(np.array_equal(a.x,count.x));np.testing.assert_array_equal(a.binary,count.binary)
            chiral=[MolecularRecord.create('x','a','F[C@H](Cl)Br','a'*64),MolecularRecord.create('x','b','F[C@@H](Cl)Br','a'*64)]
            f,_,_=featurize(chiral,settings,tmp);self.assertFalse(np.array_equal(f.binary[0],f.binary[1]))
            f,_,_=featurize(chiral,{**settings,'chirality':False},tmp);np.testing.assert_array_equal(f.binary[0],f.binary[1])
            self.assertLess(a.names.index('morgan_2'),a.names.index('morgan_10'))
            self.assertGreater(len(list(Path(tmp).glob('*.json'))),1)

    def test_deterministic_dedup_conflicts_and_protocols(self):
        records=[MolecularRecord.create('x','b','OCC','a'*64,'CCO'),MolecularRecord.create('x','a','CCO','a'*64,'CCO')]
        labels={i:{'gap_ev':1.,'observed_protocol_id':'p'} for i in ['a','b']}
        kept,_,ledger=deduplicate(records,labels);self.assertEqual(kept[0].source_record_id,'a')
        self.assertEqual(deduplicate(records[::-1],labels),(kept,{'a':labels['a']},ledger))
        labels['b']['gap_ev']=2;self.assertEqual(deduplicate(records,labels)[0],[])
        for rows in [[],[{'gap_ev':1.}], [{'gap_ev':1.,'observed_protocol_id':'a'},{'gap_ev':1.,'observed_protocol_id':'b'}]]:
            with self.assertRaises(ValueError):validate_labeled(rows)
