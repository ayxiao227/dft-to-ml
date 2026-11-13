import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.active_learning.config import validate
from src.active_learning.features import featurize, CandidateFeatures
from src.active_learning.records import MolecularRecord
from src.active_learning.models import fit,predict,serialize,load
from src.active_learning.artifacts import atomic_write
from src.active_learning.acquisition import maxmin,disagreement,select,candidate_subset


def small_config():
    return validate({'dataset':'synthetic','source':{'path':'x','sha256':'a'*64},
        'budgets':{'initial':2,'batch':2,'max_labels':6,'max_attempts':6,'rounds':2},
        'models':{'members':3,'trees':2,'downstream_trees':2},'representation':{'bits':32}})

class Algorithms(unittest.TestCase):
    def test_hand_computed_disagreement_maxmin(self):
        mean,std=disagreement([[1,2],[3,2],[5,2]])
        np.testing.assert_allclose(mean,[3,2]);np.testing.assert_allclose(std,[2,0])
        ids=['c','b','a'];fp=np.array([[0,1],[1,1],[1,0]],dtype=np.uint8)
        picked=maxmin(ids,fp,[[1,0]],3,1)
        self.assertEqual([r['source_record_id'] for r in picked],['c','b','a'])
        np.testing.assert_allclose([r['diversity_distance'] for r in picked],[1,.5,0])
        self.assertEqual(maxmin(['b','a'],np.zeros((2,2),dtype=np.uint8),[],2)[0]['source_record_id'],'a')
        for bad in [[[np.nan],[1]],[[1]]]:
            with self.assertRaises(ValueError):disagreement(bad)

    def test_strategies_caps_and_degeneracy(self):
        c=small_config();ids=tuple('abcdefgh')
        features=CandidateFeatures(ids,ids,('HeavyAtomCount',),np.arange(8).reshape(-1,1),np.array([[int(i==j) for j in range(8)] for i in range(8)],dtype=np.uint8),'x')
        for strategy in ['random','diversity','uncertainty','uncertainty_diversity']:
            rows,meta=select(strategy,list(ids[2:]),ids[:2],features,2,c,0,1,lambda ids:np.zeros(len(ids)))
            self.assertEqual(len({r['source_record_id'] for r in rows}),2)
            self.assertFalse(set(r['source_record_id'] for r in rows)&set(ids[:2]))
            if strategy.startswith('uncertainty'):self.assertIsNotNone(meta['warning'])
            other=select(strategy,list(reversed(ids[2:])),ids[:2],features,2,c,0,1,lambda ids:np.zeros(len(ids)))
            self.assertEqual((rows,meta),other)
        c['acquisition']['candidate_cap']=3
        _,m=select('random',ids[2:],ids[:2],features,2,c,0,1)
        self.assertEqual(len(m['candidate_ids']),3);self.assertTrue(m['capped'])
        self.assertEqual(candidate_subset(ids,3,42),candidate_subset(ids[::-1],3,42))
        picks={i:0 for i in ids[2:]}
        for repeat_seed in range(200):
            c['repeat_seeds']=[repeat_seed]
            rows,_=select('random',ids[2:],ids[:2],features,1,c,0,1)
            picks[rows[0]['source_record_id']]+=1
        self.assertTrue(all(v>10 for v in picks.values()))
        with self.assertRaises(ValueError):select('uncertainty',ids[2:],ids[:2],features,2,c,0,1,lambda ids:[np.nan]*len(ids))

    def test_saved_models_train_only_scaler_and_chunking(self):
        c=small_config();records=[MolecularRecord.create('x',str(i),s,'a'*64) for i,s in enumerate(['C','CC','CCC','CCCC','CCCCC'])]
        f,_,_=featurize(records,c['representation']);bundle=fit(f,{'0':1.,'1':2.},c,0,0)
        np.testing.assert_allclose(bundle['evaluators']['ridge'][0].mean_,f.rows(['0','1']).mean(axis=0))
        self.assertTrue(all(not m.bootstrap for m in bundle['committee']))
        a=predict(bundle,f,f.ids,1);b=predict(bundle,f,f.ids,100)
        for k in a:np.testing.assert_allclose(a[k],b[k])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pkl';sha=atomic_write(p,serialize(bundle))
            restored=load(p,sha,c,f.settings_hash)
            self.assertEqual(predict(restored,f,f.ids),b)
            p.write_bytes(b'tampered')
            with self.assertRaises(ValueError):load(p,sha,c,f.settings_hash)
