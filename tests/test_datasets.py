import gzip
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.datasets import prepare_gdb13, prepare_tmqm
from src.qm9 import HARTREE_TO_EV

class DatasetTests(unittest.TestCase):
    def test_mixed_protocols_rejected(self):
        from src.run_metadata import check_molecular_identity
        with self.assertRaises(ValueError):
            check_molecular_identity(pd.DataFrame({'SMILES':['C','CC'], 'protocol_id':['QM9','tmQM']}))

    def test_gdb_unlabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);source=r/'gdb.smi.gz'
            with gzip.open(source,'wt') as f:f.write('C\nCC\nCCC\nCCCC\ninvalid\n')
            path=prepare_gdb13(source,r/'out',sample=3)
            df=pd.read_csv(path)
            self.assertEqual(len(df),3)
            self.assertNotIn('gap_ev',df)
            self.assertTrue(df.label_status.eq('unlabeled').all())
            self.assertEqual(len(pd.read_csv(r/'out/rejections.csv')),1)

    def test_tmqm_units_metal_and_missing_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);source=r/'tmQM_y.csv'
            source.write_text('CSD_code;SMILES;HOMO_Energy;LUMO_Energy;HL_Gap\n'
                'A;[Ru];-0.2;-0.1;0.1\nB;;-0.2;-0.1;0.1\nC;C;-0.2;-0.1;0.1\n')
            path=prepare_tmqm(source,r/'out',sample=1)
            df=pd.read_csv(path)
            self.assertEqual(df.MetalAtomicNumber.iloc[0],44)
            self.assertAlmostEqual(df.gap_ev.iloc[0],0.1*HARTREE_TO_EV)
            self.assertNotIn('Metal_q',df)
            self.assertEqual(len(pd.read_csv(r/'out/rejections.csv')),2)
            with self.assertRaises(FileExistsError):prepare_tmqm(source,r/'out',sample=1)

if __name__=='__main__':unittest.main()
