import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.qm9 import parse_xyz, prepare, HARTREE_TO_EV


def record(ident=1, smiles='C'):
    # Original QM9 layout: ID, 15 properties, atoms, frequencies, two SMILES, InChIs.
    return ('5\ngdb '+str(ident)+' 1 2 3 0 1 -0.4 0.1 0.5 1 0.1 -40 -40 -40 -40 1\n'
        'C 0 0 0 -0.2\nH 0.6 0.6 0.6 0.05\nH -0.6 -0.6 0.6 0.05\n'
        'H -0.6 0.6 -0.6 0.05\nH 0.6 -0.6 -0.6 0.05\n100 200\n'+smiles+' '+smiles+'\nInChI=1S/CH4/h1H4 InChI=1S/CH4/h1H4\n')

class QM9Tests(unittest.TestCase):
    def test_layout_and_units(self):
        row, xyz=parse_xyz(record().replace('0.6','6.0*^-1'))
        self.assertEqual(row['qm9_id'],1)
        self.assertAlmostEqual(row['gap_ev'],0.5*HARTREE_TO_EV)
        self.assertEqual(row['SMILES'],'C')
        self.assertTrue(xyz.startswith('5\n'))
        with self.assertRaises(ValueError):parse_xyz(record().replace('0.1 0.5','0.1 0.9'))
        with self.assertRaises(ValueError):parse_xyz(record().replace('C 0 0 0','F 0 0 0'))

    def test_archive_exclusions_duplicates_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'qm9.tar.bz2'
            with tarfile.open(archive,'w:bz2') as tar:
                for name,content in [('one.xyz',record(1)),('../../escape.xyz',record(2)),('bad.xyz','broken'),('three.xyz',record(3))]:
                    data=content.encode();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            excluded=root/'uncharacterized.txt';excluded.write_text('Header\n3 C C\n')
            result=prepare(archive,excluded,root/'out',sample=1)
            df=pd.read_csv(result)
            self.assertEqual(len(df),1)
            self.assertEqual(df.qm9_id.iloc[0],1)
            self.assertEqual(len(pd.read_csv(root/'out/rejections.csv')),3)
            self.assertFalse((root/'escape.xyz').exists())
            meta=json.loads((root/'out/metadata.json').read_text())
            self.assertEqual(meta['selected'],1)
            self.assertIn('n_conj_bonds',df)
            self.assertNotIn('morgan_0',df)
            with self.assertRaises(FileExistsError):prepare(archive,excluded,root/'out')

if __name__=='__main__':unittest.main()
