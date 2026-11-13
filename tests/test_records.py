import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.gaussian_records import reconstruct, routes, identity_matches, protocol, inspect_record
from src.structure_automation import generate_slurm_scripts
from src.acquisition import fetch
from src.splits import freeze

METHANE = ''' Gaussian 16: ES64L-G16RevC.01
 Charge = 0 Multiplicity = 1
 Standard orientation:
 ---------------------------------
 Center Atomic Atomic Coordinates
 Number Number Type X Y Z
 ---------------------------------
 1 6 0 0.000 0.000 0.000
 2 1 0 0.629 0.629 0.629
 3 1 0 -0.629 -0.629 0.629
 4 1 0 -0.629 0.629 -0.629
 5 1 0 0.629 -0.629 -0.629
 ---------------------------------
'''

class RecordTests(unittest.TestCase):
    def test_geometry(self):
        smi, xyz = reconstruct(METHANE)
        self.assertEqual(smi,'C')
        self.assertTrue(xyz.startswith('5\n'))
        self.assertTrue(identity_matches('CCOC(C)O','CCO[C@@H](C)O'))
        self.assertFalse(identity_matches('CCO','COC'))
        with self.assertRaises(ValueError):reconstruct(METHANE.replace('1 6 0','1 44 0'))

    def test_protocol_and_provenance(self):
        self.assertNotEqual(protocol({'basis':'def2-TZVP'}),protocol({'basis':'def2-SVP'}))
        with tempfile.TemporaryDirectory() as tmp:
            before=os.getcwd();os.chdir(tmp)
            try:
                manifest=generate_slurm_scripts(pd.DataFrame({'SMILES':['C']}),ID='test')
                row=manifest.iloc[0];folder=Path('molecules/gjf_files/test')
                source=(folder/row.input_path).read_text()
                route=next(l for l in source.splitlines() if l.startswith('#'))
                log=Path('m.log');log.write_text(' ----->\n'+route+'\n --------\nProtocol: '+row.protocol_id+'\n'+METHANE)
                result=inspect_record(log,row,folder)
                self.assertEqual(result['optimized_smiles'],'C')
                log.write_text(log.read_text().replace('opt=calcfc freq','sp'))
                with self.assertRaises(ValueError):inspect_record(log,row,folder)
                resumed=generate_slurm_scripts(pd.DataFrame({'SMILES':['C']}),ID='test',resume=True)
                self.assertEqual(len(resumed),1)
                with self.assertRaises(ValueError):
                    generate_slurm_scripts(pd.DataFrame({'SMILES':['C']}),ID='test',resume=True,gjf_options={'basis':'def2-SVP'})
            finally:os.chdir(before)

    def test_retry_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch('src.acquisition.pcp.get_properties',side_effect=[TimeoutError('timeout'),[{'CID':1}]]) as request, patch('src.acquisition.time.sleep'):
            self.assertEqual(fetch(['x'],1,'cid',tmp),[{'CID':1}])
            self.assertEqual(fetch(['x'],1,'cid',tmp),[{'CID':1}])
            self.assertEqual(request.call_count,2)

    def test_frozen_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'m.csv'
            pd.DataFrame({'SMILES':['C'*n for n in range(1,21)]}).to_csv(src,index=False)
            freeze(src,root/'split')
            partitions=pd.read_csv(root/'split/partitions.csv')
            self.assertEqual(partitions.groupby('group').partition.nunique().max(),1)
            with self.assertRaises(FileExistsError):freeze(src,root/'split')
