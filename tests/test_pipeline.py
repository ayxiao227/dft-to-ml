import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.csv_conversion import passes_filters, validated_molecules
from src.schema import normalize_molecules, molecule_key
from src.structure_automation import generate_slurm_scripts, generate_slurm_scripts_from_csv, create_txt_file
from src.dft_parser import parse_log
from src.run_metadata import check_molecular_identity

GOOD = ''' Optimization completed.
 Alpha  occ. eigenvalues -- -0.50000
 Alpha virt. eigenvalues -- 0.10000
 Frequencies -- 100.000 200.000 300.000
 Normal termination of Gaussian
'''

class PipelineTests(unittest.TestCase):
    def test_filters(self):
        for smi in ['[CH3]', 'C.C', '[13CH4]', '[NH4+]', None, 'broken']:
            self.assertFalse(passes_filters(smi), smi)
        self.assertTrue(passes_filters('CCO'))

    def test_schema_and_identity(self):
        df = normalize_molecules(pd.DataFrame({'ConnectivitySMILES':['CCO']}))
        self.assertEqual(df.SMILES.iloc[0], 'CCO')
        self.assertEqual(molecule_key('CCO'), molecule_key('OCC'))
        with self.assertRaises(ValueError):
            check_molecular_identity(pd.DataFrame({'SMILES':['CCO','OCC']}))
        self.assertEqual(len(validated_molecules(pd.DataFrame({'SMILES':['CCO','OCC']}))), 1)

    def test_sampling(self):
        df = pd.DataFrame({'SMILES':['C','CC','CCC']})
        with patch('src.structure_automation.generate_slurm_scripts') as generate:
            generate_slurm_scripts_from_csv(df, randomize='')
            self.assertEqual(len(generate.call_args.args[0]), 3)
            generate_slurm_scripts_from_csv(df, randomize=2)
            self.assertEqual(len(generate.call_args.args[0]), 2)
            self.assertFalse(generate.call_args.args[0].duplicated().any())
        with self.assertRaises(ValueError):
            generate_slurm_scripts_from_csv(df, randomize=4)

    def test_parser_gates_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'file':['good.log','bad.log','missing.log'],
                          'ConnectivitySMILES':['C','CC','CCC'], 'molecule_id':['a','b','c'], 'run_id':['a','b','c'], 'input_path':['x']*3, 'input_sha256':['x']*3, 'protocol_id':['p']*3}).to_csv(root/'manifest.csv',index=False)
            (root/'good.log').write_text(GOOD)
            (root/'bad.log').write_text(GOOD+'Error termination\n')
            with patch('src.dft_parser.inspect_record', return_value={'xyz':'1\nx\nH 0 0 0\n','log_sha256':'hash','optimized_smiles':'C','protocol_id':'p'}):
                df=parse_log(root,'x',root/'manifest.csv')
            self.assertEqual(df.file.tolist(), ['good.log'])
            self.assertAlmostEqual(df.gap_ev.iloc[0],16.3268317476)
            self.assertEqual(len(pd.read_csv(root/'parse_rejections.csv')),2)
            (root/'good.log').write_text(GOOD.replace('100.000','-100.000'))
            with self.assertRaises(ValueError):parse_log(root,'x',root/'manifest.csv')

    def test_generate_actual_gaussian_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            before=os.getcwd();os.chdir(tmp)
            try:
                result=generate_slurm_scripts(pd.DataFrame({'SMILES':['C']}),ID='test',gjf_options={'nproc':2,'mem':'2GB'})
                self.assertTrue(result.generated.all())
                gjf=next(Path('molecules').rglob('*.gjf')).read_text()
                self.assertIn('%nprocshared=2',gjf)
                script=next(Path('scripts').rglob('*.txt'))
                self.assertIn('--cpus-per-task=2',script.read_text())
                self.assertIn('--mem=4G',script.read_text())
                subprocess.run(['bash','-n',str(script)],check=True)
                with self.assertRaises(FileExistsError):
                    generate_slurm_scripts(pd.DataFrame({'SMILES':['C']}),ID='test')
            finally:os.chdir(before)

    def test_shell_retains_scratch_on_transfer_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'m.gjf';source.write_text('input')
            dest=root/'blocked';dest.write_text('not a directory')
            script=create_txt_file('m',root,'x',user='tester',scratch_dir=tmp)
            content=script.read_text().replace('/home/tester/dft_ml/gjf_files/x/m.gjf',str(source)).replace('/home/tester/dft_ml/log_files/x',str(dest))
            # Stub Gaussian/module commands, but execute the real shell control flow.
            content=content.replace('set -euo pipefail', """set -euo pipefail
module() { :; }
g16() { echo 'Normal termination of Gaussian' > input.log; touch m.chk; }
formchk() { touch output.fchk; }""")
            script.write_text(content)
            result=subprocess.run(['bash',str(script)],env={**os.environ,'SLURM_JOB_ID':'123'},capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(list((root/'tester').glob('*/input.log'))),1)

    def test_ml_run_metadata_and_folds(self):
        from src.descriptors import build_descriptors
        from src.ml_analysis import analysis, cv_check
        with tempfile.TemporaryDirectory() as tmp:
            before=os.getcwd();os.chdir(tmp)
            try:
                pd.DataFrame({'SMILES':['C'*n for n in range(1,31)],
                              'gap_ev':[10/(n+1) for n in range(1,31)], 'optimized_smiles':['C'*n for n in range(1,31)], 'accepted':[True]*30, 'protocol_id':['test']*30}).to_csv('labels.csv',index=False)
                build_descriptors('labels.csv','features.csv')
                analysis('features.csv','smoke',n_estimators=2)
                cv_check('features.csv',n_splits=2,n_estimators=2)
                from src.splits import freeze
                from src.final_evaluate import evaluate
                freeze('features.csv','frozen')
                evaluate('frozen/development.csv','frozen/final_test.csv','final',trees=2)
                with self.assertRaises(ValueError):
                    evaluate('frozen/development.csv','frozen/development.csv','overlap',trees=2)
                runs=list(Path('results/runs').iterdir())
                self.assertEqual(len(runs),2)
                self.assertTrue(all((run/'metadata.json').exists() for run in runs))
                self.assertEqual(len(list(Path('results/runs').rglob('split.csv'))),1)
                self.assertEqual(len(list(Path('results/runs').rglob('validation_folds.csv'))),1)
            finally:os.chdir(before)

    def test_shell_rejects_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):create_txt_file('m',tmp,'x',user='x; touch BAD')

    def test_import_descriptors_has_no_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            env={**os.environ,'PYTHONPATH':os.getcwd()}
            subprocess.run([sys.executable,'-c','import src.descriptors'],cwd=tmp,env=env,check=True)
            self.assertEqual(list(Path(tmp).iterdir()),[])

if __name__ == '__main__':unittest.main()
