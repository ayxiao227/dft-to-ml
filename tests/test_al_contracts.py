import copy
import unittest
from unittest.mock import patch
from src.active_learning.config import validate, hashed
from src.active_learning.records import MolecularRecord, LabelRecord, check_collisions
from src.gaussian_records import protocol


def config():
    return {'dataset': 'synthetic', 'source': {'path': 'fixture.csv', 'sha256': 'a'*64},
            'budgets': {'initial': 2, 'batch': 2, 'max_labels': 6, 'max_attempts': 6, 'rounds': 2}}

class Contracts(unittest.TestCase):
    def test_strict_config(self):
        c = validate(config())
        self.assertEqual(hashed(c), hashed(validate(c)))
        for key, value in [('oops', 1), ('schema_version', 2), ('units', 'hartree'),
                           ('dataset', 'tmqm'), ('target_mae_ev', float('nan'))]:
            bad = config(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate(bad)
        for field, value in [('batch', -1), ('initial', True), ('max_attempts', 1)]:
            bad=config(); bad['budgets'][field]=value
            with self.assertRaises(ValueError): validate(bad)
        bad=config(); bad['models']={'oops':1}
        with self.assertRaises(ValueError): validate(bad)
        bad=config(); bad['split']={'final_fraction':.9, 'validation_fraction':.9}
        with self.assertRaises(ValueError): validate(bad)

    def test_records_and_collision(self):
        a=MolecularRecord.create('GDB-13','upstream','OCC','a'*64)
        b=MolecularRecord.create('GDB-13','other','CCO','a'*64)
        self.assertEqual(a.input_structure_key,b.input_structure_key)
        check_collisions([a,b])
        with self.assertRaises(ValueError): check_collisions([a,a])
        with patch('src.active_learning.records.molecule_key', return_value='collision'):
            a=MolecularRecord.create('x','1','C','a'*64)
            b=MolecularRecord.create('x','2','CC','a'*64)
            with self.assertRaises(ValueError): check_collisions([a,b])
        with self.assertRaises(ValueError): LabelRecord('x','s','c',1,2.,'guessed','p')
        with self.assertRaises(ValueError): LabelRecord('x','s','c',1,float('inf'),'publisher_qm9','p')

    def test_generation_contract(self):
        for opts in [{'multiplicity':True}, {'basis':None}, {'multiplicity':3}, {'solvent':'water'}, {'basis':'auto'},
                     {'method':'b3lyp sp'}, {'charge':1}, {'basis_task':'energy'}]:
            with self.subTest(opts=opts), self.assertRaises(ValueError): protocol(opts)
