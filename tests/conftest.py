import pytest
import sys
import os
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems

test_mol_dir = '/user/home/rv22218/work/inv_IMPRESSION/EMS/tests/test_mols'

@pytest.fixture(scope='function')
def mol_dir():
    yield test_mol_dir

@pytest.fixture(scope='module')
def testmol_1():
    mol_file = 'testmol_1_NMR.nmredata.sdf'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up SDF molecule testmol_1 (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id)
    print()
    print('Tearing down SDF molecule testmol_1 (scope: module)')

@pytest.fixture(scope='module')
def testmol_xyz():
    mol_file = 'dsgdb9nsd_057135.xyz'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up SDF molecule testmol_1 (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id)
    print()
    print('Tearing down SDF molecule testmol_1 (scope: module)')
