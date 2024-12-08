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
def testmol_sdf_1():
    mol_file = 'testmol_1_NMR.nmredata.sdf'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up SDF molecule testmol_sdf_1 (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id)
    print()
    print('Tearing down SDF molecule testmol_sdf_1 (scope: module)')

@pytest.fixture(scope='module')
def testmol_xyz():
    mol_file = 'dsgdb9nsd_057135.xyz'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up XYZ molecule testmol_xyz (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id)
    print()
    print('Tearing down XYZF molecule testmol_xyz (scope: module)')

@pytest.fixture(scope='module')
def testmol_smiles_asym():
    smiles = 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
    print()
    print('Setting up SMILES molecule testmol_smiles_asym (scope: module)')
    print(f'SMILES string: {smiles}')
    yield ems.EMS(smiles, line_notation='smi')
    print()
    print('Tearing down SMILES molecule testmol_smiles_asym (scope: module)')

@pytest.fixture(scope='module')
def testmol_rdmol():
    mol_file = 'dsgdb9nsd_057135.xyz'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    RDMol = ems.EMS(path, mol_id=mol_id).rdmol
    print()
    print('Setting up RDMol molecule testmol_rdmol (scope: module)')
    print(f'RDMol from: {path}')
    yield ems.EMS(RDMol, rdkit_mol=True)
    print()
    print('Tearing down RDMol molecule testmol_rdmol (scope: module)')