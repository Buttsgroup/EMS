import pytest
import os
import EMS.EMS as ems
from rdkit import Chem
from rdkit.Chem import AllChem

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
def testmol_sdf_WrongValence():
    mol_file = 'imp_dsgdb9nsd_074000.nmredata.sdf'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up SDF molecule testmol_sdf_WrongValence (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id)
    print()
    print('Tearing down SDF molecule testmol_sdf_WrongValence (scope: module)')

@pytest.fixture(scope='module')
def testmol_sdf_ReadNMR_1():
    mol_file = 'testmol_1_NMR.nmredata.sdf'
    mol_id = mol_file.split('.')[0]
    path = os.path.join(test_mol_dir, mol_file)
    print()
    print('Setting up SDF molecule testmol_sdf_ReadNMR_1 (scope: module)')
    print(f'Mol Path: {path}')
    yield ems.EMS(path, mol_id=mol_id, nmr=True)
    print()
    print('Tearing down SDF molecule testmol_sdf_ReadNMR_1 (scope: module)')

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
def testmol_smiles_sym():
    smiles = 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
    print()
    print('Setting up SMILES molecule testmol_smiles_sym (scope: module)')
    print(f'SMILES string: {smiles}')
    yield ems.EMS(smiles, line_notation='smiles')
    print()
    print('Tearing down SMILES molecule testmol_smiles_sym (scope: module)')

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

@pytest.fixture(scope='module')
def testmol_rdmol_flat():
    smiles = 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
    rdmol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.AddHs(rdmol)
    AllChem.EmbedMolecule(rdmol)              # obtain the initial 3D structure for a molecule
    AllChem.UFFOptimizeMolecule(rdmol)
    AllChem.Compute2DCoords(rdmol)
    print()
    print('Setting up 2D RDMol molecule testmol_rdmol_flat (scope: module)')
    print(f'SMILES string: {smiles}')
    yield ems.EMS(rdmol, rdkit_mol=True)
    print()
    print('Tearing down 2D RDMol molecule testmol_rdmol_flat (scope: module)')