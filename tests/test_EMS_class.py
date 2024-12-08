import sys
import os
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems
from rdkit import Chem
import pytest

mol_list = ['testmol_sdf_1', 
            'testmol_xyz',
            'testmol_smiles_asym',
            'testmol_rdmol']

class TestEMSclass:
    def setup_method(self, method):
        print()
        print(f'Setting up {method}')

    def teardown_method(self, method):
        print()
        print(f'Tearing down {method}')

    @pytest.mark.parametrize('mol', mol_list)
    def test_filenames(self, mol, request):
        if 'smiles' in mol:
            mol = request.getfixturevalue(mol)
            Xfile = mol.file
            Xfilename = Xfile
            Xstringfile = Xfile
            Xid = Xfile
            assert mol.id == Xid
        
        elif 'rdmol' in mol:
            mol = request.getfixturevalue(mol)
            Xfile = None
            Xfilename = None
            Xstringfile = None
            Xid = None

        else:
            mol = request.getfixturevalue(mol)
            Xfile = mol.file
            Xfilename = Xfile.split('/')[-1]
            Xstringfile = Xfile

        print(f'EMS.file: {mol.file}')
        print(f'EMS.id: {mol.id}')
        print(f'EMS.filename: {mol.filename}')
        print(f'EMS.stringfile: {mol.stringfile}')
        
        assert mol.file == Xfile
        assert mol.filename == Xfilename
        assert mol.stringfile == Xstringfile
        assert type(mol.rdmol) == Chem.rdchem.Mol




    # def test_check_Zcoords_zero(self, testmol_1):
    #     print(f'EMS.check_z_ords: {testmol_1.check_Zcoords_zero()}')
    #     assert testmol_1.check_Zcoords_zero() == False

    # def test_rdmol_type(self, testmol_1):
    #     print(f'RDMol type: {type(testmol_1.rdmol)}')
    #     assert type(testmol_1.rdmol) == rdkit.Chem.rdchem.Mol

    # def test_RDMolProp(self, testmol_1):
    #     print(f'RDMol.GetProp: {testmol_1.rdmol.GetProp("_Name")}')
    #     if testmol_1.rdmol.GetProp('_Name') == testmol_1.id:
    #         print('The _Name property of this RDMol is the same as its EMS.id')
    #     assert True
    

    






