import sys
import os
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems
from rdkit import Chem
import pytest
import numpy as np

mol_list = ['testmol_sdf_1',
            'testmol_sdf_WrongValence', 
            'testmol_xyz',
            'testmol_smiles_sym',
            'testmol_rdmol',
            'testmol_rdmol_flat']

symmetric_mol_list = ['testmol_sdf_1',
                      'testmol_smiles_sym',
                      'testmol_rdmol_flat']

class TestEMSclass:
    def setup_method(self, method):
        print()
        print(f'Setting up {method}')

    def teardown_method(self, method):
        print()
        print(f'Tearing down {method}')

    @pytest.mark.parametrize('mol', mol_list)
    def test_filenames(self, mol, request):

        # Test file reading
        if 'smiles' in mol:
            emol = request.getfixturevalue(mol)
            Xfile = emol.file
            Xfilename = Xfile
            Xstringfile = Xfile
            Xid = Xfile
            assert emol.id == Xid
        
        elif 'rdmol' in mol:
            emol = request.getfixturevalue(mol)
            Xfile = None
            Xfilename = None
            Xstringfile = None
            Xid = None

        else:
            emol = request.getfixturevalue(mol)
            Xfile = emol.file
            Xfilename = Xfile.split('/')[-1]
            Xstringfile = Xfile

        print(f'EMS.file: {emol.file}')
        print(f'EMS.id: {emol.id}')
        print(f'EMS.filename: {emol.filename}')
        print(f'EMS.stringfile: {emol.stringfile}')
        
        assert emol.file == Xfile
        assert emol.filename == Xfilename
        assert emol.stringfile == Xstringfile
        assert type(emol.rdmol) == Chem.rdchem.Mol

        # test valence check
        print(f'EMS.pass_valence_check: {emol.pass_valence_check}')
        if 'WrongValence' in mol:
            assert emol.pass_valence_check == False
        else:
            assert emol.pass_valence_check == True
        
        # test property reading
        print(f'EMS.type: {emol.type}')
        print(f'EMS.xyz: {emol.xyz}')
        print(f'EMS.conn: {emol.conn}')
        print(f'EMS.mol_properties["SMILES"]: {emol.mol_properties["SMILES"]}')
        assert type(emol.type) == np.ndarray
        assert type(emol.xyz) == np.ndarray
        assert type(emol.conn) == np.ndarray
        assert type(emol.mol_properties["SMILES"]) == str

        # test symmetry
        print(f'EMS.symmetric: {emol.symmetric}')
        if mol in symmetric_mol_list:
            assert emol.symmetric == True
        else:
            assert emol.symmetric == False

        # test flat molecule
        print(f'EMS.flat: {emol.flat}')
        if 'flat' in mol:
            assert emol.flat == True
        else:
            assert emol.flat == False




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
    

    






