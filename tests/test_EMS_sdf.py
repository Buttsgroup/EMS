import sys
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems
import rdkit

class TestSDF:

    def setup_method(self, method):
        print()
        print(f'Setting up {method}')

    def teardown_method(self, method):
        print()
        print(f'Tearing down {method}')

    def test_sdf_file_stringfile(self, testmol_1):
        print(f'EMS.file: {testmol_1.file}')
        print(f'EMS.stringfile: {testmol_1.stringfile}')
        assert testmol_1.stringfile == testmol_1.file
        assert testmol_1.file == '/user/home/rv22218/work/inv_IMPRESSION/EMS/tests/test_mols/testmol_1_NMR.nmredata.sdf'

    def test_sdf_filename(self, testmol_1):
        print(f'EMS.filename: {testmol_1.filename}')
        assert testmol_1.filename == 'testmol_1_NMR.nmredata.sdf'

    def test_sdf_id(self, testmol_1):
        print(f'EMS.id: {testmol_1.id}')
        assert True

    def test_check_Zcoords_zero(self, testmol_1):
        print(f'EMS.check_z_ords: {testmol_1.check_Zcoords_zero()}')
        assert testmol_1.check_Zcoords_zero() == False

    def test_rdmol_type(self, testmol_1):
        print(f'RDMol type: {type(testmol_1.rdmol)}')
        assert type(testmol_1.rdmol) == rdkit.Chem.rdchem.Mol

    def test_RDMolProp(self, testmol_1):
        print(f'RDMol.GetProp: {testmol_1.rdmol.GetProp("_Name")}')
        if testmol_1.rdmol.GetProp('_Name') == testmol_1.id:
            print('The _Name property of this RDMol is the same as its EMS.id')
        assert True
    

    






