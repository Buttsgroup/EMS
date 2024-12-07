import sys
import os
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems

test_mol_path = '/user/home/rv22218/work/inv_IMPRESSION/EMS/tests/test_mols'

class TestXYZ:

    def setup_method(self, method):
        print()
        print(f'Setting up {method}')

    def teardown_method(self, method):
        print()
        print(f'Tearing down {method}')

    # def test_sdf_file_stringfile(self, testmol_xyz):
    #     print(f'EMS.file: {testmol_xyz.file}')
    #     print(f'EMS.stringfile: {testmol_xyz.stringfile}')
    #     assert testmol_xyz.stringfile == testmol_xyz.file
    #     assert testmol_xyz.file == os.path.join(test_mol_path, 'dsgdb9nsd_057135.xyz')