import sys
import pytest
sys.path.append('/user/home/rv22218/work/inv_IMPRESSION/EMS')
import EMS as ems
import numpy as np

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
    






