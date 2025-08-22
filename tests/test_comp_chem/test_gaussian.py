from EMS.modules.properties.nmr import nmr_ops
from EMS.modules.comp_chem.gaussian.gaussian_input import write_gaussian_com_block
from EMS import EMS as ems
import numpy as np
import os


file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')


def test_scale_chemical_shifts():
    raw_shift = np.array([17.4511, 159.7362, 30.9043, 274.0412, 215.5712, -285.6313])
    atom_types = [1, 6, 1, 9, 7, 8]
    scaled_shift = nmr_ops.scale_chemical_shifts(raw_shift, atom_types)
    calculated_scaled_shift = np.array([13.9496, 27.1455, 1.2507, -92.3411, -359.2477, 0.0000])

    assert np.allclose(scaled_shift, calculated_scaled_shift, rtol=1e-4)


def test_write_gaussian_com_block():
    mol_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    emol = ems.EMS(file=mol_file, mol_id='testmol_sdf', nmr=False)
    assert emol.filename == 'imp_dsgdb9nsd_074000'

    com_block = write_gaussian_com_block(emol)
    com_block_list = com_block.split('\n')

    assert 'imp_dsgdb9nsd_074000' in com_block_list[0]
    assert 'NoSave' in com_block_list[1]
    assert 'mPW1PW' in com_block_list[4]
    assert 'imp_dsgdb9nsd_074000' in com_block_list[6]
    assert '0 1' in com_block_list[8]
    

def test_write_gaussian_com_block_nmr():
    mol_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    emol = ems.EMS(file=mol_file, mol_id='testmol_sdf', nmr=False)
    assert emol.filename == 'imp_dsgdb9nsd_074000'

    com_block = write_gaussian_com_block(emol, prefs={'calc_type': 'nmr', 'memory': 16, 'processor': 2})
    com_block_list = com_block.split('\n')

    assert 'imp_dsgdb9nsd_074000' in com_block_list[0]
    assert 'NoSave' in com_block_list[1]
    assert 'NProcShared=2' in com_block_list[3]
    assert 'nmr(giao,spinspin' in com_block_list[4]
    assert 'imp_dsgdb9nsd_074000 NMR' in com_block_list[6]
    assert '0 1' in com_block_list[8]