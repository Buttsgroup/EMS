import os
import numpy as np
from EMS import EMS as ems

file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')

def test_xyz():
    # Test the EMS class with an XYZ file
    mol_file = os.path.join(mol_dir, 'dsgdb9nsd_057135.xyz')
    emol = ems.EMS(file=mol_file, mol_id='testmol_xyz', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.file.endswith('.xyz')
    assert emol.filetype == 'xyz'
    assert emol.streamlit == False
    assert emol.id == 'testmol_xyz'
    assert emol.filename == 'testmol_xyz'
    assert emol.rdmol.GetProp('_Name') == 'testmol_xyz'

    assert type(emol.type) == np.ndarray
    assert type(emol.xyz) == np.ndarray
    assert type(emol.conn) == np.ndarray
    assert type(emol.adj) == np.ndarray
    assert type(emol.path_topology) == np.ndarray
    assert type(emol.path_distance) == np.ndarray

    assert emol.xyz.shape == (atom_num, 3)
    assert emol.conn.shape == (atom_num, atom_num)
    assert emol.adj.shape == (atom_num, atom_num)
    assert emol.path_topology.shape == (atom_num, atom_num)
    assert emol.path_distance.shape == (atom_num, atom_num)

    assert emol.flat == False
    assert emol.pass_valence_check == True
    assert emol.symmetric == 'asym'
    assert emol.nmr == False
    assert emol.addHs == False
    assert emol.sanitize == False
    assert emol.kekulize == True

    assert type(emol.mol_properties['SMILES']) == str
    assert emol.atom_properties == {}
    assert emol.pair_properties == {}

    # Test on an XYZ file when outputting to an SDF file (1)
    sdf_block = emol.to_sdf(outfile='', FileComments='Testing_xyz_molecule', prop_to_write=None)
    sdf_block_list = sdf_block.split('\n')
    assert type(sdf_block) == str
    assert sdf_block_list[0] == 'testmol_xyz'
    assert 'EMS (Efficient Molecular Storage)' in sdf_block_list[1] and 'ButtsGroup' in sdf_block_list[1]
    assert sdf_block_list[2] == 'Testing_xyz_molecule'

    # Test on an XYZ file when outputting to an SDF file (2)
    emol.to_sdf(outfile='tmp_test_xyz.sdf', FileComments='', prop_to_write=None)
    with open('tmp_test_xyz.sdf', 'r') as f:
        sdf_block_2 = f.read()
    sdf_block_2_list = sdf_block_2.split('\n')
    assert type(sdf_block_2) == str
    assert sdf_block_2_list[0] == 'testmol_xyz'
    assert 'EMS (Efficient Molecular Storage)' in sdf_block_2_list[1] and 'ButtsGroup' in sdf_block_2_list[1]
    assert sdf_block_2_list[2] == ''
    if os.path.exists('tmp_test_xyz.sdf'):
        os.remove('tmp_test_xyz.sdf')


def test_xyz_no_id():
    # Test on an XYZ file without mol_id
    mol_file = os.path.join(mol_dir, 'dsgdb9nsd_057135.xyz')
    emol_no_id = ems.EMS(file=mol_file, mol_id=None, nmr=False)

    assert emol_no_id.id == ''
    assert emol_no_id.filename == ''
    assert emol_no_id.rdmol.GetProp('_Name') == ''


def test_xyz_2():
    # Test the EMS class with an XYZ file
    mol_file = os.path.join(mol_dir, 'dsgdb9nsd_064215.xyz')
    emol = ems.EMS(file=mol_file, mol_id='testmol_xyz_2', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.file.endswith('.xyz')
    assert emol.filetype == 'xyz'
    assert emol.streamlit == False
    assert emol.id == 'testmol_xyz_2'
    assert emol.filename == 'testmol_xyz_2'
    assert emol.rdmol.GetProp('_Name') == 'testmol_xyz_2'

    assert type(emol.type) == np.ndarray
    assert type(emol.xyz) == np.ndarray
    assert type(emol.conn) == np.ndarray
    assert type(emol.adj) == np.ndarray
    assert type(emol.path_topology) == np.ndarray
    assert type(emol.path_distance) == np.ndarray

    assert emol.xyz.shape == (atom_num, 3)
    assert emol.conn.shape == (atom_num, atom_num)
    assert emol.adj.shape == (atom_num, atom_num)
    assert emol.path_topology.shape == (atom_num, atom_num)
    assert emol.path_distance.shape == (atom_num, atom_num)

    assert emol.flat == False
    assert emol.pass_valence_check == True
    assert emol.symmetric == 'sym'
    assert emol.nmr == False
    assert emol.addHs == False
    assert emol.sanitize == False
    assert emol.kekulize == True

    assert type(emol.mol_properties['SMILES']) == str
    assert emol.atom_properties == {}
    assert emol.pair_properties == {}