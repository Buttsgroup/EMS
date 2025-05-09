import os
import numpy as np
from EMS import EMS as ems
from rdkit import Chem

file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')

def test_sdf_no_nmr():
    # Test the EMS class with an SDF file
    mol_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    emol = ems.EMS(file=mol_file, mol_id='testmol_sdf', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.file.endswith('.sdf')
    assert emol.filetype == 'sdf'
    assert emol.streamlit == False
    assert emol.id == 'testmol_sdf'
    assert emol.filename == 'imp_dsgdb9nsd_074000'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'

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
    assert emol.pass_valence_check == False
    assert emol.symmetric == 'asym'
    assert emol.nmr == False
    assert emol.addHs == False
    assert emol.sanitize == False
    assert emol.kekulize == True

    assert type(emol.mol_properties['SMILES']) == str
    assert emol.atom_properties == {}
    assert emol.pair_properties == {}

    # Test on an SDF file when outputting to an SDF file 
    sdf_block = emol.to_sdf(outfile='', FileComments='Testing_sdf_molecule', prop_to_write=None)
    sdf_block_list = sdf_block.split('\n')
    assert type(sdf_block) == str
    assert sdf_block_list[0] == 'imp_dsgdb9nsd_074000'
    assert 'EMS (Efficient Molecular Storage)' in sdf_block_list[1] and 'ButtsGroup' in sdf_block_list[1]
    assert sdf_block_list[2] == 'Testing_sdf_molecule'


def test_sdf_nmr():
    # Test the EMS class with an SDF file with NMR data
    mol_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    emol = ems.EMS(file=mol_file, nmr=True)
    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.file.endswith('.sdf')
    assert emol.filetype == 'sdf'
    assert emol.streamlit == False
    assert emol.id == 'imp_dsgdb9nsd_074000'
    assert emol.filename == 'imp_dsgdb9nsd_074000'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'

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
    assert emol.pass_valence_check == False
    assert emol.symmetric == 'asym'
    assert emol.nmr == True
    assert emol.addHs == False
    assert emol.sanitize == False
    assert emol.kekulize == True

    assert type(emol.atom_properties['shift']) == np.ndarray
    assert type(emol.atom_properties['shift_var']) == np.ndarray
    assert type(emol.pair_properties['coupling']) == np.ndarray
    assert type(emol.pair_properties['coupling_var']) == np.ndarray
    assert type(emol.pair_properties['nmr_types']) == np.ndarray

    assert emol.atom_properties['shift'].shape == (atom_num,)
    assert emol.atom_properties['shift_var'].shape == (atom_num,)
    assert emol.pair_properties['coupling'].shape == (atom_num, atom_num)
    assert emol.pair_properties['coupling_var'].shape == (atom_num, atom_num)
    assert emol.pair_properties['nmr_types'].shape == (atom_num, atom_num)

    assert emol.atom_properties['shift'].dtype == np.float64
    assert emol.atom_properties['shift_var'].dtype == np.float64
    assert emol.pair_properties['coupling'].dtype == np.float64
    assert emol.pair_properties['coupling_var'].dtype == np.float64
    assert emol.pair_properties['nmr_types'].dtype == '<U4'

    # Test on an SDF file when outputting to an SDF file (1)
    emol.to_sdf(outfile='tmp_testmol_sdf_nmr.sdf', FileComments='Testing_sdf_molecule_nmr', prop_cover=True, prop_to_write='nmr')
    emol2 = ems.EMS(file='tmp_testmol_sdf_nmr.sdf', mol_id='testmol_sdf_nmr_2', nmr=True)

    assert emol2.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'
    assert 'NMREDATA_ASSIGNMENT' in emol2.rdmol.GetPropsAsDict()
    assert 'NMREDATA_J' in emol2.rdmol.GetPropsAsDict()
    assert type(emol2.rdmol.GetProp('NMREDATA_ASSIGNMENT')) == str
    assert type(emol2.rdmol.GetProp('NMREDATA_J')) == str

    # Test on an SDF file when outputting to an SDF file (2)
    for mol in Chem.SDMolSupplier('tmp_testmol_sdf_nmr.sdf', removeHs=False, sanitize=False):
        assert mol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'
        assert 'NMREDATA_ASSIGNMENT' in mol.GetPropsAsDict()
        assert 'NMREDATA_J' in mol.GetPropsAsDict()
        assert type(mol.GetProp('NMREDATA_ASSIGNMENT')) == str
        assert type(mol.GetProp('NMREDATA_J')) == str
    
    if os.path.exists('tmp_testmol_sdf_nmr.sdf'):
        os.remove('tmp_testmol_sdf_nmr.sdf')

