import os
import numpy as np
import pandas as pd
from EMS import EMS as ems
from rdkit import Chem

file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')

def test_gaussian_log():
    # Test the EMS class with a Gaussian log file
    mol_file = os.path.join(mol_dir, 'camphor.log')
    emol = ems.EMS(file=mol_file, mol_id='testmol_gaussian_log', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.file.endswith('log')
    assert emol.filetype == 'gaussian-log'
    assert emol.streamlit == False
    assert emol.id == 'testmol_gaussian_log'
    assert emol.filename == 'testmol_gaussian_log'
    assert emol.rdmol.GetProp('_Name') == 'testmol_gaussian_log'

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


def test_gaussian_log_nmr():
    # Test the EMS class with a Gaussian log file with NMR data
    mol_file = os.path.join(mol_dir, 'camphor.log')
    emol = ems.EMS(file=mol_file, nmr=True)
    atom_num = len(emol.type)

    assert emol.id == ''
    assert emol.filename == ''
    assert emol.rdmol.GetProp('_Name') == ''
    assert emol.nmr == True

    assert type(emol.atom_properties['shift']) == np.ndarray
    assert type(emol.atom_properties['raw_shift']) == np.ndarray
    assert type(emol.atom_properties['shift_var']) == np.ndarray
    assert type(emol.pair_properties['coupling']) == np.ndarray
    assert type(emol.pair_properties['coupling_var']) == np.ndarray
    assert type(emol.pair_properties['nmr_types']) == np.ndarray

    assert emol.atom_properties['shift'].shape == (atom_num,)
    assert emol.atom_properties['raw_shift'].shape == (atom_num,)
    assert emol.atom_properties['shift_var'].shape == (atom_num,)
    assert emol.pair_properties['coupling'].shape == (atom_num, atom_num)
    assert emol.pair_properties['coupling_var'].shape == (atom_num, atom_num)
    assert emol.pair_properties['nmr_types'].shape == (atom_num, atom_num)

    assert emol.atom_properties['shift'].dtype == np.float64
    assert emol.atom_properties['raw_shift'].dtype == np.float64
    assert emol.atom_properties['shift_var'].dtype == np.float64
    assert emol.pair_properties['coupling'].dtype == np.float64
    assert emol.pair_properties['coupling_var'].dtype == np.float64
    assert emol.pair_properties['nmr_types'].dtype == '<U4'

    # Test on a Gaussian log file when outputting to an SDF file
    emol.to_sdf(outfile='tmp_testmol_gaussian_log_nmr.sdf', FileComments='Testing_gaussian_log_molecule_nmr', prop_cover=True, prop_to_write='nmr')
    for mol in Chem.SDMolSupplier('tmp_testmol_gaussian_log_nmr.sdf', removeHs=False, sanitize=False):
        assert mol.GetProp('_Name') == ''
        assert mol.GetProp('_MolFileComments') == 'Testing_gaussian_log_molecule_nmr'
        assert 'NMREDATA_ASSIGNMENT' in mol.GetPropsAsDict()
        assert 'NMREDATA_J' in mol.GetPropsAsDict()
        assert type(mol.GetProp('NMREDATA_ASSIGNMENT')) == str
        assert type(mol.GetProp('NMREDATA_J')) == str
    
    if os.path.exists('tmp_testmol_gaussian_log_nmr.sdf'):
        os.remove('tmp_testmol_gaussian_log_nmr.sdf')