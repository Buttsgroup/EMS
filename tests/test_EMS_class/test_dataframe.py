import os
import numpy as np
import pandas as pd
from EMS import EMS as ems

file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')

def test_dataframe():
    # Test the EMS class with atom and pair DataFrames
    atom_df_file = os.path.join(mol_dir, 'atom_df.pkl')
    pair_df_file = os.path.join(mol_dir, 'pair_df.pkl')
    atom_df = pd.read_pickle(atom_df_file)
    pair_df = pd.read_pickle(pair_df_file)

    emol = ems.EMS(file=(atom_df, pair_df), mol_id='testmol_dataframe', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == tuple
    assert type(emol.file[0]) == pd.DataFrame
    assert emol.filetype == 'dataframe'
    assert emol.streamlit == False
    assert emol.id == 'testmol_dataframe'
    assert emol.filename == 'imp_dsgdb9nsd_122801.nmredata.sdf'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_122801.nmredata.sdf'

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


def test_dataframe_nmr():
    # Test the EMS class with atom and pair DataFrames with NMR data
    atom_df_file = os.path.join(mol_dir, 'atom_df.pkl')
    pair_df_file = os.path.join(mol_dir, 'pair_df.pkl')
    atom_df = pd.read_pickle(atom_df_file)
    pair_df = pd.read_pickle(pair_df_file)
    emol = ems.EMS(file=(atom_df, pair_df), nmr=True)
    atom_num = len(emol.type)
    
    assert emol.id == 'imp_dsgdb9nsd_122801.nmredata.sdf'
    assert emol.filename == 'imp_dsgdb9nsd_122801.nmredata.sdf'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_122801.nmredata.sdf'
    assert emol.nmr == True

    nmr_type_mask = emol.pair_properties["nmr_types_df"] != '0'
    nmr_types_match = emol.pair_properties["nmr_types_df"] == emol.pair_properties["nmr_types"]
    assert (nmr_types_match == nmr_type_mask).all()

    assert type(emol.atom_properties['shift']) == np.ndarray
    assert type(emol.atom_properties['shift_var']) == np.ndarray
    assert type(emol.pair_properties['coupling']) == np.ndarray
    assert type(emol.pair_properties['coupling_var']) == np.ndarray
    assert type(emol.pair_properties['nmr_types']) == np.ndarray
    assert type(emol.pair_properties['nmr_types_df']) == np.ndarray

    assert emol.atom_properties['shift'].shape == (atom_num,)
    assert emol.atom_properties['shift_var'].shape == (atom_num,)
    assert emol.pair_properties['coupling'].shape == (atom_num, atom_num)
    assert emol.pair_properties['coupling_var'].shape == (atom_num, atom_num)
    assert emol.pair_properties['nmr_types'].shape == (atom_num, atom_num)
    assert emol.pair_properties['nmr_types_df'].shape == (atom_num, atom_num)

    assert emol.atom_properties['shift'].dtype == np.float64
    assert emol.atom_properties['shift_var'].dtype == np.float64
    assert emol.pair_properties['coupling'].dtype == np.float64
    assert emol.pair_properties['coupling_var'].dtype == np.float64
    assert emol.pair_properties['nmr_types'].dtype == '<U4'
    assert emol.pair_properties['nmr_types_df'].dtype == '<U11'

    # Test on atom and pair dataframes when outputting to an SDF file
    emol.to_sdf(outfile='tmp_testmol_dataframe_nmr.sdf', FileComments='Testing_dataframe_molecule_nmr', prop_cover=True, prop_to_write='nmr')
    emol2 = ems.EMS(file='tmp_testmol_dataframe_nmr.sdf', nmr=True)

    assert emol2.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_122801.nmredata.sdf'
    assert 'NMREDATA_ASSIGNMENT' in emol2.rdmol.GetPropsAsDict()
    assert 'NMREDATA_J' in emol2.rdmol.GetPropsAsDict()
    assert type(emol2.rdmol.GetProp('NMREDATA_ASSIGNMENT')) == str
    assert type(emol2.rdmol.GetProp('NMREDATA_J')) == str

    if os.path.exists('tmp_testmol_dataframe_nmr.sdf'):
        os.remove('tmp_testmol_dataframe_nmr.sdf')