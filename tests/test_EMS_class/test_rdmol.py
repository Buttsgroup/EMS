import os
import numpy as np
from EMS import EMS as ems
from rdkit import Chem

file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')

def test_rdmol():
    # Test the EMS class with an RDKit molecule object
    sdf_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    rdmol = None
    for mol in Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False):
        rdmol = mol
        break
    assert type(rdmol) == Chem.Mol

    emol = ems.EMS(file=rdmol, mol_id='testmol_rdmol', nmr=False)
    atom_num = len(emol.type)

    assert type(emol.file) == Chem.Mol
    assert emol.filetype == 'rdmol'
    assert emol.streamlit == False
    assert emol.id == 'testmol_rdmol'
    assert emol.filename == 'imp_dsgdb9nsd_074000'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'

    assert 'NMREDATA_ASSIGNMENT' in emol.rdmol.GetPropsAsDict()
    assert 'NMREDATA_J' in emol.rdmol.GetPropsAsDict()
    assert type(emol.rdmol.GetProp('NMREDATA_ASSIGNMENT')) == str
    assert type(emol.rdmol.GetProp('NMREDATA_J')) == str

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


def test_rdmol_nmr():
    # Test the EMS class with an RDKit molecule object with NMR data
    sdf_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    rdmol = None
    for mol in Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False):
        rdmol = mol
        break
    emol = ems.EMS(file=rdmol, nmr=True)
    atom_num = len(emol.type)

    assert emol.id == 'imp_dsgdb9nsd_074000'
    assert emol.filename == 'imp_dsgdb9nsd_074000'
    assert emol.rdmol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'

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

    # Test on an RDKit molecule object when outputting to an SDF file
    emol.to_sdf(outfile='tmp_testmol_rdmol_nmr.sdf', FileComments='Testing_sdf_molecule_nmr', prop_cover=True, prop_to_write='nmr')
    for mol in Chem.SDMolSupplier('tmp_testmol_rdmol_nmr.sdf', removeHs=False, sanitize=False):
        assert mol.GetProp('_Name') == 'imp_dsgdb9nsd_074000'
        assert mol.GetProp('_MolFileComments') == 'Testing_sdf_molecule_nmr'
        assert 'NMREDATA_ASSIGNMENT' in mol.GetPropsAsDict()
        assert 'NMREDATA_J' in mol.GetPropsAsDict()
        assert type(mol.GetProp('NMREDATA_ASSIGNMENT')) == str
        assert type(mol.GetProp('NMREDATA_J')) == str
    
    if os.path.exists('tmp_testmol_rdmol_nmr.sdf'):
        os.remove('tmp_testmol_rdmol_nmr.sdf')
