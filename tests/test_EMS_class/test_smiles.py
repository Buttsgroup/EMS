import os
import numpy as np
from EMS import EMS as ems

def test_smiles():
    # Test the EMS class with a SMILES string
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    emol = ems.EMS(file=smiles, mol_id='testmol_smiles', nmr=False)

    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.filetype == 'smiles'
    assert emol.streamlit == False
    assert emol.id == 'testmol_smiles'
    assert emol.filename == 'CC(=O)OC1=CC=CC=C1C(=O)O'
    assert emol.rdmol.GetProp('_Name') == 'CC(=O)OC1=CC=CC=C1C(=O)O'

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


def test_smiles_no_id():
    # Test the EMS class with a SMILES string without an ID
    smiles = 'CN(C)C1=NC(=NC(=N1)N(C)C)N(C)C'
    emol = ems.EMS(file=smiles, mol_id=None, nmr=False)

    atom_num = len(emol.type)

    assert type(emol.file) == str
    assert emol.filetype == 'smiles'
    assert emol.streamlit == False
    assert emol.id == 'CN(C)C1=NC(=NC(=N1)N(C)C)N(C)C'
    assert emol.filename == 'CN(C)C1=NC(=NC(=N1)N(C)C)N(C)C'
    assert emol.rdmol.GetProp('_Name') == 'CN(C)C1=NC(=NC(=N1)N(C)C)N(C)C'

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