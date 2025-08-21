import os
import numpy as np
import pandas as pd
from EMS import EMS as ems
from rdkit import Chem

from EMS.modules.conformer.EMSconf import EMSconf


file_path = os.path.realpath(__file__)
dir_path = os.path.realpath(os.path.join(file_path, '../..'))
mol_dir = os.path.join(dir_path, 'test_mols')


def test_EMSconf():
    mol_file = os.path.join(mol_dir, 'imp_dsgdb9nsd_074000.nmredata.sdf')
    emol = ems.EMS(file=mol_file, mol_id='testmol_sdf', nmr=False)
    conf = EMSconf(emol)

    assert isinstance(conf, EMSconf)
    assert conf.molname == 'imp_dsgdb9nsd_074000'
    assert conf.params['num_conformers'] == 10
    assert len(emol.rdmol.GetConformers()) == 1

    assert "conformer_energies" not in emol.mol_properties.keys()
    conf.do_conformer_optimization()
    assert "conformer_energies" in emol.mol_properties.keys()
    assert len(emol.mol_properties["conformer_energies"]) == 10

    assert "redundant_conformers" not in emol.mol_properties.keys()
    conf.redundant_elimination(conformer_optimization=False)
    assert "redundant_conformers" in emol.mol_properties.keys()
    assert len(emol.mol_properties["redundant_conformers"]) > 0

    assert "conformer_population" not in emol.mol_properties.keys()
    conf.calc_population(eliminate_redundant=False)
    assert "conformer_population" in emol.mol_properties.keys()
    assert len(emol.mol_properties["conformer_population"]) > 0