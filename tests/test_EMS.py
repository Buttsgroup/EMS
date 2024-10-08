'needs to add functions for testing the EMS class'

import sys
sys.path.append('..')

from EMS import EMS

import numpy as np


















file_dir = './test_mols/'
file = 'testmol_1_NMR.nmredata.sdf'
path = file_dir + file

mol = EMS(path, mol_id = file, read_nmr = True, fragment = True)
# mol = EMS('CC', line_notation = 'smi', fragment = True)
print(mol.type)
print(mol.xyz[:, 0].shape)
print(mol.symmetric)
print(mol.H_index_dict)
print(mol.reduced_H_dict)
print(mol.reduced_H_list)
print(mol.eff_atom_list)
print(mol.dumb_atom_list)
# print(mol.reduced_conn[:, 47])
# print(mol.conn[:, 47])
print(mol.mol_properties["SMILES"])
print(mol.atom_properties['atom_type'])




# for i in range(len(mol.adj)):
#     print(mol.adj[i])
#     print(mol.conn[i])
#     print('\n')