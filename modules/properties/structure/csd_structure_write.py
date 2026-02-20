import numpy as np
import sys
import os
from ccdc.molecule import Molecule, Atom, Bond
from ccdc.molecule import Molecule

def csdmol_to_structure_arrays(ccdc_mol):
    """
    Extract structure information from the CSD Python API molecule object
    
    Args:
    - mol: ccdc.molecule.Molecule

    Returns:
    - type_array (np.ndarray): Atomic numbers, shape (num_atoms)
    - xyz_array (np.ndarray): Atom coordinates, shape (num_atoms, 3)
    - conn_array (np.nd_array): Bond orders between atoms, shape (num_atoms, num_atoms)
    """

    atoms = ccdc_mol.atoms
    n_atoms = len(atoms)

    type_array = np.zeros(n_atoms, dtype=np.int32)
    xyz_array = np.zeros((n_atoms, 3), dtype=np.float64)
    aromatic_conn_array = np.zeros((n_atoms, n_atoms), dtype=np.int32)
    conn_array = np.zeros((n_atoms, n_atoms), dtype=np.int32)

    for i, atom in enumerate(atoms):
        type_array[i] = atom.atomic_number

        coords = atom.coordinates
        if coords is None:
            raise ValueError(f"Atom {i} ({atom.atomic_symbol}) has no 3D coordinates")

        xyz_array[i, :] = coords.x, coords.y, coords.z

    ccdc_mol.assign_bond_types('unknown')

    for bond in ccdc_mol.bonds:
        i = bond.atoms[0].index
        j = bond.atoms[1].index
        bt = str(bond.bond_type)
        if bt == "Single":
            order = 1
        elif bt == "Double":
            order = 2
        elif bt == "Triple":
            order = 3
        elif bt == "Aromatic":
            order = 4
        else:
            order = 1
        
        conn_array[i, j] = conn_array[j, i] = order
 
    return type_array, xyz_array, conn_array