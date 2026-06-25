import numpy as np
import sys
import os
import string
import random
from ccdc.molecule import Molecule, Atom, Bond
from ccdc.molecule import Molecule
from ccdc import io

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

def csdmol_to_sdf_block(ccdc_mol, FileInfo=None, FileComment=None):
    '''
    This function is used to write an RDKit molecule object to an sdf block.
    Here are some explanations and experiences for writing this function:
    (1) In this function, Chem.SDWriter is used to write the sdf block, because this is the most efficient way (as far as I know) to automatically write the molecule properties.
        Other functions like Chem.MolToMolBlock and Chem.MolToMolFile only write the molecule structure without the properties, even if you add the properties to RDKit molecule object.
    (2) For the _Name, _MolFileInfo, and _MolFileComments properties, only _Name will be automatically written to the sdf block by Chem.SDWriter, but if you want to write
        _MolFileInfo and _MolFileComments, you need to manually change the second and third lines in the sdf block.
    (3) Some useful functions in RDKit when writing the sdf block:
        - Mol.GetPropsAsDict(): Get all the properties of the molecule as a dictionary, but not including hidden and computed properties.
        - Mol.ClearProp(prop): Clear an assigned property of the molecule. However, there seems no function to clear all the properties at once.
        - Mol.SetProp(name, value): Set a property for the molecule.
        - Mol.GetPropNames(includePrivate=True, includeComputed=True): Get all the property names of the molecule, including hidden and computed properties.
    (4) The first three lines of EMS output sdf blocks are as following:
        - The first line is the SDF file name, which is defaulted to the _Name property of the RDKit molecule. If the _Name property is empty, the first line is blank.
        - The second line is the SDF file information, which is defaulted to 'EMS (Efficient Molecular Storage) - <year> - ButtsGroup'.
        - The third line is the SDF file comments, which is defaulted to blank.
        
    Args:
    - rdmol: The RDKit molecule object to be written to the sdf block.
    - FileInfo (str): The file information of the molecule, referring to the _MolFileInfo property of the RDKit molecule object and the second line in the sdf file.
    - FileComment (str): The file comment of the molecule, referring to the _MolFileComments property of the RDKit molecule object and the third line in the sdf file.
    - SDFversion (str): The version of the sdf file, which can be "V3000" or "V2000".
    '''

    # Initialize the file information and comments
    if FileInfo is None:
        FileInfo = ''
    else:
        FileInfo = FileInfo.strip()

    if FileComment is None:
        FileComment = ''
    else:
        FileComment = FileComment.strip()

    # Set the name of the temporary SDF file to save the RDKit molecule
    characters = string.ascii_letters + string.digits  
    random_string = ''.join(random.choices(characters, k=30))
    tmp_file = f"tmp_{random_string}.sdf"    

    # Set the SDF file version according to the atom number of the RDKit molecule
    atoms = ccdc_mol.atoms
    n_atoms = len(atoms)
    

    # if SDFversion == "V2000" and n_atoms > 999:
    #     logger.warning(f"V2000 cannot be used for molecules with more than 999 atoms. SDF version is set to V3000.")
    #     SDFversion = "V3000"
    
    # if SDFversion not in ["V2000", "V3000"]:
    #     logger.warning(f"SDF version {SDFversion} is not supported. SDF version is set to V3000.")
    #     SDFversion = "V3000"

    # Write the molecule to the sdf block with the specified SDF version
    with io.MoleculeWriter(tmp_file) as writer:
        writer.write(ccdc_mol)
    
    # Read the sdf block from the temporary sdf file and set the _MolFileInfo and _MolFileComments properties
    with open(tmp_file, 'r') as f:
        lines = f.readlines()
        lines[1] = FileInfo + '\n'
        lines[2] = FileComment + '\n'
    os.remove(tmp_file)
    
    # Return the sdf block
    return ''.join(lines)