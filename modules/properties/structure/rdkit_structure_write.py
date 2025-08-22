import numpy as np
import logging
import sys
import os
from copy import deepcopy
import string
import random

from rdkit import Chem
from rdkit.Chem import AllChem


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def rdmol_to_structure_arrays(mol, kekulize=True):
    '''
    This function is used to extract the structure information: atom type, atom coordinates, and bond orders between atoms as arrays from an RDKit molecule object.
    
    Args:
    - rdmol: The RDKit molecule object to extract the structure information.
    - kekulize (bool): Whether to kekulize the molecule before extracting the structure information.
        This argument is to prevent an aromatic molecule having a bond order of 1.5, which is not an integer and may cause some problems in assigning bond orders.

    Returns:
    - type_array (np.ndarray): The array of atom types, which is the atomic number of each atom. The shape is (num_atoms,).
    - xyz_array (np.ndarray): The array of atom coordinates. The shape is (num_atoms, 3).
    - conn_array (np.ndarray): The array of bond orders between atoms. The shape is (num_atoms, num_atoms).
    '''

    rdmol = deepcopy(mol)

    # Kekulize the molecule
    if kekulize:
        try:
            Chem.Kekulize(rdmol)
        except Exception as e:
            logger.error(f"Fail to kekulize the molecule.")
            raise e
    
    # Initialze the arrays to store the structure information: atom type, atom coordinates, and bond orders between atoms
    type_array = np.zeros(rdmol.GetNumAtoms(), dtype=np.int32)
    xyz_array = np.zeros((rdmol.GetNumAtoms(), 3), dtype=np.float64)
    conn_array = np.zeros((rdmol.GetNumAtoms(), rdmol.GetNumAtoms()), dtype=np.int32)
    
    # Loop over the atoms in the molecule
    for i, atoms in enumerate(rdmol.GetAtoms()):

        # Get the atomic number of the atom
        type_array[i] = atoms.GetAtomicNum()

        # Generate conformers for the molecule if there is no conformer and get the atom coordinates from the conformer
        if rdmol.GetNumConformers() < 1:
            AllChem.Compute2DCoords(rdmol)
        xyz_array[i][0] = rdmol.GetConformer(0).GetAtomPosition(i).x
        xyz_array[i][1] = rdmol.GetConformer(0).GetAtomPosition(i).y
        xyz_array[i][2] = rdmol.GetConformer(0).GetAtomPosition(i).z

        # Get the bond orders between atoms
        for j, atoms in enumerate(rdmol.GetAtoms()):
            if i == j:
                continue

            bond = rdmol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                conn_array[i][j] = int(bond.GetBondTypeAsDouble())
                conn_array[j][i] = int(bond.GetBondTypeAsDouble())

    return type_array, xyz_array, conn_array


def rdmol_to_aromatic_bond_array(mol):
    '''
    This function is used to extract the structure information: bond orders between atoms as arrays from an RDKit molecule object.
    The aromatic bonds are included in the bond orders, and the bond order of an aromatic bond is 1.5.
    
    Args:
    - rdmol: The RDKit molecule object to extract the structure information.
   
    Returns:
    - aromatic_conn_array (np.ndarray): The array of bond orders between atoms. The shape is (num_atoms, num_atoms).
    '''

    rdmol = deepcopy(mol)

    try:
        Chem.SetAromaticity(rdmol)
    except Exception as e:
        logger.error(f"Fail to assign aromatic bonds to the molecule.")
        raise e

    aromatic_conn_array = np.zeros((rdmol.GetNumAtoms(), rdmol.GetNumAtoms()), dtype=np.float64)

    # Loop over the atoms in the molecule
    for i, atoms in enumerate(rdmol.GetAtoms()):
        # Get the bond orders (including aromatic bonds) between atoms
        for j, atoms in enumerate(rdmol.GetAtoms()):
            if i == j:
                continue

            bond = rdmol.GetBondBetweenAtoms(i, j)
            
            if bond is not None:
                aromatic_conn_array[i][j] = bond.GetBondTypeAsDouble()
                aromatic_conn_array[j][i] = bond.GetBondTypeAsDouble()
    
    return aromatic_conn_array
    

def rdmol_to_xyz_block(rdmol, FileInfo=None, FileComment=None):
    '''
    This function is used to convert an RDKit molecule object to an xyz block.

    Args:
    - rdmol: The RDKit molecule object to be converted to an xyz block.
    - FileInfo (str): The file information, which usually includes our lab information.
    - FileComment (str): Any other comments on the molecule.
    '''

    # Set up the file information and comment
    if FileInfo is None:
        FileInfo = ''
    else:
        FileInfo = f' | {FileInfo}'

    if FileComment is None:
        FileComment = ''
    else:
        FileComment = f' | {FileComment}'

    # Get the name of the RDKit molecule
    try:
        rdmol_Name = rdmol.GetProp("_Name").strip()
    except:
        rdmol_Name = ""

    # Generate the xyz block and set the comment line (the second line)
    try:
        xyz_block = Chem.MolToXYZBlock(rdmol)

        blocks = xyz_block.split('\n')
        blocks = [b for b in blocks if b.strip() != ''] 
        blocks[1] = f'{rdmol_Name}{FileInfo}{FileComment}'
        xyz_block = '\n'.join(blocks) + '\n'

        return xyz_block

    except Exception as e:
        logger.error(f"Error occurred while converting RDKit molecule: {rdmol_Name} to XYZ block.")
        raise e
    

def rdmol_to_sdf_block(rdmol, FileInfo=None, FileComment=None, SDFversion="V3000"):
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
    atom_num = rdmol.GetNumAtoms()

    if SDFversion == "V2000" and atom_num > 999:
        logger.warning(f"V2000 cannot be used for molecules with more than 999 atoms. SDF version is set to V3000.")
        SDFversion = "V3000"
    
    if SDFversion not in ["V2000", "V3000"]:
        logger.warning(f"SDF version {SDFversion} is not supported. SDF version is set to V3000.")
        SDFversion = "V3000"

    # Write the molecule to the sdf block with the specified SDF version
    with Chem.SDWriter(tmp_file) as writer:
        if SDFversion == "V3000":
            writer.SetForceV3000(True)
        elif SDFversion == "V2000":
            writer.SetForceV3000(False)
        else:
            logger.error(f"Invalid SDF version: {SDFversion}")
            raise ValueError(f"Invalid SDF version: {SDFversion}")
        writer.write(rdmol)
    
    # Read the sdf block from the temporary sdf file and set the _MolFileInfo and _MolFileComments properties
    with open(tmp_file, 'r') as f:
        lines = f.readlines()
        lines[1] = FileInfo + '\n'
        lines[2] = FileComment + '\n'
    os.remove(tmp_file)
    
    # Return the sdf block
    return ''.join(lines)