import numpy as np
import logging
import sys

from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.Chem.rdchem import BondType


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def structure_arrays_to_rdmol_NoConn(type_array, xyz_array):
    '''
    This function is used to convert the structure information: atom types and atom coordinates, to an RDKit molecule object.
    The bond orders between atoms are not included in the structure information, so they will be determined by the RDKit function DetermineBonds.

    Args:
    - type_array (np.ndarray or list): The array of atom types, which is the atomic number of each atom. The shape is (num_atoms,).
    - xyz_array (np.ndarray or list): The array of atom coordinates. The shape is (num_atoms, 3).
    '''

    # Check the types of type_array and xyz_array, and convert them to lists if they are numpy arrays
    if type(type_array) != np.ndarray and type(type_array) != list:
        logger.error(f"Invalid type for type_array: {type(type_array)}. The type_array should be a numpy array or a list.")
        raise TypeError(f"Invalid type for type_array: {type(type_array)}. The type_array should be a numpy array or a list.")
    if type(xyz_array) != np.ndarray and type(xyz_array) != list:
        logger.error(f"Invalid type for xyz_array: {type(xyz_array)}. The xyz_array should be a numpy array or a list.")
        raise TypeError(f"Invalid type for xyz_array: {type(xyz_array)}. The xyz_array should be a numpy array or a list.")

    if type(type_array) == np.ndarray:
        type_array = type_array.tolist()
    if type(xyz_array) == np.ndarray:
        xyz_array = xyz_array.tolist()
    

    # Create an RDKit molecule object and add atoms and coordinates to the molecule
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(type_array))

    atom_indices = []
    for i, (atom, coord) in enumerate(zip(type_array, xyz_array)):
        rd_atom = Chem.Atom(atom)
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, coord)
        atom_indices.append(idx)

    mol.AddConformer(conf)

    # Determine and add bonds in the molecule
    try:
        DetermineBonds(mol)
    except Exception as e:
        logger.error(f"Fail to determine the bonds by DetermineBonds function")
        raise e
    
    # Get the RDKit Mol object.
    mol = mol.GetMol()
    
    return mol


def sdf_to_rdmol(file_path, manual_read=False, streamlit=False):
    '''
    This function is used to read an sdf file and convert it to an RDKit molecule object.
    There are two modes to read the sdf file: by manual read and by Chem.ForwardSDMolSupplier.
    The manual read mode is only used for V2000 version SDF files. 
    It reads the SDF file line by line and manually adds atoms and bonds to the RDKit molecule, which is an alternative way when the Chem.ForwardSDMolSupplier mode fails.
    The Chem.ForwardSDMolSupplier mode is the recommended and default way to read the sdf file, which is usually more stable and efficient.

    Args:
    - file_path (str): The path to the sdf file.
    - manual_read (bool): Whether to read the sdf molecule manually or by Chem.ForwardSDMolSupplier. 
        If True, the function will read the sdf file line by line and manually add atoms and bonds to the RDKit molecule.
        If False, the function will read the sdf file by Chem.ForwardSDMolSupplier, which is the recommended way.
        Currently, the manual_read mode is only used for V2000 version SDF files.
    - streamlit (bool): Whether to read the molecule in the streamlit mode. Currently, the streamlit mode is not supported yet.
    '''

    # Leave space for streamlit for future use
    if streamlit:
        logger.info("Streamlit is not supported yet. Turn to the normal mode.")
        pass

    # Get the first molecule block in the sdf file in case the file includes multiple molecules
    with open(file_path, 'r') as f:
        file = f.read()
        block = file.split('$$$$\n')[0]
    
    # Get the SDF version
    SDFversion = None
    if 'V3000' in block:
        SDFversion = "V3000"
    elif 'V2000' in block:
        SDFversion = "V2000"
    
    if SDFversion is None:
        logger.error(f"Invalid sdf file: {file_path}. No SDF version found.")
        raise ValueError(f"Invalid sdf file: {file_path}. No SDF version found.")
    
    # Get the number of atoms in the SDF molecule
    num_atoms = 0
    for mol in Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=False):
        if mol is not None:
            try:
                num_atoms = mol.GetNumAtoms()
                break
            except Exception as e:
                logger.error(f"Fail to read the molecule by ForwardSDMolSupplier in the sdf file: {file_path}")
                raise e

    if num_atoms == 0:
        logger.error(f"Fail to read the number of atoms in the sdf file: {file_path}")
        raise ValueError(f"Fail to read the number of atoms in the sdf file: {file_path}")
    

    # Decide whether to read SDF molecule by SDMolSupplier or manually, based on the SDF version and the number of atoms
    # If the SDF version is V2000 and the number of atoms is larger than 999, the SDF file is invalid
    if SDFversion == "V2000" and num_atoms > 999:
        logger.error(f"Invalid V2000 sdf file: {file_path}. The number of atoms is larger than 999.")
        raise ValueError(f"Invalid V2000 sdf file: {file_path}. The number of atoms is larger than 999.")
    

    rdmol = None
    # Enter the mode of reading the SDF molecule manually, but only for V2000 version
    if manual_read:
        if SDFversion == "V3000":
            logger.error(f"The V3000 version SDF is not supported in manual read mode: {file_path}.")
            raise ValueError(f"The V3000 version SDF is not supported in manual read mode: {file_path}.")

        # Get the index of the line including 'V2000' in the sdf block
        block_lines = block.split('\n')
        version_line_idx = [i for i, line in enumerate(block_lines) if "V2000" in line][0]

        # Get the file name for the molecule
        mol_name = block_lines[0]
        
        # Get the number of atoms and bonds
        try:
            atom_num = int(block_lines[version_line_idx][0:3].strip())
            bond_num = int(block_lines[version_line_idx][3:6].strip())
        except Exception as e:
            logger.error(f"Fail to read the number of atoms and bonds in the sdf file: {file_path}")
            raise e

        # Get the atom block and bond block
        atom_lines = block_lines[version_line_idx + 1: version_line_idx + 1 + atom_num]
        bond_lines = block_lines[version_line_idx + 1 + atom_num: version_line_idx + 1 + atom_num + bond_num]

        # Get the atomic symbols and coordinates in the atom block
        # xyz_atomic_symbols includes: ((x, y, z), atomic_symbol)
        try:
            xyz_atomic_symbols = [((float(line[0:10].strip()), float(line[10:20].strip()), float(line[20:30].strip())), line[30:34].strip()) for line in atom_lines]

        except Exception as e:
            logger.error(f"Fail to read the coordinates in the sdf file: {file_path}")
            raise e
        
        # Get the bond indices and bond orders in the bond block
        # bond_indices includes: (atom_index1, atom_index2, bond_order)
        try:
            BondType_dict = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE, 4: BondType.AROMATIC}
            bond_indices = [(int(line[0:3].strip()), int(line[3:6].strip()), BondType_dict[int(line[6:9].strip())]) for line in bond_lines]

        except Exception as e:
            logger.error(f"Fail to read the bonds in the sdf file: {file_path}")
            raise e
        
        # Create an RDKit molecule object and add atoms and bonds to the molecule
        mol = Chem.RWMol()
        conf = Chem.Conformer(atom_num)

        atom_indices = []
        for coord, atom in xyz_atomic_symbols:
            rd_atom = Chem.Atom(atom)
            idx = mol.AddAtom(rd_atom)
            conf.SetAtomPosition(idx, coord)
            atom_indices.append(idx)
        
        for idx1, idx2, bond in bond_indices:
            mol.AddBond(idx1-1, idx2-1, bond)

        # Add the 3D coordinates to the molecule by the conformer. (Only conformer can store 3D coordinates)
        mol.AddConformer(conf)

        # Get the RDKit Mol object. The 'mol' object is an RWMol object for writing atoms and bonds, so we need to get the Mol object.
        rdmol = mol.GetMol()

        # Set the _Name property for the molecule by the first line in the sdf block
        rdmol.SetProp("_NAME", mol_name)


    # Enter the mode of read the SDF molecule by Chem.ForwardSDMolSupplier
    else:
        for mol in Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=False):
            if mol is not None:
                rdmol = mol
                break

    # Check whether the RDKit molecule object is successfully read
    if rdmol is None:
        logger.error(f"Fail to read the molecule in the sdf file: {file_path}")
        raise ValueError(f"Fail to read the molecule in the sdf file: {file_path}")

    return rdmol
    
        
def xyz_to_rdmol(file_path):
    '''
    This function is used to convert a xyz file to an RDKit molecule object by reading the xyz file line by line and adding atoms and bonds to the molecule.
    The purpose of writing this function is to avoid the use of RDKit's xyz file reader, which is not stable when reading xyz files.
    Openbabel.pybel is able to read xyz files in a stable way, but the installation of openbabel is not stable, so we want to avoid using openbabel in EMS package.

    Args:
    - file_path (str): The path to the xyz file.
    '''

    with open(file_path, 'r') as f:
        # Read the lines in the xyz file and check whether the file includes molecule information
        lines = f.readlines()
        if len(lines) < 3:
            logger.error(f"Invalid xyz file: {file_path}")
            raise ValueError(f"Invalid xyz file: {file_path}")
        
        # Get the number of atoms in the xyz file
        try:
            num_atoms = int(lines[0].strip())
        except Exception as e:
            logger.error(f"Fail to read the number of atoms in the xyz file: {file_path}")
            raise e
        
        # Get the atomic symbols and coordinates in the xyz file
        try:
            xyz_lines = [line.split() for line in lines[2: 2+num_atoms]]
            atomic_symbols = [line[0] for line in xyz_lines]
            x = [float(line[1]) for line in xyz_lines]
            y = [float(line[2]) for line in xyz_lines]
            z = [float(line[3]) for line in xyz_lines]
            xyz = list(zip(x, y, z))
        except Exception as e:
            logger.error(f"Fail to read the coordinates in the xyz file: {file_path}")
            raise e
        
        # Create an RDKit molecule object and add atoms and coordinates to the molecule
        mol = Chem.RWMol()
        conf = Chem.Conformer(num_atoms)

        atom_indices = []
        for i, (atom, coord) in enumerate(zip(atomic_symbols, xyz)):
            rd_atom = Chem.Atom(atom)
            idx = mol.AddAtom(rd_atom)
            conf.SetAtomPosition(idx, coord)
            atom_indices.append(idx)

        mol.AddConformer(conf)

        # Determine and add bonds in the molecule
        try:
            DetermineBonds(mol)
        except Exception as e:
            logger.error(f"Fail to determine the bonds by DetermineBonds function in the xyz file: {file_path}")
            raise e
        
        # Get the RDKit Mol object.
        mol = mol.GetMol()
        
        return mol
    

# Dataframe read functionality
def dataframe_to_rdmol(filtered_atom_df, mol_name):
    '''
    This function is used to read a molecular dataframe and convert it to an RDKit molecule object.

    Args:
    - filtered_atom_df (pd.DataFrame): The atom dataframe to be converted to an RDKit molecule object.
    - mol_name (str): The name of the molecule.
    '''

    # Get the sub-dataframe for the molecule according to the molecule name
    mol_atom_df = filtered_atom_df[filtered_atom_df['molecule_name'] == mol_name]

    # Check whether the atom indexes in the molecule are continuous
    # If not, that means two molecules in the atom dataframe share the same molecule name
    mol_index = list(mol_atom_df.index)

    continuous_check = True
    for i in range(len(mol_index)-1):
        if mol_index[i+1] - mol_index[i] != 1:
            continuous_check = False
            break

    if not continuous_check:
        logger.error(f'The atom indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.')
        raise ValueError(f'The atom indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.')


    # Get the atom coordinates, atom types and connectivity matrix from the atom dataframe
    xyz = mol_atom_df[['x', 'y', 'z']].to_numpy().tolist()
    atom_types = mol_atom_df['typestr'].tolist()

    xyz_with_atom = [((x, y, z), atom) for (x, y, z), atom in zip(xyz, atom_types)]
    conn_matrix = np.array(mol_atom_df['conn'].to_list(), dtype=int)

    # Map integers in the connectivity matrix to RDKit BondTypes
    BondType_dict = {
        1: BondType.SINGLE,
        2: BondType.DOUBLE,
        3: BondType.TRIPLE,
        4: BondType.AROMATIC
    }
    
    # Get the atom indices in bonds and bond orders in the connectivity matrix
    bond_indices = [
        (i, j, BondType_dict.get(conn_matrix[i, j], BondType.SINGLE))
        for i in range(conn_matrix.shape[0])
        for j in range(i + 1, conn_matrix.shape[1])
        if conn_matrix[i, j] > 0
    ]

    # Create an RDKit molecule object and add atoms, bonds and atom coordinates to the molecule
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(xyz_with_atom))
    
    atom_indices = []
    for coord, atom in xyz_with_atom:
        rd_atom = Chem.Atom(atom)
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, coord)
        atom_indices.append(idx)
        
    for idx1, idx2, bond in bond_indices:
        mol.AddBond(idx1, idx2, bond)

    # Add the 3D coordinates to the molecule by the conformer. (Only conformer can store 3D coordinates)
    mol.AddConformer(conf)
    rdmol = mol.GetMol()
    
    return rdmol
















################## The following functions are temporarily abandoned ##################

# def emol_to_rdmol(ems_mol):
#     '''
#     This function is used to convert an EMS molecule object to an RDKit molecule object.
#     However, the EMS molecule object itself includes an RDKit molecule object, so this function is not necessary in most cases and temporarily abandoned.

#     Args:
#     - ems_mol: The EMS molecule object.
#     '''

#     # Create an RDKit molecule object
#     periodic_table = Get_periodic_table()
#     rdmol = Chem.RWMol()

#     # Add the atoms to the molecule
#     for atom in ems_mol.type:
#         symbol = periodic_table[int(atom)]
#         rdmol.AddAtom(Chem.Atom(symbol))

#     # Add the bonds to the molecule
#     visited = []
#     for i, bond_order_array in enumerate(ems_mol.conn):
#         for j, bond_order in enumerate(bond_order_array):
#             if j in visited:
#                 continue
#             elif bond_order != 0:
#                 rdmol.AddBond(i, j, Chem.BondType(bond_order))
#             else:
#                 continue
#         visited.append(i)

#     # Add the coordinates to the atoms
#     conformer = Chem.Conformer()
#     for i, coord in enumerate(ems_mol.xyz):
#         conformer.SetAtomPosition(i, coord)
#     rdmol.AddConformer(conformer)

#     rdmol = rdmol.GetMol()
#     return rdmol



# def from_rdmol_test(rdmol):
#     '''
#     A test version of structure_from_rdmol function, which is to test whether the molecule structure information can be extracted by RDKit functions without for loop.
#     The result shows the original function with for loop is faster than this test version.
#     '''

#     type_array = np.zeros(rdmol.GetNumAtoms(), dtype=np.int32)
#     for i, atoms in enumerate(rdmol.GetAtoms()):
#         type_array[i] = atoms.GetAtomicNum()
#     if rdmol.GetNumConformers() < 1:
#         AllChem.Compute2DCoords(rdmol)
#     xyz_array = rdmol.GetConformer().GetPositions()
#     conn_array = rdmol.GetAdjacencyMatrix(useBO=True)

#     return type_array, xyz_array, conn_array



# import openbabel.pybel as pyb

# This is the old version of the function xyz_to_rdmol, which relies on openbabel.pybel to read xyz files.
# However, the installation of openbabel is not always easy and successful, so we want to avoid using openbabel in EMS package.
# def xyz_to_rdmol_old(file_path, filename, tmp_file):
#     obmol = next(pyb.readfile('xyz', file_path))
#     obmol.write('sdf', tmp_file, overwrite=True)
#     rdmol = sdf_to_rdmol(tmp_file, filename, streamlit=False)
#     os.remove(tmp_file)






# def rdmol_to_sdf_block(rdmol, MolName, FileInfo, FileComment, tmp_file, SDFversion="V3000"):
#     '''
#     This function is used to write an RDKit molecule object to an sdf block with specified molecule properties and SDF version.
#     Here are some explanations and experiences for writing this function:
#     (1) In this function, I use Chem.SDWriter to write the sdf block, because this is the only way (as far as I know) to automatically write the molecule properties.
#         Other functions like Chem.MolToMolBlock and Chem.MolToMolFile only write the molecule structure without the properties, even if you add the properties to RDKit molecule object.
#     (2) For the _Name, _MolFileInfo, and _MolFileComments properties, only _Name will be automatically written to the sdf block by Chem.SDWriter, but if you want to write
#         _MolFileInfo and _MolFileComments, you need to manually change the second and third lines in the sdf block.
#     (3) Some useful functions in RDKit when writing the sdf block:
#         - Mol.GetPropsAsDict(): Get all the properties of the molecule as a dictionary, but not including hidden and computed properties.
#         - Mol.ClearProp(prop): Clear an assigned property of the molecule. However, there seems no function to clear all the properties at once.
#         - Mol.SetProp(name, value): Set a property for the molecule.
#         - Mol.GetPropNames(includePrivate=True, includeComputed=True): Get all the property names of the molecule, including hidden and computed properties.

#     Args:
#     - rdmol: The RDKit molecule object to be written to the sdf block.
#     - MolName (str): The name of the molecule, referring to the _Name property of the RDKit molecule object and the first line in the sdf file.
#     - FileInfo (str): The file information of the molecule, referring to the _MolFileInfo property of the RDKit molecule object and the second line in the sdf file.
#     - FileComment (str): The file comment of the molecule, referring to the _MolFileComments property of the RDKit molecule object and the third line in the sdf file.
#     - tmp_file (str): The path of the temporary sdf file to store the sdf block.
#     - SDFversion (str): The version of the sdf file, which can be "V3000" or "V2000".
#     '''

#     # Set the _Name properties for the RDKit molecule objec, which refer to the first three lines in the sdf file
#     rdmol.SetProp("_Name", MolName)

#     # Write the molecule to the sdf block with the specified SDF version
#     with Chem.SDWriter(tmp_file) as writer:
#         if SDFversion == "V3000":
#             writer.SetForceV3000(True)
#         elif SDFversion == "V2000":
#             writer.SetForceV3000(False)
#         else:
#             logger.error(f"Invalid SDF version: {SDFversion}")
#             raise ValueError(f"Invalid SDF version: {SDFversion}")
#         writer.write(rdmol)
    
#     # Read the sdf block from the temporary sdf file and set the _MolFileInfo and _MolFileComments properties
#     with open(tmp_file, 'r') as f:
#         lines = f.readlines()
#         lines[1] = FileInfo + '\n'
#         lines[2] = FileComment + '\n'
#     os.remove(tmp_file)
    
#     # Return the sdf block
#     return ''.join(lines)