import numpy as np
import logging
import sys
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.Chem.rdchem import BondType

from EMS.utils.periodic_table import Get_periodic_table


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def from_rdmol(rdmol):
    type_array = np.zeros(rdmol.GetNumAtoms(), dtype=np.int32)
    xyz_array = np.zeros((rdmol.GetNumAtoms(), 3), dtype=np.float64)
    conn_array = np.zeros((rdmol.GetNumAtoms(), rdmol.GetNumAtoms()), dtype=np.int32)

    for i, atoms in enumerate(rdmol.GetAtoms()):
        type_array[i] = atoms.GetAtomicNum()
        if rdmol.GetNumConformers() < 1:
            AllChem.Compute2DCoords(rdmol)
        xyz_array[i][0] = rdmol.GetConformer(0).GetAtomPosition(i).x
        xyz_array[i][1] = rdmol.GetConformer(0).GetAtomPosition(i).y
        xyz_array[i][2] = rdmol.GetConformer(0).GetAtomPosition(i).z

        for j, atoms in enumerate(rdmol.GetAtoms()):
            if i == j:
                continue

            bond = rdmol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                conn_array[i][j] = int(bond.GetBondTypeAsDouble())
                conn_array[j][i] = int(bond.GetBondTypeAsDouble())

    return type_array, xyz_array, conn_array


def from_rdmol_test(rdmol):
    type_array = np.zeros(rdmol.GetNumAtoms(), dtype=np.int32)
    for i, atoms in enumerate(rdmol.GetAtoms()):
        type_array[i] = atoms.GetAtomicNum()
    if rdmol.GetNumConformers() < 1:
        AllChem.Compute2DCoords(rdmol)
    xyz_array = rdmol.GetConformer().GetPositions()
    conn_array = rdmol.GetAdjacencyMatrix(useBO=True)

    return type_array, xyz_array, conn_array


def emol_to_rdmol(ems_mol, sanitize=True):
    # Create an RDKit molecule object
    periodic_table = Get_periodic_table()
    rdmol = Chem.RWMol()

    # Add the atoms to the molecule
    for atom in ems_mol.type:
        symbol = periodic_table[int(atom)]
        rdmol.AddAtom(Chem.Atom(symbol))

    # Add the bonds to the molecule
    visited = []
    for i, bond_order_array in enumerate(ems_mol.conn):
        for j, bond_order in enumerate(bond_order_array):
            if j in visited:
                continue
            elif bond_order != 0:
                rdmol.AddBond(i, j, Chem.BondType(bond_order))
            else:
                continue
        visited.append(i)

        # Add the coordinates to the atoms
    conformer = Chem.Conformer()
    for i, coord in enumerate(ems_mol.xyz):
        conformer.SetAtomPosition(i, coord)
    rdmol.AddConformer(conformer)

    rdmol = rdmol.GetMol()
    # Sanitize the molecule
    if sanitize:
        Chem.SanitizeMol(rdmol)
    return rdmol


def sdf_to_rdmol(file_path, mol_id, manual_read=False, sanitize=False, addHs=False, kekulize=True, streamlit=False):
    '''
    This function is used to read an sdf file and convert it to an RDKit molecule object.
    There are two modes to read the sdf file: by manual read and by Chem.ForwardSDMolSupplier.
    The manual read mode is only used for V2000 version SDF files. 
    It reads the SDF file line by line and manually adds atoms and bonds to the RDKit molecule, which is an alternative way when the Chem.ForwardSDMolSupplier mode fails.
    The Chem.ForwardSDMolSupplier mode is the recommended way to read the sdf file, which is usually more stable and efficient.

    You can also choose whether to sanitize the molecule, add hydrogens to the molecule, and kekulize the molecule.
    !!! Attention !!! If you are reading the NMR data from the sdf file, please set sanitize=False and addHs=False.
    Otherwise, there may be some extra hydrogens added to the molecule, which are not included in the NMR data.

    Args:
    - file_path (str): The path to the sdf file.
    - mol_id (str): The id of the molecule, which is customized by the user.
    - manual_read (bool): Whether to read the sdf molecule manually or by Chem.ForwardSDMolSupplier. 
        If True, the function will read the sdf file line by line and manually add atoms and bonds to the RDKit molecule.
        If False, the function will read the sdf file by Chem.ForwardSDMolSupplier, which is the recommended way.
        Currently, the manual_read mode is only used for V2000 version SDF files.
    - sanitize (bool): Whether to sanitize the molecule after reading the sdf file.
        If you are reading the NMR data from the sdf file, please set sanitize=False.
    - addHs (bool): Whether to add hydrogens to the molecule after reading the sdf file.
        If you are reading the NMR data from the sdf file, please set addHs=False.
    - kekulize (bool): Whether to kekulize the molecule after reading the sdf file.
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
        for mol in Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=sanitize):
            if mol is not None:
                rdmol = mol
                break


    # The following section aims to set the _Name property for the molecule.
    # The _Name property is chosen in the following order if not blank: mol.GetProp("_Name"), mol_id, mol.GetProp("FILENAME").
    # The _Name property first searches the _Name property in the RDKit molecule object, which is the first line in the sdf block, 
    # then the customized 'self.id' attribute (mol_id) in the EMS object, and finally the FILENAME property.
    # The FILENAME property is not a standard property in the RDKit molecule object or the sdf file, but it is used in some sdf files in our lab.
    # Since the FILENAME property is not a standard property, it is in the lowest priority.

    # Get the molecule name from the _Name property
    try:
        NameProp = rdmol.GetProp("_Name")
    except:
        NameProp = None
        logger.info(f"Fail to read _Name property in {file_path}")
    
    if type(NameProp) == str:
        NameProp = NameProp.strip()

    # Get the molecule name from the FILENAME property
    try:
        filename = rdmol.GetProp("FILENAME")
    except:
        filename = None
        logger.info(f"FILENAME property not found in {file_path}")

    if type(filename) == str:
        filename = filename.strip()
    
    # Set the _Name property for the molecule according to the following order: NameProp, filename, mol_id
    name_order = [NameProp, mol_id, filename]
    name_order = [i for i in name_order if i is not None and i != ""]
    
    if len(name_order) == 0:
        rdmol.SetProp("_Name", '')
    else:
        rdmol.SetProp("_Name", name_order[0])
    

    # Check whether the RDKit molecule object is successfully read
    if rdmol is None:
        logger.error(f"Fail to read the molecule in the sdf file: {file_path}")
        raise ValueError(f"Fail to read the molecule in the sdf file: {file_path}")

    # Sanitize the molecule
    if sanitize:
        try:
            Chem.SanitizeMol(rdmol)
        except Exception as e:
            logger.error(f"Fail to sanitize the molecule in the sdf file: {file_path}. Return the unsanitized molecule.")
            raise e
    
    # Add hydrogens to the molecule
    if addHs:
        try:
            rdmol = Chem.AddHs(rdmol)
        except Exception as e:
            logger.error(f"Fail to add hydrogens to the molecule in the sdf file: {file_path}")
            raise e

    # Sanitize the molecule and kekulize the molecule
    if kekulize:
        try:
            Chem.Kekulize(rdmol)
        except Exception as e:
            logger.error(f"Fail to kekulize the molecule in the sdf file: {file_path}.")
            raise e
    
    return rdmol
    
        
def xyz_to_rdmol(file_path, sanitize=True):
    '''
    This function is used to convert a xyz file to an RDKit molecule object.
    The purpose of writing this function is to avoid the use of RDKit's xyz file reader, which is not stable when reading xyz files.
    Openbabel.pybel is able to read xyz files in a stable way, but the installation of openbabel is not always successful, so we want to avoid using openbabel in EMS package.

    Args:
    - file_path (str): The path to the xyz file.
    - sanitize (bool): Whether to sanitize the molecule after reading the xyz file.
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

        # Sanitize the molecule. If the molecule cannot be sanitized, return the unsanitized molecule.
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                logger.warning(f"Fail to sanitize the molecule in the xyz file: {file_path}. Return the unsanitized molecule.")
        
        return mol.GetMol()
    

def rdmol_to_sdf_block(rdmol, MolName, FileInfo, FileComment, tmp_file, prop_to_delete=[], SDFversion="V3000"):
    '''
    This function is used to write an RDKit molecule object to an sdf block with specified molecule properties and SDF version.
    Here are some explanations and experiences for writing this function:
    (1) In this function, I use Chem.SDWriter to write the sdf block, because this is the only way (as far as I know) to automatically write the molecule properties.
        Other functions like Chem.MolToMolBlock and Chem.MolToMolFile only write the molecule structure without the properties, even if you add the properties to RDKit molecule object.
    (2) For the _Name, _MolFileInfo, and _MolFileComments properties, only _Name will be automatically written to the sdf block by Chem.SDWriter, but if you want to write
        _MolFileInfo and _MolFileComments, you need to manually change the second and third lines in the sdf block.
    (3) Some useful functions in RDKit when writing the sdf block:
        - Mol.GetPropsAsDict(): Get all the properties of the molecule as a dictionary, but not including hidden and computed properties.
        - Mol.ClearProp(prop): Clear an assigned property of the molecule. However, there seems no function to clear all the properties at once.
        - Mol.SetProp(name, value): Set a property for the molecule.
        - Mol.GetPropNames(includePrivate=True, includeComputed=True): Get all the property names of the molecule, including hidden and computed properties.

    Args:
    - rdmol: The RDKit molecule object to be written to the sdf block.
    - MolName (str): The name of the molecule, referring to the _Name property of the RDKit molecule object and the first line in the sdf file.
    - FileInfo (str): The file information of the molecule, referring to the _MolFileInfo property of the RDKit molecule object and the second line in the sdf file.
    - FileComment (str): The file comment of the molecule, referring to the _MolFileComments property of the RDKit molecule object and the third line in the sdf file.
    - tmp_file (str): The path of the temporary sdf file to store the sdf block.
    - prop_to_delete (list): The list of properties to be deleted from the RDKit molecule object, so that the properties will not be written to the sdf file.
    - SDFversion (str): The version of the sdf file, which can be "V3000" or "V2000".
    '''

    # Set the _Name properties for the RDKit molecule objec, which refer to the first three lines in the sdf file
    rdmol.SetProp("_Name", MolName)

    # Delete the properties assigned in the prop_to_delete list, so that the properties will not be written to the sdf file
    if prop_to_delete is None:
        prop_to_delete = []
    for prop in prop_to_delete:
        rdmol.ClearProp(prop)

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
    



     


















# import openbabel.pybel as pyb

# This is the old version of the function xyz_to_rdmol, which relies on openbabel.pybel to read xyz files.
# However, the installation of openbabel is not always easy and successful, so we want to avoid using openbabel in EMS package.
# def xyz_to_rdmol_old(file_path, filename, tmp_file):
#     obmol = next(pyb.readfile('xyz', file_path))
#     obmol.write('sdf', tmp_file, overwrite=True)
#     rdmol = sdf_to_rdmol(tmp_file, filename, streamlit=False)
#     os.remove(tmp_file)