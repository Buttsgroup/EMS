import numpy as np
import logging
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from EMS.utils.periodic_table import Get_periodic_table
# import openbabel.pybel as pyb

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


def sdf_to_rdmol(file_path, mol_id, streamlit=False):
    if streamlit:
        SDMolMethod = Chem.ForwardSDMolSupplier
    else:
        SDMolMethod = Chem.SDMolSupplier
    
    for mol in SDMolMethod(file_path, removeHs=False, sanitize=False):
        if mol is not None:
            # This section aims to set the _Name property for the molecule
            # The _Name property is chosen in the following order if not blank: mol.GetProp("_Name"), mol.GetProp("FILENAME"), mol_id

            # Get the molecule name from the _Name property
            try:
                NameProp = mol.GetProp("_Name")
            except:
                NameProp = None
                logger.warning(f"Fail to read _Name property in {file_path}")
            
            if type(NameProp) == str:
                NameProp = NameProp.strip()

            # Get the molecule name from the FILENAME property
            try:
                filename = mol.GetProp("FILENAME")
            except:
                filename = None
                logger.warning(f"FILENAME property not found in {file_path}")

            if type(filename) == str:
                filename = filename.strip()
            
            # Set the _Name property for the molecule according to the following order: NameProp, filename, mol_id
            name_order = [NameProp, filename, mol_id]
            name_order = [i for i in name_order if i is not None or i != ""]
            
            if len(name_order) == 0:
                mol.SetProp("_Name", None)
            else:
                mol.SetProp("_Name", name_order[0])
            
            return mol
        

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
    

# This is the old version of the function xyz_to_rdmol, which relies on openbabel.pybel to read xyz files.
# However, the installation of openbabel is not always easy and successful, so we want to avoid using openbabel in EMS package.
# def xyz_to_rdmol_old(file_path, filename, tmp_file):
#     obmol = next(pyb.readfile('xyz', file_path))
#     obmol.write('sdf', tmp_file, overwrite=True)
#     rdmol = sdf_to_rdmol(tmp_file, filename, streamlit=False)
#     os.remove(tmp_file)