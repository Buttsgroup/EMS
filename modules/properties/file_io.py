import logging
import sys
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from EMS.modules.properties.structure.structure_io import sdf_to_rdmol
from EMS.modules.properties.structure.structure_io import xyz_to_rdmol
from EMS.modules.properties.structure.structure_io import dataframe_to_rdmol
from EMS.modules.properties.structure.structure_io import structure_to_rdmol_NoConn
from EMS.modules.comp_chem.gaussian.gaussian_io import gaussian_read


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.WARNING)
########### Set up the logger system ###########


def assign_rdmol_name(rdmol, mol_id=None, extra_name=None):
    '''
    This function assigns a name to the RDKit molecule object (rdmol) based on the following order:
    (1) Extra names (if provided)
    (2) The '_Name' property of the RDKit molecule object
    (3) The 'FILENAME' property of the RDKit molecule object
        The 'FILENAME' property is not a standard property in the RDKit molecule object or the sdf file, but it is used in some sdf files in our lab.
    (4) The given 'mol_id' argument (if provided)
    If none of these properties are available, the function sets the '_Name' property to an empty string ''.

    Arguments:
    - rdmol: The RDKit molecule object to which the name will be assigned.
    - mol_id: The given molecule ID (optional). If provided, it will be used as the first choice for the name.
    - extra_name: Extra names to be added to the molecule name (optional). If provided, it will be used as the last choices for the name.
    '''

    # Format the extra_name argument
    if extra_name is None:
        extra_name = []

    else:
        if type(extra_name) == str:
            extra_name = [extra_name]
        elif type(extra_name) == list:
            extra_name = extra_name
        else:
            logger.error(f"Extra name should be a string or a list of strings, but got {type(extra_name)}")
            raise ValueError(f"Extra name should be a string or a list of strings, but got {type(extra_name)}")
        
            
    # Get the molecule name from the _Name property
    try:
        NameProp = rdmol.GetProp("_Name")
    except:
        NameProp = None
    
    if type(NameProp) == str:
        NameProp = NameProp.strip()

    # Get the molecule name from the FILENAME property
    try:
        filename = rdmol.GetProp("FILENAME")
    except:
        filename = None

    if type(filename) == str:
        filename = filename.strip()
    
    # Set the _Name property for the molecule according to the following order: NameProp, filename, mol_id
    name_order = extra_name + [NameProp, filename, mol_id]
    name_order = [i for i in name_order if i is not None and i != ""]
    
    if len(name_order) == 0:
        rdmol.SetProp("_Name", '')
    else:
        rdmol.SetProp("_Name", name_order[0])

    return rdmol


def file_to_rdmol(file, mol_id=None, streamlit=False):
    '''
    This function reads from various file formats and returns the corresponding RDKit molecule object (rdmol).
    It achieves the official name for the EMS molecule from the file and assigns the official name to the RDKit molecule object.

    Currently, it supports the following file formats:
    (1) .sdf file (str)
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
        - If none of these properties are available, the _Name property will be set to an empty string ''.
        - The official name for the EMS molecule will be obtained from the '_Name' property of the name-assigned RDKit molecule object.
        - The 'streamlit' argument is used to read the sdf file in a website, but is not supported yet. Need to be implemented in the future.
    (2) .xyz file (str)
        - The .xyz files usually don't include a name for the molecule, so it is recommended to set a name for the molecule using the 'mol_id' argument.
        - All of the id and official name for the EMS molecule and the name in the RDKit molecule object will be usually the same.
    (3) .log file by Gaussian (str)
        - The Gaussian .log files usually don't include a name for the molecule, so it is recommended to set a name for the molecule using the 'mol_id' argument.
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
    (4) line notation string (str), such as SMILES or SMARTS
        - Both the name for the RDKit molecule and the official name for the EMS molecule will be assigned using the line notation string.
        - The RDKit molecule object generated from the line notation string will be sanitized, hydrogen-added and kekulized.
    (5) RDKit molecule object (rdkit.Chem.rdchem.Mol)
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
        - If none of these properties are available, the _Name property will be set to an empty string ''.
        - The official name for the EMS molecule will be obtained from the '_Name' property of the name-assigned RDKit molecule object.
    (6) atom and pair dataframes (tuple)
        - The tuple should contain two pandas dataframes: the atom dataframe and the pair dataframe.
        - Both the name for the RDKit molecule and the official name for the EMS molecule will be assigned using the molecule name in the atom dataframe.
    '''

    file_type = None
    official_name = None
    rdmol = None
    
    # Check if the file is a string
    if isinstance(file, str):

        # Check if the file is a .sdf file
        if file.endswith('.sdf'):
            file_type = 'sdf'

            # Get the RDKit molecule object from the sdf file
            try:
                rdmol = sdf_to_rdmol(file, manual_read=False, streamlit=streamlit)
            except:
                logger.error(f"Fail to read RDKit molecule from the sdf file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the sdf file: {file}")
            
            # Assign a name to the RDKit molecule object in the order of _Name, FILENAME, mol_id
            # If none of these properties are available, set the _Name property to an empty string ''
            # The official name is obtained from the _Name property of the name-assigned RDKit molecule object
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")
        

        # Check if the file is a .xyz file
        elif file.endswith('.xyz'):
            file_type = 'xyz'

            # Get the RDKit molecule object from the xyz file
            try:
                rdmol = xyz_to_rdmol(file)
            except:
                logger.error(f"Fail to read RDKit molecule from the xyz file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the xyz file: {file}")
            
            # Assign a name to the RDKit molecule object and get the official name from the _Name property
            # Because the .xyz files usually don't include a name for the molecule, the id and official name for the EMS molecule and the name in the RDKit molecule object will be the same.
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")

        
        # Check if the file is a .log file
        elif file.endswith('.log'):
            with open(file, 'r') as f:
                first_line = f.readline()
            
            # Check if the .log file is a Gaussian log file
            if 'Gaussian' in first_line:
                file_type = 'gaussian-log'
                
                # Get the atom types and coordinates from the Gaussian log file
                try:
                    atom_types, atom_coords = gaussian_read(file)
                    rdmol = structure_to_rdmol_NoConn(atom_types, atom_coords)
                except:
                    logger.error(f"Fail to read RDKit molecule from the Gaussian log file: {file}")
                    raise ValueError(f"Fail to read RDKit molecule from the Gaussian log file: {file}")
                
                # Assign a name to the RDKit molecule object in the order of _Name, FILENAME, mol_id
                rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
                official_name = rdmol.GetProp("_Name")


            # Raise an error if the .log file is not a supported type 
            else:
                logger.error(f"Unable to determine the file type from the .log file: {file}")
                raise ValueError(f"Unable to determine the file type from the .log file: {file}")


        # If the file is not a path string, check if it is a line notation string
        else:
            as_smiles = None
            as_smarts = None

            # Try reading the file as a SMILES string
            try:
                as_smiles = Chem.MolFromSmiles(file)
            except:
                logger.warning(f"Fail to read RDKit molecule from the SMILES string: {file}")

            # Try reading the file as a SMARTS string
            try:
                as_smarts = Chem.MolFromSmarts(file)
            except:
                logger.warning(f"Fail to read RDKit molecule from the SMARTS string: {file}")
            
            # If reading the file as a line notation string fails, raise an error
            line_order = [('smiles', as_smiles), ('smarts', as_smarts)]
            line_order = [i for i in line_order if i[1] is not None]
            
            if len(line_order) == 0:
                logger.error(f"Fail to read RDKit molecule from the line notation string: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the line notation string: {file}")
            
            # If reading the file as a line notation string succeeds, assign the line notation string as the name to the RDKit molecule object and the official name to the EMS molecule
            file_type = line_order[0][0]
            line_mol = line_order[0][1]
            line_mol.SetProp("_Name", file)
            official_name = file

            # Process the RDKit molecule object transformed from line notation string
            try:
                Chem.SanitizeMol(line_mol)
                line_mol = Chem.AddHs(line_mol)
                Chem.Kekulize(line_mol)
                AllChem.EmbedMolecule(line_mol)              # obtain the initial 3D structure for a molecule

            except Exception as e:
                logger.error(f"Fail to process the rdkit molecule object transformed from line notation string by RDKit: {file}")
                raise e
            
            # Assign the line_mol to the rdmol variable
            rdmol = line_mol

    
    # Check if the file is an RDKit molecule object
    elif isinstance(file, Chem.rdchem.Mol):
        file_type = 'rdmol'

        # Assign the RDKit molecule object in the file to the rdmol variable
        # The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id
        rdmol = file
        rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)

        # Get the official name from the name-assigned RDKit molecule object
        official_name = rdmol.GetProp("_Name")

    
    # Check if the file is atom and pair dataframes
    elif isinstance(file, tuple) and isinstance(file[0], pd.DataFrame):
        file_type = 'dataframe'

        # Get the atom dataframe and the molecule name from the atom dataframe
        atom_df = file[0]
        mol_name = list(atom_df['molecule_name'])[0]

        # Get the RDKit molecule object from the atom dataframe
        try:
            rdmol = dataframe_to_rdmol(atom_df, mol_name=mol_name)
        except:
            logger.error(f"Fail to read RDKit molecule from the atom dataframe")
            raise ValueError(f"Fail to read RDKit molecule from the atom dataframe")
        
        # Assign the molecule name in the atom dataframe to both the RDKit molecule object and the official name for the EMS molecule
        rdmol.SetProp("_Name", mol_name)
        official_name = mol_name
    

    # If the file is not a valid type, raise an error
    else:
        logger.error(f"Invalid file type! Fail to read RDKit molecule from the file: {file}")
        raise ValueError(f"Invalid file type! Fail to read RDKit molecule from the file: {file}")
    
    # Check if the RDKit molecule object is successfully generated
    if not isinstance(rdmol, Chem.rdchem.Mol):
        logger.error(f"Fail to read RDKit molecule from the file: {file}")
        raise ValueError(f"Fail to read RDKit molecule from the file: {file}")
    
    # Return the file type, official name and RDKit molecule object
    return file_type, official_name, rdmol

        
        
        

    

                

        

                



        



