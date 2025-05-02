import numpy as np
import copy
from io import StringIO
import logging
import sys
from datetime import date
import random
import string

from EMS.modules.properties.structure.structure_io import structure_from_rdmol
from EMS.modules.properties.structure.structure_io import rdmol_to_sdf_block
from EMS.modules.properties.file_io import file_to_rdmol
from EMS.utils.periodic_table import Get_periodic_table
from EMS.modules.properties.nmr.nmr_io import nmr_read
from EMS.modules.properties.nmr.nmr_io import nmr_read_rdmol
from EMS.modules.properties.nmr.nmr_io import nmr_read_df
from EMS.modules.properties.nmr.nmr_io import nmr_to_sdf_block
from EMS.modules.comp_chem.gaussian.gaussian_io import gaussian_read_nmr
from EMS.modules.comp_chem.gaussian.gaussian_ops import scale_chemical_shifts

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdmolfiles


########### Set up the logger system ###########
# This section is used to set up the logging system, which aims to record the information of the package

# getLogger is to initialize the logging system with the name of the package
# A package can have multiple loggers.
logger = logging.getLogger(__name__)

# StreamHandler is a type of handler to print logging output to a specific stream, such as a console.
stdout = logging.StreamHandler(stream = sys.stdout)

# Formatter is used to specify the output format of the logging messages
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")

# Add the formatter to the handler
stdout.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stdout)

# Set the logging level of the logger. The logging level below INFO will not be recorded.
logger.setLevel(logging.INFO)

########### Set up the logger system ###########


class EMS(object):
    """
    The EMS class is the main class for the EMS package. It reads the molecular structure and properties from a file and transforms it to an emol object.
    The file type includes but not limited to SDF, XYZ, SMILES/SMARTS strings, and rdkit molecule objects.

    Args:
    - file: The file including the molecular structure and properties. Currently, the file type includes:
        (1) str: Location of the file, such as .sdf and .xyz.
        (2) str: SMILES/SMARTS string.
        (3) rdkit.Chem.rdchem.Mol: The rdkit molecule object.
        (4) tuple(pandas.DataFrame, pandas.DataFrame): The atom and pair dataframes including the molecular structure and properties.
    - mol_id (str): The customized molecule id. Default: None.
        This is a preferred name for the molecule. If not provided, the id will be set to the filename or official name of the file.
        Details of the definition of filename for different file types are in EMS.modules.properties.file_io.
    - nmr (bool): Whether to read NMR data. Default: False.
    - streamlit (bool): Whether to read the file from website. Default: False.
    - addHs (bool): Whether to add hydrogens to the rdkit molecule object. Default: False.
        If you are reading NMR data, addHs should be set to False.
    - sanitize (bool): Whether to sanitize the rdkit molecule object. Default: False.
        If you are reading NMR data, sanitize should be set to False.
    - kekulize (bool): Whether to kekulize the rdkit molecule object. Default: True.
    """

    def __init__(
        self,
        file,                       # The file to read
        mol_id=None,                # Customized molecule ID
        nmr=False,                  # Whether to read NMR data
        streamlit=False,            # Streamlit mode is used to read the file from website
        addHs=False,                # Whether to add hydrogens to the rdkit molecule object
        sanitize=False,             # Whether to sanitize the rdkit molecule object
        kekulize=True               # Whether to kekulize the rdkit molecule object
    ):


        # Initialize the attributes to save the file, the file type and the RDKit molecule object
        self.file = file                   # The file to read
        self.filetype = None               # The file type of the file to read
        self.rdmol = None                  # The rdkit molecule object that will be generated from the file
        self.streamlit = streamlit         # Streamlit mode is used to read the file from website

        # Initialize the attributes to save the molecule id and the official filename
        self.id = mol_id                   # The customized molecule id
        self.filename = None               # The official name of the file to read

        # Initialize the molecular structure and properties as empty
        self.type = None                   # The atomic numbers of the atoms in the molecule. Shape: (n_atoms,)
        self.xyz = None                    # The 3D coordinates of the molecule. Shape: (n_atoms, 3)
        self.conn = None                   # The bond order matrix of the molecule. Shape: (n_atoms, n_atoms)
        self.adj = None                    # The adjacency matrix of the molecule. Shape: (n_atoms, n_atoms)
        self.path_topology = None          # Path length between atoms
        self.path_distance = None          # 3D distance between atoms
        self.bond_existence = None         # Whether a bond exists between two atoms, made up of 0 or 1
        self.atom_properties = {}
        self.pair_properties = {}
        self.mol_properties = {}
        self.flat = None                   # Whether all the Z coordinates of the molecule is zero
        self.symmetric = None              # A string among 'sym', 'asym' and 'error' to indicate whether the non-hydrogen backbone of the molecule is symmetric
        self.nmr = nmr                     # Whether to read NMR data
        self.addHs = addHs                 # Whether to add hydrogens to the rdkit molecule object
        self.sanitize = sanitize           # Whether to sanitize the rdkit molecule object
        self.kekulize = kekulize           # Whether to kekulize the rdkit molecule object


        # Achieve the filetype, RDKit molecule object and its filename
        # The filename is the official name of the file to read.
        # Details of the definition of filename for different file types are in EMS.modules.properties.file_io
        self.filetype, self.filename, self.rdmol = file_to_rdmol(self.file, mol_id=self.id, streamlit=self.streamlit)

        # Assign the official name (filename) of the file to read to self.id if self.id is None
        if self.id is None or self.id == "":
            self.id = self.filename

        # Check if self.rdmol is read correctly from the file as an rdkit molecule object
        if not isinstance(self.rdmol, Chem.rdchem.Mol):
            logger.error(f"Fail to read RDKit molecule from {self.id}")
            raise TypeError(f"Fail to read RDKit molecule from {self.id}")
        
        
        # Add hydrogens to the rdkit molecule object
        if addHs:
            try:
                self.rdmol = Chem.AddHs(self.rdmol)
            except Exception as e:
                logger.error(f"Fail to add hydrogens to the rdkit molecule object: {self.id}")
                raise e

        # Sanitize the rdkit molecule object
        if self.sanitize:
            try:
                Chem.SanitizeMol(self.rdmol)
            except Exception as e:
                logger.error(f"Fail to sanitize the rdkit molecule object: {self.id}")
                raise e
        
        # Kekulize the rdkit molecule object
        if self.kekulize:
            try:
                Chem.Kekulize(self.rdmol)
            except Exception as e:
                logger.error(f"Fail to kekulize the rdkit molecule object: {self.id}")
                raise e

        # Get the molecular structures
        self.type, self.xyz, self.conn = structure_from_rdmol(self.rdmol)
        self.adj = Chem.GetAdjacencyMatrix(self.rdmol) 
        self.path_topology, self.path_distance = self.get_graph_distance()
        self.mol_properties["SMILES"] = Chem.MolToSmiles(self.rdmol)
        self.flat = self.check_Zcoords_zero() 

        # Check if every atom in the molecule has a correct valence
        # If any atom in the molecule has a wrong implicit valence or there is any error when calling the self.check_valence() function,
        # self.pass_valence_check will be set to False
        self.pass_valence_check = self.check_valence()
        
        # Check if the non-hydrogen backbone of the molecule is symmetric
        # If there is any error when calling the self.check_symmetric() function, self.symmetric will be set to 'Error'
        # The error may be caused by wrong explicit valences which are greater than permitted
        self.symmetric = self.check_symmetric()
        
        
        # Get NMR properties
        if self.nmr:
            # Generate self.pair_properties["nmr_types"] according to the path topology of self.rdmol
            self.get_coupling_types()       

            # Read NMR data if self.file is atom and pair dataframes
            # The difference between pair_properties["nmr_types"] and pair_properties["nmr_types_df"] is:
            # (1) pair_properties["nmr_types"] is the matrix of coupling types between every two atoms, so distant atoms are also included, like '11JCH'
            # (2) pair_properties["nmr_types_df"] is the matrix of coupling types only based on the atom pairs in the pair dataframe. 
            #     If the dataframe is a 6-path one, atom pairs with 7 or more bonds are not included, like '7JCH'. The not-included atom pairs are set to a '0' string.
            if self.filetype == "dataframe":
                try:
                    atom_df = file[0]
                    pair_df = file[1]
                    shift, shift_var, coupling_array, coupling_vars, coupling_types = nmr_read_df(atom_df, pair_df, self.filename)

                    self.pair_properties["nmr_types_df"] = coupling_types

                    # Check if the non-zero elements of pair_properties["nmr_types_df"] also exists in pair_properties["nmr_types"]
                    nmr_type_mask = self.pair_properties["nmr_types_df"] != '0'
                    nmr_types_match = self.pair_properties["nmr_types_df"] == self.pair_properties["nmr_types"]

                    if not (nmr_types_match == nmr_type_mask).all():
                        logger.warning(f"Some coupling types in pair_properties['nmr_types_df'] do not match with pair_properties['nmr_types'] for molecule {self.id}")
                
                except Exception as e:
                    logger.error(f'Fail to read NMR data for molecule {self.id} from dataframe')
                    raise e
            

            # Read NMR data if self.file is an RDKit molecule object
            elif self.filetype == "rdmol":
                try:
                    shift, shift_var, coupling_array, coupling_vars = nmr_read_rdmol(self.rdmol, self.id)
                except Exception as e:
                    logger.error(f'Fail to read NMR data for molecule {self.id} from rdkit molecule object')
                    raise e
            

            # Read NMR data if self.file is an SDF file
            elif self.filetype == 'sdf':
                try:
                    shift, shift_var, coupling_array, coupling_vars = nmr_read(self.file, self.streamlit)
                except Exception as e:
                    logger.error(f'Fail to read NMR data for molecule {self.id} from SDF file {self.file}')
                    raise e
            

            # Read NMR data if self.file is a Gaussian .log file
            elif self.filetype == 'gaussian-log':
                try:
                    shift, coupling_array = gaussian_read_nmr(self.file)
                    shift_var = np.zeros_like(shift)
                    coupling_vars = np.zeros_like(coupling_array)
                except Exception as e:
                    logger.error(f'Fail to read NMR data for molecule {self.id} from Gaussian .log file {self.file}')
                    raise e


            # Raise error if the file type is not among the above
            else:
                logger.error(f'File {self.id} with file type {self.filetype} is not supported for reading NMR data')
                raise ValueError(f'File {self.id} with file type {self.filetype} is not supported for reading NMR data')


            # If file type is 'gaussian-log', save the unscaled shift values in the "raw_shift" attribute of atom properties
            # Then save the scaled shift values in the "shift" attribute
            # The coupling values generally don't need to be scaled, so save them in the "coupling" attribute
            if self.filetype == 'gaussian-log':
                self.atom_properties["raw_shift"] = shift
                self.atom_properties["shift_var"] = shift_var
                self.pair_properties["coupling"] = coupling_array
                self.pair_properties["coupling_var"] = coupling_vars

                self.atom_properties["shift"] = scale_chemical_shifts(shift, self.type)

            # Assign the NMR data to the atom and pair properties for other file types
            else:
                self.atom_properties["shift"] = shift
                self.atom_properties["shift_var"] = shift_var
                self.pair_properties["coupling"] = coupling_array
                self.pair_properties["coupling_var"] = coupling_vars

            # Check if the length of the shift array is equal to the number of atoms in the molecule
            if len(self.atom_properties["shift"]) != len(self.type):
                logger.error(f'Fail to correctly read NMR data for molecule {self.id}')
                raise ValueError(f'Fail to correctly read NMR data for molecule {self.id}')


    def __str__(self):
        return f"EMS({self.id}), {self.mol_properties['SMILES']}"

    def __repr__(self):
        return (
            f"EMS({self.id}, \n"
            f"Filename: \n {self.filename}, \n"
            f"Filetype: \n {self.filetype}, \n"
            f"SMILES: \n {self.mol_properties['SMILES']}, \n"
            f"xyz: \n {self.xyz}, \n"
            f"Atom types: \n {self.type}, \n"
            f"Atom connectivity: \n {self.conn}, \n"
            f"Path topology: \n {self.path_topology}, \n"
            f"Path distance: \n {self.path_distance}, \n"
            f"Atom properties: \n {self.atom_properties}, \n"
            f"Pair properties: \n {self.pair_properties}, \n"
            f")"
        )

    def check_valence(self):
        """
        Check whether the molecule has correct valence.
        If any of the following conditions is met, the molecule has a wrong valence and check_valence will give a False:
        (1) The function Chem.MolFromSmiles raises an error when reading the SMILES string of the molecule
        (2) The function Chem.MolFromSmiles returns None when reading the SMILES string of the molecule
        (3) The molecule is not a single molecule, but a mixture of molecules, which is indicated by '.' in the SMILES string
        (4) The RDKit atom method GetImplicitValence() raises an error when reading the implicit valence of the atom
        (5) The implicit valence of one atom is not zero
        """

        check = True

        # Check if the function Chem.MolFromSmiles works when reading the SMILES string of the molecule
        try:
            smiles_mol = Chem.MolFromSmiles(self.mol_properties["SMILES"])
        except:
            logger.error(f"Function Chem.MolFromSmiles raises error when reading SMILES string of molecule {self.id}")
            return False
        
        # Check if the molecule can be read by RDKit from its SMILES string
        if smiles_mol is None:
            logger.warning(f"Molecule {self.id} cannot be read by RDKit from its SMILES string")
            return False
        
        # Check if the molecule is a single molecule not a mixture of molecules by checking if there is '.' in the SMILES string
        if '.' in self.mol_properties["SMILES"]:
            logger.warning(f"Molecule {self.id} is not a single molecule, but a mixture of molecules")
            return False
        
        # Check if every atom in the molecule has a correct implicit valence
        for atom in self.rdmol.GetAtoms():
            # If the RDKit molecule is not sanitized, the GetImplicitValence function will raise an error
            try:
                atom.GetImplicitValence()
            except Exception as e:
                logger.error(f"Molecule {self.id} cannot read its implicit valence, which may caused by not being sanitized!")
                return False

            # List the atoms with wrong implicit valence
            if atom.GetImplicitValence() != 0:
                check = False
                logger.error(f"Molecule {self.id}: Atom {atom.GetSymbol()}, index {atom.GetIdx()}, has wrong implicit valence")
            
        return check
    

    def check_Zcoords_zero(self, threshold=1e-3):
        """
        Check if all the Z coordinates of the molecule are zero. 
        If all the Z coordinates are zero, the molecule is flat and check_Zcoords_zero will give a True.
        """

        Z_coords = self.xyz[:, 2]
        if np.all(abs(Z_coords) < threshold):
            return True
        else:
            return False


    def check_symmetric(self):
        """
        Check if the non-hydrogen backbone of the molecule is symmetric. 
        If the non-hydrogen backbone is symmetric/asymmetric, check_symmetric will give a 'sym'/'asym'.
        If the method raises an error, check_symmetric will give an 'error'. 
        Attention: This method is not for checking 3D symmetry, but only the 2D non-hydrogen backbone.
        """

        try:
            mol = copy.deepcopy(self.rdmol)
            Chem.RemoveStereochemistry(mol)
            mol = rdmolops.RemoveAllHs(mol)
            canonical_ranking = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
            
            if len(canonical_ranking) == len(set(canonical_ranking)):
                return 'asym'
            else:
                return 'sym'
        
        except Exception as e:
            logger.error(f'Symmetry check fails for molecule {self.id}, due to wrong explicit valences which are greater than permitted')
            logger.info('Symmetry check failure due to wrong explicit valences needs to be fixed...')
            return 'error'


    def get_graph_distance(self):
        """
        Get the path length matrix and 3D distance matrix.
        The path length matrix is the shortest path length between atoms. Shape: (n_atoms, n_atoms)
        The 3D distance matrix is the 3D distance between atoms. Shape: (n_atoms, n_atoms)
        """
        
        try:
            return Chem.GetDistanceMatrix(self.rdmol).astype(int), Chem.Get3DDistanceMatrix(self.rdmol)
        except Exception as e:
            logger.error(f"Fail to get the path length matrix and 3D distance matrix for molecule {self.id}")
            raise e


    def get_coupling_types(self) -> None:
        """
        Get the coupling type matrix and save in self.pair_properties["nmr_types"]. shape: (n_atoms, n_atoms)

        Example:
            If two atoms (carbon and nitrogen) are 4 bonds away, the coupling type is '4JCN'.
        """

        p_table = Get_periodic_table()

        if self.path_topology is None:
            self.path_topology, self.path_distance = self.get_graph_distance()

        cpl_types = []
        for t, type in enumerate(self.type):
            tmp_types = []
            for t2, type2 in enumerate(self.type):

                if type > type2:
                    targetflag = (
                        str(int(self.path_topology[t][t2]))
                        + "J"
                        + p_table[type]
                        + p_table[type2]
                    )
                else:
                    targetflag = (
                        str(int(self.path_topology[t][t2]))
                        + "J"
                        + p_table[type2]
                        + p_table[type]
                    )

                tmp_types.append(targetflag)
            cpl_types.append(tmp_types)

        self.pair_properties["nmr_types"] = np.array(cpl_types, dtype=str)


    def to_sdf(self, outfile='', FileComments='', prop_to_write=None, prop_cover=False, SDFversion="V3000"):
        """
        Write the emol object to an SDF file with assigned properties.
        The first line is the SDF file name, which is defaulted to the _Name property of the RDKit molecule. If the _Name property is empty, self.id will be used.
        The second line is the SDF file information, which is defaulted to 'EMS (Efficient Molecular Storage) - <year> - ButtsGroup'.
        The third line is the SDF file comments, which is defaulted to blank.
        The properties to write to the SDF file and the SDF version can be customized.

        Args:
        - outfile (str): The file path to save the SDF file. If outfile is None or blank string "", the SDF block will be returned.
        - FileComments (str): The comments to write to the third line of the SDF file.
        - prop_to_write (str or list): The properties to write to the SDF file. 
            If prop_to_write is None, no property will be written.
            If prop_to_write is "nmr", the NMR properties saved in self.atom_properties and self.pair_properties will be written in NMREDATA_ASSIGNMENT and NMREDATA_J sections.
        - prop_cover (bool): Whether to cover the existing properties in the SDF file. 
        - SDFversion (str): The version of the SDF file. The version can be "V2000" or "V3000".
        """
        
        # Deep copy the rdmol object
        rdmol = copy.deepcopy(self.rdmol)

        # Set the first line of the SDF file to the molecule name in the order of _Name property of self.rdmol, self.filename and self.id
        try:
            rdmol_Name = rdmol.GetProp("_Name").strip()
        except:
            rdmol_Name = ""

        name_list = [rdmol_Name, self.filename, self.id]
        name_list = [name for name in name_list if name != None and name != ""]

        if len(name_list) == 0:
            sdf_Name = ""
        else:
            sdf_Name = name_list[0]
        
        # Get the _MolFileInfo property of the RDKit molecule to write to the second line of the SDF file.
        sdf_MolFileInfo = f'EMS (Efficient Molecular Storage) - {date.today().year} - ButtsGroup'

        # Get the _MolFileComments property of the RDKit molecule to write to the third line of the SDF file.
        sdf_MolFileComments = FileComments

        # Set the name of the temporary SDF file to save the RDKit molecule
        characters = string.ascii_letters + string.digits  
        random_string = ''.join(random.choices(characters, k=30))
        tmp_sdf_file = f"tmp_{random_string}.sdf"

        # Set the SDF file version according to the atom number of the RDKit molecule
        atom_num = len(self.type)
        SDFversion = SDFversion

        if SDFversion == "V2000" and atom_num > 999:
            logger.warning(f"V2000 cannot be used for molecules with more than 999 atoms. SDF version is set to V3000.")
            SDFversion = "V3000"
        
        if SDFversion not in ["V2000", "V3000"]:
            logger.warning(f"SDF version {SDFversion} is not supported. SDF version is set to V3000.")
            SDFversion = "V3000"
        
        # Set the properties to write to the SDF file
        if prop_to_write is None:
            prop_to_write = []
        elif type(prop_to_write) == str:
            prop_to_write = [prop_to_write]

        origin_prop = list(rdmol.GetPropsAsDict().keys())

        for prop in prop_to_write:
            # Write NMR properties to the SDF file
            if prop == "nmr":
                atom_lines, pair_lines = nmr_to_sdf_block(self.type, self.atom_properties, self.pair_properties)

                if prop_cover:
                    rdmol.SetProp("NMREDATA_ASSIGNMENT", atom_lines)
                    rdmol.SetProp("NMREDATA_J", pair_lines)
                else:
                    if not "NMREDATA_ASSIGNMENT" in origin_prop:
                        rdmol.SetProp("NMREDATA_ASSIGNMENT", atom_lines)
                    if not "NMREDATA_J" in origin_prop:
                        rdmol.SetProp("NMREDATA_J", pair_lines)

        # Get the SDF block of the RDKit molecule
        block = rdmol_to_sdf_block(rdmol, sdf_Name, sdf_MolFileInfo, sdf_MolFileComments, tmp_sdf_file, SDFversion=SDFversion)

        # Write the SDF block to the SDF file
        if outfile is None or outfile.strip() == "":
            return block
        else:
            with open(outfile, "w") as f:
                f.write(block)


















################ The following section includes the old version of the __init__ method in EMS ################

    # def __init__(
    #     self,
    #     file=None,                  # File path (optional if DataFrames are used)
    #     mol_id=None,                # Customized molecule ID
    #     line_notation=None,         # 'smiles' or 'smarts'
    #     rdkit_mol=False,            # Whether to read the file as an rdkit molecule
    #     dataframe=False,            # Whether EMS objects are being read from a DataFrame
    #     atom_df=None,               # Atom dataframe
    #     pair_df=None,               # Pair dataframe
    #     nmr=False,                  # Whether to read NMR data
    #     streamlit=False,            # Streamlit mode is used to read the file from website
    #     addHs=False,                # Whether to add hydrogens to the rdkit molecule object
    #     sanitize=False,             # Whether to sanitize the rdkit molecule object
    #     kekulize=True               # Whether to kekulize the rdkit molecule object
    # ):


    #     # To initialize self.id, self.filename, self.file and self.stringfile
    #     # (1) If the molecule file is SMILES or SMARTS string, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the string
    #     # (2) If the molecule file is streamlit, which is used to read the file from website, self.id is the customized 'mol_id' name,
    #     #     and self.filename, self.file and self.stringfile are achieved from the 'file' object
    #     # (3) If the molecule file is neither SMILES/SMARTS string or streamlit but saved in a file like .sdf, self.id is the customized 'mol_id' name, 
    #     #     self.file and self.stringfile are the file path, and self.filename is the file name seperated by '/' in the file path
    #     # (4) If the molecule is an rdkit molecule, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the customized 'mol_id' name
    #     # (5) If the molecule is a dataframe, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the first 'molecule_name' item in the atom_df dataframe
    #     if line_notation:
    #         self.id = file
    #     elif dataframe:
    #         self.id = list(atom_df['molecule_name'])[0]
    #     else:
    #         self.id = mol_id

    #     if line_notation:
    #         self.filename = file
    #     elif streamlit and not line_notation:
    #         self.filename = file.name
    #     elif rdkit_mol:
    #         self.filename = mol_id
    #     elif dataframe:
    #         self.filename = self.id
    #     else:
    #         self.filename = file.split('/')[-1]
        
    #     if rdkit_mol:
    #         self.file = mol_id
    #     elif dataframe:
    #         self.file = self.id
    #     else:
    #         self.file = file

    #     if streamlit and not line_notation:
    #         self.stringfile = StringIO(file.getvalue().decode("utf-8"))
    #     elif rdkit_mol:
    #         self.stringfile = mol_id
    #     elif dataframe:
    #         self.stringfile = self.id
    #     else:
    #         self.stringfile = file

    #     # Initialize the molecular structure and properties as empty
    #     self.rdmol = None                  # The rdkit molecule object of the molecule
    #     self.type = None                   # The atomic numbers of the atoms in the molecule. Shape: (n_atoms,)
    #     self.xyz = None                    # The 3D coordinates of the molecule. Shape: (n_atoms, 3)
    #     self.conn = None                   # The bond order matrix of the molecule. Shape: (n_atoms, n_atoms)
    #     self.adj = None                    # The adjacency matrix of the molecule. Shape: (n_atoms, n_atoms)
    #     self.path_topology = None          # Path length between atoms
    #     self.path_distance = None          # 3D distance between atoms
    #     self.bond_existence = None         # Whether a bond exists between two atoms, made up of 0 or 1
    #     self.atom_properties = {}
    #     self.pair_properties = {}
    #     self.mol_properties = {}
    #     self.flat = None                   # Whether all the Z coordinates of the molecule is zero
    #     self.symmetric = None              # Whether the non-hydrogen backbone of the molecule is symmetric
    #     self.streamlit = streamlit
    #     self.sanitize = sanitize           # Whether to sanitize the rdkit molecule object
    #     self.kekulize = kekulize           # Whether to kekulize the rdkit molecule object

    #     # get the rdmol object from the file
    #     # If the molecule file is a SMILES/SMARTS string, the rdmol object is generated from the string
    #     if line_notation:
    #         line_mol = None

    #         if line_notation == "smiles":
    #             try:
    #                 line_mol = Chem.MolFromSmiles(file)
    #             except Exception as e:
    #                 logger.error(f"Error in calling Chem.MolFromSmiles: {file}")
    #                 raise e

    #         elif line_notation == "smarts":
    #             try:
    #                 line_mol = Chem.MolFromSmarts(file)
    #             except Exception as e:
    #                 logger.error(f"Error in calling Chem.MolFromSmarts: {file}")
    #                 raise e

    #         else:
    #             logger.error(f"Line notation, {line_notation}, not supported")
    #             raise ValueError(f"Line notation, {line_notation}, not supported")
            
    #         if line_mol is None:
    #             logger.error(f"Fail to read the molecule from string by RDKit: {file}")
    #             raise ValueError(f"Fail to read the molecule from string by RDKit: {file}")

    #         try:
    #             Chem.SanitizeMol(line_mol)
    #             line_mol = Chem.AddHs(line_mol)
    #             Chem.Kekulize(line_mol)
    #             AllChem.EmbedMolecule(line_mol)              # obtain the initial 3D structure for a molecule
    #             self.rdmol = line_mol
    #         except Exception as e:
    #             logger.error(f"Fail to process the rdkit molecule object transformed from SMILES/SMARTS string by RDKit: {file}")
    #             raise e
        
    #     # If the molecule file is an rdkit molecule, the rdmol object is the molecule itself
    #     elif rdkit_mol:
    #         self.rdmol = file
        
    #     # If the molecule is a dataframe, an rdmol object is generated
    #     # The molecule name is self.id and is assigned to self.rdmol in dataframe_to_rdmol function
    #     elif dataframe:
    #         try:
    #             self.rdmol = dataframe_to_rdmol(atom_df, self.id)
    #         except Exception as e:
    #             logger.error(f"Fail to read the molecule from dataframe: {self.id}")
    #             raise e

    #     # If the molecule is saved in a file, the rdmol object is generated from the file
    #     else:
    #         ftype = self.filename.split(".")[-1]
            
    #         if ftype == "sdf":
    #             if streamlit:
    #                 self.rdmol = sdf_to_rdmol(self.file, self.id, streamlit=True)     # Leave the streamlit mode blank for future use
    #             else:
    #                 self.rdmol = sdf_to_rdmol(self.file, self.id, streamlit=False)

    #         elif ftype == "xyz":
    #             try:
    #                 self.rdmol = xyz_to_rdmol(self.file)
    #             except Exception as e:
    #                 logger.error(f"Fail to read the xyz file: {self.file}")
    #                 raise e

    #         elif ftype == "mol2":
    #             self.rdmol = Chem.MolFromMol2File(
    #                 self.file, removeHs=False, sanitize=self.sanitize
    #             )

    #         elif ftype == "mae":
    #             for mol in Chem.MaeMolSupplier(
    #                 self.file, removeHs=False, sanitize=self.sanitize
    #             ):
    #                 if mol is not None:
    #                     if mol.GetProp("_Name") is None:
    #                         mol.SetProp("_Name", self.id)
    #                     self.rdmol = mol

    #         elif ftype == "pdb":
    #             self.rdmol = Chem.MolFromPDBFile(
    #                 self.file, removeHs=False, sanitize=False
    #             )
                
    #         # Add file reading for ASE molecules with '.traj' extension when available

    #         else:
    #             logger.error(f"File type, {ftype} not supported")
    #             raise ValueError(f"File type, {ftype} not supported")















################ The following section includes some abandoned methods of EMS class ################

    # def check_Zcoords_zero_old(self):
    #     """
    #     This is an old version of check_Zcoords_zero method.
    #     Abandoned!!!
    #     """
 
    #     # If the Z coordinates are all zero, the molecule is flat and return True
    #     # Otherwise, if there is at least one non-zero Z coordinate, return False
    #     if self.streamlit:
    #         for line in self.stringfile:
    #             # if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
    #             if len(line.split()) == 12 and line.split()[-1] != 'V2000':
    #                 z_coord = float(
    #                     line.split()[3]
    #                 )  # Assuming the z coordinate is the fourth field
    #                 if z_coord != 0:
    #                     return False
    #         return True

    #     else:
    #         with open(self.stringfile, "r") as f:
    #             lines = f.readlines()[2:]
    #             coord_flag = False
    #             for line in lines:
    #                 # if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
    #                 if 'V2000' in line:
    #                     coord_flag = True
    #                     continue
    #                 if coord_flag and len(line.split()) > 12 and line.split()[3].isalpha():
    #                     z_coord = float(
    #                         line.split()[2]
    #                     )  # Assuming the z coordinate is the fourth field
    #                     if abs(z_coord) > 1e-6:
    #                         return False
    #                 elif coord_flag and len(line.split()) < 12:
    #                     break
    #             return True



    # def check_semi_symmetric(self, atom_type_threshold={}):
    #     """
    #     If two non-hydrogen atoms of the same atom type have a chemical shift difference less than a threshold, the molecule is semi-symmetric.
    #     Attention!!! This method is abandoned and not used in the current version of EMS.
    #     """
    #     symmetric = False

    #     chemical_shift_df = pd.DataFrame({
    #         'atom_type': self.type,
    #         'shift': self.atom_properties['shift']
    #         })
        
    #     for atom_type in atom_type_threshold:
    #         threshold = atom_type_threshold[atom_type]
    #         atom_type_CS = chemical_shift_df[chemical_shift_df['atom_type'] == atom_type]['shift'].to_list()
            
    #         for i in range(len(atom_type_CS)):
    #             for j in range(i+1, len(atom_type_CS)):
    #                 if abs(atom_type_CS[i] - atom_type_CS[j]) < threshold:
    #                     symmetric = True
        
    #     return symmetric