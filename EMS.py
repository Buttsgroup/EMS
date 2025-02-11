import pandas as pd
import numpy as np
import copy
from io import StringIO
import logging
import sys
from datetime import date
import random
import string

from EMS.modules.properties.structure_io import from_rdmol, sdf_to_rdmol, xyz_to_rdmol, rdmol_to_sdf_block
from EMS.utils.periodic_table import Get_periodic_table
from EMS.modules.properties.nmr.nmr_io import nmr_read, nmr_read_rdmol

from rdkit import Chem
from rdkit.Chem import AllChem
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
        file (str): The file path of the molecule file. If the molecule file is a SMILES/SMARTS string, the file is the string.
            If the molecule is an rdkit molecule, the file is the rdkit molecule object.
        mol_id (str): The customized molecule id. The file name is preferred.
        line_notation (str): If the molecule file is a SMILES/SMARTS string, the line_notation is 'smiles' or 'smarts'. Default: None.
        rdkit_mol (bool): Whether to read the file as an rdkit molecule. Default: False.
        nmr (bool): Whether to read NMR data. Default: False.
        streamlit (bool): Whether to read the file from ForwardSDMolSupplier. Default: False.
    
    More details about the EMS class can be found in the comments below.
    """

    def __init__(
        self,
        file,                      # File path
        mol_id=None,               # Customized molecule id
        line_notation=None,        # 'smiles' or 'smarts'
        rdkit_mol=False,           # Whether to read the file as an rdkit molecule
        nmr=False,                 # Whether to read NMR data
        streamlit=False,           # Streamlit mode is used to read the file from website
    ):
        
        # To initialize self.id, self.filename, self.file and self.stringfile
        # (1) If the molecule file is SMILES or SMARTS string, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the string
        # (2) If the molecule file is streamlit, which is used to read the file from website, self.id is the customized 'mol_id' name,
        #     and self.filename, self.file and self.stringfile are achieved from the 'file' object
        # (3) If the molecule file is neither SMILES/SMARTS string or streamlit but saved in a file like .sdf, self.id is the customized 'mol_id' name, 
        #     self.file and self.stringfile are the file path, and self.filename is the file name seperated by '/' in the file path
        # (4) If the molecule is an rdkit molecule, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the customized 'mol_id' name
        if line_notation:
            self.id = file
        else:
            self.id = mol_id

        if line_notation:
            self.filename = file
        elif streamlit and not line_notation:
            self.filename = file.name
        elif rdkit_mol:
            self.filename = mol_id
        else:
            self.filename = file.split('/')[-1]
        
        if rdkit_mol:
            self.file = mol_id
        else:
            self.file = file

        if streamlit and not line_notation:
            self.stringfile = StringIO(file.getvalue().decode("utf-8"))
        elif rdkit_mol:
            self.stringfile = mol_id
        else:
            self.stringfile = file

        # Initialize the molecular structure and properties as empty
        self.rdmol = None                  # The rdkit molecule object of the molecule
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
        self.symmetric = None              # Whether the non-hydrogen backbone of the molecule is symmetric
        self.streamlit = streamlit

        # get the rdmol object from the file
        # If the molecule file is a SMILES/SMARTS string, the rdmol object is generated from the string
        if line_notation:
            line_mol = None

            if line_notation == "smiles":
                try:
                    line_mol = Chem.MolFromSmiles(file)
                except Exception as e:
                    logger.error(f"Error in calling Chem.MolFromSmiles: {file}")
                    raise e

            elif line_notation == "smarts":
                try:
                    line_mol = Chem.MolFromSmarts(file)
                except Exception as e:
                    logger.error(f"Error in calling Chem.MolFromSmarts: {file}")
                    raise e

            else:
                logger.error(f"Line notation, {line_notation}, not supported")
                raise ValueError(f"Line notation, {line_notation}, not supported")
            
            if line_mol is None:
                logger.error(f"Fail to read the molecule from string by RDKit: {file}")
                raise ValueError(f"Fail to read the molecule from string by RDKit: {file}")

            try:
                Chem.SanitizeMol(line_mol)
                line_mol = Chem.AddHs(line_mol)
                Chem.Kekulize(line_mol)
                AllChem.EmbedMolecule(line_mol)              # obtain the initial 3D structure for a molecule
                self.rdmol = line_mol
            except Exception as e:
                logger.error(f"Fail to process the rdkit molecule object by RDKit: {file}")
                raise e
        
        # If the molecule file is an rdkit molecule, the rdmol object is the molecule itself
        elif rdkit_mol:
            self.rdmol = file
 
        # If the molecule is saved in a file, the rdmol object is generated from the file
        else:
            ftype = self.filename.split(".")[-1]
            
            # !!!!!! Need to consider how to read .sdf streamlit file 
            if ftype == "sdf":
                if streamlit:
                    self.rdmol = sdf_to_rdmol(self.file, self.id, streamlit=True)
                else:
                    self.rdmol = sdf_to_rdmol(self.file, self.id, streamlit=False)

            elif ftype == "xyz":
                try:
                    self.rdmol = xyz_to_rdmol(self.file)
                except Exception as e:
                    logger.error(f"Fail to read the xyz file: {self.file}")
                    raise e

            elif ftype == "mol2":
                self.rdmol = Chem.MolFromMol2File(
                    self.file, removeHs=False, sanitize=False
                )

            elif ftype == "mae":
                for mol in Chem.MaeMolSupplier(
                    self.file, removeHs=False, sanitize=False
                ):
                    if mol is not None:
                        if mol.GetProp("_Name") is None:
                            mol.SetProp("_Name", self.id)
                        self.rdmol = mol

            elif ftype == "pdb":
                self.rdmol = Chem.MolFromPDBFile(
                    self.file, removeHs=False, sanitize=False
                )
                
            # Add file reading for ASE molecules with '.traj' extension when available

            else:
                logger.error(f"File type, {ftype} not supported")
                raise ValueError(f"File type, {ftype} not supported")

        # Check if self.rdmol is read correctly from the file as an rdkit molecule object
        if not isinstance(self.rdmol, Chem.rdchem.Mol):
            logger.error(f"File {self.file} not read correctly to rdkit molecule object")
            raise TypeError(f"File {self.file} not read correctly to rdkit molecule object")

        # Get the molecular structures
        self.type, self.xyz, self.conn = from_rdmol(self.rdmol)
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
        if nmr:
            if rdkit_mol:
                prop_dict = self.rdmol.GetPropsAsDict()
                
                try:
                    shift = prop_dict['NMREDATA_ASSIGNMENT']
                    coupling = prop_dict['NMREDATA_J']
                    self.get_coupling_types()
                    shift, shift_var, coupling, coupling_vars = nmr_read_rdmol(shift, coupling)
                except Exception as e:
                    logger.error(f'No NMR data found for molecule {self.id}')
                    raise ValueError(f'No NMR data found for molecule {self.id}')

            else:
                try:
                    self.get_coupling_types()     # Generate self.pair_properties["nmr_types"]
                    shift, shift_var, coupling, coupling_vars = nmr_read(self.stringfile, self.streamlit)
                except Exception as e:
                    raise ValueError(f'Fail to read NMR data for molecule {self.id} from file {self.stringfile}')

            self.atom_properties["shift"] = shift
            self.atom_properties["shift_var"] = shift_var
            self.pair_properties["coupling"] = coupling
            self.pair_properties["coupling_var"] = coupling_vars
            if len(self.atom_properties["shift"]) != len(self.type):
                raise ValueError(f'Fail to correctly read NMR data for molecule {self.id}')


    def __str__(self):
        return f"EMS({self.id}), {self.mol_properties['SMILES']}"

    def __repr__(self):
        return (
            f"EMS({self.id}, \n"
            f"SMILES: \n {self.mol_properties['SMILES']}, \n"
            f"xyz: \n {self.xyz}, \n"
            f"types: \n {self.type}, \n"
            f"conn: \n {self.conn}, \n"
            f"Path Topology: \n {self.path_topology}, \n"
            f"Path Distance: \n {self.path_distance}, \n"
            f"Atom Properties: \n {self.atom_properties}, \n"
            f"Pair Properties: \n {self.pair_properties}, \n"
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

        The cause of (4) may be that the molecule is not 'suitably' sanitized.
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

    def check_Zcoords_zero_old(self):
        """
        This is an old version of check_Zcoords_zero method.
        Abandoned!!!
        """
 
        # If the Z coordinates are all zero, the molecule is flat and return True
        # Otherwise, if there is at least one non-zero Z coordinate, return False
        if self.streamlit:
            for line in self.stringfile:
                # if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
                if len(line.split()) == 12 and line.split()[-1] != 'V2000':
                    z_coord = float(
                        line.split()[3]
                    )  # Assuming the z coordinate is the fourth field
                    if z_coord != 0:
                        return False
            return True

        else:
            with open(self.stringfile, "r") as f:
                lines = f.readlines()[2:]
                coord_flag = False
                for line in lines:
                    # if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
                    if 'V2000' in line:
                        coord_flag = True
                        continue
                    if coord_flag and len(line.split()) > 12 and line.split()[3].isalpha():
                        z_coord = float(
                            line.split()[2]
                        )  # Assuming the z coordinate is the fourth field
                        if abs(z_coord) > 1e-6:
                            return False
                    elif coord_flag and len(line.split()) < 12:
                        break
                return True

    def check_symmetric(self):
        """
        Check if the non-hydrogen backbone of the molecule is symmetric. 
        Attention: This method is not for checking 3D symmetry, but only the 2D non-hydrogen backbone.
        """

        try:
            mol = copy.deepcopy(self.rdmol)
            Chem.RemoveStereochemistry(mol)
            mol = rdmolops.RemoveAllHs(mol)
            canonical_ranking = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
            
            if len(canonical_ranking) == len(set(canonical_ranking)):
                return False
            else:
                return True
        
        except Exception as e:
            logger.error(f'Symmetry check fails for molecule {self.id}, due to wrong explicit valences which are greater than permitted')
            logger.info('Symmetry check failure due to wrong explicit valences needs to be fixed...')
            return 'Error'
        
    def check_semi_symmetric(self, atom_type_threshold={}):
        """
        If two non-hydrogen atoms of the same atom type have a chemical shift difference less than a threshold, the molecule is semi-symmetric.
        Attention!!! This method is abandoned and not used in the current version of EMS.
        """
        symmetric = False

        chemical_shift_df = pd.DataFrame({
            'atom_type': self.type,
            'shift': self.atom_properties['shift']
            })
        
        for atom_type in atom_type_threshold:
            threshold = atom_type_threshold[atom_type]
            atom_type_CS = chemical_shift_df[chemical_shift_df['atom_type'] == atom_type]['shift'].to_list()
            
            for i in range(len(atom_type_CS)):
                for j in range(i+1, len(atom_type_CS)):
                    if abs(atom_type_CS[i] - atom_type_CS[j]) < threshold:
                        symmetric = True
        
        return symmetric

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

        self.pair_properties["nmr_types"] = cpl_types

    def to_sdf(self, outfile, FileComments='', prop_to_write='all', SDFversion="V3000"):
        """
        Write the emol object to an SDF file with assigned properties.
        The first line is the SDF file name, which is defaulted to the _Name property of the RDKit molecule. If the _Name property is empty, self.id will be used.
        The second line is the SDF file information, which is defaulted to 'EMS (Efficient Molecular Storage) - <year> - ButtsGroup'.
        The third line is the SDF file comments, which is defaulted to blank.
        The properties to write to the SDF file and the SDF version can be customized.
        V3000 is default and V2000 is not recommended for molecules with more than 99 atoms. If the atom number is greater than 999, V3000 will be used.

        Args:
        - outfile (str): The file path to save the SDF file. If outfile is None or blank string "", the SDF block will be returned.
        - FileComments (str): The comments to write to the third line of the SDF file.
        - prop_to_write (str or list): The properties to write to the SDF file. If prop_to_write is None, no property will be written.
            If prop_to_write is "all", all the properties of the RDKit molecule will be written.
            If prop_to_write is "nmr", only the NMR properties (NMREDATA_ASSIGNMENT and NMREDATA_J) will be written.
        - SDFversion (str): The version of the SDF file. The version can be "V2000" or "V3000".
        """
        
        # Deep copy the rdmol object
        rdmol = copy.deepcopy(self.rdmol)

        # Get the _Name property of the RDKit molecule to write to the first line of the SDF file. If the _Name property is empty, set it to the emol id.
        try:
            rdmol_Name = rdmol.GetProp("_Name")
            if rdmol_Name.strip() == "":
                if self.id is None:
                    rdmol_Name = ""
                else:
                    rdmol_Name = self.id
        except:
            logger.warning(f"Fail to get the _Name property of the RDKit molecule. The molecule id will be used instead.")
            if self.id is None:
                rdmol_Name = ""
            else:
                rdmol_Name = self.id

        # Get the _MolFileInfo property of the RDKit molecule to write to the second line of the SDF file.
        rdmol_MolFileInfo = f'EMS (Efficient Molecular Storage) - {date.today().year} - ButtsGroup'

        # Get the _MolFileComments property of the RDKit molecule to write to the third line of the SDF file.
        rdmol_MolFileComments = FileComments

        # Set the name of the temporary SDF file to save the RDKit molecule
        characters = string.ascii_letters + string.digits  
        random_string = ''.join(random.choices(characters, k=30))
        tmp_sdf_file = f"tmp_{random_string}.sdf"

        # Set the SDF file version according to the atom number of the RDKit molecule
        atom_num = len(self.type)
        SDFversion = SDFversion

        if SDFversion == "V2000":
            logger.info(f"V2000 is not recommended for molecules with more than 99 atoms for the sake of failed reading. Please use V3000 instead.")

            if atom_num > 999:
                logger.error(f"V2000 cannot be used for molecules with more than 999 atoms. SDF version is set to V3000.")
                SDFversion = "V3000"

            elif atom_num > 99:
                logger.warning(f"V2000 may cause reading failure for molecules with more than 99 atoms. Please use V3000 instead.")
            
            else:
                logger.info(f"V2000 is being used for molecules with less than 100 atoms.")
        
        elif SDFversion == "V3000":
            logger.info(f"V3000 is being used.")
        
        else:
            logger.error(f"SDF version {SDFversion} is not supported. SDF version is set to V3000.")
            SDFversion = "V3000"
        
        # Set the properties to write to the SDF file
        prop_to_write = prop_to_write
        origin_prop = list(rdmol.GetPropsAsDict().keys())

        if prop_to_write is None:
            prop_to_write = []
        
        elif prop_to_write == "all":
            prop_to_write = origin_prop
        
        elif prop_to_write == "nmr":
            prop_to_write = ["NMREDATA_ASSIGNMENT", "NMREDATA_J"]
        
        elif type(prop_to_write) != list:
            logger.error(f"The roperty to write: {prop_to_write}, is not supported. All original properties will be written to the SDF file.")
            prop_to_write = origin_prop
        
        prop_to_delete = list(set([prop for prop in origin_prop if prop not in prop_to_write]))
    
        # Get the SDF block of the RDKit molecule
        block = rdmol_to_sdf_block(rdmol, rdmol_Name, rdmol_MolFileInfo, rdmol_MolFileComments, tmp_sdf_file, prop_to_delete=prop_to_delete, SDFversion=SDFversion)

        # Write the SDF block to the SDF file
        if outfile is None or outfile.strip() == "":
            return block
        else:
            with open(outfile, "w") as f:
                f.write(block)

            

            

        

        












        # try:
        #     w = Chem.SDWriter(outfile)
        #     w.write(self.rdmol)
        #     w.close()
        # except Exception as e:
        #     logger.error(f"Fail to write the emol object to SDF file: {outfile}")
        #     raise e
