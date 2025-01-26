import pandas as pd
import numpy as np
import copy
from io import StringIO
import warnings
import os

from EMS.modules.properties.structure_io import from_rdmol, to_rdmol, SDFfile_to_rdmol
from EMS.utils.periodic_table import Get_periodic_table
from EMS.modules.fragment.reduce_hydrogen import *
from EMS.modules.properties.nmr.nmr_io import nmr_read, nmr_read_rdmol

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdmolfiles

import openbabel.pybel as pyb


class EMS(object):
    """
    The EMS class is the main class for the EMS package. It reads the molecular structure and properties from a file and transforms it to an emol object.
    The file type includes but not limited to SDF, XYZ, SMILES/SMARTS strings, and rdkit molecule objects.

    Args:
        file (str): The file path of the molecule file. If the molecule file is a SMILES/SMARTS string, the file is the string.
            If the molecule is an rdkit molecule, the file is the rdkit molecule object.
        mol_id (str): The customarized molecule id. The file name is preferred.
        line_notation (str): If the molecule file is a SMILES/SMARTS string, the line_notation is 'smi' or 'smarts'. Default: None.
        rdkit_mol (bool): Whether to read the file as an rdkit molecule. Default: False.
        nmr (bool): Whether to read NMR data. Default: False.
        streamlit (bool): Whether to read the file from ForwardSDMolSupplier. Default: False.
        fragment (bool): Whether to read molecular fragments. Default: False.
        max_atoms (int): The maximum number of atoms in a molecule. If the molecule has less atoms than this number, the extra atoms are dumb atoms.
            This max_atoms argument is used with fragment. Default: 1e6.
    
    More details about the EMS class can be found in the comments below.
    """

    def __init__(
        self,
        file,                      # File path
        mol_id=None,               # Customarized molecule id
        line_notation=None,        # 'smi' or 'smarts'
        rdkit_mol=False,           # Whether to read the file as an rdkit molecule
        nmr=False,                 # Whether to read NMR data
        streamlit=False,           # Streamlit mode is used to read the file from ForwardSDMolSupplier
        fragment=False,            # Whether to read molecular fragments
        max_atoms=1e6,             # Maximum number of atoms in a molecule, if the molecule has less atoms than this number, the extra atoms are dumb atoms
    ):
        
        # To initialize self.id, self.filename, self.file and self.stringfile
        # (1) If the molecule file is SMILES or SMARTS string, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the string
        # (2) If the molecule file is streamlit, self.id is the customarized 'mol_id' name, and self.filename, self.file and self.stringfile are achieved from the 'file' object
        # (3) If the molecule file is neither SMILES/SMARTS string or streamlit, like .sdf file, self.id is the customarized 'mol_id' name, 
        #     self.file and self.stringfile are the file path, and self.filename is the file name, simplified from the file path
        # (4) If the molecule is an rdkit molecule, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the customarized 'mol_id' name
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
        self.max_atoms = max_atoms         
        self.streamlit = streamlit
        self.fragment = fragment

        # get the rdmol object from the file
        # If the molecule file is a SMILES/SMARTS string, the rdmol object is generated from the string
        if line_notation:
            if line_notation == "smi":
                try:
                    line_mol = Chem.MolFromSmiles(file)
                except Exception as e:
                    print(f"Wrong SMILES string: {file}")
                    raise e

            elif line_notation == "smarts":
                try:
                    line_mol = Chem.MolFromSmarts(file)
                except Exception as e:
                    print(f"Wrong SMARTS string: {file}")
                    raise e

            else:
                raise ValueError(f"Line notation, {line_notation}, not supported")
            
            Chem.SanitizeMol(line_mol)
            line_mol = Chem.AddHs(line_mol)
            Chem.Kekulize(line_mol)
            AllChem.EmbedMolecule(line_mol)              # obtain the initial 3D structure for a molecule
            AllChem.UFFOptimizeMolecule(line_mol)        # optimize the 3D structure of a molecule
            self.rdmol = line_mol
        
        # If the molecule file is an rdkit molecule, the rdmol object is the molecule itself
        elif rdkit_mol:
            self.rdmol = file
 
        # If the molecule is saved in a file, the rdmol object is generated from the file
        else:
            ftype = self.filename.split(".")[-1]

            if ftype == "sdf":
                if streamlit:
                    self.rdmol = SDFfile_to_rdmol(self.file, self.filename, streamlit=True)
                else:
                    self.rdmol = SDFfile_to_rdmol(self.file, self.filename, streamlit=False)

            elif ftype == "xyz":
                tmp_file = f'_tmp_{self.filename}.sdf'
                obmol = next(pyb.readfile('xyz', self.file))
                obmol.write('sdf', tmp_file, overwrite=True)
                self.rdmol = SDFfile_to_rdmol(tmp_file, self.filename, streamlit=False)
                os.remove(tmp_file)

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
                raise ValueError(f"File type, {ftype} not supported")

        # Check if self.rdmol is read correctly from the file as an rdkit molecule object
        if not isinstance(self.rdmol, Chem.rdchem.Mol):
            raise TypeError(f"File {self.file} not read correctly")

        # Get the molecular structures
        self.type, self.xyz, self.conn = from_rdmol(self.rdmol)
        self.adj = Chem.GetAdjacencyMatrix(self.rdmol) 
        self.path_topology, self.path_distance = self.get_graph_distance()
        self.mol_properties["SMILES"] = Chem.MolToSmiles(self.rdmol)
        self.flat = self.check_Zcoords_zero() 

        # Check if every atom in the molecule has a correct valence
        # If any atom in the molecule has a wrong implicit valence, self.pass_valence_check will be set to False
        # If there is any error when calling the self.check_valence() function, self.pass_valence_check will be set to 'Error'
        self.pass_valence_check = self.check_valence()
        
        # Check if the non-hydrogen backbone of the molecule is symmetric
        # If there is any error when calling the self.check_symmetric() function, self.symmetric will be set to 'Error'
        # The error may be caused by wrong explicit valences which are greater than permitted
        self.symmetric = self.check_symmetric()

        # Check if the number of atoms in the molecule is greater than the maximum number of atoms allowed
        if self.max_atoms < len(self.type):
            print(f"Number of atoms in molecule {self.id} is greater than the maximum number of atoms allowed")
        
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


        # Enter the fragment mode of EMS to generate molecular fragments
        if self.fragment:
            self.edge_index = matrix_to_edge_index(self.adj)

            # self.H_index_dict: a dictionary, key is the non-hydrogen atom index, value is a list of hydrogen atom indexes linked to this non-hydrogen atom
            # self.reduced_H_dict: a dictionary, key is the H atom index, value is a list of its equivalent H atom indexes to be deleted
            # self.reduced_H_list: a list of all H atoms to be reduced
            self.H_index_dict, self.reduced_H_dict, self.reduced_H_list = hydrogen_reduction(self.rdmol) 

            self.eff_atom_list = list(set(range(len(self.type))) - set(self.reduced_H_list))         # a list of effective atoms that excludes the reduced H atoms
            self.dumb_atom_list = list(set(range(self.max_atoms)) - set(self.eff_atom_list))         # a list of dumb atoms including both reduced H atoms and extra atoms

            self.reduced_edge_index = get_reduced_edge_index(self.edge_index, self.reduced_H_list)         # edge index that excludes the bonds with reduced H atoms
            self.reduced_adj = get_reduced_adj_mat(self.adj, self.reduced_H_list)         # adjacency matrix that excludes the bonds with reduced H atoms
            self.reduced_conn = get_reduced_adj_mat(self.conn, self.reduced_H_list)       # connectivity matrix that excludes the bonds with reduced H atoms

            if nmr:
                try:
                    if self.filename.split(".")[-2] == "nmredata" and nmr:
                        shift, shift_var, coupling, coupling_vars = nmr_read(self.stringfile, self.streamlit)
                        self.atom_properties["shift"] = shift
                        self.atom_properties["shift_var"] = shift_var
                        self.pair_properties["coupling"] = coupling
                        self.pair_properties["coupling_var"] = coupling_vars
                        assert len(self.atom_properties["shift"]) == len(self.type)
                except:
                    print(
                        f"Read NMR called but no NMR data found for molecule {self.id}"
                    )
            
            self.atom_properties['shift'] = average_atom_prop(self.atom_properties["shift"], self.reduced_H_dict, self.max_atoms)
            self.atom_properties['shift_var'] = average_atom_prop(self.atom_properties["shift_var"], self.reduced_H_dict, self.max_atoms)
            self.atom_properties['atom_type'] = reduce_atom_prop(self.type, self.reduced_H_list, self.max_atoms)

            self.pair_properties['bond_order'] = flatten_pair_properties(self.reduced_conn, self.reduced_edge_index)


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
            print(f"Function Chem.MolFromSmiles raises error when reading SMILES string of molecule {self.id}")
            return False
        
        # Check if the molecule can be read by RDKit from its SMILES string
        if smiles_mol is None:
            print(f"Molecule {self.id} cannot be read by RDKit from its SMILES string")
            return False
        
        # Check if the molecule is a single molecule not a mixture of molecules by checking if there is '.' in the SMILES string
        if '.' in self.mol_properties["SMILES"]:
            print(f"Molecule {self.id} is not a single molecule, but a mixture of molecules")
            return False
        
        # Check if every atom in the molecule has a correct implicit valence
        for atom in self.rdmol.GetAtoms():
            # If the RDKit molecule is not sanitized, the GetImplicitValence function will raise an error
            try:
                atom.GetImplicitValence()
            except Exception as e:
                print(f"Molecule {self.id} cannot read its implicit valence, which may caused by not being sanitized!")
                return False

            # List the atoms with wrong implicit valence
            if atom.GetImplicitValence() != 0:
                check = False
                print(f"Molecule {self.id}: Atom {atom.GetSymbol()}, index {atom.GetIdx()}, has wrong implicit valence")
            
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
            print(f'Symmetry check failed for molecule {self.id}, due to wrong explicit valences which are greater than permitted')
            print('This error needs to be fixed...')
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
    
    def check_atom_type_number(self, atom_type_number_threshold={}):
        """
        Check if the number of atoms of some atom types in the molecule is less than a threshold.

        Example:
            atom_type_number_threshold = {6: 3, 7: 2}
            This means that the molecule should have <= 3 carbon atoms and <= 2 nitrogen atoms.
        """

        check = True
        atom_types = self.type.tolist()

        for atom_type in atom_type_number_threshold:
            threshold = atom_type_number_threshold[atom_type]
            if atom_types.count(atom_type) > threshold:
                check = False
                break

        return check

    def get_graph_distance(self):
        """
        Get the path length matrix and 3D distance matrix.
        The path length matrix is the shortest path length between atoms. Shape: (n_atoms, n_atoms)
        The 3D distance matrix is the 3D distance between atoms. Shape: (n_atoms, n_atoms)
        """

        return Chem.GetDistanceMatrix(self.rdmol).astype(int), Chem.Get3DDistanceMatrix(self.rdmol)

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
