import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import random

import glob
import re
from io import StringIO
import warnings

from modules.properties.structure_io import from_rdmol, to_rdmol
from utils.periodic_table import Get_periodic_table
from modules.fragment.reduce_hydrogen import *

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdmolfiles



class EMS(object):

    def __init__(
        self,
        file,              # file path
        mol_id=None,
        line_notation=None,           # 'smi' or 'smarts'
        read_nmr=True,
        streamlit=False,
        fragment=False,
        max_atoms=50,          # maximum number of atoms in a molecule, if the molecule has less atoms than this number, the extra atoms are dumb atoms
    ):

        # if the molecule file is SMILES or SMARTS string, all of self.id, self.filename, self.file and self.stringfile are the same, i.e. the string
        # if the molecule file is streamlit, self.id is the customarized 'mol_id' name, and self.filename, self.file and self.stringfile are achieved from the 'file' object
        # if the molecule file is neither SMILES/SMARTS string or streamlit, self.id is the customarized 'mol_id' name, and self.filename, self.file and self.stringfile are the same, i.e. the file path
        if line_notation:
            self.id = file
        else:
            self.id = mol_id
        if streamlit and not line_notation:
            self.filename = file.name
        else:
            self.filename = file

        self.file = file

        if streamlit and not line_notation:
            self.stringfile = StringIO(file.getvalue().decode("utf-8"))
        else:
            self.stringfile = file

        # initialize the molecular structure and properties as empty
        self.type = None
        self.xyz = None
        self.conn = None                   # self.conn is the bond order matrix of the molecule
        self.adj = None                    # self.adj is the adjacency matrix of the molecule
        self.path_topology = None          # Path length between atoms
        self.path_distance = None          # 3D distance between atoms
        self.bond_existence = None         # whether a bond exists between two atoms, made up of 0 or 1
        self.atom_properties = {}
        self.pair_properties = {}
        self.mol_properties = {}
        self.flat = False
        self.max_atoms = max_atoms
        self.streamlit = streamlit
        self.fragment = fragment

        # get the rdmol object from the file
        if line_notation:
            if line_notation == "smi":
                self.rdmol = Chem.MolFromSmiles(file)
                self.rdmol = Chem.AddHs(self.rdmol)
                AllChem.EmbedMolecule(self.rdmol)              # obtain initial coordinates for a molecule
            elif line_notation == "smarts":
                self.rdmol = Chem.MolFromSmarts(file)
            else:
                raise ValueError(f"Line notation, {line_notation}, not supported")

        else:
            ftype = self.filename.split(".")[-1]

            if ftype == "sdf":
                if not self.check_z_ords():
                    self.flat = True
                    warnings.warn(
                        f"Warning: {self.id} - All Z coordinates are 0 - Flat flag set to True"
                    )
                if streamlit:
                    for mol in Chem.ForwardSDMolSupplier(
                        self.file, removeHs=False, sanitize=False
                    ):
                        if mol is not None:
                            if mol.GetProp("_Name") is None:
                                mol.SetProp("_Name", self.id)
                            self.rdmol = mol

                else:
                    for mol in Chem.SDMolSupplier(
                        self.file, removeHs=False, sanitize=False
                    ):
                        if mol is not None:
                            if mol.GetProp("_Name") is None:
                                mol.SetProp("_Name", self.id)
                            self.rdmol = mol
            elif ftype == "xyz":
                self.rdmol = Chem.MolFromXYZFile(self.path)

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
                
            elif ftype == "ase":
                    self.rdmol = ase_to_rdmol(self.file)
                
            else:
                raise ValueError(f"File type, {ftype} not supported")

        # check if every atom in the molecule has a correct valence
        self.pass_valence_check = self.check_valence()

        if self.pass_valence_check:
            # get the molecular structure and properties
            self.type, self.xyz, self.conn = from_rdmol(self.rdmol)         # self.conn is the bond order matrix of the molecule
            self.adj = Chem.GetAdjacencyMatrix(self.rdmol)                  # self.adj is the adjacency matrix of the molecule
            self.path_topology, self.path_distance = self.get_graph_distance()
            self.mol_properties["SMILES"] = Chem.MolToSmiles(self.rdmol)
            self.symmetric = self.check_symmetric()                         # check if the non-hydrogen backbone of the molecule is symmetric

            # raise an error if the number of atoms in the molecule is greater than the maximum number of atoms allowed
            if self.max_atoms < len(self.type):
                # raise ValueError(f"Number of atoms in molecule {self.id} is greater than the maximum number of atoms allowed")
                print(f"Number of atoms in molecule {self.id} is greater than the maximum number of atoms allowed")

        # enter the fragment mode of EMS, to generate molecular fragments
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

            if read_nmr:
                try:
                    if self.filename.split(".")[-2] == "nmredata" and read_nmr:
                        shift, shift_var, coupling, coupling_vars = self.nmr_read()
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


        # enter the normal mode of EMS
        elif self.pass_valence_check:
            self.get_coupling_types()

            if read_nmr:
                try:
                    if self.filename.split(".")[-2] == "nmredata" and read_nmr:
                        shift, shift_var, coupling, coupling_vars = self.nmr_read()
                        self.atom_properties["shift"] = shift
                        self.atom_properties["shift_var"] = shift_var
                        self.pair_properties["coupling"] = coupling
                        self.pair_properties["coupling_var"] = coupling_vars
                        assert len(self.atom_properties["shift"]) == len(self.type)
                except:
                    print(
                        f"Read NMR called but no NMR data found for molecule {self.id}"
                    )

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
        check = True
        for atom in self.rdmol.GetAtoms():
            # if Chem.rdchem.HasValenceViolation(atom) == True:
            if atom.HasValenceViolation() == True:
                check = False
                break
        return check

    def check_z_ords(self):
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
                lines = f.readlines()[3:]
                for line in lines:
                    # if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
                    if len(line.split()) == 12 and line.split()[-1] != 'V2000':
                        z_coord = float(
                            line.split()[3]
                        )  # Assuming the z coordinate is the fourth field
                        if z_coord != 0:
                            return False
                return True

    def check_symmetric(self):
        mol = copy.deepcopy(self.rdmol)
        Chem.RemoveStereochemistry(mol)
        mol = Chem.rdmolops.RemoveAllHs(mol)
        canonical_ranking = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
        
        if len(canonical_ranking) == len(set(canonical_ranking)):
            return False
        else:
            return True
        
    def check_semi_symmetric(self, atom_type_threshold={}):
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
        check = True
        atom_types = self.type.tolist()

        for atom_type in atom_type_number_threshold:
            threshold = atom_type_number_threshold[atom_type]
            if atom_types.count(atom_type) > threshold:
                check = False
                break

        return check

    def nmr_read(self):
        if self.streamlit:
            atoms = 0
            for line in self.stringfile:
                if len(line.split()) == 16:
                    atoms += 1
                if "V2000" in line or len(line.split()) == 12:
                    chkatoms = int(line.split()[0])

            # check for stupid size labelling issue
            if atoms != chkatoms:
                for i in range(1, len(str(chkatoms))):
                    if atoms == int(str(chkatoms)[:-i]):
                        chkatoms = atoms
                        break

            assert atoms == chkatoms
            # Define empty arrays
            shift_array = np.zeros(atoms, dtype=np.float64)
            # Variance is used for machine learning
            shift_var = np.zeros(atoms, dtype=np.float64)
            coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
            coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
            # Variance is used for machine learning
            coupling_var = np.zeros((atoms, atoms), dtype=np.float64)

            # Go through file looking for assignment sections
            shift_switch = False
            cpl_switch = False
            for line in self.stringfile:
                if "<NMREDATA_ASSIGNMENT>" in line:
                    shift_switch = True
                if "<NMREDATA_J>" in line:
                    shift_switch = False
                    cpl_switch = True
                # If shift assignment label found, process shift rows
                if shift_switch:
                    # Shift assignment row looks like this
                    #  0    , -33.56610000   , 8    , 0.00000000     \
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    shift_array[int(items[0])] = float(items[2])
                    shift_var[int(items[0])] = float(items[6])
                # If coupling assignment label found, process coupling rows
                if cpl_switch:
                    # Coupling row looks like this
                    #  0         , 4         , -0.08615310    , 3JON      , 0.00000000
                    # ['0', ',', '1', ',', '-0.26456900', ',', '5JON', ',', '0.00000000']
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    length = int(items[6].strip()[0])
                    coupling_array[int(items[0])][int(items[2])] = float(items[4])
                    coupling_array[int(items[2])][int(items[0])] = float(items[4])
                    coupling_var[int(items[0])][int(items[2])] = float(items[8])
                    coupling_var[int(items[2])][int(items[0])] = float(items[8])
                    coupling_len[int(items[0])][int(items[2])] = length
                    coupling_len[int(items[2])][int(items[0])] = length

            return shift_array, shift_var, coupling_array, coupling_var

        else:
            atoms = 0
            with open(self.stringfile, "r") as f:
                for line in f:
                    if len(line.split()) == 16:
                        atoms += 1
                    if "V2000" in line or len(line.split()) == 12:
                        chkatoms = int(line.split()[0])

            # check for stupid size labelling issue
            if atoms != chkatoms:
                for i in range(1, len(str(chkatoms))):
                    if atoms == int(str(chkatoms)[:-i]):
                        chkatoms = atoms
                        break

            assert atoms == chkatoms
            # Define empty arrays
            shift_array = np.zeros(atoms, dtype=np.float64)
            # Variance is used for machine learning
            shift_var = np.zeros(atoms, dtype=np.float64)
            coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
            coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
            # Variance is used for machine learning
            coupling_var = np.zeros((atoms, atoms), dtype=np.float64)

            # Go through file looking for assignment sections
            with open(self.stringfile, "r") as f:
                shift_switch = False
                cpl_switch = False
                for line in f:
                    if "<NMREDATA_ASSIGNMENT>" in line:
                        shift_switch = True
                    if "<NMREDATA_J>" in line:
                        shift_switch = False
                        cpl_switch = True
                    # If shift assignment label found, process shift rows
                    if shift_switch:
                        # Shift assignment row looks like this
                        #  0    , -33.56610000   , 8    , 0.00000000     \
                        items = line.split()
                        try:
                            int(items[0])
                        except:
                            continue
                        shift_array[int(items[0])] = float(items[2])
                        shift_var[int(items[0])] = float(items[6])
                    # If coupling assignment label found, process coupling rows
                    if cpl_switch:
                        # Coupling row looks like this
                        #  0         , 4         , -0.08615310    , 3JON      , 0.00000000
                        # ['0', ',', '1', ',', '-0.26456900', ',', '5JON', ',', '0.00000000']
                        items = line.split()
                        try:
                            int(items[0])
                        except:
                            continue
                        length = int(items[6].strip()[0])
                        coupling_array[int(items[0])][int(items[2])] = float(items[4])
                        coupling_array[int(items[2])][int(items[0])] = float(items[4])
                        coupling_var[int(items[0])][int(items[2])] = float(items[8])
                        coupling_var[int(items[2])][int(items[0])] = float(items[8])
                        coupling_len[int(items[0])][int(items[2])] = length
                        coupling_len[int(items[2])][int(items[0])] = length

            return shift_array, shift_var, coupling_array, coupling_var

    def get_graph_distance(self):
        return Chem.GetDistanceMatrix(self.rdmol), Chem.Get3DDistanceMatrix(self.rdmol)

    def get_coupling_types(self) -> None:
        """
        Function for generating all the coupling types for all atom-pair interactions, stores internally in pair_properties attributes.

        :param aemol: Type, aemol class to generate coupling types for
        :return: None
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

    def convert_to_rdmol(self):
        self.rdmol = to_rdmol(self)


def make_atoms_df(ems_list, atom_list='all', write=False, format="pickle"):
    p_table = Get_periodic_table()

    # construct dataframes
    # atoms has: molecule_name, atom, labeled atom,
    molecule_name = []  # molecule name
    atom_index = []  # atom index
    typestr = []  # atom type (string)
    typeint = []  # atom type (integer)
    x = []  # x coordinate
    y = []  # y coordinate
    z = []  # z coordinate
    conns = []
    atom_props = []
    smiles = []
    for propname in ems_list[0].atom_properties.keys():
        atom_props.append([])

    pbar = tqdm(ems_list, desc="Constructing atom dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1
        # Add atom values to lists
        for t, type in enumerate(ems.type):
            molecule_name.append(ems.id)
            atom_index.append(t)
            typestr.append(p_table[type])
            typeint.append(type)
            x.append(ems.xyz[t][0])
            y.append(ems.xyz[t][1])
            z.append(ems.xyz[t][2])
            conns.append(ems.conn[t])
            smiles.append(ems.mol_properties["SMILES"])
            for p, prop in enumerate(ems.atom_properties.keys()):
                if prop == 'shift' and atom_list == 'all':
                    atom_props[p].append(ems.atom_properties[prop][t])
                elif prop == 'shift' and atom_list != 'all':
                    if p_table[type] in atom_list:
                        atom_props[p].append(ems.atom_properties[prop][t])
                    else:
                        atom_props[p].append(0.0)
                else:
                    atom_props[p].append(ems.atom_properties[prop][t])

    # Construct dataframe
    atoms = {
        "molecule_name": molecule_name,
        "atom_index": atom_index,
        "typestr": typestr,
        "typeint": typeint,
        "x": x,
        "y": y,
        "z": z,
        "conn": conns,
        "SMILES": smiles,
    }
    for p, propname in enumerate(ems.atom_properties.keys()):
        atoms[propname] = atom_props[p]

    atoms = pd.DataFrame(atoms)

    pbar.close()

    atoms.astype(
        {
            "molecule_name": "category",
            "atom_index": "Int16",
            "typestr": "category",
            "typeint": "Int8",
            "x": "Float32",
            "y": "Float32",
            "z": "Float32",
            "SMILES": "category",
        }
    )

    if write:
        if format == "csv":
            atoms.to_csv(f"{write}/atoms.csv")
        elif format == "pickle":
            atoms.to_pickle(f"{write}/atoms.pkl")
        elif format == "parquet":
            atoms.to_parquet(f"{write}/atoms.parquet")

    else:
        return atoms


def make_pairs_df(ems_list, coupling_list='all', peak_frequency = {}, write=False, max_pathlen=6):
    # construct dataframe for pairs in molecule
    # only atom pairs with bonds < max_pathlen are included

    molecule_name = []  # molecule name
    atom_index_0 = []  # atom index for atom 1
    atom_index_1 = []  # atom index for atom 2
    dist = []  # distance between atoms
    path_len = []  # number of pairs between atoms (shortest path)
    pair_props = []
    bond_existence = []
    for propname in ems_list[0].pair_properties.keys():
        pair_props.append([])

    pbar = tqdm(ems_list, desc="Constructing pairs dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1

        for t, type in enumerate(ems.type):
            for t2, type2 in enumerate(ems.type):
                # Add pair values to lists
                if ems.path_topology[t][t2] > max_pathlen:
                    continue
                molecule_name.append(ems.id)
                atom_index_0.append(t)
                atom_index_1.append(t2)
                dist.append(ems.path_distance[t][t2])
                path_len.append(int(ems.path_topology[t][t2]))
                bond_existence.append(ems.adj[t][t2])
                for p, prop in enumerate(ems.pair_properties.keys()):
                    if prop == 'coupling' and coupling_list == 'all':
                        pair_props[p].append(ems.pair_properties[prop][t][t2])
                    elif prop == 'coupling' and coupling_list != 'all':
                        if ems.pair_properties['nmr_types'][t][t2] in coupling_list:
                            if ems.pair_properties['nmr_types'][t][t2] in peak_frequency:
                                freq = peak_frequency[ems.pair_properties['nmr_types'][t][t2]]
                                if random.random() < freq:
                                    pair_props[p].append(ems.pair_properties[prop][t][t2])
                                else:
                                    pair_props[p].append(0.0)
                            else:
                                pair_props[p].append(ems.pair_properties[prop][t][t2])
                        else:
                            pair_props[p].append(0.0)
                    else:
                        pair_props[p].append(ems.pair_properties[prop][t][t2])

    # Construct dataframe
    pairs = {
        "molecule_name": molecule_name,
        "atom_index_0": atom_index_0,
        "atom_index_1": atom_index_1,
        "distance": dist,
        "path_len": path_len,
        "bond_existence": bond_existence,
    }
    for p, propname in enumerate(ems.pair_properties.keys()):
        pairs[propname] = pair_props[p]

    pairs = pd.DataFrame(pairs)

    pbar.close()

    if write:
        pairs.to_pickle(f"{write}/pairs.pkl")
    else:
        return pairs
