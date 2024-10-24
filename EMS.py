import pandas as pd
import numpy as np
from tqdm import tqdm

import glob
import re
from io import StringIO
import warnings

from EMS.modules.properties.structure_io import from_rdmol, to_rdmol
from EMS.utils.periodic_table import Get_periodic_table
from EMS.modules.fragment.reduce_hydrogen import * # hydrogen_reduction, average_atom_prop, reduce_atom_prop, flatten_pair_properties AVOID IMPORT * IT'S LAZY
from EMS.modules.properties.nmr.nmr_io import nmr_read

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
        nmr=False,
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

        # get the molecular structure and properties
        self.type, self.xyz, self.conn = from_rdmol(self.rdmol)         # self.conn is the bond order matrix of the molecule
        self.adj = Chem.GetAdjacencyMatrix(self.rdmol)                  # self.adj is the adjacency matrix of the molecule
        self.path_topology, self.path_distance = self.get_graph_distance()
        self.mol_properties["SMILES"] = Chem.MolToSmiles(self.rdmol)
        self.symmetric = self.check_symmetric()                         # check if the non-hydrogen backbone of the molecule is symmetric

        # raise an error if the number of atoms in the molecule is greater than the maximum number of atoms allowed
        if self.max_atoms < len(self.type):
            raise ValueError(f"Number of atoms in molecule {self.id} is greater than the maximum number of atoms allowed")

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

        if nmr:
            try:
                assert self.filename.split(".")[-2] == "nmredata"
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
    
        if self.fragment:    
            self.atom_properties['shift'] = average_atom_prop(self.atom_properties["shift"], self.reduced_H_dict, self.max_atoms)
            self.atom_properties['shift_var'] = average_atom_prop(self.atom_properties["shift_var"], self.reduced_H_dict, self.max_atoms)
            self.atom_properties['atom_type'] = reduce_atom_prop(self.type, self.reduced_H_list, self.max_atoms)

            self.pair_properties['bond_order'] = flatten_pair_properties(self.reduced_conn, self.reduced_edge_index)
        # enter the normal mode of EMS
        self.get_coupling_types()

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

    def check_z_ords(self):
        if self.streamlit:
            for line in self.stringfile:
                if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
                    z_coord = float(
                        line.split()[3]
                    )  # Assuming the z coordinate is the fourth field
                    if z_coord != 0:
                        return False
            return True

        else:
            with open(self.stringfile, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if re.match(r"^.{10}[^ ]+ [^ ]+ ([^ ]+) ", line):
                        z_coord = float(
                            line.split()[3]
                        )  # Assuming the z coordinate is the fourth field
                        if z_coord != 0:
                            return False
                return True

    def check_symmetric(self):
        mol = Chem.rdmolops.RemoveAllHs(self.rdmol)
        canonical_ranking = list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
        
        if len(canonical_ranking) == len(set(canonical_ranking)):
            return False

        return True

    def parse_nmr(self):
        shift_array, shift_var, coupling_array, coupling_var = nmr_read(self.stringfile, self.streamlit)
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