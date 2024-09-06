import numpy as np
from collections import defaultdict

from rdkit import Chem


def matrix_to_edge_index(mat: np.ndarray) -> np.ndarray:
    '''
    Generate the edge index of a molecule from its adjacency matrix.

    Parameters
    ----------
    mat : np.ndarray
        The adjacency matrix of a molecule. The shape of the matrix should be (n_atoms, n_atoms).
    
    Returns
    -------
    edge_index: np.ndarray
        The edge index of the molecule. The shape of the matrix should be (n_edges, 2). 
        Each row of the matrix represents an edge (source atom index, destination atom index) in the molecule.

    Examples
    --------
    >>> mat = np.array([[0, 1, 0, 0],
    ...                 [1, 0, 1, 0],
    ...                 [0, 1, 0, 1],
    ...                 [0, 0, 1, 0]])
    >>> matrix_to_edge_index(mat)
    array([[0, 1],
           [1, 0],
           [1, 2],
           [2, 1],
           [2, 3],
           [3, 2]])
    '''
    
    edge_index = np.transpose(np.nonzero(mat))
    return edge_index


def hydrogen_reduction(rdmol: object):
    '''
    Delete the redundant hydrogen atoms which are equivalent to each other in a molecule. 
    For example, in a methyl group, the three hydrogen atoms are equivalent to each other.
    This function will delete two of them and keep only one. Additionally, it will return the indexes of the hydrogen atoms that are deleted.

    Parameters
    ----------
    rdmol : object
        The RDKit molecule object.
    
    Returns
    -------
    hydrogen_indexes: dict[int, list[int]]
        A dictionary that stores the indexes of the hydrogen atoms that are connected to a non-hydrogen atom in the molecule.
        The key of the dictionary is the index of the non-hydrogen atom, and the value is a list of the indexes of the hydrogen atoms.
    
    reduced_H_dict: dict[int, list[int]]
        A dictionary that stores the indexes of the hydrogen atoms kept in the molecule and its equivalent hydrogen atoms deleted.
        The key of the dictionary is the index of the hydrogen atom kept, and the value is a list of the indexes of its equivalent hydrogen atoms deleted.

    reduced_H_list: list[int]
        A list of the indexes of all the hydrogen atoms deleted in the molecule.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC')
    >>> hydrogen_indexes, reduced_H_dict, reduced_H_list = hydrogen_reduction(mol)
    >>> hydrogen_indexes
    defaultdict(<class 'list'>, {0: [2, 3, 4], 1: [5, 6, 7]})
    >>> reduced_H_dict
    {2: [3, 4], 5: [6, 7]}
    >>> reduced_H_list
    [3, 4, 6, 7]
    '''

    hydrogen_indexes = defaultdict(list)
    for atom in rdmol.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                hydrogen_indexes[atom.GetIdx()].append(neighbor.GetIdx())
    
    reduced_H_dict = {}
    for key in hydrogen_indexes.keys():
        if len(hydrogen_indexes[key]) > 1:
            kept_H = hydrogen_indexes[key][0]
            reduced_H = hydrogen_indexes[key][1:]
            reduced_H_dict[kept_H] = reduced_H
        
    reduced_H_list = []
    for key, value in reduced_H_dict.items():
        for item in value:
            reduced_H_list.append(item)

    return hydrogen_indexes, reduced_H_dict, reduced_H_list




# def get_hydrogen_indexes(rdmol: Chem.rdchem.Mol):
#     hydrogen_indexes = defaultdict(list)
#     for atom in rdmol.GetAtoms():
#         for neighbor in atom.GetNeighbors():
#             if neighbor.GetAtomicNum() == 1:
#                 hydrogen_indexes[atom.GetIdx()].append(neighbor.GetIdx())

    
    

# def get_reduced_H_dict(H_dict):
#     '''
    



#     '''


#     reduce_dict = {}
#     for key in H_dict.keys():
#         if len(H_dict[key]) > 1:
#             kept_H = H_dict[key][0]
#             reduced_H = H_dict[key][1:]
#             reduce_dict[kept_H] = reduced_H
#     return reduce_dict

# def get_reduced_H_list(H_dict):
#     reduced_H_list = []
#     for key, value in H_dict.items():
#         for item in value:
#             reduced_H_list.append(item)
#     return reduced_H_list

def binary_matrix_to_list(mat):
    pass

def get_reduced_adj_mat(mat, reduced_index):
    mat = mat.copy()
    for row in reduced_index:
        mat[row, :] = 0
        mat[:, row] = 0
    return mat