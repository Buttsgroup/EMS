import numpy as np
from collections import defaultdict


def matrix_to_edge_index(mat: np.ndarray) -> np.ndarray:
    '''
    Generate the edge index of a molecule from its adjacency matrix.

    Parameters
    ----------
    mat : np.ndarray
        The adjacency matrix of a molecule. The shape of the matrix should be (n_atoms, n_atoms).
    
    Returns
    -------
    np.ndarray
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

    return np.transpose(np.nonzero(mat))

def get_hydrogen_indexes(rdmol):
    hydrogen_indexes = defaultdict(list)
    for atom in rdmol.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                hydrogen_indexes[atom.GetIdx()].append(neighbor.GetIdx())

    return hydrogen_indexes

def get_reduced_H_dict(H_dict):
    '''
    



    '''


    reduce_dict = {}
    for key in H_dict.keys():
        if len(H_dict[key]) > 1:
            kept_H = H_dict[key][0]
            reduced_H = H_dict[key][1:]
            reduce_dict[kept_H] = reduced_H
    return reduce_dict

def get_reduced_H_list(H_dict):
    reduced_H_list = []
    for key, value in H_dict.items():
        for item in value:
            reduced_H_list.append(item)
    return reduced_H_list

def binary_matrix_to_list(mat):
    pass

def get_reduced_adj_mat(mat, reduced_index):
    mat = mat.copy()
    for row in reduced_index:
        mat[row, :] = 0
        mat[:, row] = 0
    return mat