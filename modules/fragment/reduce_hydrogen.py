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


def hydrogen_reduction(rdmol: object) -> (dict[int, list[int]], dict[int, list[int]], list[int]):
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


def get_reduced_edge_index(edge_index: np.ndarray, reduced_index: np.ndarray) -> np.ndarray:
    '''
    Delete the edges that are connected to the deleted hydrogen atoms in a molecule.

    Parameters
    ----------
    edge_index : np.ndarray
        The edge index of the molecule. The shape of the matrix should be (n_edges, 2). 
        Each row of the matrix represents an edge (source atom index, destination atom index) in the molecule.

    reduced_index : list[int]
        The indexes of the hydrogen atoms that are deleted in the molecule.

    Returns
    -------
    reduced_edge_index: np.ndarray
        The reduced edge index of the molecule, with edges connected to the deleted hydrogen atoms removed. 
        The shape of the matrix should be (n_edges - n_removed_edges, 2).

    Examples
    --------
    >>> edge_index = np.array([[0, 1],
    ...                        [1, 0],
    ...                        [0, 2],
    ...                        [2, 0],
    ...                        [0, 3],
    ...                        [3, 0],
    ...                        [0, 4],
    ...                        [4, 0]])
    >>> reduced_index = [1, 3]
    >>> get_reduced_edge_index(edge_index, reduced_index)
    array([[0, 2],
           [2, 0],
           [0, 4],
           [4, 0]])
    '''

    edge_index = edge_index.copy()
    reduced_edge_index = edge_index[np.all(~np.isin(edge_index, reduced_index), axis = 1)]
    return reduced_edge_index



def get_reduced_adj_mat(mat: np.ndarray, reduced_index: list[int]) -> np.ndarray:
    '''
    Mask some rows and columns of the adjacency or connectivity matrix of a molecule as 0 based on the indexes of hydrogen atoms that are deleted.

    Parameters
    ----------
    mat : np.ndarray
        The original adjacency or connectivity matrix of a molecule. The shape of the matrix should be (n_atoms, n_atoms).
    
    reduced_index : list[int]
        The indexes of the hydrogen atoms that are deleted in the molecule.

    Returns
    -------
    reduced_mat: np.ndarray
        The reduced adjacency or connectivity matrix of the molecule, with rows and columns of deleted hydrogen atoms masked as 0. 
        The shape of the matrix should be (n_atoms, n_atoms).
    
    Examples
    --------
    >>> mat = np.array([[0, 1, 0, 0],
    ...                 [1, 0, 1, 0],
    ...                 [0, 1, 0, 1],
    ...                 [0, 0, 1, 0]])
    >>> reduced_index = [0, 2]
    >>> get_reduced_adj_mat(mat, reduced_index)
    array([[0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0]])
    '''

    reduced_mat = mat.copy()
    for row in reduced_index:
        reduced_mat[row, :] = 0
        reduced_mat[:, row] = 0
    return reduced_mat


def average_atom_prop(prop: np.ndarray, reduced_H_dict: dict[int, list[int]], length: int) -> np.ndarray:
    '''
    Average the properties of the hydrogen atoms that are equivalent to each other in a molecule, and set the properties of the deleted hydrogen atoms as 0.
    The property array will be extended to the maximum length. The properties of dumb atoms will be set as 0.

    Parameters
    ----------
    prop : np.ndarray
        The 1D property array of the atoms in the molecule. The shape of the matrix should be (n_atoms,).
    
    reduced_H_dict : dict[int, list[int]]
        A dictionary that stores the indexes of the hydrogen atoms kept in the molecule and its equivalent hydrogen atoms deleted.
        The key of the dictionary is the index of the hydrogen atom kept, and the value is a list of the indexes of its equivalent hydrogen atoms deleted.

    length : int
        The maximum number of atoms in the molecule.
    
    Returns
    -------
    extended_prop: np.ndarray
        The extended property array of the atoms in the molecule, with averaged properties of equivalent hydrogen atoms. 
        The properties of the deleted hydrogen atoms and dumb atoms are masked as 0. The shape of the matrix should be (length,).

    Examples
    --------
    >>> prop = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> reduced_H_dict = {2: [3, 4], 5: [6, 7]}
    >>> length = 10
    >>> average_atom_prop(prop, reduced_H_dict, length)
    array([1, 2, 4, 0, 0, 7, 0, 0, 0, 0])
    '''

    prop = prop.copy()
    reduced_H_dict = reduced_H_dict.copy()

    for key, value in reduced_H_dict.items():
        prop_to_average = [prop[key]] + [prop[i] for i in value]
        prop[key] = np.mean(prop_to_average)
        for i in value:
            prop[i] = 0
        
    extended_prop = np.pad(prop, (0, length - len(prop)), 'constant', constant_values = 0)
    return extended_prop


def reduce_atom_prop(prop: np.ndarray, reduced_H_list: list[int], length: int) -> np.ndarray:
    '''
    Set the properties of the deleted hydrogen atoms in a molecule as 0. 
    The property array will be extended to the maximum length. The properties of dumb atoms will be set as 0.

    Parameters
    ----------
    prop : np.ndarray
        The 1D property array of the atoms in the molecule. The shape of the matrix should be (n_atoms,).

    reduced_H_list : list[int]
        A list of the indexes of the hydrogen atoms deleted in the molecule.

    length : int
        The maximum number of atoms in the molecule.
    
    Returns
    -------
    extended_prop: np.ndarray
        The extended property array of the atoms in the molecule. The properties of the deleted hydrogen atoms and dumb atoms are masked as 0.
        The shape of the matrix should be (length,).
    
    Examples
    --------
    >>> prop = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> reduced_H_list = [3, 4, 6, 7]
    >>> length = 10
    >>> reduce_atom_prop(prop, reduced_H_list, length)
    array([1, 2, 3, 0, 0, 6, 0, 0, 0, 0])
    '''

    prop = prop.copy()
    reduced_H_list = reduced_H_list.copy()

    for i in reduced_H_list:
        prop[i] = 0
    
    extended_prop = np.pad(prop, (0, length - len(prop)), 'constant', constant_values = 0)
    return extended_prop


def flatten_pair_properties(mat: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    '''
    Extract the pair or bond properties, which are represented by a matrix with shape (n_atoms, n_atoms), based on the edge index and flatten it to a 1D array.

    Parameters
    ----------
    mat : np.ndarray
        The matrix of the pair or bond properties in the molecule. The shape of the matrix should be (n_atoms, n_atoms).
    
    edge_index : np.ndarray
        The edge index of the molecule. The shape of the edge index should be (n_edges, 2). 
        Each row of the matrix represents an edge (source atom index, destination atom index) in the molecule.

    Returns
    -------
    flattened_mat: np.ndarray
        The flattened array of the pair or bond properties in the molecule. The shape of the matrix is (n_edges,).

    Examples
    --------
    >>> mat = np.array([[0, 1, 0, 0],
    ...                 [1, 0, 2, 0],
    ...                 [0, 2, 0, 3],
    ...                 [0, 0, 3, 0]])
    >>> edge_index = np.array([[0, 1],
    ...                        [1, 0],
    ...                        [1, 2],
    ...                        [2, 1],
    ...                        [2, 3],
    ...                        [3, 2]])
    >>> flatten_pair_properties(mat, edge_index)
    array([1, 1, 2, 2, 3, 3])
    '''
    
    mat = mat.copy()
    edge_index = edge_index.copy()

    row_index = edge_index[:, 0]
    col_index = edge_index[:, 1]
    flattened_mat = mat[row_index, col_index]
    return flattened_mat

    