import numpy as np

def get_reduced_H_dict(H_dict):
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

def binary_matrix_to_index(mat):
    return np.transpose(np.nonzero(mat))

def get_reduced_adj_mat(mat, reduced_index):
    mat = mat.copy()
    for row in reduced_index:
        mat[row, :] = 0
        mat[:, row] = 0
    return mat