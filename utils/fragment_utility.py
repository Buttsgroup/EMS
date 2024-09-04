import numpy as np

def get_reduced_H(H_dict):
    reduce_dict = {}
    for key in H_dict.keys():
        if len(H_dict[key]) > 1:
            reduce_dict[H_dict[key][0]] = H_dict[key]
    return reduce_dict

def get_reduced_H_match():
    pass

def binary_matrix_to_list(mat):
    pass

def binary_matrix_to_index(mat):
    return np.transpose(np.nonzero(mat))