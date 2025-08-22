# Get bond lengths for different bond types
# This file is directly copies from mol_translator. 
# The sources of these values need to be verified.

def get_bond_lengths():
    bond_lengths = {
        'CH':[0.90,1.20],
        'CC':[[1.30,1.70],[1.20,1.40],[1.10,1.30],[1.10,1.30]],
        'CN':[[1.30, 1.60],[1.15,1.35], [1.00,1.30], [1.20,1.40]],
        'CO':[[1.30, 1.60],[1.10,1.30]],
        'CCl':[1.60, 1.90],
        'CSi':[2.30,2.60],
        'CP':[1.70, 2.00],
        'CS':[1.70, 2.00],
        'CBr':[1.80, 2.00],
        'NO':[1.10, 1.60],
        'SS':[1.80,2.20],
        'SiSi':[2.30,2.60],
        'NN':[[1.15, 1.35],[1.25, 1.45]],
        }

    return bond_lengths