import numpy as np
import sys
import logging

########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


# Carbon and proton defaults obtained through validation on exp data in the group
# Nitrogen defaults are from Gao, Peng, Xingyong Wang, and Haibo Yu. "Towards an Accurate Prediction of Nitrogen Chemical Shifts by Density Functional Theory and Gauge‐Including Atomic Orbital." Advanced Theory and Simulations 2.2 (2019): 1800148.
# Fluorine & Phosphorus defaults are from Gao, Zhang, and Chen. "A systematic benchmarking of 31P and 19F NMR chemical shift predictions using different DFT/GIAO methods and applying linear regression to improve the prediction accuracy" International Journal of Quantum Chemistry 121.5 (2021): e26482.
# Not for the exact same functional but close enough to be useful.
# Recalculated scaling factors from original  1: [-1.0719, 32.1254], 6: [-1.0399, 187.136]
# default scaling values are for functional: wb97xd, basis_set: 6-311g(d,p)
scaling_dict = {1: [-1.0594, 32.2293], 6: [-1.0207, 187.4436], 7: [-1.0139, -148.67], 9: [-1.0940, 173.02], 15: [-1.2777, 307.74]}


def scale_chemical_shifts(raw_shift, atom_types, scaling=scaling_dict):
    '''
    This function scales the raw chemical shifts achieved from the Gaussian calculation according to the scaling factors.

    Args:
    - raw_shift (np.ndarray): The raw chemical shifts obtained from the Gaussian calculation.
    - atom_types (list): The list of atom types corresponding to the chemical shifts.
    - scaling (dict): A dictionary containing the scaling factors for each atom type. 
        Each key is the atom type, and the value is a list containing the scaling factor and the offset.
    '''
	
    # Raise an error if the length of the chemical shift and atom type arrays do not match
    if len(raw_shift) != len(atom_types):
        logger.error("Length of chemical shift and atom type arrays do not match.")
        raise ValueError("Length of chemical shift and atom type arrays do not match.")
	
    # Initialize the array to save the scaled chemical shifts
    scaled_shift = np.zeros_like(raw_shift)
    
    # Iterate over the atom types and apply the scaling factors
    for i, atom_type in enumerate(atom_types):
        if atom_type not in scaling.keys():
            continue

        scaled_shift[i] = (raw_shift[i] - scaling[atom_type][1]) / scaling[atom_type][0]

    return scaled_shift