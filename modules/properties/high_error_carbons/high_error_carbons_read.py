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


def high_error_carbons_read_list(C_index_list, shift_C_idx_list, shift_error_C_idx_list):

    num_carbons = len(C_index_list)
    high_error_carbons = np.array(C_index_list, dtype=np.int32)
    shift_C_indices = np.array(shift_C_idx_list, dtype=np.float64)
    shift_error_C_indices = np.array(shift_error_C_idx_list, dtype=np.float64)

    if not (high_error_carbons.ndim == 1):
        logger.error(f"High error carbon array should be one-dimensional!")
        raise ValueError(f"High error carbon array should be one-dimensional!")
    
    if not (shift_C_indices.ndim == 1):
        logger.error(f"Shift array for high error carbons should be one-dimensional!")
        raise ValueError(f"Shift array for high error carbons should be one-dimensional!")
    
    if not (shift_error_C_indices.ndim == 1):
        logger.error(f"Shift error array for high error carbons should be one-dimensional!")
        raise ValueError(f"Shift error array for high error carbons should be one-dimensional!")

    return high_error_carbons, shift_C_indices, shift_error_C_indices

def high_error_carbons_read_rdmol(rdmol, mol_id):
    '''
    This function is used to read high error carbon data from an RDKit molecule object.

    Args:
    - rdmol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
    - mol_id (str): Molecule ID.
    '''

    # Get all the properties of the RDKit molecule object
    prop_dict = rdmol.GetPropsAsDict()

    # Get the high error carbon data (HIGH_ERROR_CARBONS) from the atom properties
    try:
        high_error_carbons = prop_dict['HIGH_ERROR_CARBONS']
    except Exception as e:
        logger.error(f'No high error carbon data found for molecule {mol_id}')
        raise ValueError(f'No high error carbon data found for molecule {mol_id}')
    
    # Split the high error carbon data block into lines and then into items
    high_error_carbons_items = []
    for line in high_error_carbons.split('\n'):
        if line:
            high_error_carbons_items.append(line.split())    

    # Initialize arrays for saving high error carbon data
    num_atom = len(high_error_carbons_items)
    high_error_carbons_array = np.zeros(num_atom, dtype=np.int32)
    shift_C_indices = np.zeros(num_atom, dtype=np.float64)
    shift_error_C_indices = np.zeros(num_atom, dtype=np.float64)

    # Read the high error carbon data from the lines
    # high error carbon block row looks like this
    #  1    , 66.75623322    , 14.11889649
    for i, item in enumerate(high_error_carbons_items):
        high_error_carbons_array[i] = int(item[0])
        shift_C_indices[i] = float(item[2])
        shift_error_C_indices[i] = float(item[4])
    
    return high_error_carbons_array, shift_C_indices, shift_error_C_indices