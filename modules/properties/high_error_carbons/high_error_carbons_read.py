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