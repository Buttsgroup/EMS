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

def predicted_candidate_read_df(mol_name, pred_atom_df, pred_pair_df):
    '''
    This function is used to read predicted candidate data from a pair dataframe. Returns atom_index_0, atom_index_1 and bond order as pair properties.


    Args:
    - mol_name (str): The molecule id of the original molecule in the testset.
    - pred_atom_df (pd.DataFrame): DataFrame containing atom-level predicted data for the candidate (contains information for one testset molecule only).
    - pred_pair_df (pd.DataFrame): DataFrame containing pair-level predicted data for the candidate (contains information for one testset molecule only).
    '''

    # Check whether the indexes in the molecule are continuous
    # If not, that means two molecules in the dataframe share the same molecule name
    atom_index = list(pred_atom_df.index)
    pair_index = list(pred_pair_df.index)

    atom_check = True
    pair_check = True

    for i in range(len(atom_index)-1):
        if atom_index[i+1] - atom_index[i] != 1:
            atom_check = False
            break
    
    for i in range(len(pair_index)-1):
        if pair_index[i+1] - pair_index[i] != 1:
            pair_check = False
            break
    
    if not (atom_check and pair_check):
        logger.error(f"The indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.")
        raise ValueError(f"The indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.")
    
    num_atoms = len(pred_atom_df)
    bond_order_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)

    i = np.array(pred_pair_df['atom_index_0'], dtype=np.int32)
    j = np.array(pred_pair_df['atom_index_1'], dtype=np.int32)
    bond_order = np.array(pred_pair_df['bond_order'], dtype=np.int32)

    bond_order_matrix[i, j] = bond_order

    # Return the bonding data of the predicted candidate
    return bond_order_matrix


