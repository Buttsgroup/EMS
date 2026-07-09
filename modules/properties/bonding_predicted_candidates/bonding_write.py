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

def bonding_pred_candidate_to_sdf_block(atom_types, pair_properties):
    '''
    This function reads bonding information of predicted candidate saved in EMS molecule's pair properties and converts this to the <BONDING_PREDICTED_CANDIDATE> section in SDF block.
    
    Args:
    - atom_types (list): List of atom types of one EMS molecule
    - pair_properties (dict): Dictionary of pair properties, including bond order matrix of predicted candidate
    '''

    bonding_lines = []

    if 'bond_order_predicted_candidate' not in pair_properties:
        logger.warning('Bond order property not found in pair properties when writing to SDF block')

    else:
        num_atoms = len(atom_types)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):  # avoid duplicate and self-pairs
                bonding = pair_properties['bond_order_predicted_candidate'][i][j]
                if bonding == 0:
                    continue
                line = f"{i:<10d}, {j:<10d}, {bonding:<10d}"
                bonding_lines.append(line)

    bonding_block = '\n'.join(bonding_lines)

    return bonding_block