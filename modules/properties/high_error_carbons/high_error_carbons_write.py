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


def high_error_carbons_to_sdf_block(atom_properties):
    '''
    This function reads the lists of atom indexes and chemical shifts for high error carbons saved in EMS molecule's atom properties and converts them to the <HIGH_ERROR_CARBONS> section in SDF block.

    Args:
    - atom_properties (dict): Dictionary of atom properties, including atom indexes and chemical shifts of high error carbons
    '''

    high_error_carbons_lines = []

    if 'high_error_carbons' not in atom_properties:
        logger.warning('High error carbons not found in atom properties when writing to SDF block')

    else:
        for (atom_index, shift, shift_error) in (zip(atom_properties['high_error_carbons'], atom_properties['shift_high_error_carbons'], atom_properties['shift_error_high_error_carbons'])):
            line = f"{atom_index:<5d}, {shift:<15.8f}, {shift_error:<15.8f}"
            high_error_carbons_lines.append(line)
        
    high_error_carbons_block = '\n'.join(high_error_carbons_lines)

    return high_error_carbons_block





