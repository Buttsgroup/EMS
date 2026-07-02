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
    This function reads the list of high error carbons saved in EMS molecule's atom properties and converts them to the <HIGH_ERROR_CARBONS> section in SDF block.

    Args:
    - atom_properties (dict): Dictionary of atom properties, including atom indexes of high error carbons
    '''
    high_error_carbons_lines = ", ".join(map(str, atom_properties["high_error_carbons"]))

    return high_error_carbons_lines





