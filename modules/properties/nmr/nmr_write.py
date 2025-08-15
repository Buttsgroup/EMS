import sys
import logging

from EMS.modules.comp_chem.gaussian.gaussian_read import gaussian_read_nmr


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def nmr_to_sdf_block(atom_types, atom_properties, pair_properties):
    '''
    This function reads the NMR data saved in EMS molecule's atom and pair properties and converts them to the <NMREDATA_ASSIGNMENT> and <NMREDATA_J> sections in SDF block.

    Args:
    - atom_types (list): List of atom types of one EMS molecule
    - atom_properties (dict): Dictionary of atom properties, including chemical shifts and their variances
    - pair_properties (dict): Dictionary of pair properties, including coupling constants, their variances, and types
    '''

    # Create the SDF block for chemical shift data
    atom_lines = []

    if 'shift' not in atom_properties:
        logger.warning('Chemical shift property not found in atom properties when writing to SDF block')

    else:
        for i, (typ, shift, var) in enumerate(zip(atom_types, atom_properties['shift'], atom_properties['shift_var'])):
            line = f"{i:<5d}, {shift:<15.8f}, {typ:<5d}, {var:<15.8f}\\"
            atom_lines.append(line)
    
    atom_block = '\n'.join(atom_lines)
    

    # Create the SDF block for coupling constant data
    pair_lines = []

    if 'coupling' not in pair_properties:
        logger.warning('Coupling constant property not found in pair properties when writing to SDF block')

    else:
        num_atoms = len(atom_types)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):  # avoid duplicate and self-pairs
                coupling = pair_properties['coupling'][i][j]
                if coupling == 0:
                    continue
                label = pair_properties['nmr_types'][i][j]
                var = pair_properties['coupling_var'][i][j]
                line = f"{i:<10d}, {j:<10d}, {coupling:<15.8f}, {label:<10s}, {var:<15.8f}"
                pair_lines.append(line)

    pair_block = '\n'.join(pair_lines)
    
    # Return the atom and pair lines in the SDF block
    return atom_block, pair_block