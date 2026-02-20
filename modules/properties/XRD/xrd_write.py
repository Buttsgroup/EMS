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

def xray_to_sdf_block(mol_properties):
    """
    This function reads the 'Xray' data saved in EMS molecules mol_properties and converts them to the <XRD_PATTERN> in the SDF block
    """

# Create the SDF block for molecule data
    mol_lines = []

    if 'intensity' not in mol_properties:
        logger.warning('Xray property not found in molecule properties when writing to SDF block')
    else:
        for i, (molecule_name, Q, intensity) in enumerate(zip(mol_properties['molecule_name'], mol_properties["Q"], mol_properties["intensity"])):
            line = f"{i:<20d}, {molecule_name:<5d}, {Q:<15.8f},{intensity:<15.8f}\\"
            mol_lines.append(line)
    
    mol_block = '\n'.join(mol_lines)

    # Return the mol_block lines in the SDF block
    return mol_block