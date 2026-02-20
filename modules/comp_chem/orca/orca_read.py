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


def orca_read_energy(file):
    '''
    Read the final SCF energy from an ORCA output file.
    The energy unit is Hartree in ORCA and the conversion factor to kcal/mol is 627.509474.

    Args:
    - file (str): Path to the ORCA output file.
    '''

    # Get the file block
    with open(file, 'r') as f:
        orca_output_block = f.read()

    # Get the final energy line by searching for the line containing 'Energy             :'
    energy_symbol = 'Energy             :'
    if energy_symbol in orca_output_block:
        energy_line = [line for line in orca_output_block.split('\n') if energy_symbol in line][-1]
        energy = float(energy_line.split()[-2])
        
    # If the line is not found, search for the last line containing 'Energy', ':' and 'Eh' and having less than 4 elements
    else:
        energy_lines = [line for line in orca_output_block.split('\n') if 'Energy' in line and ':' in line and 'Eh' in line and len(line.split()) <= 4]
        
        # If no energy line is found, return None
        if len(energy_lines) == 0:
            logger.error(f"Final energy not found in ORCA output file {file}. Please check the output file for details.")
            return None
        
        energy_line = energy_lines[-1]
        energy = float(energy_line.split()[-2])
    
    return energy * 627.509474  # Convert Hartree to kcal/mol


def orca_read_geometry(file):
    '''
    Read the final optimized geometry from an ORCA output file.

    Args:
    - file (str): Path to the ORCA output file.
    '''

    # Get the file lines
    with open(file, 'r') as f:
        orca_xyz_lines = f.readlines()

    # Get the number of atoms from the first line
    atom_num = int(orca_xyz_lines[0].strip())

    # Get the atom types and coordinates from the lines
    coord_lines = orca_xyz_lines[2: 2+atom_num]
    if len(coord_lines) != atom_num:
        logger.error(f"Number of atoms in ORCA output file {file} does not match the number of coordinate lines. Please check the output file.")
        return None
    
    # Iterate through the coordinate lines to get the atom coordinates
    atom_coords = []
    for line in coord_lines:
        parts = line.strip().split()
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        atom_coords.append((x, y, z))
    
    return atom_coords
        




    