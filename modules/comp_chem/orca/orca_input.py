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


def get_control_block(block=None):
    ''' 
    Write the control block for the ORCA input file.
    The input block can be a file path or a string block.
    '''

    # Write the control block. The control block can be a file path or a string block.
    if block is None:
        block = ''
    
    # Check if the control block is a file path or string block
    if block.split('.')[-1].isalnum() and len(block.split('.')[-1]) <= 10:
        with open(block, 'r') as f:
            control_block = f.read()
        control_block = control_block.strip()

    else:
        control_block = block.strip()
    
    return control_block + '\n\n' if control_block != '' else ''


def get_geometry_block(geometry, charge=0, multiplicity=1):
    '''
    Write the geometry block for the ORCA input file according to the type of geometry input.
    '''

    # When the geometry is given as a tuple of (atom_types, coordinates), the geometry block type is 'xyz'
    if isinstance(geometry, tuple) and len(geometry) == 2:
        atom_types, coordinates = geometry

        geometry_block = f'* xyz {charge} {multiplicity}\n'
        for atom, coord in zip(atom_types, coordinates):
            geometry_block += f' {atom}  {coord[0]:.4f}  {coord[1]:.4f}  {coord[2]:.4f}\n'
        geometry_block += '*\n'
    
    # When the geometry is given as a .xyz file path, the geometry block type is 'xyzfile'
    elif isinstance(geometry, str) and geometry.split('.')[-1] == 'xyz':
        geometry_block = f'* xyzfile {charge} {multiplicity} {geometry}\n'
        
    # Return the geometry block
    return geometry_block


def write_orca_inp_block(geometry_block, rootline=None, control_block=None, charge=0, multiplicity=1):
    '''
    Write the ORCA input file for any type of calculation.
    The manual of ORCA 6.1 can be found at https://www.faccts.de/docs/orca/6.1/manual/index.html.
    
    Args:
    - geometry_block (str): The geometry of the molecule to be calculated by ORCA. The following file types are supported:
        xyz (atom type, x, y, z): a tuple with a list of atom types and a matrix of coordinates
        xyz file: a file path string to an xyz file
    - rootline (str | None): The input root line for the ORCA input file. If None, a default root line for structure optimization is used.
    - control_block (str | None): The control block for the ORCA input file. If None, no control block is used.
        The control block can be either a file path or a string block.
        An example control block file is available in EMS/modules/comp_chem/orca/control_block_demo.inc
    - charge (int): The total charge of the molecule. Default is 0.
    - multiplicity (int): The spin multiplicity of the molecule. Default is 1.
    '''

    # Write the ORCA input file root line
    if rootline is None:
        # rootline = 'B3LYP D4 def2-TZVP Opt TightSCF RIJCOSX def2/J'
        rootline = 'BLYP def2-SVP Opt'
    
    rootline = '! ' + rootline.strip() + '\n\n'

    # Write the control block
    control_block = get_control_block(control_block)

    # Write the geometry block
    geometry_block = get_geometry_block(geometry_block, charge=charge, multiplicity=multiplicity)

    # Combine all blocks to form the ORCA input file
    orca_inp_block = ''.join([rootline, control_block, geometry_block])

    return orca_inp_block

    
        
    
    
    