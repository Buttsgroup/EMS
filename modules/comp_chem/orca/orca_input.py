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


def default_orca_parameters():
    '''
    Return the default ORCA parameters as a dictionary.
    The default parameters are:
    - scf_convergence (str | None): The SCF convergence criteria, controlling the accuracy of each electronic structure step. 
        Default is None, related to the default 'TightSCF' in ORCA.
    - opt_level (str): The optimization criteria, controlling the accuracy of the geometry optimization. Default is 'OPT'.
    - freq (bool): Whether to perform frequency calculation after optimization. Default is False.
    - functional (str): The DFT functional. Default is 'mPW1PW'.
    - basis_set (str): The basis set. Default is '6-311g(d,p)'.
    - solvent (str | None): The solvent for the calculation. Default is None.
    - solvent_model (str | None): The solvent model for the calculation. Default is None.
        The choices can be 'CPCM', 'SMD', 'COSMORS' or None.
    - dispersion_correction (bool): Whether to include dispersion correction. Default is None.
        The choices can be 'D2', 'D4', 'D3ZERO', 'D3BJ' or None.
    - spin_thres (float | None): The distance threshold for considering spin-spin coupling in NMR calculation. Default is 5.0.
    - nmr_atoms (list): The list of atom types to calculate NMR chemical shifts. Default is ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'].
    '''
    
    return {
        'scf_convergence': None,
        'opt_level': 'OPT',
        'freq': False,
        'functional': 'mPW1PW',
        'basis_set': '6-311g(d,p)',
        'solvent': None,
        'solvent_model': None,
        'dispersion_correction': None,
        'spin_thres': 5.0,
        'nmr_atoms': ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'],
    }


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


def write_orca_auto_opt(prefs):
    '''
    Write the ORCA input file for conformer optimization automatically according to a few parameters.
    This function returns the root line and control block for the ORCA input file.
    '''

    # Write the root line
    root_line = f"! {prefs['functional']} {prefs['basis_set']}"
    
    if prefs['dispersion_correction'] is not None:
        root_line += f" {prefs['dispersion_correction']}"

    if prefs['scf_convergence'] is not None:
        root_line += f" {prefs['scf_convergence']}"
    
    root_line += f" {prefs['opt_level']}"

    if prefs['freq']:
        root_line += " FREQ"

    if prefs['solvent'] is not None and prefs['solvent_model'] is not None:
        root_line += f" {prefs['solvent_model']}({prefs['solvent']})"

    # Write the control block
    control_block = ''
    control_block += f"%geom\n"
    control_block += f" AddExtraBonds true\n"
    control_block += f" AddExtraBonds_MaxLength 10\n"
    control_block += f" AddExtraBonds_MaxDist 5\n"
    control_block += f"end\n"

    if prefs['solvent_model'].lower() == 'smd' and prefs['solvent'] is not None:
        control_block += f"\n"
        control_block += f"%cpcm\n"
        control_block += f" smd true\n"
        control_block += f" SMDsolvent \"{prefs['solvent']}\"\n"
        control_block += f"end\n"

    return root_line, control_block

    
def write_orca_auto_nmr(prefs):
    '''
    Write the ORCA input file for NMR calculation automatically according to a few parameters.
    This function returns the root line and control block for the ORCA input file.
    '''

    # Write the root line
    root_line = f"! {prefs['functional']} {prefs['basis_set']} NMR"

    if prefs['scf_convergence'] is not None:
        root_line += f" {prefs['scf_convergence']}"

    if prefs['solvent'] is not None and prefs['solvent_model'] is not None:
        root_line += f" {prefs['solvent_model']}({prefs['solvent']})"

    # Write the control block
    control_block = ''
    control_block += f"%eprnmr\n"

    if prefs['functional'].lower() == 'wb97xd':
        control_block += f" GIAO_2el = GIAO_2el_RIJCOSX\n"
    
    for atom in prefs['nmr_atoms']:
        control_block += f" Nuclei = all {atom} {{shift, ssall}}\n"
    
    if prefs['spin_thresh'] is not None:
        control_block += f" SpinSpinRThresh = {prefs['spin_thresh']}\n"
    
    control_block += f"end\n"
    
    if prefs['solvent_model'].lower() == 'smd' and prefs['solvent'] is not None:
        control_block += f"\n"
        control_block += f"%cpcm\n"
        control_block += f" smd true\n"
        control_block += f" SMDsolvent \"{prefs['solvent']}\"\n"
        control_block += f"end\n"
    
    return root_line, control_block
    

def write_orca_inp_block(geometry_block, rootline, control_block=None, charge=0, multiplicity=1):
    '''
    Write the ORCA input file for any type of calculation.
    The manual of ORCA 6.1 can be found at https://www.faccts.de/docs/orca/6.1/manual/index.html.
    
    Args:
    - geometry_block (str): The geometry of the molecule to be calculated by ORCA. The following file types are supported:
        xyz (atom type, x, y, z): a tuple with a list of atom types and a matrix of coordinates
        xyz file: a file path string to an xyz file
    - rootline (str): The input root line for the ORCA input file. 
    - control_block (str | None): The control block for the ORCA input file. If None, no control block is used.
        The control block can be either a file path or a string block.
        An example control block file is available in EMS/modules/comp_chem/orca/control_block_demo.inc
    - charge (int): The total charge of the molecule. Default is 0.
    - multiplicity (int): The spin multiplicity of the molecule. Default is 1.
    '''

    # Write the ORCA input file root line
    # Here are two examples of root lines:
    # rootline = 'B3LYP D4 def2-TZVP Opt TightSCF RIJCOSX def2/J'
    # rootline = 'BLYP def2-SVP Opt'
    rootline = '! ' + rootline.strip() + '\n\n'

    # Write the control block
    control_block = get_control_block(control_block)

    # Write the geometry block
    geometry_block = get_geometry_block(geometry_block, charge=charge, multiplicity=multiplicity)

    # Combine all blocks to form the ORCA input file
    orca_inp_block = ''.join([rootline, control_block, geometry_block])

    return orca_inp_block

    
        
    
    
    