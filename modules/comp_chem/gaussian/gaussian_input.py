import sys
import logging

from EMS.utils.periodic_table import Get_periodic_table

########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def check_prefs(prefs=None):
    '''
    Set the prefs dictionary for gaussian calculations to default values. If the some keys are not initialized, the default values are used.

    The default prefs dictionary includes the following keys and values:
    - charge (int): Total net charge of the molecule. Default is 0.
    - multiplicity (int): Spin multiplicity. Default is 1.
    - calc_type (str): Gaussian calculation type. Default is 'opt'.
    - memory (int): Memory to request (in GB). Default is 12.
    - processor (int): Number of CPU cores to request. Default is 8.
    - nodes (int): Number of nodes to request. Default is 1.
    - walltime (str): Maximum wall time for the job. Default is '120:00:00'.
    - opt (str): Optimization level. Default is 'tight'.
    - freq (bool): Whether to perform frequency calculations. Default is True.
    - functional (str): DFT functional. Default is 'mPW1PW'.
    - basis_set (str): Basis set for representing molecular orbitals. Default is '6-311g(d,p)'.
    - mixed (bool): Whether to calculate both shielding tensors and J couplings in one job. Default is True.
    - solvent (str | None): Solvent for implicit solvent calculations. Default is None.
    - solvent_model (str | None): Solvent model. Default is None.
    - grid (str): Integration grid for DFT calculation. Default is 'ultrafine'.
    - custom_cmd_line (bool): Whether to use a manually provided Gaussian command line instead of auto-generated keywords. Default is False.
    '''

    if prefs is None:
        prefs = {}
    
    prefs.setdefault('charge', 0)
    prefs.setdefault('multiplicity', 1)
    prefs.setdefault('calc_type', 'opt')
    prefs.setdefault('memory', 12)
    prefs.setdefault('processor', 8)
    prefs.setdefault('nodes', 1)
    prefs.setdefault('walltime', '120:00:00')
    prefs.setdefault('opt', 'tight')
    prefs.setdefault('freq', True)
    prefs.setdefault('functional', 'mPW1PW')
    prefs.setdefault('basis_set', '6-311g(d,p)')
    prefs.setdefault('mixed', True)
    prefs.setdefault('solvent', None)
    prefs.setdefault('solvent_model', None)
    prefs.setdefault('grid', 'ultrafine')
    prefs.setdefault('custom_cmd_line', False)

    return prefs


def make_gaussian_rootline(prefs=None):
    """
    Function for the automation of the root line required for setting up the gaussian calculation

    Args:
    - prefs (dict): The dictionary containing key parameters for the gaussian calculation
    """

    # Initialize the prefs dictionary
    prefs = check_prefs(prefs)

    # Write the root line if the calculation type is structure optimization
    if prefs['calc_type'] in ('optimisation', 'opt'):
        if prefs['freq']:
            root_line = f"Opt={str(prefs['opt'])} freq {str(prefs['functional'])}/{str(prefs['basis_set'])} integral={str(prefs['grid'])} MaxDisk=50GB"

        else:
            root_line = f"Opt={str(prefs['opt'])} {str(prefs['functional'])}/{str(prefs['basis_set'])} integral={str(prefs['grid'])} MaxDisk=50GB"

        if prefs['solvent'] is not None:
            root_line += f" scrf=( {str(prefs['solvent_model'])} , solvent={str(prefs['solvent'])} )"

        return root_line

    # Write the root line if the calculation type is NMR parameter calculation
    elif prefs['calc_type'] in ('nmr', 'NMR'):
        if prefs['mixed'] == True:
            root_line = f"nmr(giao,spinspin,mixed) {str(prefs['functional'])}/{str(prefs['basis_set'])} maxdisk=50GB"

        else:
            root_line = f"nmr(giao,spinspin) {str(prefs['functional'])}/{str(prefs['basis_set'])} maxdisk=50GB"

        if prefs['solvent'] is not None:
            root_line += f" scrf=( {str(prefs['solvent_model'])} , solvent={str(prefs['solvent'])} )"

        return root_line
    
    # Raise an error for unsupported calculation types
    else:
        logger.error(f"Calculation type {prefs['calc_type']} is not supported. Currently only 'optimisation' and 'nmr' are supported.")
        raise ValueError(f"Calculation type {prefs['calc_type']} is not supported. Currently only 'optimisation' and 'nmr' are supported.")


def write_gaussian_com_block(EMSmol, prefs=None):
    """
    The function generates Gaussian input .com files based off prefs dictionary values.
   
    Args:
    - EMSmol: The EMS molecule object containing atomic coordinates and other information
    - prefs: A dictionary of preferences for the Gaussian calculation
    """

    # Get the periodic table
    periodic_table = Get_periodic_table()

    # Initialize the prefs dictionary
    prefs = check_prefs(prefs)

    # Get the molecule name
    molname = EMSmol.filename
    if molname is None or molname.strip() == "":
        molname = ""
    
    # Get the root line
    root_line = make_gaussian_rootline(prefs)

    # Check the types of the parameters in the prefs dictionary
    if not isinstance(prefs['memory'], int):
        logger.error(f"Invalid type for 'memory' in prefs: {prefs['memory']} is {type(prefs['memory'])}, should be an integer value")
        raise ValueError(f"Invalid type for 'memory' in prefs: {prefs['memory']} is {type(prefs['memory'])}, should be an integer value")

    if not isinstance(prefs['processor'], int):
        logger.error(f"Invalid type for 'processor' in prefs: {prefs['processor']} is {type(prefs['processor'])}, should be an integer value")
        raise ValueError(f"Invalid type for 'processor' in prefs: {prefs['processor']} is {type(prefs['processor'])}, should be an integer value")


    # Write the Gaussian input .com file
    strings = []

    # Add the checkpoint file information
    if prefs['calc_type'] in ('optimisation', 'opt'):
        strings.append(f"%Chk={molname}_OPT.chk")

    elif prefs['calc_type'] in ('nmr', 'NMR'):
        strings.append(f"%Chk={molname}_NMR.chk")

    else:
        logger.error(f"Calculation type {prefs['calc_type']} is not supported. Currently only 'optimisation' and 'nmr' are supported.")
        raise ValueError(f"Calculation type {prefs['calc_type']} is not supported. Currently only 'optimisation' and 'nmr' are supported.")

    # Add lines for requesting computing resources
    strings.append("%NoSave")
    strings.append(f"%mem={prefs['memory']}GB")
    strings.append(f"%NProcShared={prefs['processor']}")

    # Add the root line
    if prefs['calc_type'] in ('optimisation', 'opt'):
        strings.append(f"# {root_line}")

    elif prefs['calc_type'] in ('nmr', 'NMR'):
        strings.append(f"#T {root_line}")
    
    strings.append("")

    # Add the title line
    if prefs['calc_type'] in ('optimisation', 'opt'):
        strings.append(f"{molname} OPT")

    elif prefs['calc_type'] in ('NMR', 'nmr'):
        strings.append(f"{molname} NMR")

    strings.append("")

    # Add the line for charge and multiplicity
    strings.append(f"{prefs['charge']} {prefs['multiplicity']}")

    # Add the atomic coordinates
    for i in range(len(EMSmol.xyz)):
        atom_type = periodic_table[EMSmol.type[i]]
        string = f" {atom_type:<2.2s}{EMSmol.xyz[i][0]:>18.6f}{EMSmol.xyz[i][1]:>18.6f}{EMSmol.xyz[i][2]:>18.6f}"
        strings.append(string)

    for i in range(4):
        strings.append("")

    strings = "\n".join(strings)
    return strings