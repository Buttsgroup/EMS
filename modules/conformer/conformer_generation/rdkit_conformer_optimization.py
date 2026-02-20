from rdkit import Chem
from rdkit.Chem import AllChem

import sys
import logging


########### Set up the logger system ###########
# This section is used to set up the logging system, which aims to record the information of the package

# getLogger is to initialize the logging system with the name of the package
# A package can have multiple loggers.
logger = logging.getLogger(__name__)

# StreamHandler is a type of handler to print logging output to a specific stream, such as a console.
stdout = logging.StreamHandler(stream = sys.stdout)

# Formatter is used to specify the output format of the logging messages
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")

# Add the formatter to the handler
stdout.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stdout)

# Set the logging level of the logger. The logging level below INFO will not be recorded.
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def check_rdkit_conformer_optimization_params(params):
    '''
    This function checks and sets default parameters for RDKit conformer optimization methods, like MMFF and UFF.
    The parameters can be found in RDKit documentation: https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html

    Parameters:
    - rdkit_CO_method (str): The force field method to optimize conformers. "MMFF94", "MMFF94s" and "UFF" are supported. Default: "MMFF94"
    - rdkit_CO_maxIters (int): Maximum number of optimization iterations. Default: 200
    - rdkit_CO_numThreads (int): Number of threads to use for optimization. Default: 1
    - rdkit_CO_ignoreInterfragInteractions (bool): Whether to ignore non-bonded terms between fragments in the forcefield. Default: True
    - rdkit_CO_nonBondedThresh (float): Threshold to exclude non-bonded interactions. Default: 100.0 (Specific to the MMFF)
    - rdkit_CO_vdwThresh (float): Threshold to exclude Van der Waals interactions. Default: 10.0 (Specific to the UFF)
    '''

    params.setdefault("rdkit_CO_method", "MMFF94")
    params.setdefault("rdkit_CO_maxIters", 200)
    params.setdefault("rdkit_CO_numThreads", 1)
    params.setdefault("rdkit_CO_ignoreInterfragInteractions", True)
    params.setdefault("rdkit_CO_nonBondedThresh", 100.0)
    params.setdefault("rdkit_CO_vdwThresh", 10.0)


def rdkit_conformer_optimization_helper(force_field, rdmol, params):
    '''
    Helper function to perform RDKit conformer optimization by a specific force field method.
    '''

    # Get the conformer IDs
    ids = [conf.GetId() for conf in rdmol.GetConformers()]

    # Carry out structure optimization on each conformer 
    if force_field == "MMFF94":
        results = AllChem.MMFFOptimizeMoleculeConfs(rdmol, 
                                                    numThreads=params["rdkit_CO_numThreads"],
                                                    maxIters=params["rdkit_CO_maxIters"], 
                                                    mmffVariant="MMFF94",
                                                    nonBondedThresh=params["rdkit_CO_nonBondedThresh"], 
                                                    ignoreInterfragInteractions=params["rdkit_CO_ignoreInterfragInteractions"])
    elif force_field == "MMFF94s":
        results = AllChem.MMFFOptimizeMoleculeConfs(rdmol, 
                                                    numThreads=params["rdkit_CO_numThreads"],
                                                    maxIters=params["rdkit_CO_maxIters"], 
                                                    mmffVariant="MMFF94s",
                                                    nonBondedThresh=params["rdkit_CO_nonBondedThresh"], 
                                                    ignoreInterfragInteractions=params["rdkit_CO_ignoreInterfragInteractions"])
    elif force_field == "UFF":
        results = AllChem.UFFOptimizeMoleculeConfs(rdmol, 
                                                   numThreads=params["rdkit_CO_numThreads"],
                                                   maxIters=params["rdkit_CO_maxIters"], 
                                                   vdwThresh=params["rdkit_CO_vdwThresh"],
                                                   ignoreInterfragInteractions=params["rdkit_CO_ignoreInterfragInteractions"])
        
    # Check if the number of conformers equal to the optimized structures
    if len(results) != len(ids):
        logger.error("The number of conformers does not match the number of optimized structures.")
        raise ValueError("The number of conformers does not match the number of optimized structures.")
    
    return ids, results


def rdkit_conformer_optimization(rdmol, params):
    '''
    Optimize the conformers of the given RDKit molecule using RDKit and return their energies (kcal/mol, kcal=kJ*4.184)
    '''

    # Initialize parameters for conformer optimization by RDKit
    check_rdkit_conformer_optimization_params(params)

    # Check whether the molecule is suitable for MMFF. If not, use UFF.
    do_mmff = AllChem.MMFFHasAllMoleculeParams(rdmol)

    # Optimize the conformers
    if not do_mmff:
        logger.warning("Molecule is not suitable for MMFF. Switching to UFF.")
        conf_ids, results = rdkit_conformer_optimization_helper("UFF", rdmol, params)

    else:
        if params["rdkit_CO_method"] in ["MMFF94", "MMFF94s", "UFF"]:
            conf_ids, results = rdkit_conformer_optimization_helper(params["rdkit_CO_method"], rdmol, params)
        else:
            logger.warning(f"Unsupported force field method: {params['rdkit_CO_method']}. Using MMFF94 as default.")
            conf_ids, results = rdkit_conformer_optimization_helper("MMFF94", rdmol, params)

    # Return a dictionary with the key as the conformer id and the value as the optimized energy
    # The conformers that fail to optimize will be deleted
    energy_dict = {}
    for conf_id, (converged_flag, energy) in zip(conf_ids, results):
        if converged_flag == 0:
            energy_dict[conf_id] = energy

    return energy_dict