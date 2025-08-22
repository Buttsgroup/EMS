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


def check_rdkit_conformer_embedding_params(params):
    '''
    This function checks and sets default parameters for RDKit conformer embedding methods.
    The parameters can be found in RDKit documentation: https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html
    The default values are consistent with RDKit's default.
    '''

    params.setdefault("rdkit_CE_method", "ETKDGv3")
    params.setdefault("rdkit_CE_clearConfs", True)
    params.setdefault("rdkit_CE_maxIterations", 0)
    params.setdefault("rdkit_CE_pruneRmsThresh", -1.0)
    params.setdefault("rdkit_CE_basinThresh", 5.0)
    params.setdefault("rdkit_CE_boundsMatForceScaling", 1.0)
    params.setdefault("rdkit_CE_boxSizeMult", 2.0)
    params.setdefault("rdkit_CE_forceTransAmides", True)
    params.setdefault("rdkit_CE_embedFragmentsSeparately", True)
    params.setdefault("rdkit_CE_enableSequentialRandomSeeds", False)
    params.setdefault("rdkit_CE_enforceChirality", True)
    params.setdefault("rdkit_CE_ignoreSmoothingFailures", False)
    params.setdefault("rdkit_CE_numThreads", 1)
    params.setdefault("rdkit_CE_numZeroFail", 1)
    params.setdefault("rdkit_CE_onlyHeavyAtomsForRMS", True)
    params.setdefault("rdkit_CE_optimizerForceTol", 0.001)
    params.setdefault("rdkit_CE_randNegEig", True)
    params.setdefault("rdkit_CE_randomSeed", -1)
    params.setdefault("rdkit_CE_symmetrizeConjugatedTerminalGroupsForPruning", True)
    params.setdefault("rdkit_CE_timeout", 0)
    params.setdefault("rdkit_CE_trackFailures", False)
    params.setdefault("rdkit_CE_useBasicKnowledge", True)
    params.setdefault("rdkit_CE_useExpTorsionAnglePrefs", True)
    params.setdefault("rdkit_CE_useMacrocycle14config", True)
    params.setdefault("rdkit_CE_useMacrocycleTorsions", True)
    params.setdefault("rdkit_CE_useRandomCoords", False)
    params.setdefault("rdkit_CE_useSmallRingTorsions", False)
    params.setdefault("rdkit_CE_useSymmetryForPruning", True)
    params.setdefault("rdkit_CE_verbose", False)


def rdkit_conformer_embedding(rdmol, params):
    '''
    This function embeds conformers for a given RDKit molecule object (rdmol) using RDKit.
    '''

    # Set default parameters for RDKit conformer embedding methods
    # Here are examples of some common parameters:
    # (1) rdkit_CE_clearConfs: Whether to clear existing conformers before generating new ones. Default: True
    # (2) rdkit_CE_maxIterations: Maximum number of embedding attempts to use for a single conformation. Default: 0
    # (3) rdkit_CE_pruneRmsThresh: The RMSD threshold for removing similar conformers when embedding multiple conformers. Default: -1.0
    check_rdkit_conformer_embedding_params(params)

    # Choose the RDKit conformer embedding method
    if params['rdkit_CE_method'] == 'ETDG':
        CE_params = AllChem.ETDG()
    elif params['rdkit_CE_method'] == 'ETKDG':
        CE_params = AllChem.ETKDG()
    elif params['rdkit_CE_method'] == 'ETKDGv2':
        CE_params = AllChem.ETKDGv2()
    elif params['rdkit_CE_method'] == 'ETKDGv3':
        CE_params = AllChem.ETKDGv3()
    else:
        logging.warning(f"Unknown conformer embedding method: {params['rdkit_CE_method']}. Change to ETKDGv3.")
        CE_params = AllChem.ETKDGv3()

    # Set the conformer embedding parameters
    for key, value in params.items():
        # Get the the name of the parameters to set
        if key != 'rdkit_CE_method' and 'rdkit_CE_' in key:
            key = key.split('rdkit_CE_')[-1]
            
            if hasattr(CE_params, key):
                setattr(CE_params, key, value)

    # Embed the conformers for the RDKit molecule
    AllChem.EmbedMultipleConfs(rdmol, numConfs=params["num_conformers"], params=CE_params)