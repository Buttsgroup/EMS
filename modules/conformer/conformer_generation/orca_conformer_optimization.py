import os
import sys
import logging
import contextlib
import subprocess
import shutil

from rdkit.Geometry import Point3D
from rdkit import Chem

from EMS.modules.comp_chem.orca.orca_input import write_orca_inp_block
from EMS.modules.comp_chem.orca.orca_read import orca_read_energy, orca_read_geometry


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


def check_orca_conformer_optimization_params(params):
    '''
    This function checks and sets default parameters for conformer optimization by ORCA.
    The parameters can be found in ORCA 6.1 documentation: https://www.faccts.de/docs/orca/6.1/manual/index.html.

    Parameters:
    - orca_CO_rootline (str): The root line for the ORCA input file. Default is 'BLYP def2-SVP Opt'.
    - orca_CO_controlblock (str | None): The control block for the ORCA input file. If None, no control block will be used.
        The control block can be either a file path or a string block.
        An example control block file is available in EMS/modules/comp_chem/orca/control_block_demo.inc
    '''

    params.setdefault('orca_CO_rootline', 'BLYP def2-SVP Opt')
    params.setdefault('orca_CO_controlblock', None)
    

def orca_conformer_optimization(rdmol, params):
    '''
    Optimize the conformers of the given RDKit molecule using ORCA and return their energies (kcal/mol, kcal=kJ*4.184)
    '''
    
    # Set default parameters for ORCA conformer optimization
    check_orca_conformer_optimization_params(params)

    # Get the molecule's name
    mol_name = rdmol.GetProp("_Name")
    if mol_name is None or mol_name.strip() == "":
        logger.error("The molecule being processed does not have a name. Please set the name by EMS before optimization.")
        raise ValueError("The molecule being processed does not have a name. Please set the name by EMS before optimization.")
    mol_name = mol_name.strip().split('.')[0]

    # Get the charge of the molecule
    charge = Chem.GetFormalCharge(rdmol)
    if charge != 0:
        logger.warning(f"The molecule {mol_name} has a non-zero charge ({charge}). The charge will be used in the ORCA input file.")
    
    # Get the multiplicity of the molecule
    num_radical_e = sum(atom.GetNumRadicalElectrons() for atom in rdmol.GetAtoms())
    multiplicity = num_radical_e + 1
    if multiplicity != 1:
        logger.warning(f"The molecule {mol_name} has a multiplicity of {multiplicity} instead of 1. The multiplicity will be used in the ORCA input file.")

    # Get the atom types
    atom_types = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

    # Get conformers of the RDKit molecule
    conformers = [conf for conf in rdmol.GetConformers()]

    # Initialize the energy dictionary
    energy_dict = {}

    # Iterate over the conformers to optimize
    for conformer in conformers:
        # Get the conformer ID
        conf_id = conformer.GetId()

        # Get the job name
        job_name = f"{mol_name}_conf_{conf_id}"

        # Get the 3D coordinates of the conformer as a 2D list
        coordinates = [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z] for i in range(rdmol.GetNumAtoms())]

        # Get the input geometry as a tuple of (atom_types, coordinates)
        geometry = (atom_types, coordinates)

        # Create a temporary directory to store the input and output files
        orca_folder = f"run_orca/{job_name}"
        os.makedirs(orca_folder, exist_ok=True)
        
        # Work in the temporary directory
        with contextlib.chdir(orca_folder):
            # Create and write the ORCA input file
            orca_input_file = f"{job_name}.inp"
            with open(orca_input_file, 'w') as f:
                block = write_orca_inp_block(geometry, 
                                             rootline=params['orca_CO_rootline'], 
                                             control_block=params['orca_CO_controlblock'], 
                                             charge=charge, 
                                             multiplicity=multiplicity)
                f.write(block)

            # Set the ORCA output file name
            orca_output_file = f"{job_name}.out"
            
            # Check if the ORCA software is available
            orca_executable = shutil.which("orca")
            if orca_executable is None:
                logger.error("ORCA executable not found. Please ensure ORCA is installed and added to the system PATH.")
                raise FileNotFoundError("ORCA executable not found. Please ensure ORCA is installed and added to the system PATH.")
            
            # Run the ORCA calculation
            try:
                with open(orca_output_file, 'w') as f:
                    subprocess.run(['orca', orca_input_file], stdout=f, stderr=subprocess.STDOUT, check=True)

            except subprocess.CalledProcessError as e:
                logger.error(f"ORCA optimization failed for {job_name}. Error: {e}")
                continue

            # Check if the output file is generated
            if not os.path.isfile(orca_output_file):
                logger.error(f"ORCA output file not found for {job_name}. Optimization may have failed.")
                continue

            # Check if the xyz file is generated
            orca_xyz_file = f"{job_name}.xyz"
            if not os.path.isfile(orca_xyz_file):
                logger.error(f"ORCA xyz file not found for {job_name}. Optimization may have failed.")
                continue

            # Check if the ORCA program is terminated normally
            with open(orca_output_file, 'r') as f:
                orca_output = f.read()
            if 'ORCA TERMINATED NORMALLY' not in orca_output:
                logger.error(f"ORCA optimization for {job_name} did not terminate normally. Please check the output file for details.")
                continue

            # Extract the final energy from the output file (Hartree to kcal/mol conversion factor: 627.509) and store it in the energy dictionary
            energy = orca_read_energy(orca_output_file)
            if energy is None:
                logger.error(f"Failed to extract energy for {job_name}. Please check the output file for details.")
                continue
            energy_dict[conf_id] = energy

            # Extract the final optimized geometry from the xyz file
            optimized_coordinates = orca_read_geometry(orca_xyz_file)

            if optimized_coordinates is None:
                logger.error(f"Failed to extract optimized geometry for {job_name}. Please check the xyz file for details.")
                continue
            if len(optimized_coordinates) != rdmol.GetNumAtoms():
                logger.error(f"Number of atoms in optimized geometry does not match for {job_name}. Please check the xyz file for details.")
                continue

            # Update the conformer coordinates in the RDKit molecule
            for i, (x, y, z) in enumerate(optimized_coordinates):
                conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        
        # Delete the running folder
        if params["clean_files"] == True:
            shutil.rmtree(orca_folder, ignore_errors=True)
    
    return energy_dict





