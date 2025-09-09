import os
import sys
import logging
import json
import subprocess
import contextlib
import shutil
import copy

from xtb_ase import XTBProfile, _io
import ase
from ase.io import read

from rdkit.Geometry import Point3D


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


class suppress_output:
    '''
    Mute noisy stdout/stderr at the OS level, like RDKit warnings.
    '''

    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *args):
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        os.close(self.null_fd)


def check_xtb_conformer_optimization_params(params):
    '''
    This function checks and sets default parameters for conformer optimization by xTB.
    The parameters can be found in xTB documentation: https://xtb-docs.readthedocs.io/en/latest/commandline.html
    and https://xtb-docs.readthedocs.io/en/latest/optimization.html
    If one parameter's value is None, default value will be used in xTB calculation.

    Parameters:
    - xtb_CO_opt (str): The level of theory to be used for xTB calculations (e.g., "tight", "normal", "lax", "loose").
    - xtb_CO_alpb (str): The solvent used in Poisson-Boltzmann (ALPB) model (e.g., "chcl3", "water", "ethanol").
    - xtb_CO_cycles (int): The maximum number of optimization cycles.
    - xtb_CO_chrg (int): The molecular charge.
    - xtb_CO_uhf (int): The spin state (0 for singlet, 1 for doublet, 2 for triplet ...).
    - xtb_CO_gfn (int or str): The GFN-xTB parametrization (0 for GFN0, 1 for GFN1, 2 for GFN2 and "gfnff" for classical force field). Default in xTB: 2
    - xtb_CO_etemp (float): The electronic temperature. Default in xTB: 300K
    - xtb_CO_acc (float): Accuracy for SCC calculation. Default in xTB: 1.0
    - xtb_CO_vparam (float): Parameter file for vTB calculation.
    - xtb_CO_gbsa (str): The solvent used in the Generalized Born (GB) model with Solvent Accessible Surface (SASA) model.
    - xtb_CO_cma (bool): Whether to shift molecule to center of mass and rotate its axes to the principal axes of inertia.
    - xtb_CO_molden (bool): Whether to print Molden file.
    - xtb_CO_lmo (bool): Whether to use local molecular orbitals (LMOs) for the calculation.
    - xtb_CO_fod (bool): Whether to use a FOD calculation.
    - xtb_CO_input (str): The file used as input source for xcontrol instructions.
    - xtb_CO_copy (bool): Whether to copy the xcontrol file at startup.
    - xtb_CO_norestart (bool): Whether to disable restarting calculation from xtbrestart.
    - xtb_CO_parallel (int): The number of parallel processes.
    - xtb_CO_namespace (str): The prefix of all files from this xTB calculation.
    - xtb_CO_define (bool): Whether to perform automatic check of input and terminate
    - xtb_CO_citation (bool): Whether to print citation and terminate.
    - xtb_CO_license (bool): Whether to print license and terminate.
    - xtb_CO_verbose (bool): Whether to print more detailed progress and information.
    - xtb_CO_silent (bool): Whether to reduce the amount of output printed, which is the opposite of verbose.
    - xtb_CO_strict (bool): Whether to turn all warnings into hard errors.
    - xtb_CO_help (bool): Whether to show help page.
    '''

    params.setdefault("xtb_CO_opt", "tight")
    params.setdefault("xtb_CO_alpb", "chcl3")
    params.setdefault("xtb_CO_cycles", None)
    params.setdefault("xtb_CO_chrg", None)
    params.setdefault("xtb_CO_uhf", None)
    params.setdefault("xtb_CO_gfn", None)
    params.setdefault("xtb_CO_etemp", None)
    params.setdefault("xtb_CO_acc", None)
    params.setdefault("xtb_CO_vparam", None)
    params.setdefault("xtb_CO_gbsa", None)
    params.setdefault("xtb_CO_cma", None)
    params.setdefault("xtb_CO_molden", None)
    params.setdefault("xtb_CO_lmo", None)
    params.setdefault("xtb_CO_fod", None)
    params.setdefault("xtb_CO_input", None)
    params.setdefault("xtb_CO_copy", None)
    params.setdefault("xtb_CO_norestart", None)
    params.setdefault("xtb_CO_parallel", None)
    params.setdefault("xtb_CO_namespace", None)
    params.setdefault("xtb_CO_define", None)
    params.setdefault("xtb_CO_citation", None)
    params.setdefault("xtb_CO_license", None)
    params.setdefault("xtb_CO_verbose", None)
    params.setdefault("xtb_CO_silent", None)
    params.setdefault("xtb_CO_strict", None)
    params.setdefault("xtb_CO_help", None)


def write_xtb_profile(params):
    '''
    Write the xTB calculation profile to a list.
    '''

    params = copy.deepcopy(params)

    # Get the parameter dictionary for xTB conformer optimization
    xtb_CO_dict = {}
    for key, value in params.items():
        if key.startswith("xtb_CO_") and value is not None:
            xtb_CO_dict[key] = value
    
    # Initialize the xTB profile list
    xtb_profile = ["xtb", "coords"]

    # Add parameters to the profile according to the parameter dictionary
    for key, value in xtb_CO_dict.items():
        # Handle the --gfn or --gfnff parameter
        if key == "xtb_CO_gfn":
            if isinstance(value, int):
                xtb_profile.extend(["--gfn", str(value)])
            elif value == "gfnff":
                xtb_profile.append("--gfnff")
            
        # Handle other parameters
        else:
            real_key = key.replace("xtb_CO_", "--")
            if isinstance(value, bool) and value == True:
                xtb_profile.append(real_key)
            else:
                xtb_profile.extend([real_key, str(value)])
    
    return xtb_profile


def xtb_conformer_optimization(rdmol, params):
    '''
    Optimize the conformers of the given RDKit molecule using xTB and return their energies (kcal/mol, kcal=kJ*4.184)
    '''

    # Set default parameters for xTB conformer optimization
    check_xtb_conformer_optimization_params(params)

    # Write xTB calculation profile
    xtb_profile = write_xtb_profile(params)
    profile = XTBProfile(xtb_profile)

    # Get the molecule's name
    mol_name = rdmol.GetProp("_Name")
    if mol_name is None or mol_name.strip() == "":
        logger.error("The molecule being processed does not have a name. Please set the name by EMS before optimization.")
        raise ValueError("The molecule being processed does not have a name. Please set the name by EMS before optimization.")
    mol_name = mol_name.strip().split('.')[0]

    # Get conformers of the RDKit molecule
    conformers = [conf for conf in rdmol.GetConformers()]

    # Make a directory for the molecule to run xTB optimization
    xtb_folder = f"run_xtb/{mol_name}"
    os.makedirs(xtb_folder, exist_ok=True)
    os.makedirs(f"{xtb_folder}/xtb-out", exist_ok=True)

    # Initialize the energy dictionary
    energy_dict = {}

    # Change the working directory
    with contextlib.chdir(xtb_folder):
        # Iterate over the conformers to optimize
        for conformer in conformers:
            # Get the conformer ID
            conf_id = conformer.GetId()

            # Set the conformer name
            conformer_name = f"{mol_name}_conf_{conf_id}"

            # Get the positions and atomic numbers of the conformers and create an ASE atoms object
            ase_atoms = ase.Atoms(
                numbers=[
                    atom.GetAtomicNum() for atom in rdmol.GetAtoms()
                ], 
                positions=conformer.GetPositions()
            )

            # Carry out xTB structure optimization
            try:
                # Write the input files and run xTB
                _io.write_xtb_inputs(atoms=ase_atoms, input_filepath=f"xtb-out/input.inp", geom_filepath=f"xtb-out/coord.xyz", parameters={})
                with suppress_output():
                    profile.run(directory=".", input_filename=f"xtb-out/input.inp", output_filename=f"xtb-out/xtb-test.out", geom_filename=f"xtb-out/coord.xyz")
                with open(f"xtbout.json", "r") as f:
                    data = json.load(f)

            except subprocess.CalledProcessError as e:
                logger.error(f"Process error for molecule conformer {conformer_name} during xTB optimization: {e}")
                continue

            except Exception as e:
                logger.error(f"Error during xTB optimization for molecule {conformer_name}: {e}")
                continue

            # Get the energy and atomic positions
            energy = data["total energy"] * 627.509474  # Convert energy from Hartree to kcal/mol
            atoms = read("xtbopt.xyz")
            atom_coords = atoms.get_positions().tolist()

            # Save the optimized energy
            energy_dict[conf_id] = energy

            # Check if the number of atoms matches
            if len(atom_coords) != rdmol.GetNumAtoms():
                logger.error(f"The number of atoms in molecule conformer {conformer_name} after xTB optimization does not match the number of atoms in the molecule.")
                raise ValueError(f"The number of atoms in molecule conformer {conformer_name} after xTB optimization does not match the number of atoms in the molecule.")

            # Update the conformer with the optimized coordinates
            for i, (x, y, z) in enumerate(atom_coords):
                conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    # Delete the xTB working directory
    if params["clean_files"] == True:
        shutil.rmtree(xtb_folder, ignore_errors=True)

    return energy_dict