import numpy as np
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


def gaussian_read_structure(file):
    '''
    This function reads the Gaussian log file and extracts the types and coordinates of the atoms in the molecule.

    Args:
    - file (str): The path to the Gaussian log file.
    '''

    # !!! The following section reads the number of atoms in the molecule from the Gaussian log file
    # The atom number information is located in the line that contains 'NAtoms='
    atomnumber = None
    with open(file, 'r') as f:
        for line in f:
            if 'NAtoms=' in line:
                items = line.split()

                # The atom number is the second item in the line
                # It is converted to an integer. If this fails, an error is raised
                try:
                    atomnumber = int(items[1])
                    break
                except:
                    logger.error(f"Fail to read atom number in log file: {file}")
                    raise ValueError(f"Fail to read atom number in log file: {file}")
    
    # If atom number is not found, raise an error
    if atomnumber is None:
        logger.error(f"No atom number found in log file: {file}")
        raise ValueError(f"No atom number found in log file: {file}")
    

    # !!! The following section reads the atom types and atom coordinates together from the Gaussian log file
    # Initialize switches for reading atom types and atom coordinates
    # The atom_switch is activated when the atom types and atom coordinates are found, which are located between 'Standard orientation:' and 'Rotational constants (GHZ)' lines
    atom_switch = False

    # Go through file to find atom types and atom coordinates
    with open(file, 'r') as f_handle:
        for line in f_handle:

            # Control the switch for reading the atom types and atom coordinates
            if "Standard orientation:" in line:
                atom_switch = True
                # Initialize empty arrays for atom types, atom coordinates, chemical shielding tensors and coupling constants every time when reading the structure block
                atom_types = np.zeros(atomnumber, dtype=np.int32)
                atom_coords = np.zeros((atomnumber, 3), dtype=np.float64)
                continue

            if "Rotational constants (GHZ)" in line:
                atom_switch = False
                continue

            # Read the atom types and atom coordinates
            if atom_switch:
                # Skip the lines that not contain the atom types and atom coordinates
                if all(substring not in line for substring in ['----', 'Center', 'Number']):
                    items = line.split()
                    try:
                        # A line that contains the atom types and atom coordinates looks like the following:
                        # '      1          6           0       -2.299297   -0.454669   -0.237828'
                        atom_index = int(items[0])
                        atom_type = int(items[1])
                        atom_x = float(items[3])
                        atom_y = float(items[4])
                        atom_z = float(items[5])
                    except:
                        logger.error(f"Fail to read atom types and atom coordinates in log file: {file}")
                        raise ValueError(f"Fail to read atom types and atom coordinates in log file: {file}")
                    
                    atom_types[atom_index-1] = atom_type
                    atom_coords[atom_index-1] = np.array([atom_x, atom_y, atom_z])
    
    return atom_types, atom_coords


def gaussian_read_nmr(file):
    '''
    This function reads the Gaussian log file and extracts the chemical shielding tensors and coupling constants.

    Args:
    - file (str): The path to the Gaussian log file.
    '''

    # !!! The following section reads the number of atoms in the molecule from the log file

    # The atom number information is located in the line that contains 'NAtoms='
    atomnumber = None
    with open(file, 'r') as f:
        for line in f:
            if 'NAtoms=' in line:
                items = line.split()

                # The atom number is the second item in the line
                # It is converted to an integer. If this fails, an error is raised
                try:
                    atomnumber = int(items[1])
                    break
                except:
                    logger.error(f"Fail to read atom number in log file: {file}")
                    raise ValueError(f"Fail to read atom number in log file: {file}")
    
    # If atom number is not found, raise an error
    if atomnumber is None:
        logger.error(f"No atom number found in log file: {file}")
        raise ValueError(f"No atom number found in log file: {file}")
    

    # !!! The following section reads the atom types, atom coordinates, chemical shielding tensors and coupling constants together from the log file
    # Initialize switches for reading chemical shielding tensors and coupling constants
    # The shift_switch is activated when the chemical shielding tensors are found, which are located between 'SCF GIAO Magnetic shielding tensor (ppm)' and 'Fermi Contact' lines
    # The coupling_switch is activated when the coupling constants are found, which are located between 'Total nuclear spin-spin coupling J (Hz):' and 'End of Minotr' lines
    shift_switch = False
    coupling_switch = False

    # Go through file to find magnetic shielding tensors
    with open(file, 'r') as f_handle:
        for line in f_handle:

            # Control the switch for reading the chemical shielding tensors
            if "SCF GIAO Magnetic shielding tensor (ppm)" in line:
                shift_switch = True
                # Initialze empty arrays for chemical shielding tensors every time when reading the shielding tensor block
                shift_array = np.zeros(atomnumber, dtype=np.float64)
                continue

            if "Fermi Contact" in line:
                shift_switch= False
                continue

            # Control the switch for reading the coupling constants
            if "Total nuclear spin-spin coupling J (Hz):" in line:
                coupling_switch = True
                # Initialize empty arrays for chemical shielding tensors and coupling constants
                couplings = np.zeros((atomnumber, atomnumber), dtype=np.float64)
                continue

            if "End of Minotr" in line:
                coupling_switch = False
                continue
            
            # Read the chemical shielding tensors
            if shift_switch:
                # Find lines including 'Isotropic', which contain the chemical shielding tensors
                if "Isotropic" in line:
                    items = line.split()
                    try:
                        # A line that contains the chemical shielding tensors looks like the following:
                        # '      2  C    Isotropic =   141.3195   Anisotropy =    19.2464'
                        num = int(items[0])
                        shielding = float(items[4])
                    except:
                        logger.error(f"Fail to read chemical shielding tensors in log file: {file}")
                        raise ValueError(f"Fail to read chemical shielding tensors in log file: {file}")

                    shift_array[num-1] = shielding


            # Read the coupling constants
            if coupling_switch:
                # All coupling lines contain "D", all index lines do not
                if "D" not in line:
                    # Get indices for this section
                    # A line that contains the indices looks like the following:
                    # '                1             2             3             4             5 '
                    items = line.split()
                    try:
                        i_indices = np.asarray(items, dtype=np.int32)
                    except:
                        logger.error(f"Fail to read coupling indices in log file: {file}")
                        raise ValueError(f"Fail to read coupling indices in log file: {file}")

                else:
                    # Assign couplings (array is diagonalised in log file, so this is fiddly)
                    # A line that contains the coupling constants looks like the following:
                    # '      11  0.143943D+01 -0.445679D+00  0.167482D+01  0.313373D+02 -0.642198D+00'
                    items = line.split()
                    try:
                        index_j = int(items[0]) - 1
                        for i in range(len(items)-1):
                            index_i = i_indices[i] - 1
                            coupling = float(items[i+1].replace("D", "E"))
                            couplings[index_i][index_j] = coupling
                            couplings[index_j][index_i] = coupling
                    except:
                        logger.error(f"Fail to read coupling constants in log file: {file}")
                        raise ValueError(f"Fail to read coupling constants in log file: {file}")
        
    return shift_array, couplings















######################### The following code is the old code to read atom types, atom coordinates and nmr parameters together #########################


# def gaussian_read_nmr(file):
#     """
#     Read gaussian .log files and extracts the nmr parameters if present

#     :param file: filepath of the .log file which contains the nmr logs
#     :return shift_array and coupling array: numpy array of the chemical tensors for respective atom position and atom-pair interaction
#     """
    
#     # !!! The following section reads the number of atoms in the molecule from the log file

#     # The atom number information is located in the line that contains 'NAtoms='
#     atomnumber = None
#     with open(file, 'r') as f:
#         for line in f:
#             if 'NAtoms=' in line:
#                 items = line.split()

#                 # The atom number is the second item in the line
#                 # It is converted to an integer. If this fails, an error is raised
#                 try:
#                     atomnumber = int(items[1])
#                     break
#                 except:
#                     logger.error(f"Fail to read atom number in log file: {file}")
#                     raise ValueError(f"Fail to read atom number in log file: {file}")
    
#     # If atom number is not found, raise an error
#     if atomnumber is None:
#         logger.error(f"No atom number found in log file: {file}")
#         raise ValueError(f"No atom number found in log file: {file}")


#     # !!! The following section reads the atom types, atom coordinates, chemical shielding tensors and coupling constants together from the log file

#     # Initialize empty arrays for atom types, atom coordinates, chemical shielding tensors and coupling constants
#     shift_array = np.zeros(atomnumber, dtype=np.float64)
#     couplings = np.zeros((atomnumber, atomnumber), dtype=np.float64)
#     atom_types = np.zeros(atomnumber, dtype=np.int32)
#     atom_coords = np.zeros((atomnumber, 3), dtype=np.float64)

#     # Initialize switches for reading atom types, atom coordinates, chemical shielding tensors and coupling constants
#     # The shift_switch is activated when the chemical shielding tensors are found, which are located between 'SCF GIAO Magnetic shielding tensor (ppm)' and 'Fermi Contact' lines
#     # The coupling_switch is activated when the coupling constants are found, which are located between 'Total nuclear spin-spin coupling J (Hz):' and 'End of Minotr' lines
#     # The atom_switch is activated when the atom types and atom coordinates are found, which are located between 'Standard orientation:' and 'Rotational constants (GHZ)' lines
#     shift_switch = False
#     coupling_switch = False
#     atom_switch = False


#     # Go through file to find magnetic shielding tensors
#     with open(file, 'r') as f_handle:
#         for line in f_handle:

#             # Control the switch for reading the chemical shielding tensors
#             if "SCF GIAO Magnetic shielding tensor (ppm)" in line:
#                 shift_switch = True
#                 continue
#             if "Fermi Contact" in line:
#                 shift_switch= False
#                 continue

#             # Control the switch for reading the coupling constants
#             if "Total nuclear spin-spin coupling J (Hz):" in line:
#                 coupling_switch = True
#                 continue
#             if "End of Minotr" in line:
#                 coupling_switch = False
#                 continue

#             # Control the switch for reading the atom types and atom coordinates
#             if "Standard orientation:" in line:
#                 atom_switch = True
#                 continue
#             if "Rotational constants (GHZ)" in line:
#                 atom_switch = False
#                 continue

#             # Read the atom types and atom coordinates
#             if atom_switch:
#                 # Skip the lines that not contain the atom types and atom coordinates
#                 if all(substring not in line for substring in ['----', 'Center', 'Number']):
#                     items = line.split()
#                     try:
#                         # A line that contains the atom types and atom coordinates looks like the following:
#                         # '      1          6           0       -2.299297   -0.454669   -0.237828'
#                         atom_index = int(items[0])
#                         atom_type = int(items[1])
#                         atom_x = float(items[3])
#                         atom_y = float(items[4])
#                         atom_z = float(items[5])
#                     except:
#                         logger.error(f"Fail to read atom types and atom coordinates in log file: {file}")
#                         raise ValueError(f"Fail to read atom types and atom coordinates in log file: {file}")
                    
#                     atom_types[atom_index-1] = atom_type
#                     atom_coords[atom_index-1] = np.array([atom_x, atom_y, atom_z])
            

#             # Read the chemical shielding tensors
#             if shift_switch:
#                 # Find lines including 'Isotropic', which contain the chemical shielding tensors
#                 if "Isotropic" in line:
#                     items = line.split()
#                     try:
#                         # A line that contains the chemical shielding tensors looks like the following:
#                         # '      2  C    Isotropic =   141.3195   Anisotropy =    19.2464'
#                         num = int(items[0])
#                         shielding = float(items[4])
#                     except:
#                         logger.error(f"Fail to read chemical shielding tensors in log file: {file}")
#                         raise ValueError(f"Fail to read chemical shielding tensors in log file: {file}")

#                     shift_array[num-1] = shielding


#             # Read the coupling constants
#             if coupling_switch:
#                 # All coupling lines contain "D", all index lines do not
#                 if "D" not in line:
#                     # Get indices for this section
#                     # A line that contains the indices looks like the following:
#                     # '                1             2             3             4             5 '
#                     items = line.split()
#                     try:
#                         i_indices = np.asarray(items, dtype=np.int32)
#                     except:
#                         logger.error(f"Fail to read coupling indices in log file: {file}")
#                         raise ValueError(f"Fail to read coupling indices in log file: {file}")

#                 else:
#                     # Assign couplings (array is diagonalised in log file, so this is fiddly)
#                     # A line that contains the coupling constants looks like the following:
#                     # '      11  0.143943D+01 -0.445679D+00  0.167482D+01  0.313373D+02 -0.642198D+00'
#                     items = line.split()
#                     try:
#                         index_j = int(items[0]) - 1
#                         for i in range(len(items)-1):
#                             index_i = i_indices[i] - 1
#                             coupling = float(items[i+1].replace("D", "E"))
#                             couplings[index_i][index_j] = coupling
#                             couplings[index_j][index_i] = coupling
#                     except:
#                         logger.error(f"Fail to read coupling constants in log file: {file}")
#                         raise ValueError(f"Fail to read coupling constants in log file: {file}")
        
#     return atom_types, atom_coords, shift_array, couplings