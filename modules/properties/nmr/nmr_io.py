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


def nmr_read(stringfile, streamlit=False):
    '''
    This function is used to read NMR data from an SDF file. When reading NMR data for an EMS object, the SDF file is the self.stringfile attribute.
    It reads the SDF file line by line and extracts the NMR data from the <NMREDATA_ASSIGNMENT> and <NMREDATA_J> properties in the SDF file.
    Details of this function:
    (1) This function loops over the lines of the SDF file only once, so as to minimize the time taken to read the NMR data.
    (2) This function first reads the info line of the SDF file to get the number of atoms in the molecule ('V2000' line in V2000 SDF, or 'M  V30 COUNTS' line in V3000 SDF),
        and then reads the structure block to get the number of atom lines (before 'M  END' line). If the two numbers are not equal, an error will be raised.

    Args:
    - stringfile (str): The SDF string if streamlit is on, or the path to the SDF file if streamlit is off.
    - streamlit (bool): If True, the SDF file is as read as streamlit. If False (default), the SDF file is read as a file path.
    '''

    # Get the file as a list of lines
    if not streamlit:
        with open(stringfile, "r") as f:
            lines = f.readlines()
    else:
        lines = [line for line in stringfile]

    # Initialize variables when reading the SDF file
    structure_end_check = False          # Check if the 'M  END' line or the end of structure information block is reached
    atom_block_start_check = False       # Check if the atom block start line is reached
    atom_block_end_check = False         # Check if the atom block end line is reached
    shift_switch = False                 # Switch to read the chemical shift lines
    cpl_switch = False                   # Switch to read the coupling constant lines

    chkatoms = 0                         # Number of atoms read in the 'V2000' line in V2000 SDF file or 'M  V30 COUNTS' line in V3000 SDF file
    sdf_version = None                   # SDF version (V2000 or V3000) read in the info line
    atom_block_start = 0                 # The index of the line where the atom block starts
    atom_block_end = 0                   # The index of the line where the atom block ends

    # Initialize arrays for saving NMR data
    shift_array = None
    shift_var = None
    coupling_array = None
    coupling_len = None
    coupling_var = None

    # Loop over the lines of the SDF file only once
    for idx, line in enumerate(lines):

        # Break the loop if the end of the molecule ('$$$$') is reached
        if '$$$$' in line:
            break

        # Check if the block of structure information has ended
        if 'M  END' in line:
            structure_end_check = True

        # Enter the mode of getting the atom number, which is before the 'M  END' line
        if not structure_end_check:
            # Get the SDF version
            if 'V3000' in line:
                sdf_version = 'V3000'
            elif 'V2000' in line:
                sdf_version = 'V2000'

            # For V3000 SDF, get the number of atoms from the 'M  V30 COUNTS' line 
            # and indices of the start and end lines of the atom block from the 'M  V30 BEGIN ATOM' and 'M  V30 END ATOM' lines
            if sdf_version == 'V3000':
                if 'M  V30 COUNTS' in line:
                    chkatoms = int(line.split()[3])
                
                if 'M  V30 BEGIN ATOM' in line:
                    atom_block_start = idx + 1
                    atom_block_start_check = True
                
                if 'M  V30 END ATOM' in line:
                    atom_block_end = idx
                    atom_block_end_check = True
            
            # For V2000 SDF, get the number of atoms from the 'V2000' line 
            # and the indices of the start and end lines of the atom block from change of word numbers in the lines
            elif sdf_version == 'V2000':
                if 'V2000' in line:
                    chkatoms = int(line[0:3].strip())
                
                if atom_block_start_check == False and len(line.split()) >= 14:
                    atom_block_start = idx
                    atom_block_start_check = True
                
                if atom_block_end_check == False and atom_block_start_check == True and len(line.split()) <= 8:
                    atom_block_end = idx
                    atom_block_end_check = True
        

        # Enter the mode of checking if the atom number is correct and reading NMR data, which is after the 'M  END' line
        else:      # if structure_end_check == True
            
            # When in the 'M  END' line, initialize the arrays for saving NMR data and check if the atom number is correctly read
            if 'M  END' in line:
                # Check if the atom number is correctly read
                atoms = atom_block_end - atom_block_start
                if chkatoms != atoms or chkatoms == 0:
                    logger.error(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
                    raise ValueError(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
                
                # Define empty arrays for saving NMR data
                # Variance is used for machine learning
                shift_array = np.zeros(atoms, dtype=np.float64)
                shift_var = np.zeros(atoms, dtype=np.float64)
                coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
                coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
                coupling_var = np.zeros((atoms, atoms), dtype=np.float64)
            
            # After the 'M  END' line, read the NMR data from the <NMREDATA_ASSIGNMENT> and <NMREDATA_J> properties
            else:
                if "<NMREDATA_ASSIGNMENT>" in line:
                    shift_switch = True
                if "<NMREDATA_J>" in line:
                    shift_switch = False
                    cpl_switch = True

                # If shift assignment label found, process shift rows
                if shift_switch:
                    # Shift assignment row looks like this
                    #  0    , -33.56610000   , 8    , 0.00000000     \
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    shift_array[int(items[0])] = float(items[2])
                    shift_var[int(items[0])] = float(items[6])

                # If coupling assignment label found, process coupling rows
                if cpl_switch:
                    # Coupling row looks like this
                    #  0         , 4         , -0.08615310    , 3JON      , 0.00000000
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    length = int(items[6].strip()[0])
                    coupling_array[int(items[0])][int(items[2])] = float(items[4])
                    coupling_array[int(items[2])][int(items[0])] = float(items[4])
                    coupling_var[int(items[0])][int(items[2])] = float(items[8])
                    coupling_var[int(items[2])][int(items[0])] = float(items[8])
                    coupling_len[int(items[0])][int(items[2])] = length
                    coupling_len[int(items[2])][int(items[0])] = length

    # Raise errors if the following conditions are not met. These conditions make sure that the NMR data is correctly read.
    if structure_end_check == False:
        logger.error(f"No 'M  END' line found in the file: {stringfile}")
        raise ValueError(f"No 'M  END' line found in the file: {stringfile}")
    
    if not (atom_block_start_check and atom_block_end_check):
        logger.error(f"Structure block not found in the file: {stringfile}")
        raise ValueError(f"Structure block not found in the file: {stringfile}")
    
    if sdf_version is None:
        logger.error(f'SDF version not found in the file: {stringfile}')
        raise ValueError(f'SDF version not found in the file: {stringfile}')
    
    if chkatoms == 0 or atom_block_end - atom_block_start == 0:
        logger.error(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
        raise ValueError(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
    
    if shift_array is None:
        logger.error(f'NMR data in the SDF file is not correctly read: {stringfile}')
        raise ValueError(f'NMR data in the SDF file is not correctly read: {stringfile}')

    return shift_array, shift_var, coupling_array, coupling_var

    
def nmr_read_rdmol(shift, coupling):
    shift_items = []
    for line in shift.split('\n'):
        if line:
            shift_items.append(line.split())
    
    coupling_items = []
    for line in coupling.split('\n'):
        if line:
            coupling_items.append(line.split())
    
    num_atom = len(shift_items)
    shift_array = np.zeros(num_atom, dtype=np.float64)
    shift_var = np.zeros(num_atom, dtype=np.float64)
    coupling_array = np.zeros((num_atom, num_atom), dtype=np.float64)
    coupling_var = np.zeros((num_atom, num_atom), dtype=np.float64)

    for item in shift_items:
        shift_array[int(item[0])] = float(item[2])
        shift_var[int(item[0])] = float(item[6])
    
    for item in coupling_items:
        coupling_array[int(item[0])][int(item[2])] = float(item[4])
        coupling_array[int(item[2])][int(item[0])] = float(item[4])
        coupling_var[int(item[0])][int(item[2])] = float(item[8])
        coupling_var[int(item[2])][int(item[0])] = float(item[8])
    
    return shift_array, shift_var, coupling_array, coupling_var