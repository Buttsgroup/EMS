import numpy as np


def nmr_read(stringfile, streamlit=False):
    if streamlit:
        atoms = 0
        for line in stringfile:
            if len(line.split()) == 16:
                atoms += 1
            if "V2000" in line or len(line.split()) == 12:
                chkatoms = int(line.split()[0])

        # check for stupid size labelling issue
        if atoms != chkatoms:
            for i in range(1, len(str(chkatoms))):
                if atoms == int(str(chkatoms)[:-i]):
                    chkatoms = atoms
                    break

        assert atoms == chkatoms
        # Define empty arrays
        shift_array = np.zeros(atoms, dtype=np.float64)
        # Variance is used for machine learning
        shift_var = np.zeros(atoms, dtype=np.float64)
        coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
        coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
        # Variance is used for machine learning
        coupling_var = np.zeros((atoms, atoms), dtype=np.float64)

        # Go through file looking for assignment sections
        shift_switch = False
        cpl_switch = False
        for line in stringfile:
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
                # ['0', ',', '1', ',', '-0.26456900', ',', '5JON', ',', '0.00000000']
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

        return shift_array, shift_var, coupling_array, coupling_var

    else:
        atoms = 0
        with open(stringfile, "r") as f:
            for line in f:
                if len(line.split()) == 16:
                    atoms += 1
                if "V2000" in line or len(line.split()) == 12:
                    chkatoms = int(line.split()[0])

        # check for stupid size labelling issue
        if atoms != chkatoms:
            for i in range(1, len(str(chkatoms))):
                if atoms == int(str(chkatoms)[:-i]):
                    chkatoms = atoms
                    break

        assert atoms == chkatoms
        # Define empty arrays
        shift_array = np.zeros(atoms, dtype=np.float64)
        # Variance is used for machine learning
        shift_var = np.zeros(atoms, dtype=np.float64)
        coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
        coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
        # Variance is used for machine learning
        coupling_var = np.zeros((atoms, atoms), dtype=np.float64)

        # Go through file looking for assignment sections
        with open(stringfile, "r") as f:
            shift_switch = False
            cpl_switch = False
            for line in f:
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
                    # ['0', ',', '1', ',', '-0.26456900', ',', '5JON', ',', '0.00000000']
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

        return shift_array, shift_var, coupling_array, coupling_var