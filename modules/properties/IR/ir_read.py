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


def ir_read_cif(cif):
    """
    This function is used to read IR data from a cif file

    Args:
    - cif file
    """
    with open(cif) as f:
        lines = f.readlines()

    wavenumber_ir = []
    intensity_ir = []

    start = False
    for line in lines:
        if line.strip().startswith('loop_'):
            continue
        if '_ir_data' in line or '_ir_intensity' in line:
            start = True
            continue
        if start:
            if line.strip() == '':
                break
            parts = line.split()
            if len(parts) == 2:
                try:
                    wavenumber_ir.append(float(parts[0]))
                    intensity_ir.append(float(parts[1]))
                except ValueError:
                    continue
                
    return wavenumber_ir, intensity_ir
