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


def xrd_read_sdf():
    pass

def xrd_read_cif(cif):
    """
    This function is used to read XRD data from a cif file

    Args:
    - cif file
    """
    with open(cif) as f:
        lines = f.readlines()

    q_values_xray = []
    intensity_xray = []

    start = False
    for line in lines:
        if line.strip().startswith('loop_'):
            continue
        if '_xray_q' in line or 'x_ray_intensity' in line:
            start = True
            continue
        if start:
            if line.strip() == '#END' or line.strip() == '':
                break
            parts = line.split()
            if len(parts) == 2:
                try:
                    q_values_xray.append(float(parts[0]))
                    intensity_xray.append(float(parts[1]))
                except ValueError:
                    continue

    return q_values_xray, intensity_xray



