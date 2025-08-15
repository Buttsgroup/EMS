from EMS.modules.properties.nmr import nmr_ops
import numpy as np


def test_scale_chemical_shifts():
    raw_shift = np.array([17.4511, 159.7362, 30.9043, 274.0412, 215.5712, -285.6313])
    atom_types = [1, 6, 1, 9, 7, 8]
    scaled_shift = nmr_ops.scale_chemical_shifts(raw_shift, atom_types)
    calculated_scaled_shift = np.array([13.9496, 27.1455, 1.2507, -92.3411, -359.2477, 0.0000])

    assert np.allclose(scaled_shift, calculated_scaled_shift, rtol=1e-4)