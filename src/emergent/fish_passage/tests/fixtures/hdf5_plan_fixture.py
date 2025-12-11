import h5py
import numpy as np
from pathlib import Path


def create_minimal_plan(path):
    """Create a minimal HECRAS-like HDF5 plan file at `path`.

    Creates dataset:
    - 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate' shape (4,2)
    - a small 'Fields/depth' dataset matching node count
    """
    path = Path(path)
    with h5py.File(str(path), 'w') as f:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype='f4')
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        # simple depth field per node
        f.create_dataset('Fields/depth', data=np.array([0.1, 0.2, 0.3, 0.4], dtype='f4'))
    return str(path)


def create_sim_hdf(path):
    path = Path(path)
    f = h5py.File(str(path), 'w')
    return f
