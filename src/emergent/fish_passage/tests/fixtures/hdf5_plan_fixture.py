import h5py
import numpy as np


def make_minimal_plan(path, coords=None, values=None):
    coords = coords if coords is not None else np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    values = values if values is not None else np.array([[0.0], [1.0], [0.2]])
    with h5py.File(path, 'w') as f:
        f.create_dataset('Geometry/Nodes/Coordinates', data=coords)
        # also create 2D Flow Areas center coord for other consumers
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        results = f.create_group('Results')
        res = results.create_group('Results_0001')
        chd = res.create_group('Cell Hydraulic Depth')
        chd.create_dataset('Values', data=values)
    return path
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
