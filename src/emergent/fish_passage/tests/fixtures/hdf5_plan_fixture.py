import numpy as np
import h5py


def create_hecras_plan(path, vector=True):
    """Create a minimal HECRAS-like HDF5 plan file for tests.

    If vector=True, include perimeter/facepoints datasets so the vector
    extraction path is exercised. If vector=False, omit perimeter datasets
    to force the raster fallback path.
    """
    with h5py.File(path, 'w') as f:
        # create a simple cell coordinate (one cell at center)
        coords = np.array([[0.5, 0.5]])
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)

        # Create depth dataset under Results as a small timeseries (T x n_cells)
        # so the wrapper reads a per-cell array when indexing by timestep.
        n_cells = coords.shape[0]
        depths = np.full((2, n_cells), 0.2, dtype=float)
        base = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/'
        f.create_dataset(base + 'Cell Hydraulic Depth', data=depths)

        if vector:
            # FacePoints: four corners forming a rectangle
            facepoints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            f.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Coordinate', data=facepoints)
            # Mark all facepoints as perimeter with -1
            is_perim = np.full((facepoints.shape[0],), -1, dtype=int)
            f.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter', data=is_perim)
            # Cells Face and Orientation Info: start index and count
            face_info = np.array([[0, facepoints.shape[0]]], dtype=int)
            f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info', data=face_info)
            # Perimeter coords: same as facepoints for simplicity
            f.create_dataset('Geometry/2D Flow Areas/2D area/Perimeter', data=facepoints)
        else:
            # Omit perimeter fields to force raster fallback
            pass

    return path
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
