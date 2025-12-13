import numpy as np
import h5py
from typing import Tuple


def load_hecras_cells(path: str, value_dataset_name: str = 'Cell Hydraulic Depth') -> Tuple[np.ndarray, np.ndarray]:
    """Load HECRAS plan cell centers and a value array.

    Returns (coords, values) where coords is (N,2) and values is (N,)
    The function searches for the common geometry path and the value dataset
    under several likely locations.
    """
    with h5py.File(path, 'r') as hdf:
        # geometry
        coords_key = '/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if coords_key not in hdf:
            raise KeyError('HECRAS plan missing Cells Center Coordinate')
        coords = np.asarray(hdf[coords_key])

        # try several candidate paths for value dataset
        candidates = [
            '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/' + value_dataset_name,
            '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth',
            '/' + value_dataset_name
        ]
        val = None
        for c in candidates:
            if c in hdf:
                ds = hdf[c]
                arr = np.asarray(ds[:])
                # if time series, take first timestep
                if arr.ndim > 1 and arr.shape[0] > 1:
                    val = np.asarray(arr[0]).reshape(-1)
                else:
                    val = np.asarray(arr).reshape(-1)
                break
        if val is None:
            raise KeyError('HECRAS plan missing value dataset')
    return coords, val
