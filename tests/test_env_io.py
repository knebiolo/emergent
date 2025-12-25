import os
import tempfile
import numpy as np
from emergent.salmon_abm import io, hdf5_io

def test_write_raster_to_hdf5_roundtrip():
    # use existing raster in repo
    path = os.path.join('data', 'salmon_abm', 'vel_x.tif')
    assert os.path.exists(path), 'test raster missing'
    # use dict-like hdf5 store for test
    store = {}
    arr, tr_tup, crs = io.write_raster_to_hdf5(store, path, dataset_name='vel_x', sim=None)
    # verify array written
    assert 'environment/vel_x' in store
    ds = store['environment/vel_x']
    assert ds.shape == arr.shape
    # verify transform tuple
    assert tr_tup is not None and len(tr_tup) == 6
    # pixel -> geo -> pixel roundtrip using Affine if available from enviro_import
    arr2, transform, crs2 = io.enviro_import(path)
    # pick a sample pixel
    row, col = 10, 20
    x, y = transform * (col + 0.5, row + 0.5)
    inv = ~transform
    rt_col, rt_row = inv * (x, y)
    assert abs(rt_row - (row + 0.5)) < 1e-6
    assert abs(rt_col - (col + 0.5)) < 1e-6

if __name__ == '__main__':
    test_write_raster_to_hdf5_roundtrip()
    print('ok')
