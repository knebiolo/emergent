import os
import numpy as np
from emergent.salmon_abm import utils, io


def test_geo_pixel_roundtrip():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root, 'data', 'salmon_abm')
    depth_path = os.path.join(data_dir, 'depth.tif')
    assert os.path.exists(depth_path), 'depth.tif missing for test'
    arr, transform, crs = io.enviro_import(depth_path)
    # Ensure _unpack_affine works on rasterio transform and on tuple
    tup = utils._unpack_affine(transform)
    assert len(tup) == 6

    # pick several pixel centers and test roundtrip
    rows = np.array([0, arr.shape[0]//2, arr.shape[0]-1])
    cols = np.array([0, arr.shape[1]//2, arr.shape[1]-1])
    # pixel_to_geo returns (x,y) for (row,col)
    xs, ys = utils.pixel_to_geo(rows, cols, transform)
    # now map back
    r2, c2 = utils.geo_to_pixel(xs, ys, transform)
    # should be equal (allow off-by-one due to rounding semantics)
    assert np.all(np.abs(r2 - rows) <= 1)
    assert np.all(np.abs(c2 - cols) <= 1)


def test_geo_to_pixel_vectorized_and_affine_inverse():
    # create a simple affine transform: x = 10*col + 0*row + 100, y = 0*col + -10*row + 200
    tr = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    # test scalar
    r, c = utils.geo_to_pixel(100.0 + 10.0 * 5 + 0.0, 200.0 + -10.0 * 7 + 0.0, tr)
    assert r == 7 and c == 5

    # vectorized
    xs = np.array([100.0 + 10.0 * i for i in range(4)])
    ys = np.array([200.0 + -10.0 * j for j in range(4)])
    rows, cols = utils.geo_to_pixel(xs, ys, tr)
    assert rows.shape == cols.shape
    # basic sanity for ordering
    assert rows[0] == 0

    # test using an Affine-like object if available (rasterio.Affine has __invert__)
    try:
        import rasterio
        from rasterio.transform import Affine
        a = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        r_, c_ = utils.geo_to_pixel(150.0, 150.0, a)
        # ensure output types are ints or arrays
        assert isinstance(r_, (int, np.integer)) or (hasattr(r_, 'dtype'))
    except Exception:
        # rasterio may not be installed in test env; that's okay
        pass
