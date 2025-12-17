import numpy as np
import h5py
from emergent.salmon_abm.viewer_v3.hecras_adapter import extract_depth_points


def make_test_hdf():
    # create an in-memory h5py file
    f = h5py.File('inmemory.h5', mode='w', driver='core', backing_store=False)
    # create minimal required datasets
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8], [2.0, 2.0]])
    f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
    # depth timeseries: single timestep
    depth = np.array([[0.0, 0.2, 0.1, 0.0]])
    f.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=depth)
    return f


def test_extract_depth_basic():
    f = make_test_hdf()
    pts, vals = extract_depth_points(f, timestep=0, depth_thresh=0.05)
    assert pts.shape[0] == 2
    assert vals.shape[0] == 2
    # ensure returned pts are subset of original coords
    orig = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.8],[2.0,2.0]])
    for p in pts:
        assert any(np.allclose(p, o) for o in orig)
    f.close()
