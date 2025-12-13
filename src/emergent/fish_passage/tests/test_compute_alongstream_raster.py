import h5py
import numpy as np
from pathlib import Path
from emergent.fish_passage.io import compute_alongstream_raster


class DummySim:
    def __init__(self, hdf_path):
        self.hdf5 = h5py.File(hdf_path, 'w')
        self.depth_rast_transform = None


def test_compute_alongstream_basic(tmp_path: Path):
    p = tmp_path / 'sim.h5'
    sim = DummySim(str(p))
    env = sim.hdf5.create_group('environment')
    # simple 3x3 depth raster, center cell is deep
    depth = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype='f4')
    env.create_dataset('depth', data=depth)
    # create x/y coords
    xs = np.tile(np.arange(3, dtype='f4'), (3,1))
    ys = np.tile(np.arange(3, dtype='f4').reshape((3,1)), (1,3))
    env.create_dataset('x_coords', data=xs)
    env.create_dataset('y_coords', data=ys)

    arr = compute_alongstream_raster(sim, outlet_xy=None, depth_name='depth', out_name='along_stream_dist')
    assert 'along_stream_dist' in sim.hdf5['environment']
    res = np.asarray(sim.hdf5['environment']['along_stream_dist'])
    assert res.shape == depth.shape
    # center should be zero distance to itself (outlet chosen as min y -> row 0, but check finite values exist)
    assert np.isfinite(res).any()
    sim.hdf5.close()
