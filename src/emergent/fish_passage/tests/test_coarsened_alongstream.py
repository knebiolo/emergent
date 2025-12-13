import h5py
import numpy as np
from pathlib import Path
from emergent.fish_passage.io import compute_coarsened_alongstream_raster


class DummySim:
    def __init__(self, hdf_path):
        self.hdf5 = h5py.File(hdf_path, 'w')


def test_coarsened_basic(tmp_path: Path):
    p = tmp_path / 'sim.h5'
    sim = DummySim(str(p))
    env = sim.hdf5.create_group('environment')
    # 6x6 grid with central wetted patch
    depth = np.zeros((6, 6), dtype='f4')
    depth[2:4, 2:4] = 1.0
    env.create_dataset('depth', data=depth)
    xs = np.tile(np.arange(6, dtype='f4'), (6, 1))
    ys = np.tile(np.arange(6, dtype='f4').reshape((6, 1)), (1, 6))
    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)

    out = compute_coarsened_alongstream_raster(simulation=sim, factor=2, depth_name='depth', out_name='asc')
    assert out.shape == depth.shape or out.shape == (6,6)
    assert 'asc' in sim.hdf5['environment']
    sim.hdf5.close()
