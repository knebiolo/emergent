import numpy as np
import h5py
import tempfile
import os
from emergent.fish_passage.io import enviro_import, sample_environment, initialize_hdf5
from emergent.fish_passage.geometry import geo_to_pixel


class DummySim:
    pass


def make_sim_with_env(arr):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 4
    sim.X = np.array([0.1, 0.4, 0.8, 0.2])
    sim.Y = np.array([0.1, 0.5, 0.2, 0.9])
    # create a simple affine that maps indices to 0..1 domain
    from affine import Affine
    sim.depth_rast_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    enviro_import(sim, arr, 'depth', transform=sim.depth_rast_transform)
    return sim, tmp.name


def test_sample_environment_array():
    arr = np.arange(100).reshape((10, 10)).astype('f4')
    sim, fname = make_sim_with_env(arr)
    try:
        vals = sample_environment(sim, sim.depth_rast_transform, 'depth')
        assert vals.shape[0] == sim.num_agents
        # sample via geo_to_pixel to compare
        rows, cols = geo_to_pixel(sim.depth_rast_transform, sim.X, sim.Y)
        expected = arr[np.clip(rows, 0, arr.shape[0]-1), np.clip(cols, 0, arr.shape[1]-1)].flatten()
        assert np.allclose(vals, expected)
    finally:
        sim.hdf5.close()
        os.unlink(fname)
