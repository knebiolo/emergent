import numpy as np
import h5py
import tempfile
import os
from emergent.fish_passage.io import initialize_refugia_map, update_refugia_map, enviro_import


class DummySim:
    pass


def make_sim_for_refugia(arr):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 2
    sim.X = np.array([0.2, 0.8])
    sim.Y = np.array([0.3, 0.7])
    sim.height, sim.width = arr.shape
    from affine import Affine
    sim.depth_rast_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    initialize_refugia_map(sim, refugia_cell_size=(max(sim.height, sim.width) / 3.0))
    enviro_import(sim, arr, 'depth', transform=sim.depth_rast_transform)
    return sim, tmp.name


def test_update_refugia_map_basic():
    arr = np.arange(9).reshape((3, 3)).astype('f4')
    sim, fname = make_sim_for_refugia(arr)
    try:
        update_refugia_map(sim, current_velocity=None, raster_name='depth')
        ref = sim.hdf5['refugia']
        for i in range(sim.num_agents):
            ds = np.asarray(ref[str(i)])
            assert ds.sum() >= 0
    finally:
        sim.hdf5.close()
        os.unlink(fname)
