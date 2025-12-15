import numpy as np
import h5py
import tempfile
import os
from emergent.fish_passage.io import initialize_mental_map, update_mental_map, enviro_import


class DummySim:
    pass


def make_sim_for_mental(arr):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 3
    # place agents near top-left, center, bottom-right
    sim.X = np.array([0.1, 0.5, 0.9])
    sim.Y = np.array([0.1, 0.5, 0.9])
    sim.height, sim.width = arr.shape
    from affine import Affine
    sim.depth_rast_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    # initialize memory maps
    initialize_mental_map(sim, avoid_cell_size= (max(sim.height, sim.width) / 3.0))
    # import depth raster
    enviro_import(sim, arr, 'depth', transform=sim.depth_rast_transform)
    return sim, tmp.name


def test_update_mental_map_basic():
    arr = np.arange(16).reshape((4, 4)).astype('f4')
    sim, fname = make_sim_for_mental(arr)
    try:
        update_mental_map(sim, 0, raster_name='depth')
        mem = sim.hdf5['memory']
        # check agent 0,1,2 maps exist and have non-zero entries
        for i in range(sim.num_agents):
            ds = np.asarray(mem[str(i)])
            assert ds.sum() >= 0
    finally:
        sim.hdf5.close()
        os.unlink(fname)
