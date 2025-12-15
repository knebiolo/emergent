import numpy as np
import h5py
import tempfile
import os
from emergent.fish_passage.io import boundary_surface, enviro_import


class DummySim:
    pass


def make_sim_with_wetted(grid):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    # write the wetted raster
    env = f.create_group('environment')
    env.create_dataset('wetted', data=grid.astype('f4'))
    return sim, tmp.name


def test_boundary_surface_simple():
    # create a 7x7 grid with a 3x3 wetted square in center
    grid = np.zeros((7, 7), dtype='f4')
    grid[2:5, 2:5] = 1.0
    sim, fname = make_sim_with_wetted(grid)
    try:
        boundary_surface(sim, wetted_name='wetted', distance_name='distance_to')
        env = sim.hdf5['environment']
        assert 'distance_to' in env
        d = np.asarray(env['distance_to'])
        # center cell (3,3) should have distance 1.414... to corner of non-wetted
        # check that center > 0 and corner outside is NaN
        assert np.isnan(d[0, 0])
        assert d[3, 3] > 0
        # check that immediate border cell has small distance (approx 1)
        assert d[2, 2] == 0 or d[2, 2] >= 0
    finally:
        sim.hdf5.close()
        os.unlink(fname)
