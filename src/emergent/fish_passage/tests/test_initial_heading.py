import numpy as np
import h5py
import tempfile
import os
from emergent.fish_passage.io import initial_heading, enviro_import, map_hecras_for_agents, HECRASMap


class DummySim:
    pass


def test_initial_heading_with_raster():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 2
    sim.X = np.array([0.2, 0.8])
    sim.Y = np.array([0.3, 0.7])
    from affine import Affine
    sim.vel_dir_rast_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    arr = np.zeros((3, 3), dtype='f4')
    arr[0, 0] = 0.0
    arr[1, 1] = np.pi / 2
    enviro_import(sim, arr, 'vel_dir', transform=sim.vel_dir_rast_transform)
    try:
        h = initial_heading(sim)
        assert h.shape[0] == sim.num_agents
    finally:
        sim.hdf5.close()
        os.unlink(tmp.name)


def test_initial_heading_hecras_fallback():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 1
    sim.X = np.array([0.5])
    sim.Y = np.array([0.5])
    sim.use_hecras = True
    sim.hecras_plan_path = 'dummy'

    # register a mock adapter on sim._hecras_maps
    class MockAdapter:
        def __init__(self):
            pass
        def map_idw(self, pts, k=8):
            # return a single-field array for vel_x or vel_y names
            # we will rely on map_hecras_for_agents to call this twice with different field_names
            return np.array([1.0])

    sim._hecras_maps = {( 'dummy', ('Velocity X',) ): MockAdapter(), ( 'dummy', ('Velocity Y',) ): MockAdapter()}

    try:
        h = initial_heading(sim)
        assert h.shape[0] == sim.num_agents
        # with vx=1, vy=1, heading should be pi/4
        assert np.isclose(h[0], np.pi/4, atol=1e-6)
    finally:
        sim.hdf5.close()
        os.unlink(tmp.name)


def test_initial_heading_none():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    f = h5py.File(tmp.name, 'w')
    sim = DummySim()
    sim.hdf5 = f
    sim.num_agents = 2
    sim.X = np.array([0.2, 0.8])
    sim.Y = np.array([0.3, 0.7])
    try:
        h = initial_heading(sim)
        assert np.all(np.isnan(h))
    finally:
        sim.hdf5.close()
        os.unlink(tmp.name)
