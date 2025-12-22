import numpy as np

from emergent.salmon_abm.behavior import behavior


class DummyHDF5(dict):
    def __getitem__(self, key):
        # support keys like 'memory/0'
        if key.startswith('memory/'):
            return self[key]
        return super().__getitem__(key)


class DummySim:
    pass


def test_already_been_here_returns_force_array():
    sim = DummySim()
    sim.num_agents = 2
    sim.X = np.array([0.0, 1.0])
    sim.Y = np.array([0.0, 1.0])
    from affine import Affine
    sim.depth_rast_transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    # create a minimal hdf5-like dict with 'memory/0' and 'memory/1'
    hdf5 = {}
    # small arrays representing last-visit times
    hdf5['memory/0'] = np.zeros((20, 20))
    hdf5['memory/1'] = np.zeros((20, 20))
    sim.hdf5 = hdf5

    sim.num_agents = 2
    beh = behavior(dt=1.0, simulation_object=sim)
    forces = beh.already_been_here(weight=1.0, t=5)

    assert forces.shape == (2, 2)
