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


def test_rheotaxis_points_upstream():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 3
    sim.length = np.array([200.0, 200.0, 200.0])

    def sample_environment(transform, raster_name):
        if raster_name == 'vel_x':
            return np.array([2.0, 2.0, 2.0])
        if raster_name == 'vel_y':
            return np.array([0.0, 0.0, 0.0])
        return np.full(sim.num_agents, np.nan)

    sim.sample_environment = sample_environment
    sim.vel_x_rast_transform = None
    sim.vel_y_rast_transform = None
    sim.vel_dir_rast_transform = None

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.rheo_cue(weight=10.0, downstream=False)
    assert vec.shape == (3, 2)
    assert np.allclose(vec[:, 0], -10.0)
    assert np.allclose(vec[:, 1], 0.0)


def test_collision_repulsive_points_away_from_neighbor():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 2
    sim.X = np.array([0.0, 1.0])
    sim.Y = np.array([0.0, 0.0])
    sim.closest_agent = np.array([1.0, 0.0])
    sim.nearest_neighbor_distance = np.array([1.0, 1.0])
    sim.agents_within_buffers = [np.array([1], dtype=int), np.array([0], dtype=int)]

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.collision_cue(weight=5.0)
    assert vec.shape == (2, 2)
    assert np.allclose(vec[0], np.array([-5.0, 0.0]))
    assert np.allclose(vec[1], np.array([5.0, 0.0]))


def test_refugia_attractive_points_toward_refuge():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([0.0])
    sim.Y = np.array([0.0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    sim.refugia_map_transform = sim.depth_rast_transform
    sim.hdf5 = {
        'environment/refugia': np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int8,
        )
    }

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.find_nearest_refuge(weight=10.0)
    assert vec.shape == (1, 2)
    # refuge is to the +x direction (col=2) from (0,0)
    assert vec[0, 0] > 0
