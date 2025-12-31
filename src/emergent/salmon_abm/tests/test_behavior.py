import numpy as np

from emergent.salmon_abm.behavior import behavior
from emergent.salmon_abm import utils


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


def test_alignment_attractive_points_in_neighbor_heading_direction():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 2
    sim.X = np.array([0.0, 1.0])
    sim.Y = np.array([0.0, 0.0])
    sim.heading = np.array([0.0, np.pi / 2])  # agent0 east, agent1 north
    sim.x_vel = np.zeros(2)
    sim.y_vel = np.zeros(2)
    sim.sog = np.array([1.0, 1.0])
    sim.length = np.array([200.0, 200.0])
    sim.agents_within_buffers = [np.array([1], dtype=int), np.array([0], dtype=int)]
    sim.debug_behavior = False

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.alignment_cue(weight=5.0)
    assert vec.shape == (2, 2)
    # agent0 aligns to neighbor1 (north)
    assert vec[0, 1] > 0 and abs(vec[0, 0]) < 1e-6
    # agent1 aligns to neighbor0 (east)
    assert vec[1, 0] > 0 and abs(vec[1, 1]) < 1e-6


def test_cohesion_attractive_points_toward_neighbors():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 2
    sim.X = np.array([0.0, 2.0])
    sim.Y = np.array([0.0, 0.0])
    sim.x_vel = np.zeros(2)
    sim.y_vel = np.zeros(2)
    sim.agents_within_buffers = [np.array([1], dtype=int), np.array([0], dtype=int)]

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.cohesion_cue(weight=5.0)
    assert vec.shape == (2, 2)
    assert vec[0, 0] > 0 and abs(vec[0, 1]) < 1e-6
    assert vec[1, 0] < 0 and abs(vec[1, 1]) < 1e-6


def _make_identity_grids(nrows=7, ncols=7):
    cols = np.arange(ncols, dtype=float)
    rows = np.arange(nrows, dtype=float)
    col_grid, row_grid = np.meshgrid(cols, rows)
    x_coords = col_grid.copy()
    y_coords = row_grid.copy()
    return x_coords, y_coords


def test_border_repulsive_points_toward_interior():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([1.0])
    sim.Y = np.array([3.0])
    sim.heading = np.array([0.0])  # facing +x
    sim.length = np.array([200.0])
    sim.in_eddy = np.array([0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    sim.vel_mag_rast_transform = sim.depth_rast_transform

    # distance_to increases with x (further from left boundary)
    dist = np.tile(np.arange(7, dtype=float)[None, :], (7, 1))
    x_coords, y_coords = _make_identity_grids(7, 7)
    sim.hdf5 = {
        'environment/distance_to': dist,
        'environment/x_coords': x_coords,
        'environment/y_coords': y_coords,
    }

    def sample_environment(transform, raster_name):
        if raster_name != 'distance_to':
            return np.full(sim.num_agents, np.nan)
        rows, cols = utils.geo_to_pixel(sim.X, sim.Y, transform)
        rows = np.atleast_1d(rows).astype(int)
        cols = np.atleast_1d(cols).astype(int)
        return np.array([dist[rows[0], cols[0]]], dtype=float)

    sim.sample_environment = sample_environment

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.border_cue(weight=10.0, t=0.0)
    assert vec.shape == (1, 2)
    # should push toward increasing distance_to (positive x direction)
    assert vec[0, 0] > 0


def test_shallow_repulsive_points_away_from_shallow_cell_ahead():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([3.0])
    sim.Y = np.array([3.0])
    sim.heading = np.array([0.0])  # facing +x
    sim.too_shallow = np.array([1.0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    x_coords, y_coords = _make_identity_grids(7, 7)

    depth = np.full((7, 7), 10.0, dtype=float)
    # make a shallow cell ahead (east) of agent
    depth[3, 4] = 0.5

    sim.hdf5 = {
        'environment/depth': depth,
        'environment/x_coords': x_coords,
        'environment/y_coords': y_coords,
    }

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.shallow_cue(weight=10.0)
    assert vec.shape == (1, 2)
    # shallow cell is to the +x direction; repulsive vector should point -x
    assert vec[0, 0] < 0


def test_avoid_repulsive_points_away_from_recent_memory():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([12.0])
    sim.Y = np.array([12.0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    sim.mental_map_transform = sim.depth_rast_transform

    mmap = np.full((25, 25), np.nan, dtype=float)
    # mark a recently visited cell ahead (+x / col+1)
    t = 1000.0
    mmap[12, 13] = t - 100.0
    sim.hdf5 = {'memory/0': mmap}

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.already_been_here(weight=10.0, t=t)
    assert vec.shape == (1, 2)
    # visited cell is to the +x direction; repulsive should point -x
    assert vec[0, 0] < 0


def test_avoid_sparse_repulsive_points_away_from_recent_history():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([12.0])
    sim.Y = np.array([12.0])
    sim.use_sparse_avoid_memory = True
    sim.avoid_memory_horizon_s = 7200.0
    sim.avoid_history_chunk = 8
    sim.mental_map_transform = (1, 0, 0, 0, 1, 0)

    t = 1000.0
    sim.avoid_hist_rows = np.array([[12]], dtype=np.int16)
    sim.avoid_hist_cols = np.array([[13]], dtype=np.int16)
    sim.avoid_hist_t = np.array([[t - 100.0]], dtype=np.float32)
    sim.avoid_hist_pos = np.array([1], dtype=np.int32)

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.already_been_here(weight=10.0, t=t)
    assert vec.shape == (1, 2)
    # visited cell is to the +x direction; repulsive should point -x
    assert vec[0, 0] < 0


def test_low_speed_attractive_points_toward_low_velocity_cell_ahead():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([3.0])
    sim.Y = np.array([3.0])
    sim.heading = np.array([0.0])  # facing +x
    sim.length = np.array([200.0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    sim.vel_mag_rast_transform = sim.depth_rast_transform
    x_coords, y_coords = _make_identity_grids(7, 7)

    vel = np.full((7, 7), 5.0, dtype=float)
    # put a low-velocity cell ahead (+x)
    vel[3, 4] = 0.1

    sim.hdf5 = {
        'environment/vel_mag': vel,
        'environment/x_coords': x_coords,
        'environment/y_coords': y_coords,
    }

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.vel_cue(weight=10.0)
    assert vec.shape == (1, 2)
    assert vec[0, 0] > 0


def test_wave_drag_attractive_points_toward_optimal_depth_cell_ahead():
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.X = np.array([3.0])
    sim.Y = np.array([3.0])
    sim.heading = np.array([0.0])  # facing +x
    sim.opt_wat_depth = np.array([2.0])
    sim.depth_rast_transform = (1, 0, 0, 0, 1, 0)
    sim.vel_mag_rast_transform = sim.depth_rast_transform
    x_coords, y_coords = _make_identity_grids(7, 7)

    depth = np.full((7, 7), 10.0, dtype=float)
    # optimal depth cell ahead (+x)
    depth[3, 4] = 2.0

    sim.hdf5 = {
        'environment/depth': depth,
        'environment/x_coords': x_coords,
        'environment/y_coords': y_coords,
    }

    beh = behavior(dt=1.0, simulation_object=sim)
    vec = beh.wave_drag_cue(weight=10.0)
    assert vec.shape == (1, 2)
    assert vec[0, 0] > 0
