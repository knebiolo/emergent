import numpy as np
from emergent.salmon_abm import movement as new_movement, behavior as new_behavior, fatigue as new_fatigue
import emergent.salmon_abm.sockeye as old_sockeye


def make_dummy_sim(num_agents=2):
    class Dummy:
        pass
    sim = Dummy()
    sim.num_agents = num_agents
    sim.X = np.array([0.0, 1.0])
    sim.Y = np.array([0.0, 1.0])
    sim.prev_X = sim.X.copy()
    sim.prev_Y = sim.Y.copy()
    sim.length = np.array([200.0, 200.0])
    sim.weight = np.array([0.3, 0.3])
    sim.body_depth = np.array([20.0, 20.0])
    sim.too_shallow = np.array([0.1, 0.1])
    sim.Hz = np.array([1.0, 1.0])
    sim.heading = np.array([0.0, 0.0])
    sim.ideal_sog = np.array([0.2, 0.2])
    sim.sog = np.array([0.2, 0.2])
    sim.x_vel = np.array([0.0, 0.0])
    sim.y_vel = np.array([0.0, 0.0])
    sim.wave_drag = np.array([1.0, 1.0])
    sim.swim_behav = np.array([1, 1])
    sim.max_s_U = 2.0
    sim.max_p_U = 4.0
    sim.a_p = 0.0
    sim.b_p = -1.0
    sim.a_s = 0.0
    sim.b_s = -1.0
    sim.battery = np.array([1.0, 1.0])
    sim.recover_stopwatch = np.array([0.0, 0.0])
    sim.past_longitudes = np.zeros((num_agents, 5))
    sim.swim_speeds = np.zeros((num_agents, 5))
    sim.in_eddy = np.zeros(num_agents, dtype=bool)
    sim.time_since_eddy_escape = np.zeros(num_agents)
    sim.max_eddy_escape_seconds = 1000
    sim.longitudinal = np.zeros(num_agents)
    sim.opt_wat_depth = np.array([0.5, 0.5])
    sim.water_temp = np.array([10.0, 10.0])
    from rasterio.transform import Affine
    sim.vel_mag_rast_transform = Affine(1,0,0,0,1,0)
    sim.depth_rast_transform = Affine(1,0,0,0,1,0)
    sim.refugia_map_transform = Affine(1,0,0,0,1,0)
    # minimal hdf5-like dict
    sim.hdf5 = {}
    sim.hdf5['memory/0'] = np.zeros((10,10))
    sim.hdf5['memory/1'] = np.zeros((10,10))
    sim.hdf5['refugia/0'] = np.zeros((10,10))
    sim.hdf5['refugia/1'] = np.zeros((10,10))
    sim.hdf5['environment/depth'] = np.zeros((10,10))
    sim.hdf5['x_coords'] = np.indices((10,10))[1]
    sim.hdf5['y_coords'] = np.indices((10,10))[0]
    sim.hdf5['environment/vel_mag'] = np.ones((10,10))
    sim.hdf5['environment/distance_to'] = np.ones((10,10))
    return sim


def test_parity_basic():
    sim = make_dummy_sim()

    # old implementations (call monolithic functions as methods from sockeye's classes if available)
    # monolithic sockeye defines these as nested classes under simulation
    old_mov = old_sockeye.simulation.movement(sim)
    old_beh = old_sockeye.simulation.behavior(1.0, sim)
    old_fat = old_sockeye.simulation.fatigue(1.0, 1.0, sim)

    # new implementations
    new_mov = new_movement.movement(sim)
    new_beh = new_behavior.behavior(1.0, sim)
    new_fat = new_fatigue.fatigue(1.0, 1.0, sim)

    mask = np.array([True, True])
    # thrust
    old_mov.thrust_fun(mask, 0, 1.0)
    old_thrust = sim.thrust.copy()
    new_mov.thrust_fun(mask, 0, 1.0)
    new_thrust = sim.thrust.copy()
    assert np.allclose(old_thrust, new_thrust)

    # drag
    old_mov.drag_fun(mask, 0, 1.0)
    old_drag = sim.drag.copy()
    new_mov.drag_fun(mask, 0, 1.0)
    new_drag = sim.drag.copy()
    assert np.allclose(old_drag, new_drag)

    # swim speeds
    old_ss = old_fat.swim_speeds()
    new_ss = new_fat.swim_speeds()
    assert np.allclose(old_ss, new_ss)

    # behavior already_been_here
    old_rep = old_beh.already_been_here(25000, 0)
    new_rep = new_beh.already_been_here(25000, 0)
    assert np.allclose(old_rep, new_rep)
