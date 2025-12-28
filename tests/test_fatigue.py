import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.fatigue import fatigue


def make_minimal_sim(num_agents=2):
    sim = simulation(model_dir='outputs/diagnostics', model_name='test_fatigue', crs=None, basin='nuyakuk',
                     water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None,
                     num_timesteps=1, num_agents=num_agents, db_path='outputs/diagnostics/test_fatigue.h5')
    return sim


def test_swim_speeds_and_bl_s():
    sim = make_minimal_sim(2)
    # set headings and sog so fish velocity is simple
    sim.sog[:] = np.array([1.0, 2.0])
    sim.heading[:] = np.array([0.0, 0.0])  # cos=1,sin=0
    # set water velocities so relative speeds are known
    sim.x_vel[:] = np.array([0.0, 1.0])
    sim.y_vel[:] = np.array([0.0, 0.0])
    f = fatigue(0, 1.0, sim)
    speeds = f.swim_speeds()
    # fish velocities are [1,2] in x-direction; relative to water: [1,1]
    assert np.allclose(speeds, np.array([1.0, 1.0]))
    bl = f.bl_s(speeds)
    # default lengths set by agents.sim_length produce positive numbers; ensure shape and sign
    assert bl.shape == speeds.shape
    assert np.all(bl >= 0)
    sim.close()


def test_time_to_fatigue_castrosantos():
    sim = make_minimal_sim(3)
    # set regression params on sim
    sim.a_p = np.repeat(0.0, sim.num_agents)
    sim.b_p = np.repeat(-0.1, sim.num_agents)
    sim.a_s = np.repeat(0.0, sim.num_agents)
    sim.b_s = np.repeat(-0.2, sim.num_agents)
    sim.length = np.repeat(100.0, sim.num_agents)
    f = fatigue(0, 1.0, sim)
    # swim speeds for three agents
    swim = np.array([0.5, 1.0, 2.0])
    mask = {'prolonged': np.array([True, False, False]), 'sprint': np.array([False, True, False]), 'sustained': np.array([False, False, True])}
    ttf = f.time_to_fatigue(swim, mask, method='CastroSantos')
    # prolonged agent ttf = exp(a_p + swim*b_p) ; sprint uses a_s,b_s
    expected0 = np.exp(sim.a_p[0] + swim[0] * sim.b_p[0])
    expected1 = np.exp(sim.a_s[1] + swim[1] * sim.b_s[1])
    assert np.isclose(ttf[0], expected0)
    assert np.isclose(ttf[1], expected1)
    sim.close()


def test_calc_battery_reduction_and_clip():
    sim = make_minimal_sim(2)
    # start with battery=1.0 for both
    sim.battery[:] = np.array([1.0, 1.0])
    f = fatigue(0, 1.0, sim)
    # per_rec small recovery
    per_rec = np.array([0.0, 0.0])
    # ttf: agent0 is sustained -> will be in mask_sustained; agent1 non-sustained
    ttf = np.array([10.0, 5.0])
    mask_dict = {'sustained': np.array([True, False])}
    # call calc_battery which should reduce battery for non-sustained agent
    f.calc_battery(per_rec, ttf, {'sustained': mask_dict['sustained']})
    # battery for agent0 should remain ~1.0, agent1 should be <= 1.0
    assert sim.battery[0] >= sim.battery[1]
    assert np.all(sim.battery >= 0) and np.all(sim.battery <= 1)
    sim.close()
