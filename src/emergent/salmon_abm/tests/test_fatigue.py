import numpy as np

from emergent.salmon_abm.fatigue import fatigue


class DummySim:
    pass


def test_bl_s_and_bout_distance_and_ttf():
    sim = DummySim()
    sim.num_agents = 2
    sim.length = np.array([100.0, 200.0])  # mm
    sim.prev_X = np.array([0.0, 0.0])
    sim.prev_Y = np.array([0.0, 0.0])
    sim.X = np.array([1.0, 2.0])
    sim.Y = np.array([0.0, 0.0])
    sim.bout_dur = 0.0
    sim.dist_per_bout = np.zeros(2)
    sim.swim_speeds = np.zeros((2, 3))
    sim.sog = np.array([0.5, 0.2])
    sim.x_vel = np.array([0.0, 0.0])
    sim.y_vel = np.array([0.0, 0.0])
    sim.heading = np.array([0.0, 0.0])
    sim.prev_X = np.array([0.0, 0.0])
    sim.prev_Y = np.array([0.0, 0.0])

    f = fatigue(t=0, dt=1.0, simulation_object=sim)

    swim_speeds = f.swim_speeds()
    bls = f.bl_s(swim_speeds)
    assert bls.shape == swim_speeds.shape

    # bout distance should update distance and duration
    f.bout_distance()
    assert sim.bout_dur > 0
    assert np.any(sim.dist_per_bout > 0)

    # time_to_fatigue with CastroSantos should not error
    mask_dict = {'prolonged': np.array([True, False]), 'sprint': np.array([False, True]), 'sustained': np.array([False, False])}
    sim.a_p = -1.0
    sim.b_p = 0.1
    sim.a_s = -2.0
    sim.b_s = 0.2
    ttf = f.time_to_fatigue(swim_speeds, mask_dict, method='CastroSantos')
    assert ttf.shape == swim_speeds.shape
