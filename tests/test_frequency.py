import numpy as np
from emergent.salmon_abm.movement import movement


class DummySim:
    def __init__(self, n):
        self.num_agents = n
        self.length = np.ones(n) * 475.0  # mm
        self.x_vel = np.zeros(n)
        self.y_vel = np.zeros(n)
        self.ideal_sog = np.ones(n) * 0.5
        self.heading = np.zeros(n)
        self.X = np.zeros(n)
        self.Y = np.zeros(n)
        self.prev_X = np.zeros(n)
        self.prev_Y = np.zeros(n)
        self.swim_behav = np.ones(n, dtype=int)
        self.is_stuck = np.zeros(n, dtype=bool)
        self.max_s_U = np.ones(n) * 1.0
        self.wave_drag = np.ones(n)
        self.water_temp = 10.0
        self.weight = np.ones(n) * 1.661164
        self.sog = np.ones(n) * 0.5
        self.prev_Hz = np.zeros(n)

    def drag_coeff(self, reynolds):
        # simple constant drag coefficient for tests
        try:
            return np.ones_like(reynolds) * 0.1
        except Exception:
            return 0.1


def test_frequency_positive_ratio():
    sim = DummySim(3)
    mv = movement(sim)
    # emulate fish velocity smaller than Webb V so V>U
    sim.X = np.array([0.0, 0.0, 0.0])
    sim.prev_X = np.array([0.0, 0.0, 0.0])
    sim.x_vel = np.zeros(3)
    sim.y_vel = np.zeros(3)
    mask = np.array([True, True, True])
    # call frequency with t=0 (uses ideal_sog)
    mv.frequency(mask, t=0, dt=1.0)
    Hz = sim.Hz
    assert np.all(np.isfinite(Hz))
    assert np.all(Hz >= 0)


def test_frequency_zero_power():
    sim = DummySim(2)
    mv = movement(sim)
    mask = np.array([True, True])
    # force drag power to zero by zeroing rel velocities and drags
    sim.ideal_sog = np.zeros(2)
    mv.frequency(mask, t=0, dt=1.0)
    Hz = sim.Hz
    # when num_si == 0 we set Hz == 0 (unless swim_behav == 3)
    assert np.all(Hz == 0.0)


def test_frequency_minHz_and_stuck():
    sim = DummySim(2)
    mv = movement(sim)
    mask = np.array([True, True])
    # set swim_behav=3 for first agent (minHz), second agent stuck
    sim.swim_behav = np.array([3, 1])
    sim.is_stuck = np.array([False, True])
    mv.frequency(mask, t=0, dt=1.0)
    Hz = sim.Hz
    # first agent should be min_Hz > 0, second should be 0
    assert Hz[0] > 0
    assert Hz[1] == 0.0
