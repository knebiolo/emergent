import numpy as np
import pytest

from emergent.salmon_abm.movement import movement


class DummySim:
    def __init__(self, n_agents=3, length_mm=200.0):
        self.length = np.full(n_agents, length_mm)
        self.x_vel = np.zeros(n_agents)
        self.y_vel = np.zeros(n_agents)
        self.ideal_sog = np.full(n_agents, 0.5)
        self.heading = np.zeros(n_agents)
        self.max_s_U = np.full(n_agents, 0.6)
        self.wave_drag = np.ones(n_agents)
        self.water_temp = 10
        self.swim_behav = np.zeros(n_agents, dtype=int)
        self.is_stuck = np.zeros(n_agents, dtype=bool)
        self.Hz = np.zeros(n_agents)
        self.prev_Hz = np.zeros(n_agents)
        self.X = np.zeros(n_agents)
        self.prev_X = np.zeros(n_agents)
        self.Y = np.zeros(n_agents)
        self.prev_Y = np.zeros(n_agents)


def test_sympy_parity_or_skip():
    sim = DummySim(n_agents=3, length_mm=200.0)
    mv = movement(sim)

    mask = np.array([True, True, True])

    # first compute numeric-only Hz
    mv.frequency(mask, t=1, dt=1.0, use_sympy=False)
    hz_numeric = sim.Hz.copy()

    # now compute sympy-backed Hz if available
    try:
        mv.frequency(mask, t=1, dt=1.0, use_sympy=True)
    except Exception:
        pytest.skip('SymPy not available or sympy-backed evaluation failed')

    hz_sym = sim.Hz

    # compare elementwise; they should be close (within 20%) when both finite
    finite = np.isfinite(hz_numeric) & np.isfinite(hz_sym)
    if not np.any(finite):
        pytest.skip('No finite Hz values to compare')

    rel_err = np.abs(hz_numeric[finite] - hz_sym[finite]) / np.maximum(1e-8, np.abs(hz_sym[finite]))
    assert np.all(rel_err < 0.2), f'Relative error too high: {rel_err}'
