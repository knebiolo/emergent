import pytest

pytestmark = pytest.mark.slow

import sys, os
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from emergent.ship_abm.simulation_core import simulation as Simulation
from emergent.ship_abm.ship_model import ship as ShipModel
from emergent.ship_abm.config import SIMULATION_BOUNDS


def test_pid_rudder_basic():
    sim = Simulation('Galveston', dt=0.1, T=1.0, n_agents=1, test_mode=None, load_enc=False, light_bg=True, verbose=False)
    sim.psi[:] = np.deg2rad(-35.6)
    sim.prev_psi[:] = sim.psi - 0.001
    sim.ship = ShipModel(sim.state, sim.pos, sim.psi, sim.goals)
    hd_cmd = np.deg2rad(-39.3)
    rud = sim._compute_rudder(np.array([hd_cmd]), roles=['normal'])
    assert rud.shape[0] == 1
