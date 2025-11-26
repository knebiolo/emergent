import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from emergent.ship_abm.ship_model import ship
from emergent.ship_abm.config import ADVANCED_CONTROLLER, PROPULSION


def test_dead_reck_compute_desired():
    psi0 = np.array([np.deg2rad(-116.86)])
    pos0 = np.array([[0.0], [0.0]])
    state0 = np.zeros((4, 1))
    goals = np.array([[500.0], [-200.0]])
    s = ship(state0, pos0, psi0, goals)
    current_vec = np.array([[2.021], [3.612]])
    hd, sp = s.compute_desired(goals, pos0[0,0], pos0[1,0], 0.0, 0.0, 0.0, psi0, current_vec=current_vec)
    assert isinstance(hd, np.ndarray) or isinstance(hd, float)
