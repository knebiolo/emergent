"""Quick smoke test: call ship.compute_desired() with a Rosario-like current
vector and print the dead-reck diagnostics. Run from the repo root:

python scripts/test_dead_reck_power.py
"""

from emergent.ship_abm.ship_model import ship
import numpy as np
from emergent.ship_abm.config import ADVANCED_CONTROLLER, PROPULSION

# Fabricate a single-ship state at origin, heading roughly -117° (as in your log)
psi0 = np.array([np.deg2rad(-116.86)])
pos0 = np.array([[0.0], [0.0]])
state0 = np.zeros((4, 1))

# goals: goal out to x=500, y=-200 (roughly southwest)
goals = np.array([[500.0], [-200.0]])

s = ship(state0, pos0, psi0, goals)
# ensure ship has same tuning as config
s.dead_reck_sensitivity = ADVANCED_CONTROLLER.get('dead_reck_sensitivity', 0.25)
try:
    s.desired_speed = np.array([PROPULSION['desired_speed']])
except Exception:
    s.desired_speed = float(PROPULSION['desired_speed'])

# Use the Rosario-like combined drift vector (E=2.021, N=3.612)
current_vec = np.array([[2.021], [3.612]])

hd, sp = s.compute_desired(goals, pos0[0,0], pos0[1,0], 0.0, 0.0, 0.0, psi0, current_vec=current_vec)
print(f"Returned hd (deg) = {np.degrees(hd)[0]:.3f}, sp = {sp}")
