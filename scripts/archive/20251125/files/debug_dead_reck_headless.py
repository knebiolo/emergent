"""Headless debug: call ship.compute_desired with controlled current/wind vectors and print outputs.
"""
import numpy as np
from emergent.ship_abm.ship_model import ship
from emergent.ship_abm.config import PID_DEBUG

# enable PID_DEBUG at runtime for this test
import emergent.ship_abm.config as cfg
cfg.PID_DEBUG = True

# Minimal ship instance: 1 vessel
state0 = np.zeros((4,1))
pos0 = np.array([[0.0],[0.0]])
psi0 = np.array([0.0])
goals = np.array([[1000.0],[0.0]])  # goal due east
s = ship(state0, pos0, psi0, goals)

# current pushing north (v positive). Use earth-frame ordering [east, north]
current = np.array([[0.0],[0.5]])  # 0.5 m/s north
print('DEBUG: current shape=', current.shape, 'current=', current.tolist())
print('DEBUG: psi0=', psi0, 'pos0=', pos0.tolist())
hd, sp = s.compute_desired(goals, 0.0, 0.0, 5.0, 0.0, 0.0, psi0, current_vec=current)
print('hd (deg)=', np.degrees(hd), 'sp=', sp)
