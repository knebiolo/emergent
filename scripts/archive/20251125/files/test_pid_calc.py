"""
Quick test to compute rudder output for a single-agent state matching the user's paste.
This script constructs a Simulation object for a tiny area, sets state variables to
match the provided sample, and calls _compute_rudder to print internal terms.
"""
from datetime import datetime
import numpy as np
import os
import sys
# ensure package import works when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from emergent.ship_abm.simulation_core import simulation as Simulation
from emergent.ship_abm.config import SIMULATION_BOUNDS

# pick a small bbox (Galveston or any) - Simulation requires bounds
bbox = SIMULATION_BOUNDS.get('Galveston', {'minx':-95.4,'miny':29.2,'maxx':-94.8,'maxy':29.6})
minx, miny, maxx, maxy = bbox['minx'], bbox['miny'], bbox['maxx'], bbox['maxy']

print('About to instantiate Simulation (no ENC)')
sim = Simulation('Galveston', dt=0.1, T=200.0, n_agents=1, test_mode=None, load_enc=False, light_bg=True, verbose=False)
print('Simulation instantiated')
# set psi (heading) ~ -35.5 degrees (hd_cur in log)
sim.psi[:] = np.deg2rad(-35.6)
# set previous psi to reflect r_meas small
sim.prev_psi[:] = sim.psi - 0.001
# ensure sim.ship exists by creating a minimal ship instance
from emergent.ship_abm.ship_model import ship as ShipModel
sim.ship = ShipModel(sim.state, sim.pos, sim.psi, sim.goals)
# desired heading (hd_cmd) from logs: -39.3 deg
hd_cmd = np.deg2rad(-39.3)
# call internal _compute_rudder with single-agent array
print(f"psi (deg) = {np.degrees(sim.psi[0]):.2f}")
print(f"hd_cmd (deg) = {np.degrees(hd_cmd):.2f}")
print("\nSweep Kd (D-gain) and deriv_tau (D low-pass) → rudder (deg):")
print("Kd\tderiv_tau\trudder_deg")
for kd in [3.0, 1.0, 0.5, 0.2, 0.0]:
    for tau in [0.1, 1.0, 5.0]:
        sim.tuning['Kd'] = kd
        sim.tuning['deriv_tau'] = tau
        # reset filtered r for consistent comparison
        if hasattr(sim, '_r_filtered'):
            sim._r_filtered = np.array([0.0])
        rud = sim._compute_rudder(np.array([hd_cmd]), roles=['normal'])
        print(f"{kd}\t{tau}\t{np.degrees(rud[0]):.3f}")

print(f"\nmax_rudder (deg) = {np.degrees(sim.ship.max_rudder):.1f}")
