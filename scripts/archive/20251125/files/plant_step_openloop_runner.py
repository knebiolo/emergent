"""Explicit open-loop plant runner: call control→dynamics loop manually so we capture
the exact rudder argument passed into the dynamics and the instantaneous yaw-rate.

Usage: python scripts/plant_step_openloop_runner.py [duration_s] [rudder_deg]
Saves CSV to scripts/plant_step_openloop.csv
"""
import sys, os
import numpy as np
from emergent.ship_abm.simulation_core import simulation

DT = 0.5
T = float(sys.argv[1]) if len(sys.argv)>1 else 60.0
RUDDER_DEG = float(sys.argv[2]) if len(sys.argv)>2 else 10.0

sim = simulation('Rosario Strait', dt=DT, T=T, n_agents=1, load_enc=False, verbose=False)

# zero environment
def make_zero_field():
    def sampler(lons, lats, when):
        import numpy as _np
        return _np.zeros((len(np.atleast_1d(lons)),2))
    return sampler

sim.wind_fn = make_zero_field()
sim.current_fn = make_zero_field()

# simple short route so spawn works
cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
half_sep = 50.0
start = (float(cx - half_sep), float(cy))
end   = (float(cx + half_sep), float(cy))
sim.waypoints = [[start, end]]
sim.spawn()

# open-loop: we'll pass a constant rudder value each step directly into _step_dynamics
r_const = np.deg2rad(RUDDER_DEG)
steps = int(T / DT)
out = []
for step in range(steps):
    t = step * DT
    # hd/sp placeholders (not used by dynamics here)
    hd = np.array([sim.psi[0]])
    sp = np.array([sim.state[0,0]])
    rud = np.array([r_const])
    # call dynamics directly with our chosen rudder
    sim._step_dynamics(hd, sp, rud)
    # record
    psi_deg = float(np.degrees(sim.psi[0]))
    r_meas_deg_s = float(np.degrees(sim.state[3,0]))
    rud_deg = float(np.degrees(rud[0]))
    out.append((t, psi_deg, r_meas_deg_s, rud_deg))

# save
import csv
out_path = os.path.join('scripts','plant_step_openloop.csv')
with open(out_path,'w',newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t','psi_deg','r_meas_deg_s','rud_deg'])
    for row in out:
        w.writerow(row)

print('[OPENLOOP] Saved', out_path)
