"""Plant step-response rudder test in calm conditions.

Usage: python scripts/plant_step_test.py [duration_s] [rudder_deg]
Saves output CSV to scripts/plant_step_response.csv
"""
import os
import sys
import numpy as np
from emergent.ship_abm.simulation_core import simulation

DT = 0.5
T = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
RUDDER_DEG = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

def make_zero_field():
    def sampler(lons, lats, when):
        import numpy as _np
        return _np.zeros((len(np.atleast_1d(lons)),2))
    return sampler

print(f"[PLANT_TEST] Running plant step test T={T}s, rudder={RUDDER_DEG}°")
sim = simulation('Rosario Strait', dt=DT, T=T, n_agents=1, load_enc=False, verbose=False)
sim.wind_fn = make_zero_field()
sim.current_fn = make_zero_field()
# create a simple short route (250m each side) so spawn() can initialize ship
cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
half_sep = 50.0
start = (float(cx - half_sep), float(cy))
end   = (float(cx + half_sep), float(cy))
sim.waypoints = [[start, end]]
sim.spawn()

# Set a constant rudder command visible to the ship (simulate open-loop step)
sim.tc_rudder_deg = RUDDER_DEG
sim.ship.constant_rudder_cmd = np.deg2rad(RUDDER_DEG)

# Run simulation
# Monkey-patch the control loop to force open-loop rudder commands so the plant
# sees the constant rudder directly (disable PID/controller fusion for this test).
def _open_loop_compute(self, nu, t):
    # use current headings as hd_cmds (so heading error ≈ 0) and keep speed steady
    try:
        hd_cmds = self.psi.copy()
    except Exception:
        hd_cmds = np.zeros(self.n)
    try:
        sp_cmds = np.full(self.n, float(self.state[0, 0]))
    except Exception:
        sp_cmds = np.zeros(self.n)
    rud_cmds = np.full(self.n, getattr(self.ship, 'constant_rudder_cmd', np.deg2rad(RUDDER_DEG)))
    return hd_cmds, sp_cmds, rud_cmds

import types
sim._compute_controls_and_update = types.MethodType(_open_loop_compute, sim)

sim.run()

# collect time, psi, r_meas, rudder
out = []
N = len(sim.t_history)
for i in range(N):
    t = sim.t_history[i]
    psi = sim.psi_history[i] if len(sim.psi_history) > i else np.nan
    r_meas = sim.r_meas_history[i] if hasattr(sim, 'r_meas_history') and len(sim.r_meas_history) > i else np.nan
    # smoothed_rudder may be scalar or array; try to sample per-step
    try:
        rud_arr = sim.ship.smoothed_rudder
        if hasattr(rud_arr, '__len__'):
            # if per-step array saved, use that
            rud = np.degrees(rud_arr[i]) if len(rud_arr) > i else np.degrees(rud_arr[-1])
        else:
            rud = np.degrees(float(rud_arr))
    except Exception:
        rud = np.nan
    out.append((t, psi, r_meas, rud))

import csv
out_path = os.path.join('scripts','plant_step_response.csv')
with open(out_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t','psi_deg','r_meas_deg_s','rud_deg'])
    for row in out:
        w.writerow(row)

print(f'[PLANT_TEST] Saved: {out_path}')
