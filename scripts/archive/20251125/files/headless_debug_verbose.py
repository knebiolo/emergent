"""
Headless runner that enables verbose simulation-level logging for diagnostics.
Writes output to scripts/headless_debug_verbose_log.txt
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config as _config

# Ensure PID tracing disabled here (optional)
try:
    _config.PID_TRACE['enabled'] = False
except Exception:
    pass

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'headless_debug_verbose_log.txt'))
print('Logging to', OUT)

# create sim with ENC disabled and verbose True
sim = simulation(
    port_name='Galveston',
    dt=0.1,
    T=120.0,
    n_agents=1,
    light_bg=True,
    verbose=True,
    load_enc=False
)

# force a constant eastward current of 0.6 m/s
def forced_current(lon, lat, t):
    try:
        lon_arr = np.asarray(lon)
        n = lon_arr.size
        u = np.full((n,), 0.6)
        v = np.zeros((n,))
        return np.column_stack([u, v])
    except Exception:
        return np.array([[0.6, 0.0]])

sim.current_fn = forced_current

# spawn
try:
    sim.spawn()
except Exception as e:
    print('spawn() failed - will attempt to fabricate waypoints and spawn:', e)
    sim.waypoints = [[(0.0, 0.0), (1000.0, 0.0)] for _ in range(sim.n)]
    sim.spawn()

with open(OUT, 'w', buffering=1) as fh:
    steps = int(120.0 / sim.dt)
    for k in range(steps):
        t = k * sim.dt
        hd, sp, rud = sim._compute_controls_and_update(sim.state, t)
        # write some diagnostics
        try:
            fh.write(f"T={t:.2f} goal_hd={np.degrees(hd[0]):.2f} psi={np.degrees(sim.psi[0]):.2f} fused_hd={np.degrees(hd[0]):.2f}\n")
        except Exception:
            pass
        sim._step_dynamics(hd, sp, rud)

print('Done')