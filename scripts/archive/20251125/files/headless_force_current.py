"""
Headless runner that forces a constant eastward current so the PID sees a heading error.
Writes PID internals to scripts/pid_trace_forced.csv
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from emergent.ship_abm.simulation_core import simulation

# Enable PID trace to a separate file
from emergent.ship_abm import config as _config
pid_out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pid_trace_forced.csv'))
_config.PID_TRACE['enabled'] = True
_config.PID_TRACE['path'] = pid_out
print(f"PID_TRACE enabled -> {pid_out}")

# Create sim
sim = simulation(port_name='Galveston', dt=0.1, T=60.0, n_agents=1, light_bg=True, verbose=False, load_enc=False)
# Enable zigzag test mode to force non-zero heading commands so PID is exercised
sim.test_mode = 'zigzag'
sim.zz_delta = np.radians(10.0)  # 10° deflection
sim.zz_hold = 5.0
sim.zz_base_psi = 0.0
# ensure zz_next_sw exists (simulation.__init__ only sets it when test_mode passed)
if not hasattr(sim, 'zz_next_sw'):
    sim.zz_next_sw = sim.zz_hold
if not hasattr(sim, 'zz_sign'):
    sim.zz_sign = 1
if not hasattr(sim, 'zz_sp_cmd'):
    # set zigzag commanded speed to the ship's desired speed
    sim.zz_sp_cmd = float(sim.ship.desired_speed[0]) if getattr(sim, 'ship', None) is not None else 5.0

# spawn
try:
    sim.spawn()
except Exception as e:
    sim.waypoints = [[(0.0, 0.0), (1000.0, 0.0)]]
    sim.spawn()

# Force a constant eastward current of 0.5 m/s
def const_current(lons, lats, when):
    lons = np.atleast_1d(lons)
    N = lons.size
    # Force a northward (positive v) current so it has a perpendicular
    # component to an eastward route and provokes heading corrections.
    u = np.zeros((N,))
    v = np.full((N,), 0.5)
    return np.column_stack((u, v))

sim.current_fn = const_current

# run loop
steps = int(60.0 / sim.dt)
for k in range(steps):
    t = k * sim.dt
    hd, sp, rud = sim._compute_controls_and_update(sim.state, t)
    sim._step_dynamics(hd, sp, rud)

print('run complete')

# show top of CSV
if os.path.exists(pid_out):
    with open(pid_out, 'r') as fh:
        for i, line in enumerate(fh):
            print(line.strip())
            if i > 40:
                break
else:
    print('PID CSV missing')
