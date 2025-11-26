"""
Long headless runner with a forced lateral current (northward) to induce heading corrections.
Writes PID internals to scripts/pid_trace_force_long.csv
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from emergent.ship_abm.simulation_core import simulation
# Enable PID internals CSV tracing for this headless run
try:
    from emergent.ship_abm import config as _config
    pid_out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pid_trace_force_long.csv'))
    _config.PID_TRACE['enabled'] = True
    _config.PID_TRACE['path'] = pid_out
    print(f"PID_TRACE enabled → {pid_out}")
except Exception as _e:
    print(f"Could not enable PID_TRACE: {_e}")

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'headless_force_long_log.txt'))
print('Logging to', OUT)

# create sim with ENC disabled and minimal nuisance
sim = simulation(
    port_name='Galveston',
    dt=0.1,
    T=180.0,
    n_agents=1,
    light_bg=True,
    verbose=False,
    load_enc=False
)

# force a constant northward current of 0.8 m/s
def forced_current(lon, lat, t):
    try:
        lon_arr = np.asarray(lon)
        n = lon_arr.size
        u = np.zeros((n,))
        v = np.full((n,), 0.8)
        return np.column_stack([u, v])
    except Exception:
        return np.array([[0.0, 0.8]])

sim.current_fn = forced_current

# spawn to create ship and routes for the agents
try:
    sim.spawn()
except Exception as e:
    print('spawn() failed - will attempt to fabricate waypoints and spawn:', e)
    sim.waypoints = [[(0.0, 0.0), (1000.0, 0.0)] for _ in range(sim.n)]
    sim.spawn()

# run simulation loop and capture debug prints
steps = int(180.0 / sim.dt)
with open(OUT, 'w', buffering=1) as fh:
    for k in range(steps):
        t = k * sim.dt
        hd, sp, rud = sim._compute_controls_and_update(sim.state, t)
        sim._step_dynamics(hd, sp, rud)
        # write DR/CTRL lines
        psi = sim.psi[0]
        pos = sim.pos[:,0]
        try:
            goal_hd, goal_sp = sim.ship.compute_desired(
                sim.goals,
                sim.pos[0, 0],
                sim.pos[1, 0],
                sim.state[0, 0],
                sim.state[1, 0],
                sim.state[3, 0],
                sim.psi[0],
                sim.current_fn(None, None, None).T[0]
            )
            dr_line = f"[DR] t={t:.2f} hd_goal={np.degrees(float(np.asarray(goal_hd).ravel()[0])):.2f}°"
        except Exception as e:
            dr_line = f"[DR] t={t:.2f} compute_desired failed: {e}"
        fh.write(dr_line + '\n')
        # controller line
        err = ((hd - sim.psi + 3.141592653589793) % (2 * 3.141592653589793)) - 3.141592653589793
        hd_cmd_deg = float(np.degrees(hd[0]))
        hd_cur_deg = float(np.degrees(sim.psi[0]))
        err_deg = float(np.degrees(err[0]))
        rud_rad = float(rud[0])
        rud_deg = np.degrees(rud_rad)
        ctrl_line = (
            f"[CTRL] t={t:.2f} -> hd_cmd={hd_cmd_deg:.1f}°, hd_cur={hd_cur_deg:.1f}°, "
            f"err={err_deg:.1f}°, rud={rud_rad:.4f}rad ({rud_deg:.2f}°)"
        )
        fh.write(ctrl_line + '\n')

print('Done')
