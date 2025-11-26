"""
Headless runner: instantiate simulation without ENC, run for a short time, and capture the DR/CTRL prints to a file.
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
    pid_out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pid_trace_simulation.csv'))
    _config.PID_TRACE['enabled'] = True
    _config.PID_TRACE['path'] = pid_out
    print(f"PID_TRACE enabled → {pid_out}")
except Exception as _e:
    print(f"Could not enable PID_TRACE: {_e}")

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'headless_run_log.txt'))
print('Logging to', OUT)

# create sim with ENC disabled and minimal nuisance
sim = simulation(
    port_name='Galveston',
    dt=0.1,
    T=600.0,
    n_agents=1,
    light_bg=True,
    verbose=False,
    load_enc=False
)

# spawn to create ship and routes for the agents
try:
    sim.spawn()
except Exception as e:
    print('spawn() failed - will attempt to fabricate waypoints and spawn:', e)
    # fabricate a trivial straight route if spawn requires UI
    sim.waypoints = [[(0.0, 0.0), (1000.0, 0.0)] for _ in range(sim.n)]
    sim.spawn()

# run simulation loop and capture the existing printouts by temporarily redirecting stdout
import io, contextlib
buf = []

# We'll run for 300 simulated seconds (3000 steps at dt=0.1)
steps = int(300.0 / sim.dt)

with open(OUT, 'w', buffering=1) as fh:
    for k in range(steps):
        t = k * sim.dt
        # advance one step: call the top-level step function used by viewer
        # The public loop in simulation uses run()/Qt; here we use internal methods
        hd, sp, rud = sim._compute_controls_and_update(sim.state, t)
        # step dynamics is performed inside viewer loop; call _step_dynamics explicitly
        sim._step_dynamics(hd, sp, rud)
        # write any debug prints we normally see by emitting DR/CTRL lines ourselves
        # For fairness, mimic existing debug formatting
        psi = sim.psi[0]
        pos = sim.pos[:,0]
        # We need wind/current sample at pos; simulation has wind_fn/current_fn as methods
        lon, lat = sim._utm_to_ll.transform(pos[0], pos[1])
        wind_vec = sim.wind_fn(lon, lat, None).T[0]
        current_vec = sim.current_fn(lon, lat, None).T[0]
        # Dead-reck diagnostics are printed in compute_desired normally; mimic one line per step
        # Use available compute_desired call to get hd_base and hd_final if possible
        try:
            # compute_desired may return numpy scalars/arrays; coerce to Python floats for safe formatting
            goal_hd, goal_sp = sim.ship.compute_desired(
                sim.goals,
                sim.pos[0, 0],
                sim.pos[1, 0],
                sim.state[0, 0],
                sim.state[1, 0],
                sim.state[3, 0],
                sim.psi[0],
                current_vec,
            )
            goal_hd_val = float(np.asarray(goal_hd).ravel()[0])
            goal_sp_val = float(np.asarray(goal_sp).ravel()[0])
            wind_mag = float(np.linalg.norm(wind_vec))
            cur_mag = float(np.linalg.norm(current_vec))
            dr_line = (
                f"[DR] t={t:.2f} pos=[{pos[0]:.3f} {pos[1]:.3f}]; "
                f"wind={wind_mag:.2f}m/s; current={cur_mag:.2f}m/s; "
                f"hd_goal={np.degrees(goal_hd_val):.2f}°; sp_goal={goal_sp_val:.2f}"
            )
        except Exception as e:
            dr_line = f"[DR] t={t:.2f} pos=[{pos[0]:.3f} {pos[1]:.3f}]; wind=...; current=... (compute_desired failed: {e})"
        fh.write(dr_line + '\n')
        # Controller line
        err = ((hd - sim.psi + 3.141592653589793) % (2 * 3.141592653589793)) - 3.141592653589793
        # report heading command/current/error in degrees and rudder in degrees (easy to read)
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
