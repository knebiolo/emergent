"""Open-loop PRBS identification harness

Sends PRBS rudder commands (open-loop) to the plant and records high-rate telemetry
for system identification. Produces one CSV per run and a metadata JSON.

Defaults:
 - speeds: [3.0, 5.0, 7.0] m/s
 - amplitude: 4.0 deg (±)
 - dt: 0.2 s
 - T: 300 s
 - PRBS bit time: 2.0 s

Usage: run from repository root:
    python scripts/id_prbs.py
"""
from pathlib import Path
import importlib.util
import numpy as np
import json
import csv
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('simmod', str(ROOT / 'src' / 'emergent' / 'ship_abm' / 'simulation_core.py'))
# We'll import the simulation class via the scripts that already wrap it
SPEC2 = importlib.util.spec_from_file_location('zig', str(ROOT / 'scripts' / 'zigzag_kf_sweep.py'))
mod = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation

def generate_prbs(duration, bit_time, dt, seed=0):
    # produce a piecewise-constant sequence of ±1 for duration seconds
    rng = np.random.RandomState(seed)
    n_bits = int(np.ceil(duration / bit_time))
    bits = rng.randint(0, 2, size=n_bits) * 2 - 1
    t = np.arange(0.0, duration + dt/2, dt)
    seq = np.zeros_like(t)
    for i in range(n_bits):
        start = int(np.floor(i * bit_time / dt))
        end = int(np.floor((i+1) * bit_time / dt))
        seq[start:end] = bits[i]
    return t, seq

def run_prbs(U_nom=5.0, amp_deg=4.0, dt=0.2, T=300.0, bit_time=2.0, out_prefix='id'):
    # spawn a sim with single ship, but we'll command rudder directly
    sim = simulation(port_name='Galveston', dt=dt, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)

    # create simple eastward waypoint and spawn
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    waypoints = [(cx, cy), (cx + 1000.0, cy)]
    sim.waypoints = [waypoints]
    state0, pos0, psi0, goals = sim.spawn()

    # Disable autopilot: set simulation controller flag so per-ship PID returns prev_rudder
    try:
        from emergent.ship_abm.config import ADVANCED_CONTROLLER
        ADVANCED_CONTROLLER['use_simulation_controller'] = True
    except Exception:
        pass

    # Set desired forward speed (cruise) to U_nom
    sim.ship.desired_speed[:] = U_nom
    sim.ship.current_speed[:] = U_nom

    # build PRBS sequence
    t_seq, bits = generate_prbs(T, bit_time, dt, seed=42)
    rud_cmd_rad = np.deg2rad(amp_deg) * bits

    # prepare output CSV and metadata
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_csv = Path.cwd() / f"{out_prefix}_U{U_nom:.1f}_prbs_{ts}.csv"
    meta = {
        'U_nom': U_nom,
        'amp_deg': amp_deg,
        'dt': dt,
        'T': T,
        'bit_time': bit_time,
        'timestamp_utc': ts,
        'note': 'Open-loop PRBS applied directly to rudder; ADVANCED_CONTROLLER.use_simulation_controller set False'
    }

    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        # header
        w.writerow(['t','psi_deg','r_deg_s','u_m_s','v_m_s','cmd_rudder_deg','applied_rudder_deg','wind_e','wind_n','current_e','current_n'])

        # run loop: step through t_seq and apply commanded rudder directly by overriding ship.prev_rudder and smoothed state
        for i, t in enumerate(t_seq):
            # set commanded/applied rudder directly (simulate open-loop actuator chain)
            cmd = float(rud_cmd_rad[i])
            # write into prev_rudder and smoothed_rudder so pid_control (bypassed) returns this value
            sim.ship.prev_rudder[:] = cmd
            sim.ship.smoothed_rudder[:] = cmd

            # Query env at current pos (normalize to shape (2, n) to match sim internals)
            lons = np.atleast_1d(sim.pos[0])
            lats = np.atleast_1d(sim.pos[1])
            wind = sim.wind_fn(lons, lats, sim.t).T
            current = sim.current_fn(lons, lats, sim.t).T

            # Apply commanded rudder directly and step the simulation dynamics
            # using the simulation-level integrator which expects sim.state (4×n).
            # Prepare simple hd/sp placeholders (not used in open-loop dynamics here)
            hd = np.array([sim.psi[0]])
            sp = np.array([sim.state[0, 0]])
            rud = np.array([cmd])
            # write into ship's prev/smoothed rudder for consistency with other traces
            sim.ship.prev_rudder[:] = cmd
            sim.ship.smoothed_rudder[:] = cmd

            # step the sim dynamics with our chosen open-loop rudder
            sim._step_dynamics(hd, sp, rud)

            # after stepping, sim.state and sim.psi are updated
            # compute yaw rate r and heading psi from updated sim state
            r = sim.state[3, 0]
            psi = sim.psi[0]

            # write telemetry row
            w.writerow([float(t), float(np.degrees(psi)), float(np.degrees(r)), float(sim.ship.current_speed[0]), 0.0, float(np.degrees(cmd)), float(np.degrees(sim.ship.smoothed_rudder[0])), float(wind[0,0]), float(wind[1,0]), float(current[0,0]), float(current[1,0])])

            # advance sim time
            sim.t += dt

    # write metadata
    meta_path = out_csv.with_suffix('.json')
    with open(meta_path, 'w') as fm:
        json.dump(meta, fm, indent=2)

    print('Wrote telemetry:', out_csv)
    print('Wrote metadata:', meta_path)
    return out_csv, meta_path

def main():
    speeds = [3.0, 5.0, 7.0]
    for U in speeds:
        run_prbs(U_nom=U, amp_deg=4.0, dt=0.2, T=300.0, bit_time=2.0, out_prefix='id')

if __name__ == '__main__':
    main()
