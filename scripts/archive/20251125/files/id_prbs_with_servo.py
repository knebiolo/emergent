"""PRBS harness that applies PRBS commands but simulates actuator/servo dynamics
(rate-limit + first-order lag) using the ship's configured `rudder_tau` and
`max_rudder_rate`. This produces telemetry with `cmd_rudder_deg` and
`applied_rudder_deg` differing, enabling actuator ID.

Usage: python scripts/id_prbs_with_servo.py
"""
from pathlib import Path
import importlib.util
import numpy as np
import json
import csv
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SPEC2 = importlib.util.spec_from_file_location('zig', str(ROOT / 'scripts' / 'zigzag_kf_sweep.py'))
mod = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation


def generate_prbs(duration, bit_time, dt, seed=0):
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


def run_prbs_with_servo(U_nom=5.0, amp_deg=4.0, dt=0.2, T=300.0, bit_time=2.0, out_prefix='id_servo'):
    sim = simulation(port_name='Galveston', dt=dt, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    waypoints = [(cx, cy), (cx + 1000.0, cy)]
    sim.waypoints = [waypoints]
    state0, pos0, psi0, goals = sim.spawn()

    # desired speed
    sim.ship.desired_speed[:] = U_nom
    sim.ship.current_speed[:] = U_nom

    t_seq, bits = generate_prbs(T, bit_time, dt, seed=42)
    cmd_rad = np.deg2rad(amp_deg) * bits

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_csv = Path.cwd() / f"{out_prefix}_U{U_nom:.1f}_prbs_{ts}.csv"
    meta = {'U_nom': U_nom, 'amp_deg': amp_deg, 'dt': dt, 'T': T, 'bit_time': bit_time, 'timestamp_utc': ts, 'note': 'PRBS with servo dynamics simulated'}

    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t','psi_deg','r_deg_s','u_m_s','v_m_s','cmd_rudder_deg','applied_rudder_deg','wind_e','wind_n','current_e','current_n'])

        for i, t in enumerate(t_seq):
            # commanded rudder (rad)
            cmd = float(cmd_rad[i])
            # rate-limit relative to previous applied
            prev = float(sim.ship.prev_rudder[0])
            max_delta = sim.ship.max_rudder_rate * dt
            cmd_rate_limited = np.clip(cmd, prev - max_delta, prev + max_delta)
            # servo low-pass
            tau = float(sim.ship.rudder_tau)
            alpha = dt / (tau + dt) if (tau + dt) > 0 else 1.0
            applied = (1.0 - alpha) * float(sim.ship.smoothed_rudder[0]) + alpha * cmd_rate_limited
            # write applied into ship for dynamics
            sim.ship.prev_rudder[:] = applied
            sim.ship.smoothed_rudder[:] = applied

            # sample env
            lons = np.atleast_1d(sim.pos[0])
            lats = np.atleast_1d(sim.pos[1])
            wind = sim.wind_fn(lons, lats, sim.t).T
            current = sim.current_fn(lons, lats, sim.t).T

            # step dynamics using 'applied' as rud
            hd = np.array([sim.psi[0]])
            sp = np.array([sim.state[0, 0]])
            rud = np.array([applied])
            sim._step_dynamics(hd, sp, rud)

            r = sim.state[3, 0]
            psi = sim.psi[0]

            w.writerow([float(t), float(np.degrees(psi)), float(np.degrees(r)), float(sim.ship.current_speed[0]), 0.0, float(np.degrees(cmd)), float(np.degrees(sim.ship.smoothed_rudder[0])), float(wind[0,0]), float(wind[1,0]), float(current[0,0]), float(current[1,0])])
            sim.t += dt

    meta_path = out_csv.with_suffix('.json')
    with open(meta_path, 'w') as fm:
        json.dump(meta, fm, indent=2)
    print('Wrote telemetry:', out_csv)
    print('Wrote metadata:', meta_path)
    return out_csv, meta_path


def main():
    speeds = [3.0, 5.0, 7.0]
    dt = 0.2
    T = 180.0
    bit_time = 1.0
    amp_deg = 6.0
    for U in speeds:
        print('Running servo PRBS for U=', U)
        run_prbs_with_servo(U_nom=U, amp_deg=amp_deg, dt=dt, T=T, bit_time=bit_time, out_prefix='id_servo_short')

if __name__ == '__main__':
    main()
