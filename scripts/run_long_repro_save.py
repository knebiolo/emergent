import os
import json
import numpy as np
import csv
from datetime import datetime
from emergent.ship_abm.simulation_core import simulation

ROUTES_PATH = os.path.join(os.path.dirname(__file__), '..', '.emergent_routes.json')
HOME_PATH = os.path.join(os.path.expanduser('~'), '.emergent_routes.json')


def load_seattle_waypoints():
    for p in (ROUTES_PATH, HOME_PATH):
        try:
            if os.path.exists(p):
                with open(p, 'r') as fh:
                    data = json.load(fh)
                entry = data.get('Seattle')
                if entry and entry.get('waypoints'):
                    wps = []
                    for agent in entry['waypoints']:
                        wps.append([np.array(pt, dtype=float) for pt in agent])
                    print(f"Loaded Seattle route from {p}")
                    return wps
        except Exception as e:
            print('Failed to load', p, e)
    raise FileNotFoundError('Seattle route not found in expected paths')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('steps', nargs='?', type=int, default=4000)
    p.add_argument('dt', nargs='?', type=float, default=0.1)
    args = p.parse_args()
    steps = args.steps
    dt = args.dt
    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', f'colregs_repro_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.csv'))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    wps = load_seattle_waypoints()
    sim = simulation(port_name='Seattle', dt=dt, T=steps*dt, n_agents=2, load_enc=False, verbose=False)
    sim.waypoints = wps
    sim.spawn()
    print('Spawned sim: n=', sim.n)

    header = ['step', 't', 'agent', 'x', 'y', 'psi_deg', 'flagged_give_way', 'crossing_lock', 'commanded_rpm']
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        # run a dynamics-stepping loop that mirrors simulation.run()
        sim.dt = dt
        sim.t = 0.0
        sim.steps = int((steps * dt) / dt)
        for step in range(steps):
            t = step * sim.dt
            sim.t = t

            # update goals & compute controls
            try:
                if sim.test_mode not in ("zigzag", "turning_circle"):
                    sim._update_goals()
            except Exception:
                pass

            try:
                # pack body-fixed velocities into nu
                nu = np.vstack([sim.state[0], sim.state[1], sim.state[3]])
            except Exception:
                nu = np.zeros((3, sim.n))

            try:
                hd, sp, rud = sim._compute_controls_and_update(nu, t)
            except Exception:
                # fallback to naive commands
                hd = np.zeros(sim.n)
                sp = getattr(sim.ship, 'desired_speed', np.zeros(sim.n))
                rud = np.zeros(sim.n)

            # run the physics integration
            try:
                sim._step_dynamics(hd, sp, rud)
            except Exception:
                pass

            # call COLREGS on the ship with the current (post-step) state
            try:
                u = sim.state[0] if sim.state.shape[0] > 0 else np.zeros(sim.n)
                v = sim.state[1] if sim.state.shape[0] > 1 else np.zeros(sim.n)
                nu2 = np.vstack([u, v])
                sim.ship.colregs(sim.dt, sim.pos, nu2, sim.psi, sim.ship.commanded_rpm)
            except Exception:
                pass

            # record rows (position/state after dynamics step)
            for i in range(sim.n):
                row = [step, sim.t, i, float(sim.pos[0,i]), float(sim.pos[1,i]), float(np.degrees(sim.psi[i])), bool(getattr(sim.ship, 'flagged_give_way', [False]*sim.n)[i]), int(getattr(sim.ship, 'crossing_lock', [-1]*sim.n)[i]) if hasattr(sim.ship, 'crossing_lock') else -1, float(getattr(sim.ship, 'commanded_rpm', [0.0]*sim.n)[i])]
                writer.writerow(row)

    print('Saved CSV to', out_csv)
