import json
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation

# Reproducer: load Seattle route from workspace copy of ~/.emergent_routes.json
ROUTES_PATH = os.path.join(os.path.dirname(__file__), '..', '.emergent_routes.json')
# The real file lives in the user's home; the attachments were provided by the user and
# stored as repo-root copies named .emergent_routes.json in their home. Try several fallbacks.
HOME_PATH = os.path.join(os.path.expanduser('~'), '.emergent_routes.json')


def load_seattle_waypoints():
    for p in (ROUTES_PATH, HOME_PATH):
        try:
            if os.path.exists(p):
                with open(p, 'r') as fh:
                    data = json.load(fh)
                entry = data.get('Seattle')
                if entry and entry.get('waypoints'):
                    # convert lists -> arrays
                    wps = []
                    for agent in entry['waypoints']:
                        wps.append([np.array(pt, dtype=float) for pt in agent])
                    print(f"Loaded Seattle route from {p}")
                    return wps
        except Exception as e:
            print('Failed to load', p, e)
    raise FileNotFoundError('Seattle route not found in expected paths')


def run(steps=30, dt=1.0, verbose=True):
    wps = load_seattle_waypoints()
    sim = simulation(port_name='Seattle', dt=dt, T=steps*dt, n_agents=2, load_enc=False, verbose=verbose)
    sim.waypoints = wps
    sim.spawn()
    print('Spawned sim: n=', sim.n)

    for step in range(steps):
        # compute nu from state if available
        try:
            u = sim.state[0] if sim.state.shape[0] > 0 else np.zeros(sim.n)
            v = sim.state[1] if sim.state.shape[0] > 1 else np.zeros(sim.n)
            nu = np.vstack([u, v])
        except Exception:
            nu = np.zeros((2, sim.n))
        col_hd, col_sp, col_rpm, roles = sim.ship.colregs(sim.dt, sim.pos, nu, sim.psi, sim.ship.commanded_rpm)
        print(f"t={sim.t:.1f} step={step} roles={roles} crossing_lock={sim.ship.crossing_lock.tolist()} flagged={getattr(sim.ship,'flagged_give_way',None)}")
        # print any recent simulation log lines (short)
        lines = getattr(sim, 'log_lines', [])[:5]
        for ln in lines:
            print('  [simlog]', ln)
        # step basic dynamics a tiny bit so positions change
        for i in range(sim.n):
            sim.pos[0,i] += 1.0 * np.cos(sim.psi[i])
            sim.pos[1,i] += 1.0 * np.sin(sim.psi[i])
        sim.t += dt
    print('Done repro')


if __name__ == '__main__':
    import sys
    steps = 30
    try:
        if len(sys.argv) > 1:
            steps = int(sys.argv[1])
    except Exception:
        pass
    run(steps=steps)
