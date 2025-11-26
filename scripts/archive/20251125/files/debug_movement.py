import os
import json
import numpy as np
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
    steps = 50
    dt = 0.1
    wps = load_seattle_waypoints()
    sim = simulation(port_name='Seattle', dt=dt, T=steps*dt, n_agents=2, load_enc=False, verbose=False)
    sim.waypoints = wps
    sim.spawn()
    print('Initial pos:', sim.pos)
    print('Initial psi (deg):', np.degrees(sim.psi))

    pos_hist = np.zeros((steps+1, 2, sim.n), dtype=float)
    pos_hist[0,:,:] = sim.pos.copy()

    for step in range(steps):
        # compute nu if available
        try:
            u = sim.state[0] if sim.state.shape[0] > 0 else np.zeros(sim.n)
            v = sim.state[1] if sim.state.shape[0] > 1 else np.zeros(sim.n)
            nu = np.vstack([u, v])
        except Exception:
            nu = np.zeros((2, sim.n))
        # call colregs
        try:
            sim.ship.colregs(sim.dt, sim.pos, nu, sim.psi, sim.ship.commanded_rpm)
        except Exception:
            pass
        # kinematic step same as plotting script
        for i in range(sim.n):
            sim.pos[0, i] += 1.0 * np.cos(sim.psi[i])
            sim.pos[1, i] += 1.0 * np.sin(sim.psi[i])
        sim.t += dt
        pos_hist[step+1,:,:] = sim.pos.copy()

    for i in range(sim.n):
        start = pos_hist[0,:,i]
        end = pos_hist[-1,:,i]
        delta = end - start
        dist = np.linalg.norm(delta)
        print(f'Agent {i}: start={start}, end={end}, delta={delta}, dist={dist:.3f} m')
    # print first 5 positions for agent 0
    print('Agent0 first positions:')
    for k in range(min(5, steps+1)):
        print(k, pos_hist[k,:,0])
