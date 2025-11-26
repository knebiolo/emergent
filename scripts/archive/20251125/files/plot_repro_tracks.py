import os
import json
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def record_and_plot(steps=400, dt=1.0, out_png=None, verbose=False):
    wps = load_seattle_waypoints()
    sim = simulation(port_name='Seattle', dt=dt, T=steps*dt, n_agents=2, load_enc=False, verbose=verbose)
    sim.waypoints = wps
    sim.spawn()
    print('Spawned sim: n=', sim.n)

    # Print waypoint summary and initial agent states
    for i, agent_wps in enumerate(wps):
        print(f'Agent {i} waypoints ({len(agent_wps)}):')
        for k, pt in enumerate(agent_wps):
            print(f'  wp{i}_{k}: {pt[0]:.3f}, {pt[1]:.3f}')
    print('Initial positions:')
    for i in range(sim.n):
        print(f'  agent{i} pos: {sim.pos[0,i]:.3f}, {sim.pos[1,i]:.3f} psi(deg)={(sim.psi[i]*180.0/3.14159265):.2f}')

    pos_hist = np.zeros((steps+1, 2, sim.n), dtype=float)
    pos_hist[0, :, :] = sim.pos.copy()

    for step in range(steps):
        # compute nu from state if available
        try:
            u = sim.state[0] if sim.state.shape[0] > 0 else np.zeros(sim.n)
            v = sim.state[1] if sim.state.shape[0] > 1 else np.zeros(sim.n)
            nu = np.vstack([u, v])
        except Exception:
            nu = np.zeros((2, sim.n))

        # call colregs to preserve logic run (we don't use outputs here beyond potential side-effects)
        try:
            sim.ship.colregs(sim.dt, sim.pos, nu, sim.psi, sim.ship.commanded_rpm)
        except Exception:
            pass

        # simple kinematic step like the reproducer does
        for i in range(sim.n):
            sim.pos[0, i] += 1.0 * np.cos(sim.psi[i])
            sim.pos[1, i] += 1.0 * np.sin(sim.psi[i])

        sim.t += dt
        pos_hist[step+1, :, :] = sim.pos.copy()

    # Plot
    if out_png is None:
        out_png = os.path.join(os.path.dirname(__file__), '..', 'figs', 'colregs_tracks.png')
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['C0', 'C1', 'C2', 'C3']
    # plot tracks and start/end markers
    for i in range(sim.n):
        xs = pos_hist[:, 0, i]
        ys = pos_hist[:, 1, i]
        ax.plot(xs, ys, '-', color=colors[i % len(colors)], label=f'Agent {i} track')
        ax.plot(xs[0], ys[0], marker='o', color=colors[i % len(colors)], markersize=6, label=f'Agent {i} start')
        ax.plot(xs[-1], ys[-1], marker='s', color=colors[i % len(colors)], markersize=6, label=f'Agent {i} end')

    # Plot waypoints for each agent
    for i, agent_wps in enumerate(wps):
        wp_arr = np.array(agent_wps)
        if wp_arr.size > 0:
            ax.plot(wp_arr[:, 0], wp_arr[:, 1], 'x', color=colors[i % len(colors)], markersize=8, label=f'Agent {i} wps')
            for k, pt in enumerate(wp_arr):
                ax.text(pt[0], pt[1], f'wp{i}_{k}', fontsize=8)

    # Auto-zoom to the extents of the tracks + waypoints so small motions are visible
    try:
        all_x = pos_hist[:, 0, :].ravel().tolist()
        all_y = pos_hist[:, 1, :].ravel().tolist()
        for agent_wps in wps:
            for p in agent_wps:
                all_x.append(float(p[0]))
                all_y.append(float(p[1]))
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        # padding: 3% of max span or at least 20 m
        spanx = maxx - minx if (maxx - minx) > 0 else 1.0
        spany = maxy - miny if (maxy - miny) > 0 else 1.0
        padx = max(20.0, 0.03 * spanx)
        pady = max(20.0, 0.03 * spany)
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)
        print(f'[plot] bbox x=({minx:.1f},{maxx:.1f}) y=({miny:.1f},{maxy:.1f}) padx={padx:.1f} pady={pady:.1f}')
    except Exception:
        # fallback: keep matplotlib autoscaling
        pass

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Agent tracks (Seattle route)')
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print('Saved track plot to', out_png)


if __name__ == '__main__':
    import sys
    steps = None
    dt = 0.1
    try:
        if len(sys.argv) > 1:
            # first arg may be steps or seconds
            val = float(sys.argv[1])
            # if val is small (<1000) assume seconds when dt provided later; otherwise assume steps
            steps = int(val)
        if len(sys.argv) > 2:
            dt = float(sys.argv[2])
    except Exception:
        pass
    # default: run 400 seconds at dt=0.1 -> 4000 steps
    if steps is None:
        steps = int(400.0 / dt)
    record_and_plot(steps=steps, dt=dt)
