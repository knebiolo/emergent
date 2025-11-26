"""Run a single straight-line crosswind test with a given PID candidate and save CSV/plots.
Usage: edit gains below and run the script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'figs' / 'straight_line_instrumented'
OUT.mkdir(parents=True, exist_ok=True)

# Candidate gains (edit as needed)
Kp = 1.0
Ki = 0.0
Kd = 0.5

# sim params
DT = 0.5
T = 300.0
crosswind_speed = 2.0
crosswind_dir_deg = 90.0

import importlib.util
spec = importlib.util.spec_from_file_location('simmod', str(ROOT / 'src' / 'emergent' / 'ship_abm' / 'simulation_core.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation

def wind_fn(lons, lats, when):
    N = int(np.atleast_1d(lons).size)
    dir_rad = math.radians(crosswind_dir_deg)
    e = math.cos(dir_rad) * crosswind_speed
    n = math.sin(dir_rad) * crosswind_speed
    return np.tile(np.array([[e, n]]), (N,1))

def run():
    tag = f'Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}'
    print('Running candidate', tag)
    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    sim.waypoints = [[(cx, cy), (cx + 1000.0, cy)]]
    state0, pos0, psi0, goals = sim.spawn()
    sim.ship.Kp = Kp
    sim.ship.Ki = Ki
    sim.ship.Kd = Kd
    sim.wind_fn = wind_fn
    sim.current_fn = lambda lons, lats, when: np.tile(np.array([[0.0, 0.0]]), (int(np.atleast_1d(lons).size),1))
    sim.run()

    t = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd_cmd = np.array(sim.hd_cmd_history)
    traj = np.array(sim.history[0])
    cmd_rud = np.asarray(getattr(sim, 'rudder_history', []))
    applied = np.asarray(getattr(sim, 'applied_rudder_history', []))

    n = t.size
    def pad1d(arr, n):
        if arr is None:
            return np.full(n, np.nan)
        a = np.asarray(arr)
        if a.size >= n:
            return a[:n]
        else:
            return np.concatenate([a, np.full(n - a.size, np.nan)])

    def pad2d(arr, n, ncols=2):
        if arr is None:
            return np.full((n, ncols), np.nan)
        a = np.asarray(arr)
        if a.ndim == 1:
            # assume flattened sequence of rows
            if a.size == ncols:
                a = a.reshape((1, ncols))
            else:
                return np.full((n, ncols), np.nan)
        if a.shape[0] >= n:
            return a[:n, :ncols]
        else:
            pad_rows = np.full((n - a.shape[0], ncols), np.nan)
            return np.vstack([a[:, :ncols], pad_rows])

    cmd_rud_p = pad1d(cmd_rud, n)
    applied_p = pad1d(applied, n)
    traj_p = pad2d(traj, n, ncols=2)

    df = pd.DataFrame({'t_s': t,
                       'psi_deg': np.degrees(psi),
                       'hd_cmd_deg': np.degrees(hd_cmd),
                       'rud_cmd_deg': np.degrees(cmd_rud_p),
                       'rud_applied_deg': np.degrees(applied_p),
                       'x_m': traj_p[:,0], 'y_m': traj_p[:,1]})
    csv = OUT / f'pid_candidate_{tag}.csv'
    df.to_csv(csv, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(t, np.degrees(cmd_rud_p), label='cmd_rud')
    plt.plot(t, np.degrees(applied_p), label='applied_rud')
    plt.legend(); plt.grid(True)
    plt.title(f'Rudder trace {tag}')
    png_r = OUT / f'pid_candidate_rudder_{tag}.png'
    plt.savefig(png_r, dpi=150); plt.close()

    err = (((psi - hd_cmd + np.pi) % (2*np.pi)) - np.pi)
    err_deg = np.degrees(err)
    plt.figure(figsize=(10,4))
    plt.plot(t, np.degrees(hd_cmd), label='hd_cmd')
    plt.plot(t, np.degrees(psi), label='psi')
    plt.plot(t, err_deg, label='err_deg')
    plt.legend(); plt.grid(True)
    png_h = OUT / f'pid_candidate_heading_{tag}.png'
    plt.savefig(png_h, dpi=150); plt.close()

    # cross-track
    x = traj[:,0]; y = traj[:,1]
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    line_vec = np.array([x1-x0, y1-y0])
    line_len = np.hypot(line_vec[0], line_vec[1])
    cross_track = ((x1-x0)*(y0-y) - (x0 - x)*(y1 - y0)) / (line_len + 1e-12)
    rmse_cross = float(np.sqrt(np.mean(cross_track**2)))
    max_cross = float(np.max(np.abs(cross_track)))
    rmse_heading = float(np.sqrt(np.mean(err_deg**2)))

    summary = {'Kp':Kp, 'Ki':Ki, 'Kd':Kd, 'rmse_cross_m':rmse_cross, 'max_cross_m':max_cross, 'rmse_heading_deg':rmse_heading, 'csv':str(csv), 'rud_png':str(png_r), 'hd_png':str(png_h)}
    outsum = OUT / 'pid_candidate_summary.csv'
    pd.DataFrame([summary]).to_csv(outsum, index=False)
    print('Wrote', outsum)

if __name__ == '__main__':
    run()
