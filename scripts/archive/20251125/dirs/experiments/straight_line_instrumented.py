"""Instrumented straight-line runs: record commanded vs applied rudder.
Runs a small sweep (Ki=0) and saves per-run CSVs and plots plus a summary CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'figs' / 'straight_line_instrumented'
OUT.mkdir(parents=True, exist_ok=True)

# targeted runs (Ki=0)
cases = [
    {'Kp':0.2, 'Ki':0.0, 'Kd':0.05},
    {'Kp':0.4, 'Ki':0.0, 'Kd':0.05},
    {'Kp':0.4, 'Ki':0.0, 'Kd':0.12},
]

# sim params
DT = 0.5
T = 180.0
crosswind_speed = 2.0
crosswind_dir_deg = 90.0

import importlib.util
spec = importlib.util.spec_from_file_location('simmod', str(ROOT / 'src' / 'emergent' / 'ship_abm' / 'simulation_core.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation

summary = []

for c in cases:
    Kp, Ki, Kd = c['Kp'], c['Ki'], c['Kd']
    tag = f'Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}'
    print('Running', tag)
    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
    # set straight-waypoints (centerline)
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    sim.waypoints = [[(cx, cy), (cx + 1000.0, cy)]]
    state0, pos0, psi0, goals = sim.spawn()
    # set gains
    sim.ship.Kp = Kp
    sim.ship.Ki = Ki
    sim.ship.Kd = Kd
    # wind
    def wind_fn(lons, lats, when):
        N = int(np.atleast_1d(lons).size)
        dir_rad = math.radians(crosswind_dir_deg)
        e = math.cos(dir_rad) * crosswind_speed
        n = math.sin(dir_rad) * crosswind_speed
        return np.tile(np.array([[e, n]]), (N,1))
    sim.wind_fn = wind_fn
    sim.current_fn = lambda lons, lats, when: np.tile(np.array([[0.0, 0.0]]), (int(np.atleast_1d(lons).size),1))
    # run
    sim.run()

    t = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd_cmd = np.array(sim.hd_cmd_history)
    traj = np.array(sim.history[0])

    # commanded rudder recorded in sim.rudder_history (may be shorter); applied in sim.applied_rudder_history
    cmd_rud = np.asarray(getattr(sim, 'rudder_history', []))
    applied = np.asarray(getattr(sim, 'applied_rudder_history', []))

    # align lengths: use t as reference
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
                # single row
                a = a.reshape((1, ncols))
            else:
                # cannot interpret, return nan-filled
                return np.full((n, ncols), np.nan)
        if a.shape[0] >= n:
            return a[:n, :ncols]
        else:
            pad_rows = np.full((n - a.shape[0], ncols), np.nan)
            return np.vstack([a[:, :ncols], pad_rows])

    cmd_rud_p = pad1d(cmd_rud, n)
    applied_p = pad1d(applied, n)

    traj_p = pad2d(traj, n, ncols=2)
    psi_p = pad1d(psi, n)
    hd_cmd_p = pad1d(hd_cmd, n)

    # save per-run CSV
    df = pd.DataFrame({
        't_s': t,
        'psi_deg': np.degrees(psi_p),
        'hd_cmd_deg': np.degrees(hd_cmd_p),
        'rud_cmd_deg': np.degrees(cmd_rud_p),
        'rud_applied_deg': np.degrees(applied_p),
        'x_m': traj_p[:,0], 'y_m': traj_p[:,1]
    })
    csv = OUT / f'straight_instr_{tag}.csv'
    df.to_csv(csv, index=False)

    # plot rudder commanded vs applied
    plt.figure(figsize=(10,4))
    plt.plot(t, np.degrees(cmd_rud_p), label='cmd_rud')
    plt.plot(t, np.degrees(applied_p), label='applied_rud')
    plt.legend(); plt.grid(True)
    plt.title(f'Rudder trace {tag}')
    plt.xlabel('t (s)')
    png_r = OUT / f'rudder_{tag}.png'
    plt.savefig(png_r, dpi=150)
    plt.close()

    # heading trace
    err = (((psi - hd_cmd + np.pi) % (2*np.pi)) - np.pi)
    err_deg = np.degrees(err)
    plt.figure(figsize=(10,4))
    plt.plot(t, np.degrees(hd_cmd), label='hd_cmd')
    plt.plot(t, np.degrees(psi), label='psi')
    plt.plot(t, err_deg, label='err_deg')
    plt.legend(); plt.grid(True)
    png_h = OUT / f'heading_{tag}.png'
    plt.savefig(png_h, dpi=150)
    plt.close()

    # simple diagnostics
    # cross-track to straight centerline
    x = traj[:,0]; y = traj[:,1]
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    line_vec = np.array([x1-x0, y1-y0])
    line_len = np.hypot(line_vec[0], line_vec[1])
    cross_track = ((x1-x0)*(y0-y) - (x0 - x)*(y1 - y0)) / (line_len + 1e-12)
    rmse_cross = float(np.sqrt(np.mean(cross_track**2)))
    max_cross = float(np.max(np.abs(cross_track)))
    rmse_heading = float(np.sqrt(np.mean(err_deg**2)))

    summary.append({
        'Kp':Kp, 'Ki':Ki, 'Kd':Kd,
        'rmse_cross_m':rmse_cross, 'max_cross_m':max_cross,
        'rmse_heading_deg':rmse_heading,
        'traj_csv': str(csv), 'rud_png': str(png_r), 'hd_png': str(png_h)
    })

    print('Done', tag)

dfs = pd.DataFrame(summary)
dfs.to_csv(OUT / 'straight_line_instrumented_summary.csv', index=False)
print('Wrote instrumented summary to', OUT / 'straight_line_instrumented_summary.csv')
