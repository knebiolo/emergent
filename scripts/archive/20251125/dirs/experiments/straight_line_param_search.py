"""Straight-line crosswind parameter search.
Runs a grid over PID gains while following a straight waypoint in a constant crosswind.
Produces per-run diagnostics and a summary CSV to help pick stable tuning that minimizes overshoot and oscillation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'figs' / 'straight_line_search'
OUT.mkdir(parents=True, exist_ok=True)

# grid (coarse by default)
Kp_list = [0.1, 0.2, 0.4]
Ki_list = [0.0, 0.01]
Kd_list = [0.05, 0.12]

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

results = []

for Kp in Kp_list:
    for Ki in Ki_list:
        for Kd in Kd_list:
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
            traj = np.array(sim.history[0])
            t = np.array(sim.t_history)
            psi = np.array(sim.psi_history)
            hd_cmd = np.array(sim.hd_cmd_history)
            # diagnostics: cross-track to straight centerline
            x = traj[:,0]; y = traj[:,1]
            x0, y0 = x[0], y[0]
            x1, y1 = x[-1], y[-1]
            line_vec = np.array([x1-x0, y1-y0])
            line_len = np.hypot(line_vec[0], line_vec[1])
            if line_len == 0:
                line_unit = np.array([1.0, 0.0])
            else:
                line_unit = line_vec / line_len
            # perp distance formula
            cross_track = ((x1-x0)*(y0-y) - (x0 - x)*(y1 - y0)) / (line_len + 1e-12)
            rmse_cross = float(np.sqrt(np.mean(cross_track**2)))
            max_cross = float(np.max(np.abs(cross_track)))
            # heading error
            err = (((psi - hd_cmd + np.pi) % (2*np.pi)) - np.pi)
            err_deg = np.degrees(err)
            rmse_heading = float(np.sqrt(np.mean(err_deg**2)))
            max_heading = float(np.max(np.abs(err_deg)))
            # yaw-rate
            dt = np.median(np.diff(t)) if len(t)>1 else DT
            r_deg_s = np.gradient(np.degrees(psi), dt)
            r_std = float(np.nanstd(r_deg_s))
            # oscillation proxy: zero crossings of cross-track
            signs = np.sign(cross_track)
            zc = int(((signs[1:]*signs[:-1])<0).sum())
            # save top-down plot
            png = OUT / f'straight_{tag}.png'
            plt.figure(figsize=(8,6))
            plt.plot(x, y, label='actual traj')
            # nominal ideal: straight centerline
            plt.plot([x0, x1], [y0, y1], ':k', label='centerline')
            plt.title(f'{tag}  rmse_x={rmse_cross:.1f} m, max_x={max_cross:.1f} m')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.savefig(png, dpi=150)
            plt.close()
            # save heading trace
            png2 = OUT / f'heading_{tag}.png'
            plt.figure(figsize=(10,4))
            plt.plot(t, np.degrees(hd_cmd), label='hd_cmd')
            plt.plot(t, np.degrees(psi), label='psi_actual')
            plt.plot(t, err_deg, label='err_deg')
            plt.legend(); plt.grid(True)
            plt.savefig(png2, dpi=150)
            plt.close()
            results.append({
                'Kp':Kp, 'Ki':Ki, 'Kd':Kd,
                'rmse_cross_m':rmse_cross, 'max_cross_m':max_cross,
                'rmse_heading_deg':rmse_heading, 'max_heading_deg':max_heading,
                'r_std_deg_s':r_std, 'cross_zc':zc,
                'traj_png':str(png), 'hd_png':str(png2)
            })

# write summary
df = pd.DataFrame(results)
summary_csv = OUT / 'straight_line_search_summary.csv'
df.to_csv(summary_csv, index=False)
print('Wrote summary to', summary_csv)
