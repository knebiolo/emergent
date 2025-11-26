"""Zig-zag PID sweep experiment.
Runs a small grid over Kp/Kd (Ki fixed), applies a constant crosswind, and records:
 - RMSE cross-track (m)
 - max cross-track (m)
 - yaw-rate std (deg/s)
 - zero-cross count of cross-track (proxy for oscillation)
Saves per-run CSVs and a summary CSV and PNGs under figs/zigzag/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / 'figs' / 'zigzag' / 'pid_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# grid
Kp_list = [0.2, 0.4, 0.8]
Ki = 0.01
Kd_list = [0.05, 0.12, 0.25]

# crosswind
crosswind_speed = 2.0
crosswind_dir_deg = 90.0

import importlib.util
spec = importlib.util.spec_from_file_location('simmod', str(ROOT / 'src' / 'emergent' / 'ship_abm' / 'simulation_core.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation

runs = []
for Kp in Kp_list:
    for Kd in Kd_list:
        print(f"Running Kp={Kp}, Kd={Kd}...")
        # setup sim
        sim = simulation(port_name='Galveston', dt=0.5, T=180.0, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
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
        hd_cmd = np.array([math.degrees(x) for x in sim.hd_cmd_history])
        # compute cross-track relative to centerline cy
        y = traj[:,1]
        cross = y - cy
        rmse_cross = float(np.sqrt(np.mean(cross**2)))
        max_cross = float(np.max(np.abs(cross)))
        # zero-crossings
        signs = np.sign(cross)
        zc = int(((signs[1:] * signs[:-1]) < 0).sum())
        # yaw-rate: compute from psi history derivative
        dt = np.median(np.diff(t)) if len(t)>1 else 0.5
        r_deg_s = np.gradient(np.degrees(psi), dt)
        r_std = float(np.nanstd(r_deg_s))
        # save per-run PNG
        png = OUT_DIR / f'zigzag_pid_Kp{Kp:.3f}_Kd{Kd:.3f}.png'
        plt.figure(figsize=(8,6))
        plt.plot(traj[:,0], traj[:,1], label='actual traj')
        # nominal ideal using hd_cmd integrated at desired speed
        try:
            nominal_speed = float(sim.ship.desired_speed[0])
        except Exception:
            nominal_speed = float(np.nanmean(np.hypot(np.diff(traj[:,0],prepend=traj[0,0])/dt, np.diff(traj[:,1],prepend=traj[0,1])/dt)))
        x_nom = np.zeros_like(t)
        y_nom = np.zeros_like(t)
        x_nom[0] = traj[0,0]
        y_nom[0] = traj[0,1]
        for k in range(1,len(t)):
            th = math.radians(hd_cmd[k-1])
            x_nom[k] = x_nom[k-1] + nominal_speed * math.cos(th) * dt
            y_nom[k] = y_nom[k-1] + nominal_speed * math.sin(th) * dt
        plt.plot(x_nom, y_nom, linestyle='--', label='nominal ideal')
        plt.axhline(cy, color='k', linestyle=':', label='centerline')
        plt.title(f'Kp={Kp}, Ki={Ki}, Kd={Kd}, rmse_x={rmse_cross:.2f} m')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(png, dpi=150)
        plt.close()
        # append summary
        runs.append({
            'Kp':Kp, 'Ki':Ki, 'Kd':Kd,
            'rmse_cross_m':rmse_cross, 'max_cross_m':max_cross,
            'cross_zc_count':zc, 'r_std_deg_s':r_std,
            'png': str(png)
        })

# write summary
summary = pd.DataFrame(runs)
summary_csv = OUT_DIR / 'zigzag_pid_sweep_summary.csv'
summary.to_csv(summary_csv, index=False)
print('Wrote summary to', summary_csv)
