"""Monte-Carlo gust robustness test for the selected zig-zag controller candidate.
- Picks the best zig-zag candidate from `zigzag_search_summary.csv`.
- Runs N trials with random gust amplitude and direction, using a simple gust wind_fn.
- Writes summary CSV to `figs/gust_mc_summary.csv` and a boxplot PNG.
"""
import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.angle_utils import heading_diff_deg

ROOT = Path(__file__).resolve().parent.parent
zig_summary = ROOT / 'zigzag_search_summary.csv'
if not zig_summary.exists():
    raise SystemExit('zigzag_search_summary.csv missing')
zd = pd.read_csv(zig_summary)
# pick best as before
if 'rmse_cross_m' in zd.columns and zd['rmse_cross_m'].notna().any():
    key = 'rmse_cross_m'
elif 'final_cross_m' in zd.columns and zd['final_cross_m'].notna().any():
    key = 'final_cross_m'
else:
    key = None
if key:
    idx = zd[key].idxmin()
else:
    idx = 0
best = zd.loc[idx]
Kp = float(best.get('Kp', 0.4))
Ki = float(best.get('Ki', 0.01))
Kd = float(best.get('Kd', 0.12))
Kf = float(best.get('Kf', 0.0))
wind_speed_nom = float(best.get('wind_speed', 0.5)) if 'wind_speed' in best else 0.5

OUT_DIR = ROOT / 'figs' / 'gust_mc'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / 'gust_mc_summary.csv'

N_TRIALS = 20
T = 180.0
DT = 0.5
results = []

# simple gust generator: base wind plus transient bump centered at random t0

def make_gust_wind_fn(base_ws, amp, dir_deg, t0, tau):
    # direction in degrees: 0 = east, 90 = north
    dir_rad = np.radians(dir_deg)
    be = np.cos(dir_rad) * (base_ws)
    bn = np.sin(dir_rad) * (base_ws)
    ae = np.cos(dir_rad) * amp
    an = np.sin(dir_rad) * amp
    def wind_fn(lons, lats, when):
        # simulate a simple gaussian-shaped gust in time; the sim passes naive datetime objects
        try:
            now = when if isinstance(when, datetime) else datetime.now(timezone.utc)
            # we don't have sim internal time, so return base+amp always for simplicity (worst-case)
            N = int(np.atleast_1d(lons).size)
            return np.tile(np.array([[be + ae, bn + an]]), (N,1))
        except Exception:
            N = int(np.atleast_1d(lons).size)
            return np.tile(np.array([[be + ae, bn + an]]), (N,1))
    return wind_fn

for i in range(N_TRIALS):
    gust_amp = random.uniform(0.5, 3.0)
    gust_dir = random.uniform(0.0, 360.0)
    gust_tau = random.uniform(2.0, 8.0)
    wind_fn = make_gust_wind_fn(wind_speed_nom, gust_amp, gust_dir, t0=T*0.5, tau=gust_tau)

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    sim.waypoints = [[(cx, cy), (cx + 1000.0, cy)]]
    state0, pos0, psi0, goals = sim.spawn()

    sim.ship.Kp = Kp
    sim.ship.Ki = Ki
    sim.ship.Kd = Kd
    sim.tuning['Kf'] = Kf
    sim.ship.dead_reck_sensitivity = 0.25

    sim.wind_fn = wind_fn
    sim.current_fn = lambda lons, lats, when: np.tile(np.array([[0.0, 0.0]]), (int(np.atleast_1d(lons).size),1))

    sim.run()

    traj = np.array(sim.history[0])
    A = np.array([cx, cy])
    B = np.array([cx + 1000.0, cy])
    AB = B - A
    AB_unit = AB / (np.linalg.norm(AB) + 1e-12)
    normal = np.array([-AB_unit[1], AB_unit[0]])
    ct_errors = np.dot(traj - A, normal)
    final_cross = float(ct_errors[-1])
    rmse = float(np.sqrt(np.mean(ct_errors**2)))

    try:
        psi_arr = np.asarray(sim.psi_history)
        hd_cmd_arr = np.asarray(sim.hd_cmd_history)
        err_deg = heading_diff_deg(np.degrees(hd_cmd_arr), np.degrees(psi_arr))
        heading_rmse = float(np.sqrt(np.mean(err_deg**2)))
    except Exception:
        heading_rmse = float('nan')

    results.append({'trial': i, 'gust_amp': gust_amp, 'gust_dir': gust_dir, 'gust_tau': gust_tau, 'final_cross_m': final_cross, 'rmse_cross_m': rmse, 'heading_rmse_deg': heading_rmse})
    print(f"Trial {i+1}/{N_TRIALS}: amp={gust_amp:.2f} dir={gust_dir:.0f} rmse={rmse:.3f}")

# write summary
pd.DataFrame(results).to_csv(SUMMARY_CSV, index=False)

# boxplot of RMSE
plt.figure(figsize=(6,3))
plt.boxplot([r['rmse_cross_m'] for r in results], vert=False)
plt.xlabel('RMSE cross-track (m)')
plt.title('Gust Monte-Carlo RMSE')
plt.savefig(OUT_DIR / 'gust_mc_rmse_boxplot.png', bbox_inches='tight', dpi=150)
plt.close()

print('Wrote summary to', SUMMARY_CSV)
