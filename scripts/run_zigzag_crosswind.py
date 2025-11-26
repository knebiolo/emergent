"""Run a cross-wind zig-zag using the best zigzag candidate and plot top-down actual vs ideal (hd_cmd) path.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

ROOT = Path(__file__).resolve().parent.parent
# pick best zigzag candidate (same logic as plot_best_zigzag)
summary_csv = ROOT / 'zigzag_search_summary.csv'
if not summary_csv.exists():
    raise SystemExit(f"Summary CSV not found: {summary_csv}")
df = pd.read_csv(summary_csv)
if 'rmse_cross_m' in df.columns and df['rmse_cross_m'].notna().any():
    key = 'rmse_cross_m'
elif 'final_cross_m' in df.columns and df['final_cross_m'].notna().any():
    key = 'final_cross_m'
else:
    key = None
if key is not None:
    df_valid = df[df[key].notna()].copy()
    if df_valid.empty:
        idx = 0
    else:
        idx = df_valid[key].idxmin()
else:
    idx = 0
row = df.loc[idx]
# extract controller params or use defaults
Kp = float(row.get('Kp', 0.4))
Ki = float(row.get('Ki', 0.01))
Kd = float(row.get('Kd', 0.12))
Kf = float(row.get('Kf', 0.0))
# choose crosswind: 2.0 m/s from north (i.e., wind vector 0 east, +2 north)
crosswind_speed = 2.0
crosswind_dir_deg = 90.0  # 90deg = north

# run sim
import importlib.util
spec = importlib.util.spec_from_file_location('simmod', str(ROOT / 'src' / 'emergent' / 'ship_abm' / 'simulation_core.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from emergent.ship_abm.simulation_core import simulation

DT = 0.5
T = 180.0
sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False, test_mode='zigzag', zigzag_deg=10, zigzag_hold=40)
# set waypoints same as other scripts
cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
sim.waypoints = [[(cx, cy), (cx + 1000.0, cy)]]
state0, pos0, psi0, goals = sim.spawn()
# set controller gains
sim.ship.Kp = Kp
sim.ship.Ki = Ki
sim.ship.Kd = Kd
sim.tuning['Kf'] = Kf
# wind fn: constant crosswind
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
# write traj CSV
traj = sim.history[0]
out_csv = ROOT / f'headless_zigzag_crosswind_Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}_w{row.get("w",1.5)}_dr{row.get("dr",0.5)}.csv'
with open(out_csv, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['t','x_m','y_m','psi_deg','hd_cmd_deg','u_m_s','cmd_rudder_deg','applied_rudder_deg'])
    for i, t in enumerate(sim.t_history):
        pos = traj[i]
        psi = sim.psi_history[i]
        hd = sim.hd_cmd_history[i]
        # simulation does not keep a time-series of body-fixed speeds; compute
        # approximate ground speed from successive position differences
        if i == 0:
            u = 0.0
        else:
            prev = traj[i-1]
            dx = float(pos[0]) - float(prev[0])
            dy = float(pos[1]) - float(prev[1])
            u = math.hypot(dx, dy) / float(sim.dt)
        # ensure rudder history present
        try:
            cmd_rud = sim.rudder_history[i]
        except Exception:
            cmd_rud = float('nan')
        try:
            # no per-time applied rudder history stored; fall back to NaN or last known
            applied = float('nan')
            if hasattr(sim.ship, 'smoothed_rudder'):
                # if it's an array pick agent-0 value, but note it's the last value
                try:
                    applied = float(sim.ship.smoothed_rudder[0])
                except Exception:
                    applied = float(sim.ship.smoothed_rudder)
        except Exception:
            applied = float('nan')
        except Exception:
            applied = float('nan')
        w.writerow([float(t), float(pos[0]), float(pos[1]), float(np.degrees(psi)), float(np.degrees(hd)), float(u), float(np.degrees(cmd_rud)), float(np.degrees(applied))])

# compute ideal path by integrating hd_cmd_deg with forward speed u_m_s
traj_df = pd.read_csv(out_csv)
t = traj_df['t'].values
dt = np.median(np.diff(t))
hd_deg = traj_df['hd_cmd_deg'].values
u = traj_df['u_m_s'].values
# integrate
x_ideal = np.zeros_like(t)
y_ideal = np.zeros_like(t)
x_ideal[0] = traj_df['x_m'].iloc[0]
y_ideal[0] = traj_df['y_m'].iloc[0]
for k in range(1, len(t)):
    theta = math.radians(hd_deg[k-1])
    x_ideal[k] = x_ideal[k-1] + u[k-1] * math.cos(theta) * dt
    y_ideal[k] = y_ideal[k-1] + u[k-1] * math.sin(theta) * dt

# build a second ideal path using the ship's nominal desired speed
try:
    nominal_speed = float(sim.ship.desired_speed[0])
except Exception:
    # fallback to first u sample mean if not available
    nominal_speed = float(np.nanmean(traj_df['u_m_s'].values)) if len(traj_df) > 0 else 0.0

# compute nominal ideal path (dead-reckoned using desired cruise speed)
x_nominal = np.zeros_like(t)
y_nominal = np.zeros_like(t)
x_nominal[0] = traj_df['x_m'].iloc[0]
y_nominal[0] = traj_df['y_m'].iloc[0]
for k in range(1, len(t)):
    theta = math.radians(hd_deg[k-1])
    x_nominal[k] = x_nominal[k-1] + nominal_speed * math.cos(theta) * dt
    y_nominal[k] = y_nominal[k-1] + nominal_speed * math.sin(theta) * dt

# plot top-down
out_dir = ROOT / 'figs' / 'zigzag'
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / f'crosswind_zigzag_topdown_Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}.png'
plt.figure(figsize=(8,6))
plt.plot(traj_df['x_m'], traj_df['y_m'], label='actual trajectory', color='tab:blue')
plt.plot(x_ideal, y_ideal, label='ideal (hd_cmd integrated, using recorded ground speed)', color='tab:orange', linestyle='--')
plt.plot(x_nominal, y_nominal, label=f'nominal ideal (hd_cmd integrated, desired speed={nominal_speed:.2f} m/s)', color='tab:purple', linestyle=':')
plt.scatter([traj_df['x_m'].iloc[0]], [traj_df['y_m'].iloc[0]], color='green', s=50, label='start')
plt.scatter([traj_df['x_m'].iloc[-1]], [traj_df['y_m'].iloc[-1]], color='red', s=50, label='end')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Crosswind zig-zag: Kp={Kp}, Ki={Ki}, Kd={Kd}, wind={crosswind_speed} m/s from {crosswind_dir_deg}°')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig(out_png, dpi=150)
print('Wrote trajectory CSV and top-down PNG:')
print(out_csv)
print(out_png)
