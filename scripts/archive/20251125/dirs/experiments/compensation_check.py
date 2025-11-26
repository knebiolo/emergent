"""Compensation check: log computed goal heading and compensation offsets from compute_desired.
Runs one straight-line crosswind case and records goal_hd, raw_no_drift_hd (if computed), and actual psi.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'figs' / 'compensation_check'
OUT.mkdir(parents=True, exist_ok=True)

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

sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, light_bg=True, verbose=False, use_ais=False, load_enc=False)
cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
sim.waypoints = [[(cx, cy), (cx + 1000.0, cy)]]
state0, pos0, psi0, goals = sim.spawn()

# wind
def wind_fn(lons, lats, when):
    N = int(np.atleast_1d(lons).size)
    dir_rad = math.radians(crosswind_dir_deg)
    e = math.cos(dir_rad) * crosswind_speed
    n = math.sin(dir_rad) * crosswind_speed
    return np.tile(np.array([[e, n]]), (N,1))

sim.wind_fn = wind_fn
sim.current_fn = lambda lons, lats, when: np.tile(np.array([[0.0, 0.0]]), (int(np.atleast_1d(lons).size),1))

# We'll instrument compute_desired by calling the ship method directly at each timestep
goal_hds = []
raw_hds = []
psi_hist = []
t_hist = []
traj = []

# helper to call compute_desired with and without drift
for step in range(int(T/DT)):
    t = step * DT
    # sample current wind/current at ship position
    lon, lat = sim._utm_to_ll.transform(sim.pos[0], sim.pos[1])
    wind_vec = sim.wind_fn(lon, lat, None).T
    current_vec = sim.current_fn(lon, lat, None).T
    combined_drift_vec = -(current_vec + wind_vec)

    # call compute_desired with drift
    goal_hd, goal_sp = sim.ship.compute_desired(
        sim.goals,
        sim.pos[0], sim.pos[1],
        sim.state[0], sim.state[1], sim.state[3], sim.psi,
        current_vec = combined_drift_vec
    )

    # call compute_desired with zero drift for raw heading
    raw_goal_hd, _ = sim.ship.compute_desired(
        sim.goals,
        sim.pos[0], sim.pos[1],
        sim.state[0], sim.state[1], sim.state[3], sim.psi,
        current_vec = np.zeros_like(combined_drift_vec)
    )

    # fuse/compute rudder as the sim would
    hd, sp = sim._fuse_and_pid(goal_hd, goal_sp, np.zeros_like(goal_hd), np.zeros_like(goal_sp), ['neutral'])
    rud = sim._compute_rudder(hd, ['neutral'])
    # step dynamics (as sim.run would)
    sim._step_dynamics(hd, sp, rud)
    sim.t += DT

    goal_hds.append(float(goal_hd[0]))
    raw_hds.append(float(raw_goal_hd[0]))
    psi_hist.append(float(sim.psi[0]))
    t_hist.append(t)
    traj.append([float(sim.pos[0]), float(sim.pos[1])])

# Save results
df = pd.DataFrame({
    't_s': t_hist,
    'goal_hd_deg': np.degrees(np.array(goal_hds)),
    'raw_goal_hd_deg': np.degrees(np.array(raw_hds)),
    'psi_deg': np.degrees(np.array(psi_hist)),
    'x_m': np.array(traj)[:,0], 'y_m': np.array(traj)[:,1]
})
csv = OUT / 'compensation_check.csv'
df.to_csv(csv, index=False)

# Plot heading traces and compensation offset
plt.figure(figsize=(10,4))
plt.plot(df['t_s'], df['raw_goal_hd_deg'], label='raw_goal (no drift)')
plt.plot(df['t_s'], df['goal_hd_deg'], label='goal_with_comp')
plt.plot(df['t_s'], df['psi_deg'], label='psi_actual')
plt.legend(); plt.grid(True)
plt.xlabel('t (s)')
plt.title('Heading: raw vs compensated vs actual')
plt.savefig(OUT / 'compensation_heading.png', dpi=150)
plt.close()

# compensation offset (goal_with - raw)
df['comp_offset_deg'] = df['goal_hd_deg'] - df['raw_goal_hd_deg']
plt.figure(figsize=(10,3))
plt.plot(df['t_s'], df['comp_offset_deg'])
plt.axhline(0, color='k', lw=0.5)
plt.grid(True)
plt.xlabel('t (s)')
plt.title('Compensation offset (deg)')
plt.savefig(OUT / 'compensation_offset.png', dpi=150)
plt.close()

print('Wrote compensation_check.csv and plots to', OUT)
