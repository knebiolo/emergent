"""Experiment: single vessel between two waypoints (0.5 km apart) with crosswind and cross-current
Saves PID trace to scripts/pid_trace_exp_waypoint_crosswind.csv and prints summary metrics.
Run from workspace root: python scripts/exp_waypoint_crosswind.py
"""
import os
from emergent.ship_abm.simulation_core import simulation
import numpy as np
from datetime import datetime

OUT_TRACE = os.path.join(os.getcwd(), 'scripts', 'pid_trace_exp_waypoint_crosswind.csv')
# ensure output dir exists
os.makedirs(os.path.dirname(OUT_TRACE), exist_ok=True)

# experiment parameters
dt = 0.5            # seconds per step (reasonable compromise)
T = 600.0           # 10 minutes
n_agents = 1
# Waypoints: start at (0,0), goal at (500, 0) meters (0.5 km east)
waypoints = [[(0.0, 0.0), (500.0, 0.0)]]

# Crosswind: blowing from the north -> vector pointing south (u=0, v=-w_speed)
w_speed = 8.0  # m/s (stiff breeze)
# Cross current: from north to south (same direction)
c_speed = 0.5  # m/s

def constant_wind_fn(lon, lat, now):
    # simulation expects (2, n) arrays in meters/sec in earth frame
    # we ignore lon/lat and return constant vector for each agent
    return np.tile(np.array([[0.0], [-w_speed]]), (1, n_agents))

def constant_current_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-c_speed]]), (1, n_agents))

# Create simulation
sim = simulation(port_name='Galveston', dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode=None)
# inject wind/current functions
sim.wind_fn = constant_wind_fn
sim.current_fn = constant_current_fn

# set waypoints and spawn
sim.waypoints = waypoints
print(f"[exp] Waypoints set: {sim.waypoints}")
sim.spawn()

# enable PID tracing (simulation writes per-step CSV when enabled)
from emergent.ship_abm.config import PID_TRACE
PID_TRACE['enabled'] = True
PID_TRACE['path'] = OUT_TRACE

print(f"[exp] Running simulation for T={T}s (dt={dt}s). Output trace: {OUT_TRACE}")
arrival_time = None
distance_threshold_m = 10.0  # arrival when within 10 meters of final waypoint

print('[exp] Running (with distance-based arrival detection)')

# We'll step the sim ourselves to capture positions and arrival time reliably.
# The simulation `sim.run()` performs the loop internally; instead we step manually.
# Re-create a fresh simulation instance (simulation has no reset() API).
sim = simulation(port_name='Galveston', dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode=None)
# inject wind/current functions again
sim.wind_fn = constant_wind_fn
sim.current_fn = constant_current_fn
sim.waypoints = waypoints
sim.spawn()
# Run the simulation to completion, then post-process history for arrival
sim.run()

# post-process sim.history (each agent history has initial pos + per-step entries)
hist = sim.history.get(0, [])
times = getattr(sim, 't_history', [])
target = waypoints[0][-1]
# times[k] corresponds to hist[k+1]
for k, tval in enumerate(times):
    pos_k = hist[k+1]
    dx = pos_k[0] - target[0]
    dy = pos_k[1] - target[1]
    dist = (dx*dx + dy*dy) ** 0.5
    if arrival_time is None and dist <= distance_threshold_m:
        arrival_time = tval
        print(f"[exp] Agent 0 arrived at t={arrival_time:.1f}s (dist={dist:.2f} m) [postproc]")

# After stepping, print a summary from the PID trace file
import csv
from math import fabs
max_err = 0.0
sum_cte = 0.0
count = 0
with open(OUT_TRACE, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        err = fabs(float(row['err_deg']))
        max_err = max(max_err, err)
        sum_cte += err
        count += 1

mean_err = sum_cte / count if count>0 else None
print(f"max_heading_error_deg={max_err:.3f}")
print(f"mean_abs_heading_error_deg={mean_err:.3f}")
print(f"arrival_time_s={arrival_time}")
print('[exp] Done')
