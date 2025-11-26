import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, SHIP_PHYSICS

T = 300.0
dt = 0.1
wind_speed = 5.0  # m/s

# create sim (no zigzag test_mode so it will follow waypoints)
# Set a straight waypoint far ahead along East (5 km). spawn() requires waypoints pre-set.
# Use geographic lon/lat waypoints (Rosario Strait approximate) so spawn() treats
# them as geographic coordinates and performs the proper projection.
# Start near -122.7 lon, 48.2 lat (example); goal ~0.05° east (~4–5 km)
start = np.array([-122.7, 48.2])  # (lon, lat)
goal = start + np.array([0.05, 0.0])
sim = simulation(port_name='Rosario Strait', dt=dt, T=T, n_agents=1, load_enc=False, test_mode=None)
# assign waypoints before spawn so spawn() can initialize position/heading
sim.waypoints = [[start, goal]]
sim.spawn()

# determine initial heading after spawn and set constant crosswind perpendicular to travel (starboard)
psi0 = float(sim.psi[0])
cross_theta = psi0 + math.pi/2.0  # starboard perpendicular
wx = wind_speed * math.cos(cross_theta)
wy = wind_speed * math.sin(cross_theta)

def constant_crosswind(lon, lat, when):
    # return shape (n,2) where each row is [wx, wy]
    n = 1
    a = np.tile(np.array([[wx, wy]]), (n, 1))
    return a

# attach wind and zero current
sim.wind_fn = constant_crosswind
sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

# enable PID trace
trace_path = 'scripts/pid_trace_straight_wind.csv'
PID_TRACE['enabled'] = True
PID_TRACE['path'] = trace_path

print(f"Running straight-line crosswind test: wind={wind_speed} m/s at {math.degrees(cross_theta):.1f}°")
try:
    if os.path.exists(trace_path):
        os.remove(trace_path)
except Exception:
    pass

sim.run()
time.sleep(0.1)

# collect history
hist = getattr(sim, 'history', {})
t_hist = getattr(sim, 't_history', [])
seq = hist.get(0, [])

if len(seq) == 0:
    raise RuntimeError('No trajectory recorded')

xs = [p[0] for p in seq]
ys = [p[1] for p in seq]
# ensure time and position lists are the same length before creating a DataFrame
n = min(len(xs), len(ys), len(t_hist))
if n == 0:
    raise RuntimeError('No synchronized trajectory/time data available')
xs = xs[:n]
ys = ys[:n]
ts = list(t_hist)[:n]

os.makedirs('scripts', exist_ok=True)
track_csv = 'scripts/straight_track.csv'
pd.DataFrame({'t': ts, 'x': xs, 'y': ys}).to_csv(track_csv, index=False)
print('Wrote', track_csv)

# read pid trace for agent 0
df = pd.read_csv(trace_path)
df0 = df[df['agent'] == 0]

# plots
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(6,6))
plt.plot(xs, ys, '-k')
plt.scatter(xs[0], ys[0], c='g', label='start')
plt.scatter(xs[-1], ys[-1], c='r', label='end')
plt.axis('equal')
plt.title(f'Straight-line track (crosswind {wind_speed} m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.tight_layout()
plot_track = 'plots/straight_track.png'
plt.savefig(plot_track)
plt.close()
print('Wrote', plot_track)

plt.figure(figsize=(10,4))
plt.plot(df0['t'], df0['psi_deg'], label='psi (deg)')
plt.plot(df0['t'], df0['err_deg'], label='err_deg')
plt.xlabel('t (s)')
plt.ylabel('deg')
plt.title('Heading & error (deg)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_head = 'plots/straight_heading.png'
plt.savefig(plot_head)
plt.close()
print('Wrote', plot_head)

plt.figure(figsize=(10,4))
plt.plot(df0['t'], df0['raw_deg'], '--', label='raw cmd (deg)', alpha=0.7)
plt.plot(df0['t'], df0['rud_deg'], label='applied rudder (deg)')
plt.xlabel('t (s)')
plt.ylabel('deg')
plt.title('Rudder command vs applied')
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_rud = 'plots/straight_rudder.png'
plt.savefig(plot_rud)
plt.close()
print('Wrote', plot_rud)

# numeric summary
max_err = float(df0['err_deg'].abs().max())
mean_err = float(df0['err_deg'].abs().mean())
max_raw = float(df0['raw_deg'].abs().max())
max_rud = float(df0['rud_deg'].abs().max())
sat_frac = float((df0['rud_deg'].abs() >= math.degrees(SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

summary = {
    'wind_speed_m_s': wind_speed,
    'max_err_deg': max_err,
    'mean_err_deg': mean_err,
    'max_raw_deg': max_raw,
    'max_rud_deg': max_rud,
    'sat_frac': sat_frac
}
pd.DataFrame([summary]).to_csv('scripts/straight_summary.csv', index=False)
print('Wrote scripts/straight_summary.csv')
print(summary)
