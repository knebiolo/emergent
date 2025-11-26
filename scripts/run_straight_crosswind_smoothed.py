"""Run a straight-line crosswind test (short T) with hd_cmd smoothing enabled
and write trace to sweep_results. This uses the main straight_line_crosswind setup
but sets T shorter (120s) and writes trace into sweep_results.
"""
import os, math, numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1
T = 120.0
WIND_SPEED = 5.0

trace = os.path.join(OUT_DIR, 'pid_trace_straight_wind_smoothed.csv')
PID_TRACE['enabled'] = True
PID_TRACE['path'] = trace

# build sim, straight goal
# Use geographic lon/lat waypoints (Rosario Strait approximate)
start = np.array([-122.7, 48.2])
goal = start + np.array([0.05, 0.0])

sim = simulation(port_name='Rosario Strait', dt=DT, T=T, n_agents=1, load_enc=False, test_mode=None)
# enable verbose minimal prints
sim.verbose = True
# keep default slew/ramp settings (hd_cmd_slew_deg_per_s and zz_ramp_time)

sim.waypoints = [[start, goal]]
sim.spawn()

psi0 = float(sim.psi[0])
cross_theta = psi0 + math.pi/2.0
wx = WIND_SPEED * math.cos(cross_theta)
wy = WIND_SPEED * math.sin(cross_theta)

sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

print('Running smoothed straight-line crosswind test T=', T)
sim.run()
print('Trace written to', trace)
