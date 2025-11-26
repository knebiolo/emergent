"""Run a straight-line crosswind test with hd_cmd smoothing enabled and T=300s
Writes trace to sweep_results/pid_trace_straight_wind_smoothed.csv (overwrites previous)
"""
import os, math, numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1
T = 300.0
WIND_SPEED = 5.0

trace = os.path.join(OUT_DIR, 'pid_trace_straight_wind_smoothed.csv')
PID_TRACE['enabled'] = True
PID_TRACE['path'] = trace

# build sim, straight goal
start = np.array([-122.7, 48.2])
goal = start + np.array([0.05, 0.0])

sim = simulation(port_name='Rosario Strait', dt=DT, T=T, n_agents=1, load_enc=False, test_mode=None)
# keep verbose False for clean runs
sim.verbose = False

sim.waypoints = [[start, goal]]
sim.spawn()

psi0 = float(sim.psi[0])
cross_theta = psi0 + math.pi/2.0
wx = WIND_SPEED * math.cos(cross_theta)
wy = WIND_SPEED * math.sin(cross_theta)

sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

print('Running smoothed straight-line crosswind test T=', T)
try:
	if os.path.exists(trace):
		os.remove(trace)
except Exception:
	pass
sim.run()
print('Trace written to', trace)
