"""Run a short headless simulation with dead-reck enabled and crosswind/current
to capture the DR debug prints added to compute_desired().
"""
import os, time, math
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

DT = 0.1
T = 6.0
WIND_SPEED = 5.0

config.PID_TRACE['enabled'] = False
config.PID_DEBUG = True

# make sim with 1 agent
sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False, test_mode='straight', verbose=True)
# simple crosswind function: wind from starboard (right) -> north if psi=0
def constant_crosswind_for_sim(sim, wind_speed):
    psi0 = float(sim.psi[0])
    cross_theta = psi0 + math.pi / 2.0
    wx = wind_speed * math.cos(cross_theta)
    wy = wind_speed * math.sin(cross_theta)
    def wind_fn(lon, lat, when):
        # return shape (2, n)
        return np.tile(np.array([[wx],[wy]]), (1, sim.n))
    return wind_fn

# Cross current as well
def constant_current_for_sim(sim, cur_speed):
    psi0 = float(sim.psi[0])
    # push northwards
    cx, cy = 0.0, cur_speed
    def cur_fn(lon, lat, when):
        return np.tile(np.array([[cx],[cy]]), (1, sim.n))
    return cur_fn

# set waypoints as lon/lat to exercise projection code path
try:
    sim.waypoints = [[(-94.8, 29.3), (-94.7, 29.35)]]
except Exception:
    pass

sim.spawn()
# ensure dead-reck enabled (default)
try:
    if hasattr(sim.ship, 'disable_dead_reck'):
        delattr(sim.ship, 'disable_dead_reck')
except Exception:
    pass

sim.wind_fn = constant_crosswind_for_sim(sim, WIND_SPEED)
sim.current_fn = constant_current_for_sim(sim, 0.5)

start = time.time()
sim.run()
dur = time.time() - start
print(f"run finished dt={DT} T={T} duration_s={dur:.2f}")
