"""Quick 30s straight-line headless test to verify hd_cmd normalization fix.
Writes: sweep_results/pid_trace_quick_short.csv and sweep_results/quick_short_summary.csv
"""
import os, time, math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

DT = 0.1
T = 30.0
WIND_SPEED = 5.0

# short candidate settings (leave controller as-configured)
ndelta_mult = 1.0
rudder_tau = 1.0

def constant_crosswind_for_sim(sim, wind_speed):
    psi0 = float(sim.psi[0])
    cross_theta = psi0 + math.pi / 2.0
    wx = wind_speed * math.cos(cross_theta)
    wy = wind_speed * math.sin(cross_theta)
    def wind_fn(lon, lat, when):
        return np.tile(np.array([[wx, wy]]), (1, 1))
    return wind_fn

trace = os.path.join(OUT, 'pid_trace_quick_short_nodead_no_wind.csv')
config.PID_TRACE['enabled'] = True
config.PID_TRACE['path'] = trace
sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                 test_mode='straight', verbose=True)
# enable PID debug prints for this diagnostic run
try:
    config.PID_DEBUG = True
except Exception:
    pass
try:
    sim.waypoints = [[(-94.8, 29.3), (-94.7, 29.35)]]
except Exception:
    pass
sim.spawn()
# Experiment: disable dead-reck correction to see controller-only behavior
try:
    sim.ship.disable_dead_reck = True
except Exception:
    pass
# For this run set wind to zero (no environmental forcing)
sim.wind_fn = lambda lon, lat, when: np.zeros((1, 2))
sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))
try:
    sim.ship.rudder_tau = float(rudder_tau)
except Exception:
    pass
start = time.time()
sim.run()
dur = time.time() - start

# produce a tiny summary
import pandas as pd

df = pd.read_csv(trace)
df0 = df[df['agent'] == 0]
summary = {
    'trace': trace,
    'max_err_deg': float(df0['err_deg'].abs().max()),
    'mean_err_deg': float(df0['err_deg'].abs().mean()),
    'first_err_deg': float(df0['err_deg'].abs().iloc[0]),
    'max_rud_deg': float(df0['rud_deg'].abs().max()),
    'sat_frac': float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0)),
    'run_time_s': dur
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT, 'quick_short_summary.csv'), index=False)
print('done')
