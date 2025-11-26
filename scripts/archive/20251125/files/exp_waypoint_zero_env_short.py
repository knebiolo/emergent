"""Short experiment: single vessel 180s with zero wind/current, writes enriched PID trace."""
import numpy as np
import os
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

PID_TRACE['path'] = 'scripts/pid_trace_waypoint_zero_env_short.csv'
PID_TRACE['enabled'] = True

# Parameters
PORT = 'Rosario Strait'
DT = 0.5
T = 180.0  # 3 minutes
N_AGENTS = 1

# Environmental forcing (zero)
WIND_EAST = 0.0
WIND_NORTH = 0.0
CUR_EAST = 0.0
CUR_NORTH = 0.0

def make_const_field(u, v):
    def sampler(lons, lats, when):
        import numpy as _np
        lons = _np.atleast_1d(lons)
        N = lons.size
        out = _np.tile(_np.array([[float(u), float(v)]]), (N, 1))
        return out
    return sampler

wind_fn = make_const_field(WIND_EAST, WIND_NORTH)
cur_fn  = make_const_field(CUR_EAST, CUR_NORTH)

print(f"[EXP_ZERO] Creating simulation: port={PORT}, dt={DT}, T={T}, agents={N_AGENTS}")
sim = simulation(PORT, dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, verbose=False)

cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
half_sep = 250.0
start = (float(cx - half_sep), float(cy))
end   = (float(cx + half_sep), float(cy))
sim.waypoints = [[start, end]]
print(f"[EXP_ZERO] Waypoints (UTM meters): start={start}, end={end}")

sim.wind_fn = wind_fn
sim.current_fn = cur_fn

sim.spawn()
print("[EXP_ZERO] Running short simulation...")
sim.run()
print('[EXP_ZERO] Done')
