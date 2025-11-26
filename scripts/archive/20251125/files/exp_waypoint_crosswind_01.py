"""Experiment: single vessel between two waypoints 0.5 km apart with crosswind/current.
Saves PID trace to scripts/pid_trace_waypoint_crosswind.csv and prints summary metrics.
"""
import numpy as np
import os
from datetime import datetime
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

PID_TRACE['path'] = 'scripts/pid_trace_waypoint_crosswind.csv'
PID_TRACE['enabled'] = True

# Parameters
PORT = 'Rosario Strait'
DT = 0.5
T = 600.0  # 10 minutes
N_AGENTS = 1

# Environmental forcing (constant crosswind and cross-current)
# For our coordinate frame: sampler returns (u_east, v_north) per query point
WIND_EAST = 0.0
WIND_NORTH = 6.0   # 6 m/s crosswind northwards
CUR_EAST = 0.0
CUR_NORTH = -0.4   # 0.4 m/s southward current

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

# Create simulation (disable ENC loading for speed)
print(f"[EXP] Creating simulation: port={PORT}, dt={DT}, T={T}, agents={N_AGENTS}")
sim = simulation(PORT, dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, verbose=False)

# choose two waypoints 5 km apart along +x (east) centered in domain
cx = 0.5 * (sim.minx + sim.maxx)
cy = 0.5 * (sim.miny + sim.maxy)
half_sep = 2500.0  # 2.5 km each side -> 5.0 km total
start = (float(cx - half_sep), float(cy))
end   = (float(cx + half_sep), float(cy))
sim.waypoints = [[start, end]]
print(f"[EXP] Waypoints (UTM meters): start={start}, end={end}")

# Attach constant environmental fields
sim.wind_fn = wind_fn
sim.current_fn = cur_fn

# Spawn agents and run headless
try:
    sim.spawn()
except Exception as e:
    print(f"[EXP] spawn() failed: {e}")
    raise

print("[EXP] Running simulation...")
sim.run()

# Post-run analysis
try:
    hist = np.array(sim.history[0])  # shape (steps+1, 2)
except Exception:
    hist = np.vstack([np.array([sim.pos[:,0]])])

# line between start and end
p0 = np.array(start)
p1 = np.array(end)
seg = p1 - p0
seg_len = np.linalg.norm(seg) + 1e-12

# cross-track distance for each recorded position
vecs = hist - p0
# projection length along segment
proj = np.dot(vecs, seg) / (seg_len**2)
proj_clamped = np.minimum(np.maximum(proj, 0.0), 1.0)
closest = p0.reshape(1,2) + np.outer(proj_clamped, seg)
dists = np.linalg.norm(hist - closest, axis=1)
mean_cte = float(np.mean(np.abs(dists)))
max_cte = float(np.max(np.abs(dists)))

# heading error: use psi_history (radians) vs hd_cmd_history (radians)
psi_arr = np.array(sim.psi_history)
hd_cmd = np.array(sim.hd_cmd_history)
# wrap error into [-180,180]
if psi_arr.size and hd_cmd.size and psi_arr.shape == hd_cmd.shape:
    err_deg = (np.degrees(psi_arr) - np.degrees(hd_cmd) + 180) % 360 - 180
    max_heading_err = float(np.max(np.abs(err_deg)))
else:
    max_heading_err = float('nan')

# arrival time: first t where distance to final wp < wp_tol
final_wp = np.array(end)
dists_to_goal = np.linalg.norm(hist - final_wp, axis=1)
arrival_idx = np.where(dists_to_goal <= getattr(sim, 'wp_tol', 50.0))[0]
arrival_time = float(sim.t_history[arrival_idx[0]]) if arrival_idx.size else float('nan')

print("[EXP] Summary:")
print(f"  max_heading_error_deg = {max_heading_err:.3f}")
print(f"  mean_cross_track_error_m = {mean_cte:.3f}")
print(f"  max_cross_track_error_m = {max_cte:.3f}")
print(f"  arrival_time_s = {arrival_time}")
print(f"  pid_trace = {PID_TRACE.get('path')}")

# Save sim histories for offline correlation and debugging
hist_path = PID_TRACE.get('path', 'scripts/pid_trace_waypoint_crosswind.csv').replace('.csv', '')
npz_path = f"{hist_path}_simhist.npz"
try:
    t_arr = np.array(sim.t_history)
    psi_arr = np.array(sim.psi_history)
    hd_arr = np.array(sim.hd_cmd_history)
    pos_hist = np.array(sim.history[0]) if hasattr(sim, 'history') and len(sim.history) > 0 else np.vstack([sim.pos[:,0]])
    np.savez_compressed(npz_path, t=t_arr, psi=psi_arr, hd_cmd=hd_arr, pos=pos_hist)
    print(f"[EXP] Saved sim histories to: {npz_path}")
except Exception as e:
    print(f"[EXP] Failed to save sim histories: {e}")

# exit
print('[EXP] Done at', datetime.utcnow().isoformat())
