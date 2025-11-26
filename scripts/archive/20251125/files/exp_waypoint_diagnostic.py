"""Diagnostic: run three short waypoint tests (60s) with different wind settings.
Saves PID traces and position histories for offline inspection.
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

DT = 0.5
T = 60.0
N_AGENTS = 1
PORT = 'Rosario Strait'

# scenario definitions: (name, wind_east, wind_north, cur_east, cur_north)
SCENARIOS = [
    ('no_wind', 0.0, 0.0, 0.0, 0.0),
    ('light_wind', 0.0, 1.0, 0.0, -0.1),
    ('orig_wind', 0.0, 6.0, 0.0, -0.4),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), 'diag_outputs')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def make_const_field(u, v):
    def sampler(lons, lats, when):
        import numpy as _np
        lons = _np.atleast_1d(lons)
        N = lons.size
        out = _np.tile(_np.array([[float(u), float(v)]]), (N, 1))
        return out
    return sampler

for name, we, wn, ce, cn in SCENARIOS:
    print(f"\n[DIAG] Running scenario: {name}")
    PID_TRACE['enabled'] = True
    pid_path = os.path.join(OUT_DIR, f'pid_trace_{name}.csv')
    PID_TRACE['path'] = pid_path

    wind_fn = make_const_field(we, wn)
    cur_fn  = make_const_field(ce, cn)

    sim = simulation(PORT, dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, verbose=False)

    # place 0.5 km apart centered as before
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    half_sep = 250.0
    start = (float(cx - half_sep), float(cy))
    end   = (float(cx + half_sep), float(cy))
    sim.waypoints = [[start, end]]
    sim.wind_fn = wind_fn
    sim.current_fn = cur_fn

    sim.spawn()
    sim.run()

    # save history
    hist = np.array(sim.history[0])
    np.save(os.path.join(OUT_DIR, f'history_{name}.npy'), hist)

    # compute simple CTE as in experiment
    p0 = np.array(start)
    p1 = np.array(end)
    seg = p1 - p0
    seg_len = np.linalg.norm(seg) + 1e-12
    vecs = hist - p0
    proj = np.dot(vecs, seg) / (seg_len**2)
    proj_clamped = np.minimum(np.maximum(proj, 0.0), 1.0)
    closest = p0.reshape(1,2) + np.outer(proj_clamped, seg)
    dists = np.linalg.norm(hist - closest, axis=1)
    mean_cte = float(np.mean(np.abs(dists)))
    max_cte = float(np.max(np.abs(dists)))

    print(f"[DIAG] Scenario={name}: mean_cte_m={mean_cte:.3f}, max_cte_m={max_cte:.3f}, pid_trace={pid_path}")

print('\n[DIAG] All scenarios complete. Files in', OUT_DIR)
