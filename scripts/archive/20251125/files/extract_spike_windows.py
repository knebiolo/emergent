"""Extract simulation windows around top heading-error spikes.
Reads the PID trace `scripts/pid_trace_exp_waypoint_crosswind.csv`, finds top N spikes
by absolute err_deg, then re-runs short simulations up to each spike time + post_window
and saves a .npz per spike with pos, psi, hd_cmd, t arrays for offline inspection.

Run: python scripts/extract_spike_windows.py
"""
import os
import math
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

PID_PATH = os.path.join(os.getcwd(), 'scripts', 'pid_trace_exp_waypoint_crosswind.csv')
OUT_DIR = os.path.join(os.getcwd(), 'scripts', 'spike_windows')
os.makedirs(OUT_DIR, exist_ok=True)

# tolerant CSV reader: pick rows that have at least 10 comma-separated fields
rows = []
with open(PID_PATH, 'r', newline='') as fh:
    for line in fh:
        parts = line.strip().split(',')
        # skip empty lines
        if len(parts) < 2:
            continue
        # header detection
        if parts[0].strip() == 't' and parts[1].strip() == 'agent':
            continue
        # require at least t,agent,err_deg,r_des_deg,derr_deg,P_deg,I_deg,D_deg,raw_deg,rud_deg
        if len(parts) < 10:
            continue
        try:
            t = float(parts[0])
            agent = int(parts[1])
            err_deg = float(parts[2])
        except Exception:
            continue
        rows.append({'t': t, 'agent': agent, 'err_deg': err_deg, 'raw': parts})

if not rows:
    print('No valid PID rows found at', PID_PATH)
    raise SystemExit(1)

# sort by absolute error descending
rows_sorted = sorted(rows, key=lambda r: abs(r['err_deg']), reverse=True)
# pick unique times (first occurrence per time)
seen_t = set()
spikes = []
for r in rows_sorted:
    if r['t'] in seen_t:
        continue
    seen_t.add(r['t'])
    spikes.append(r)
    if len(spikes) >= 5:
        break

print('Top spike times (t, err_deg):')
for s in spikes:
    print(f"  {s['t']:.3f}s  err={s['err_deg']:.3f}°")

# Simulation parameters matching exp script
DT = 0.5
N_AGENTS = 1
WAYPOINTS = [[(0.0, 0.0), (500.0, 0.0)]]

# constant env functions (same as experiment)
W_SPEED = 8.0
C_SPEED = 0.5

def constant_wind_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-W_SPEED]]), (1, N_AGENTS))

def constant_current_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-C_SPEED]]), (1, N_AGENTS))

# disable PID_TRACE while re-running short sims (avoid appending to main trace)
PID_TRACE['enabled'] = False

for idx, s in enumerate(spikes):
    t0 = s['t']
    pre = 3.0
    post = 3.0
    t_start = max(0.0, t0 - pre)
    t_end = t0 + post
    steps = int(math.ceil(t_end / DT))
    T = steps * DT
    print(f'Running short sim for spike {idx}: t0={t0} window=({t_start},{t_end}) T={T}s')

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, test_mode=None)
    sim.wind_fn = constant_wind_fn
    sim.current_fn = constant_current_fn
    sim.waypoints = WAYPOINTS
    sim.spawn()
    sim.run()

    # access history
    hist = np.array(sim.history.get(0, []))  # shape (steps+1, 2)
    times = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd_cmd = np.array(sim.hd_cmd_history)

    # find index nearest to t0 in times
    if len(times) == 0:
        print('No times recorded for sim; skipping')
        continue
    idx_nearest = int(np.argmin(np.abs(times - t0)))
    # compute slice indices in hist: times[k] corresponds to hist[k+1]
    k0 = max(0, idx_nearest - int(pre / DT))
    k1 = min(len(times)-1, idx_nearest + int(post / DT))
    # hist indices for these times are +1
    hist_slice = hist[k0+1:k1+2]
    times_slice = times[k0:k1+1]
    psi_slice = psi[k0:k1+1]
    hd_slice = hd_cmd[k0:k1+1]

    out_path = os.path.join(OUT_DIR, f'spike_{idx}_t{t0:.1f}.npz')
    np.savez(out_path,
             t0=t0,
             times=times_slice,
             pos=hist_slice,
             psi=psi_slice,
             hd_cmd=hd_slice)
    print('Saved', out_path)

print('Done extracting windows')
