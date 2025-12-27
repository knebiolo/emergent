#!/usr/bin/env python3
"""Parse behavior_debug_step_*.npz and trace CSV for collision cue diagnostics.

This will look for per-step NPZs and compute per-agent collision vectors
(closest-agent vectors) and compare with agent headings.
"""
import os, glob, csv, math, json
import numpy as np

OUTDIR = os.path.join('outputs', 'diagnostics')

# find latest trace
traces = sorted(glob.glob(os.path.join(OUTDIR, '*probe_collision_only*_trace.csv')))
if not traces:
    traces = sorted(glob.glob(os.path.join(OUTDIR, '*_trace.csv')))
if not traces:
    print('No trace CSV found')
    raise SystemExit(0)
trace = traces[-1]
print('Using trace:', trace)

# load trace headings per timestep-agent
pos = {}
headings = {}
with open(trace, 'r', newline='') as fh:
    r = csv.DictReader(fh)
    for row in r:
        try:
            t = int(float(row['timestep']))
            a = int(row['agent'])
            pos.setdefault(t, {})[a] = (float(row.get('x', 'nan')), float(row.get('y','nan')))
            if 'heading' in row and row['heading'] != '':
                try:
                    headings.setdefault(t, {})[a] = float(row['heading'])
                except Exception:
                    pass
        except Exception:
            pass

# gather NPZs
npzs = sorted(glob.glob(os.path.join(OUTDIR, 'behavior_debug_step_*.npz')))
if not npzs:
    print('No behavior_debug_step NPZs found')
    raise SystemExit(0)

rows = []
# load HDF5 fallback headings if CSV missing them
h5_headings = {}
h5_matches = sorted(glob.glob(os.path.join(OUTDIR, '*_headless.h5')))
if h5_matches:
    try:
        import h5py
        with h5py.File(h5_matches[-1], 'r') as h5f:
            if 'agent_data/heading' in h5f:
                arr = np.array(h5f['agent_data/heading'])
                # arr shape (n_agents, n_timesteps)
                if arr.ndim == 2:
                    tsteps = arr.shape[1]
                    for t0 in range(tsteps):
                        h5_headings.setdefault(t0, {})
                        for a0 in range(arr.shape[0]):
                            try:
                                h5_headings[t0][a0] = float(arr[a0, t0])
                            except Exception:
                                h5_headings[t0][a0] = float('nan')
            elif 'heading' in h5f:
                arr = np.array(h5f['heading'])
                h5_headings[0] = {a: float(arr[a]) for a in range(arr.size)}
    except Exception:
        h5_headings = {}
if h5_headings:
    print('Loaded HDF5 headings fallback with %d timesteps' % len(h5_headings))
for p in npzs:
    base = os.path.basename(p)
    try:
        step = int(base.split('_')[3])
    except Exception:
        continue
    d = np.load(p, allow_pickle=True)
    # attempt to get closest_agent or neighbor summaries
    if 'closest_agent' in d and 'nearest_neighbor_distance' in d:
        ca = d['closest_agent']
        nnd = d['nearest_neighbor_distance']
        # ca may be float array with NaNs
        for a in range(ca.size):
            try:
                nb = int(ca[a])
            except Exception:
                rows.append((step, a, float('nan'), float('nan'), headings.get(step, {}).get(a, float('nan'))))
                continue
            if nb < 0:
                rows.append((step, a, float('nan'), float('nan'), headings.get(step, {}).get(a, float('nan'))))
                continue
            # compute vector from agent to closest neighbor using trace positions when available
            if step in pos and a in pos[step] and nb in pos[step]:
                ax, ay = pos[step][a]
                bx, by = pos[step][nb]
                vx = bx - ax
                vy = by - ay
                # prefer CSV heading, fallback to HDF5 headings when available
                head = headings.get(step, {}).get(a, float('nan'))
                if (head is None or (isinstance(head, float) and math.isnan(head))) and step in h5_headings and a in h5_headings[step]:
                    head = h5_headings[step][a]
                rows.append((step, a, vx, vy, head))
            else:
                head = headings.get(step, {}).get(a, float('nan'))
                if (head is None or (isinstance(head, float) and math.isnan(head))) and step in h5_headings and a in h5_headings[step]:
                    head = h5_headings[step][a]
                rows.append((step, a, float('nan'), float('nan'), head))
    else:
        # fallback: use neighbors_concat and neighbor_counts to compute min-distance neighbor vector
        if 'neighbor_counts' in d and 'neighbors_concat' in d:
            nc = np.asarray(d['neighbor_counts'])
            concat = np.asarray(d['neighbors_concat'])
            # reconstruct positions from trace if possible
            idx = 0
            for a in range(nc.size):
                c = int(nc[a])
                if c == 0:
                    rows.append((step, a, float('nan'), float('nan'), headings.get(step, {}).get(a, float('nan'))))
                    continue
                nbrs = concat[idx:idx+c]
                # select closest neighbor by distance using trace positions
                dists = []
                AXAY = pos.get(step, {}).get(a, (float('nan'), float('nan')))
                ax, ay = AXAY
                for nb in nbrs:
                    if nb in pos.get(step, {}):
                        bx, by = pos[step][int(nb)]
                        dists.append(((bx-ax)**2 + (by-ay)**2, int(nb), bx-ax, by-ay))
                if not dists:
                    head = headings.get(step, {}).get(a, float('nan'))
                    if (head is None or (isinstance(head, float) and math.isnan(head))) and step in h5_headings and a in h5_headings[step]:
                        head = h5_headings[step][a]
                    rows.append((step, a, float('nan'), float('nan'), head))
                else:
                    best = min(dists, key=lambda x: x[0])
                    head = headings.get(step, {}).get(a, float('nan'))
                    if (head is None or (isinstance(head, float) and math.isnan(head))) and step in h5_headings and a in h5_headings[step]:
                        head = h5_headings[step][a]
                    rows.append((step, a, float(best[2]), float(best[3]), head))
                idx += c

# compute diffs
diffs = []
for step,a,vx,vy,heading in rows:
    if math.isnan(vx) or math.isnan(heading):
        continue
    a_ang = math.atan2(vy, vx)
    h_ang = heading
    d = abs((a_ang - h_ang + math.pi) % (2*math.pi) - math.pi)
    diffs.append(d)

print('rows total:', len(rows))
print('n diffs (with heading):', len(diffs))
if diffs:
    print('mean abs angle deg:', float(np.degrees(np.mean(diffs))))
    print('median abs angle deg:', float(np.degrees(np.median(diffs))))
    print('max abs angle deg:', float(np.degrees(np.max(diffs))))

out_csv = os.path.join(OUTDIR, 'behavior_collision_timeseries_probe.csv')
with open(out_csv, 'w', newline='') as fh:
    import csv
    w = csv.writer(fh)
    w.writerow(['step','agent','coll_vx','coll_vy','heading_radians'])
    for r in rows:
        w.writerow(r)
print('Wrote', out_csv)

summary = {
    'n_rows': len(rows),
    'n_with_heading': len(diffs),
    'mean_abs_angle_deg': float(np.degrees(np.mean(diffs))) if diffs else None,
    'median_abs_angle_deg': float(np.degrees(np.median(diffs))) if diffs else None,
    'max_abs_angle_deg': float(np.degrees(np.max(diffs))) if diffs else None,
}
with open(os.path.join(OUTDIR, 'behavior_collision_alignment_summary.json'), 'w', encoding='utf-8') as jf:
    json.dump(summary, jf, indent=2)
print('Wrote summary JSON')
