#!/usr/bin/env python3
"""Parse behavior_debug_step_*.npz and trace CSV for alignment cue diagnostics.

This script will load the latest probe_alignment_only_trace.csv and the closest
behavior_debug_step_*.npz files, reconstruct per-agent alignment vectors (if
present) or compute them from neighbor headings in the NPZ, and compute angle
between alignment vector and agent heading.
"""
import os, glob, csv, math, json
import numpy as np
import h5py


def load_step_payload(path, step=None):
    # Accept .npz or .h5 path. If path is a directory pattern, caller should pass file.
    if path.endswith('.h5') or path.endswith('.hdf5'):
        with h5py.File(path, 'r') as h5:
            grp = h5.get('steps')
            if grp is None:
                return {}
            key = str(step or 0)
            if key not in grp:
                return {}
            out = {}
            g = grp[key]
            for k in g.keys():
                out[k] = g[k][()]
            return out
    else:
        try:
            return dict(np.load(path, allow_pickle=True))
        except Exception:
            return {}

OUTDIR = os.path.join('outputs', 'diagnostics')

# find latest trace
traces = sorted(glob.glob(os.path.join(OUTDIR, '*probe_alignment_only*_trace.csv')))
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

# gather NPZs or h5 diagnostics
npzs = sorted(glob.glob(os.path.join(OUTDIR, 'behavior_debug_step_*.npz')))
h5s = sorted(glob.glob(os.path.join(OUTDIR, '*_diagnostics.h5')))
if not npzs and not h5s:
    print('No behavior diagnostics found (NPZ or HDF5)')
    raise SystemExit(0)

rows = []
# prefer HDF5 if present
if h5s:
    for h5p in h5s:
        # assume step 0 present; if multiple steps then this will need expanding
        payload = load_step_payload(h5p, step=0)
        if not payload:
            continue
        base = os.path.basename(h5p)
        step = 0
        d = payload
        # attempt to get alignment vectors per-agent
        if 'alignment_vec' in d:
            head_vec = d.get('alignment_vec')
        else:
            head_vec = None
        # process similarly below
        if head_vec is not None:
            for a in range(np.array(head_vec).shape[0]):
                vx, vy = float(head_vec[a,0]), float(head_vec[a,1])
                heading = headings.get(step, {}).get(a, float('nan'))
                rows.append((step, a, vx, vy, heading))
        else:
            nc = d.get('neighbor_counts')
            concat = d.get('neighbors_concat')
            hn = d.get('headings_neighbors_used')
            if nc is None or concat is None or hn is None:
                continue
            counts = np.asarray(nc)
            concat = np.asarray(concat, dtype=int)
            neigh_headings = np.asarray(hn)
            idx = 0
            for a in range(counts.size):
                c = int(counts[a])
                if c == 0:
                    rows.append((step, a, float('nan'), float('nan'), headings.get(step, {}).get(a, float('nan'))))
                else:
                    seg = neigh_headings[idx:idx+c]
                    mean_x = np.mean(np.cos(seg))
                    mean_y = np.mean(np.sin(seg))
                    rows.append((step, a, float(mean_x), float(mean_y), headings.get(step, {}).get(a, float('nan'))))
                idx += c
else:
    for p in npzs:
        base = os.path.basename(p)
        try:
            step = int(base.split('_')[3])
        except Exception:
            continue
        d = np.load(p, allow_pickle=True)
        # attempt to get alignment vectors per-agent (support both alignment_vec and legacy head_vec)
        if 'alignment_vec' in d:
            head_vec = d['alignment_vec']
        elif 'head_vec' in d:
            head_vec = d['head_vec']
        else:
            head_vec = None
        if head_vec is not None:
            for a in range(np.array(head_vec).shape[0]):
                vx, vy = float(head_vec[a,0]), float(head_vec[a,1])
                heading = headings.get(step, {}).get(a, float('nan'))
                rows.append((step, a, vx, vy, heading))
        else:
            # reconstruct per-agent desired heading from neighbors (use headings_neighbors_used and neighbor counts/concat)
            nc = d['neighbor_counts'] if 'neighbor_counts' in d else None
            concat = d['neighbors_concat'] if 'neighbors_concat' in d else None
            hn = d['headings_neighbors_used'] if 'headings_neighbors_used' in d else None
            if nc is None or concat is None or hn is None:
                # skip if insufficient data
                continue
            # concat corresponds to neighbor indices; hn corresponds to per-neighbor headings
            counts = np.asarray(nc)
            concat = np.asarray(concat, dtype=int)
            neigh_headings = np.asarray(hn)
            idx = 0
            for a in range(counts.size):
                c = int(counts[a])
                if c == 0:
                    rows.append((step, a, float('nan'), float('nan'), headings.get(step, {}).get(a, float('nan'))))
                else:
                    seg = neigh_headings[idx:idx+c]
                    mean_x = np.mean(np.cos(seg))
                    mean_y = np.mean(np.sin(seg))
                    rows.append((step, a, float(mean_x), float(mean_y), headings.get(step, {}).get(a, float('nan'))))
                idx += c

# compute diffs
diffs = []
for step,a,vx,vy,heading in rows:
    if math.isnan(vx) or math.isnan(heading):
        continue
    # compute angle between vector (vx,vy) and heading vector (cos, sin)
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

# dump small CSV
out_csv = os.path.join(OUTDIR, 'behavior_alignment_timeseries_probe.csv')
with open(out_csv, 'w', newline='') as fh:
    import csv
    w = csv.writer(fh)
    w.writerow(['step','agent','align_vx','align_vy','heading_radians'])
    for r in rows:
        w.writerow(r)
print('Wrote', out_csv)
