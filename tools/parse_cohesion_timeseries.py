"""Parse per-step behavior NPZs and trace CSV to compute cohesion vs heading timeseries.

Outputs:
 - outputs/diagnostics/behavior_cohesion_alignment_timeseries.csv
 - outputs/diagnostics/behavior_cohesion_alignment_summary.json

Assumptions:
 - Runner produced a trace CSV named <model_name>_trace.csv (we will detect the most recent cue_test_cohesion* trace)
 - Per-step NPZs are in outputs/diagnostics/behavior_debug_step_*.npz and include neighbor_counts and neighbors_concat

This script is defensive and will skip steps without neighbor data.
"""
import os
import glob
import numpy as np
import csv
import math
import json

OUTDIR = os.path.join('outputs', 'diagnostics')


def load_trace_csv(model_name_pattern='cue_test_cohesion'):
    # find latest matching trace CSV
    matches = sorted(glob.glob(os.path.join(OUTDIR, f"{model_name_pattern}*_trace.csv")))
    if not matches:
        # fallback: any trace.csv
        matches = sorted(glob.glob(os.path.join(OUTDIR, "*_trace.csv")))
    if not matches:
        raise RuntimeError('No trace CSV found in outputs/diagnostics')
    trace_file = matches[-1]
    # read into dict: (t,agent)->(x,y,heading)
    pos = {}
    headings = {}
    with open(trace_file, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t = int(float(row['timestep']))
                a = int(row['agent'])
                x = float(row.get('x', 'nan'))
                y = float(row.get('y', 'nan'))
            except Exception:
                continue
            pos.setdefault(t, {})[a] = (x, y)
            # some runs persist heading in CSV; otherwise we'll read HDF5 if needed
            if 'heading' in row and row['heading'] != '':
                try:
                    headings.setdefault(t, {})[a] = float(row['heading'])
                except Exception:
                    pass
    return trace_file, pos, headings


def load_headings_from_h5():
    # find latest headless h5 and read agent_data/heading or top-level heading
    matches = sorted(glob.glob(os.path.join(OUTDIR, '*_headless.h5')))
    if not matches:
        return {}
    fn = matches[-1]
    try:
        import h5py
        with h5py.File(fn, 'r') as h5:
            # attempt to read agent_data/heading (num_agents x num_timesteps)
            if 'agent_data/heading' in h5:
                arr = np.array(h5['agent_data/heading'])
                # convert to dict: pos[t][agent]=heading
                headings = {}
                # arr shape (num_agents, num_timesteps)
                num_timesteps = arr.shape[1] if arr.ndim > 1 else 1
                for t in range(num_timesteps):
                    headings.setdefault(t, {})
                    for a in range(arr.shape[0]):
                        try:
                            val = float(arr[a, t])
                        except Exception:
                            val = float('nan')
                        headings[t][a] = val
                return headings
            elif 'heading' in h5:
                arr = np.array(h5['heading'])
                # treat as headings at timestep 0
                headings = {0: {a: float(arr[a]) for a in range(arr.size)}}
                return headings
    except Exception:
        return {}
    return {}


def iter_step_npzs():
    files = sorted(glob.glob(os.path.join(OUTDIR, 'behavior_debug_step_*.npz')))
    for f in files:
        try:
            # parse step from filename behavior_debug_step_<step>_<ts>.npz
            base = os.path.basename(f)
            parts = base.split('_')
            step = int(parts[3])
        except Exception:
            # skip unparseable
            continue
        yield step, f


def concat_neighbors(neighbor_counts, neighbors_concat):
    # neighbor_counts: (n,), neighbors_concat: (m,) flattened indices
    n = int(neighbor_counts.size)
    out = []
    idx = 0
    for i in range(n):
        c = int(neighbor_counts[i])
        if c > 0:
            out.append(np.asarray(neighbors_concat[idx:idx+c], dtype=int))
        else:
            out.append(np.array([], dtype=int))
        idx += c
    return out


def angle_between_vectors(ax, ay, bx, by):
    # returns absolute smallest angle between vectors in radians
    a = math.atan2(ay, ax)
    b = math.atan2(by, bx)
    d = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return d


def main():
    trace_file, pos, headings_csv = load_trace_csv()
    print('Using trace:', trace_file)
    # load headings from HDF5 as fallback when CSV doesn't contain them
    headings_h5 = load_headings_from_h5()
    if headings_h5:
        print('Loaded headings from HDF5 with %d timesteps' % len(headings_h5))
    else:
        print('No headings found in HDF5 fallback')

    rows = []
    steps_seen = set()
    for step, npz in iter_step_npzs():
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        if 'neighbor_counts' not in d or 'neighbors_concat' not in d:
            continue
        nc = np.asarray(d['neighbor_counts'])
        nc = nc.astype(int)
        neighbors_concat = np.asarray(d['neighbors_concat']) if 'neighbors_concat' in d else np.array([], dtype=int)
        neighbors = concat_neighbors(nc, neighbors_concat)
        # collect per-agent cohesion vectors
        # need positions at this step
        if step not in pos:
            # skip if no positions
            continue
        P = pos[step]
        n_agents = len(nc)
        for a in range(n_agents):
            nbrs = neighbors[a]
            if nbrs.size == 0:
                # no neighbors -> skip or record NaNs
                rows.append((step, a, float('nan'), float('nan'), float('nan')))
                continue
            # compute centroid of neighbor positions
            xs = []
            ys = []
            for nb in nbrs:
                if nb in P:
                    xnb, ynb = P[nb]
                    xs.append(xnb)
                    ys.append(ynb)
            if len(xs) == 0:
                rows.append((step, a, float('nan'), float('nan'), float('nan')))
                continue
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            axp, ayp = P[a]
            vx = cx - axp
            vy = cy - ayp
            # compute heading: prefer CSV headings, fallback to HDF5
            head_angle = None
            if step in headings_csv and a in headings_csv[step]:
                head_angle = headings_csv[step][a]
            elif headings_h5 and step in headings_h5 and a in headings_h5[step]:
                head_angle = headings_h5[step][a]
            rows.append((step, a, vx, vy, head_angle if head_angle is not None else float('nan')))
        steps_seen.add(step)

    # write timeseries CSV
    out_csv = os.path.join(OUTDIR, 'behavior_cohesion_alignment_timeseries.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['step', 'agent', 'cohesion_vx', 'cohesion_vy', 'heading_radians'])
        for r in sorted(rows):
            w.writerow(r)
    print('Wrote timeseries CSV:', out_csv)

    # compute summary stats (angle diffs where heading present)
    diffs = []
    for step, agent, vx, vy, heading in rows:
        if math.isnan(vx) or heading is None or (isinstance(heading, float) and math.isnan(heading)):
            continue
        d = angle_between_vectors(vx, vy, math.cos(heading), math.sin(heading))
        diffs.append(d)
    out_summary = {
        'n_steps': len(steps_seen),
        'n_rows': len(rows),
        'n_with_heading': len(diffs),
        'mean_abs_angle_deg': float(np.degrees(np.mean(diffs))) if len(diffs) > 0 else None,
        'median_abs_angle_deg': float(np.degrees(np.median(diffs))) if len(diffs) > 0 else None,
        'max_abs_angle_deg': float(np.degrees(np.max(diffs))) if len(diffs) > 0 else None,
    }
    summary_file = os.path.join(OUTDIR, 'behavior_cohesion_alignment_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as fh:
        json.dump(out_summary, fh, indent=2)
    print('Wrote summary JSON:', summary_file)


if __name__ == '__main__':
    main()
