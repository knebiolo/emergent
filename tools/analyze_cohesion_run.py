import os
import math
import csv
import numpy as np
import h5py
from scipy.spatial import cKDTree

def load_h5(path):
    return h5py.File(path, 'r')

def compute_neighbors(xs, ys, radius):
    pts = np.column_stack((xs, ys))
    tree = cKDTree(pts)
    nbrs = tree.query_ball_tree(tree, r=radius)
    # convert to arrays excluding self
    agents_within = [np.array([j for j in lst if j != i], dtype=int) for i, lst in enumerate(nbrs)]
    return agents_within

def angle_between(a, b):
    ang1 = math.degrees(math.atan2(a[1], a[0]))
    ang2 = math.degrees(math.atan2(b[1], b[0]))
    d = ang2 - ang1
    d = (d + 180) % 360 - 180
    return d

def analyze(h5path, weights_file=None, out_csv=None):
    h5 = load_h5(h5path)
    # positions: use agent_data/X time 0 if present
    ad = h5['agent_data']
    X = ad['X'][:]
    Y = ad['Y'][:]
    if X.ndim == 2:
        xs = X[:, 0]
    else:
        xs = X[:]
    if Y.ndim == 2:
        ys = Y[:, 0]
    else:
        ys = Y[:]

    n = xs.size

    # length to compute neighbor radius
    length = None
    if 'length' in h5:
        length = np.asarray(h5['length'][:])
    elif 'agent_data' in h5 and 'length' in h5['agent_data']:
        length = np.asarray(h5['agent_data']['length'][:])
    if length is None:
        # fallback to 50 m buffer
        radius = 50.0
    else:
        # mimic simulation default: max(10.0, (mean_length/100.0)*5.0)
        try:
            radius = max(10.0, (float(np.mean(length)) / 100.0) * 5.0)
        except Exception:
            radius = 50.0

    agents_within = compute_neighbors(xs, ys, radius)

    # load weight
    cohesion_w = 11000.0
    if weights_file and os.path.exists(weights_file):
        try:
            import json
            with open(weights_file, 'r', encoding='utf-8') as fh:
                w = json.load(fh)
            cohesion_w = float(w.get('cohesion', cohesion_w))
        except Exception:
            pass

    # compute cohesion vec per agent
    coh_x = np.zeros(n)
    coh_y = np.zeros(n)
    for i in range(n):
        nbrs = agents_within[i]
        if nbrs.size == 0:
            coh_x[i] = 0.0
            coh_y[i] = 0.0
            continue
        cx = float(np.mean(xs[nbrs]))
        cy = float(np.mean(ys[nbrs]))
        vx = cx - xs[i]
        vy = cy - ys[i]
        mag = math.hypot(vx, vy)
        if mag == 0:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = vx / mag, vy / mag
        coh_x[i] = cohesion_w * ux
        coh_y[i] = cohesion_w * uy

    # load final headings
    headings = None
    if 'agent_data' in h5 and 'heading' in h5['agent_data']:
        head_arr = np.asarray(h5['agent_data']['heading'][:])
        if head_arr.ndim == 2:
            headings = head_arr[:, -1]
        else:
            headings = head_arr[:]
    elif 'heading' in h5:
        headings = np.asarray(h5['heading'][:])
    else:
        raise RuntimeError('No heading data in HDF5')

    head_x = np.cos(headings)
    head_y = np.sin(headings)

    rows = []
    for i in range(n):
        samp = (coh_x[i], coh_y[i])
        head = (head_x[i], head_y[i])
        ang = angle_between(samp, head)
        rows.append({'agent': i, 'coh_x': samp[0], 'coh_y': samp[1], 'head_x': head[0], 'head_y': head[1], 'ang_head_vs_cohesion_deg': ang, 'neighbor_count': int(agents_within[i].size)})

    if out_csv:
        keys = ['agent','neighbor_count','coh_x','coh_y','head_x','head_y','ang_head_vs_cohesion_deg']
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    angs = np.array([r['ang_head_vs_cohesion_deg'] for r in rows])
    print('n_agents', n)
    print('mean_abs_ang_head_vs_cohesion_deg', np.mean(np.abs(angs)))
    print('max_abs_ang_head_vs_cohesion_deg', np.max(np.abs(angs)))
    h5.close()
    return rows

if __name__ == '__main__':
    import sys
    h5path = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/cue_test_cohesion_20251226T000000Z_headless.h5'
    weights = sys.argv[2] if len(sys.argv)>2 else 'outputs/diagnostics/fresh_test_weights_cohesion_only.json'
    outcsv = sys.argv[3] if len(sys.argv)>3 else 'outputs/diagnostics/cue_test_cohesion_20251226T000000Z_alignment_peragent.csv'
    print('Analyzing cohesion run', h5path)
    analyze(h5path, weights, outcsv)
