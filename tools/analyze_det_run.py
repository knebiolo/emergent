import os
import json
import math
import numpy as np
import h5py
import csv

def load_h5(path):
    return h5py.File(path, 'r')

def sample_velocity(h5):
    env = h5['environment']
    vx = env['vel_x'][:]
    vy = env['vel_y'][:]
    xg = env['x_coords'][:]
    yg = env['y_coords'][:]
    agents = h5['agent_data']
    xs = agents['X'][:]
    ys = agents['Y'][:]
    # if time-series (n_agents, n_steps), use initial positions (t=0)
    if xs.ndim == 2:
        xs = xs[:,0]
    if ys.ndim == 2:
        ys = ys[:,0]
    # nearest-neighbor sample: compute pixel indices from grids
    # assume xg and yg are full grids with constant spacing
    dx = xg[0,1] - xg[0,0]
    dy = yg[1,0] - yg[0,0]
    xi = np.clip(((xs - xg[0,0]) / dx).astype(int), 0, vx.shape[1]-1)
    yi = np.clip(((ys - yg[0,0]) / dy).astype(int), 0, vx.shape[0]-1)
    samp_vx = vx[yi, xi]
    samp_vy = vy[yi, xi]
    return samp_vx, samp_vy

def angle_between(a, b):
    # angle from vector a to vector b in degrees [-180,180]
    ang1 = math.degrees(math.atan2(a[1], a[0]))
    ang2 = math.degrees(math.atan2(b[1], b[0]))
    d = ang2 - ang1
    d = (d + 180) % 360 - 180
    return d

def unit(vx, vy):
    mag = np.sqrt(vx*vx + vy*vy)
    nz = mag > 0
    ux = np.zeros_like(vx)
    uy = np.zeros_like(vy)
    ux[nz] = vx[nz]/mag[nz]
    uy[nz] = vy[nz]/mag[nz]
    return ux, uy

def analyze(h5path, out_csv=None):
    h5 = load_h5(h5path)
    samp_vx, samp_vy = sample_velocity(h5)
    # rheotaxis vector is upstream: -flow
    rheo_vx = -samp_vx
    rheo_vy = -samp_vy
    # final headings persisted
    if 'agent_data' in h5 and 'heading' in h5['agent_data']:
        headings = h5['agent_data']['heading'][:]
        # if heading is time-series, prefer final non-zero column, otherwise fallback to first
        if headings.ndim == 2:
            # if last column is all zero (or near-zero), use first column
            last_col = headings[:, -1]
            if np.allclose(last_col, 0.0):
                headings = headings[:, 0]
            else:
                headings = last_col
    elif 'heading' in h5:
        headings = h5['heading'][:]
    else:
        raise RuntimeError('No persisted headings in HDF5')

    # convert headings to unit vectors
    head_x = np.cos(headings)
    head_y = np.sin(headings)

    # unitize rheotaxis
    urx, ury = unit(rheo_vx, rheo_vy)

    n = len(headings)
    rows = []
    for i in range(n):
        samp = (samp_vx[i], samp_vy[i])
        rheo = (rheo_vx[i], rheo_vy[i])
        head = (head_x[i], head_y[i])
        ang_head_vs_rheo = angle_between(rheo, head)
        ang_flow_vs_rheo = angle_between(samp, rheo)
        rows.append({'agent': i, 'samp_vx': float(samp[0]), 'samp_vy': float(samp[1]), 'rheo_vx': float(rheo[0]), 'rheo_vy': float(rheo[1]), 'head_x': float(head[0]), 'head_y': float(head[1]), 'ang_head_vs_rheo_deg': float(ang_head_vs_rheo), 'ang_flow_vs_rheo_deg': float(ang_flow_vs_rheo)})

    if out_csv:
        keys = ['agent','samp_vx','samp_vy','rheo_vx','rheo_vy','head_x','head_y','ang_head_vs_rheo_deg','ang_flow_vs_rheo_deg']
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # simple aggregates
    angs = np.array([r['ang_head_vs_rheo_deg'] for r in rows])
    print('n_agents', n)
    print('mean_abs_ang_head_vs_rheo_deg', np.mean(np.abs(angs)))
    print('max_abs_ang_head_vs_rheo_deg', np.max(np.abs(angs)))
    h5.close()
    return rows

if __name__ == '__main__':
    import sys
    h5path = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/det_rheo_50x2_headless.h5'
    outcsv = sys.argv[2] if len(sys.argv)>2 else 'outputs/diagnostics/det_rheo_50x2_alignment_peragent.csv'
    print('Analyzing', h5path)
    analyze(h5path, outcsv)
