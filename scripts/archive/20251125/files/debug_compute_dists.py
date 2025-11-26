import sys
import math
import numpy as np
import pandas as pd

def point_to_polyline_distance(P, poly):
    px, py = P
    best = float('inf')
    for i in range(len(poly)-1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        denom = vx*vx + vy*vy
        if denom == 0:
            t = 0.0
        else:
            t = (vx*wx + vy*wy) / denom
            t = max(0.0, min(1.0, t))
        cx = ax + t * vx
        cy = ay + t * vy
        d = math.hypot(px - cx, py - cy)
        if d < best:
            best = d
    return best

def recompute(csv_path, leg_length=200.0, zig_amp=30.0):
    df = pd.read_csv(csv_path)
    # take first point as start
    x0 = float(df.iloc[0]['x_m'])
    y0 = float(df.iloc[0]['y_m'])
    # reconstruct poly using same params as zigzag script
    waypoints = [(x0, y0)]
    for i in range(1, 7):
        x = x0 + i * leg_length
        y = y0 + ((-1) ** (i+1)) * zig_amp
        waypoints.append((x, y))

    traj = df[['x_m','y_m']].to_numpy()
    dists = []
    for px, py in traj:
        try:
            d = point_to_polyline_distance((float(px), float(py)), waypoints)
        except Exception:
            d = float('nan')
        dists.append(d)
    dists = np.array(dists, dtype=float)
    finite_ok = np.isfinite(dists).all()
    print('computed dists: finite?', finite_ok)
    print('nan present?', np.isnan(dists).any())
    if finite_ok:
        print('min,max,mean:', float(np.nanmin(dists)), float(np.nanmax(dists)), float(np.nanmean(dists)))
    else:
        print('min,max,mean: (not all finite)')
    print('first 10 dists:', dists[:10].tolist())
    return dists

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: debug_compute_dists.py <traj_csv>')
        sys.exit(1)
    recompute(sys.argv[1])
