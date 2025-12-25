#!/usr/bin/env python3
"""Compare sampled raster velocities in trace CSV to rheotaxis_vec in debug NPZs.

Writes a CSV per NPZ summarizing per-agent angle of sampled vel and rheotaxis_vec and their difference.
"""
import argparse, csv, os, glob, math
import numpy as np

def angle_deg(vx, vy):
    return math.degrees(math.atan2(vy, vx))

def load_trace(trace_csv):
    trace = {}
    with open(trace_csv, 'r', newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            t = int(row['timestep'])
            a = int(row['agent'])
            vx = float(row.get('vel_x_sample', 'nan'))
            vy = float(row.get('vel_y_sample', 'nan'))
            trace[(t,a)] = (vx, vy)
    return trace


def process(npzfile, trace, outdir):
    d = np.load(npzfile)
    base = os.path.basename(npzfile)
    # filename format: behavior_debug_step_<step>_<ts>.npz
    parts = base.split('_')
    try:
        idx = parts.index('step')
        timestep = int(parts[idx+1])
    except Exception:
        # fallback: try to find the numeric part after 'behavior_debug_step'
        for i,p in enumerate(parts):
            if p.startswith('step'):
                try:
                    timestep = int(p.replace('step',''))
                    break
                except Exception:
                    continue
        else:
            # last resort: pull the third underscore token as int
            timestep = int(parts[2])
    # rheotaxis_vec should be (N,2)
    if 'rheotaxis_vec' not in d.files:
        print('no rheotaxis_vec in', npzfile); return 1
    rv = np.asarray(d['rheotaxis_vec']).astype(float)
    N = rv.shape[0]
    rows = [['agent','vel_x_sample','vel_y_sample','vel_sample_ang_deg','rheo_x','rheo_y','rheo_ang_deg','delta_deg']]
    for a in range(N):
        vx, vy = trace.get((timestep, a), (float('nan'), float('nan')))
        if vx == -9999.0 or vy == -9999.0:
            vel_ang = float('nan')
        else:
            vel_ang = angle_deg(vx, vy)
        rx, ry = float(rv[a,0]), float(rv[a,1])
        if math.isfinite(rx) and math.isfinite(ry) and (rx!=0 or ry!=0):
            rheo_ang = angle_deg(rx, ry)
            if math.isfinite(vel_ang):
                ddeg = ((rheo_ang - vel_ang + 180) % 360) - 180
            else:
                ddeg = float('nan')
        else:
            rheo_ang = float('nan'); ddeg = float('nan')
        rows.append([a, vx, vy, vel_ang, rx, ry, rheo_ang, ddeg])
    out = os.path.join(outdir, base.replace('.npz','_rheo_check.csv'))
    with open(out,'w',newline='') as fh:
        w=csv.writer(fh); w.writerows(rows)
    # summary
    diffs = [abs(r[-1]) for r in rows[1:] if isinstance(r[-1], float) and not math.isnan(r[-1])]
    cnt = len(diffs)
    mean = sum(diffs)/cnt if cnt else float('nan')
    within_30 = sum(1 for v in diffs if v<=30)
    print(f'Wrote {out}; mean_abs_delta_deg={mean:.2f} count={cnt} within_30={within_30}')
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dir','-d',default='outputs/diagnostics')
    p.add_argument('--trace',default='outputs/diagnostics/behavior_diag_50x20_fix_trace.csv')
    args=p.parse_args()
    trace = load_trace(args.trace)
    files = sorted([f for f in glob.glob(os.path.join(args.dir,'behavior_debug_step_*.npz')) if not f.endswith('_alignment.npz')])
    if not files:
        print('no files'); return 2
    picks = [files[0]]
    if len(files)>2:
        picks.append(files[len(files)//2])
    if len(files)>1:
        picks.append(files[-1])
    for f in picks:
        process(f, trace, args.dir)

if __name__=='__main__':
    raise SystemExit(main())
