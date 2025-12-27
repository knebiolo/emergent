#!/usr/bin/env python3
"""Deep rheotaxis analysis across diagnostics NPZs and trace CSV.

Produces a consolidated CSV `rheo_deep_summary.csv` with per-file aggregated stats
and writes a per-agent mismatch CSV for the worst offenders.
"""
import argparse, glob, os, csv, math
import numpy as np

def ang_deg(vx, vy):
    return math.degrees(math.atan2(vy, vx))


def load_trace(trace_file):
    import csv
    out = {}
    with open(trace_file, 'r', newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            t = int(row['timestep'])
            a = int(row['agent'])
            vx = float(row.get('vel_x_sample', 'nan'))
            vy = float(row.get('vel_y_sample', 'nan'))
            out[(t,a)] = (vx, vy)
    return out


def analyze(npzfile, trace, outdir):
    import h5py
    base = os.path.basename(npzfile)
    # detect .h5 vs .npz
    if npzfile.endswith('.h5') or npzfile.endswith('.hdf5'):
        # expect path to an h5 file; read step 0 by default or encoded step
        # try to infer step from basename
        parts = base.split('_')
        step = None
        for i,p in enumerate(parts):
            if p=='step' and i+1 < len(parts):
                try:
                    step = int(parts[i+1]); break
                except Exception:
                    continue
        if step is None:
            step = 0
        with h5py.File(npzfile,'r') as f:
            grp = f.get('steps')
            if grp is None or str(step) not in grp:
                return None
            g = grp[str(step)]
            if 'rheo_vec' not in g:
                return None
            rv = np.asarray(g['rheo_vec']).astype(float)
    else:
        d = np.load(npzfile)
    # base and step detection for NPZ preserved below
    N = rv.shape[0]
    deltas = []
    per_agent = []
    for a in range(N):
        vx, vy = trace.get((step,a), (float('nan'), float('nan')))
        if vx == -9999.0 or vy == -9999.0:
            continue
        vel_ang = ang_deg(vx, vy)
        rx, ry = float(rv[a,0]), float(rv[a,1])
        if rx==0 and ry==0:
            continue
        rheo_ang = ang_deg(rx, ry)
        diff = ((rheo_ang - vel_ang + 180) % 360) - 180
        per_agent.append((a, vel_ang, rheo_ang, diff, math.hypot(rx,ry)))
        deltas.append(abs(diff))
    if not deltas:
        return None
    mean = sum(deltas)/len(deltas)
    pct_within_30 = sum(1 for x in deltas if x<=30)/len(deltas)
    pct_within_90 = sum(1 for x in deltas if x<=90)/len(deltas)
    # write per-agent file for worst offenders
    per_agent.sort(key=lambda x: -abs(x[3]))
    worst = per_agent[:50]
    per_path = os.path.join(outdir, base.replace('.npz','_rheo_peragent.csv'))
    with open(per_path,'w',newline='') as fh:
        w=csv.writer(fh)
        w.writerow(['agent','vel_ang_deg','rheo_ang_deg','delta_deg','rheo_mag'])
        w.writerows(worst)
    summary = {'file': base, 'step': step, 'count': len(deltas), 'mean_abs_deg': mean, 'pct_within_30': pct_within_30, 'pct_within_90': pct_within_90, 'per_agent_csv': os.path.basename(per_path)}
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dir','-d', default='outputs/diagnostics')
    p.add_argument('--trace','-t', default='outputs/diagnostics/behavior_diag_50x20_fix_trace.csv')
    args=p.parse_args()
    trace = load_trace(args.trace)
    files = sorted([f for f in glob.glob(os.path.join(args.dir,'behavior_debug_step_*.npz')) if not f.endswith('_alignment.npz')])
    out = []
    for f in files:
        s = analyze(f, trace, args.dir)
        if s:
            out.append(s)
    summary_csv = os.path.join(args.dir, 'rheo_deep_summary.csv')
    with open(summary_csv,'w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['file','step','count','mean_abs_deg','pct_within_30','pct_within_90','per_agent_csv'])
        for r in out:
            w.writerow([r['file'], r['step'], r['count'], r['mean_abs_deg'], r['pct_within_30'], r['pct_within_90'], r['per_agent_csv']])
    print('Wrote', summary_csv)

if __name__=='__main__':
    raise SystemExit(main())
