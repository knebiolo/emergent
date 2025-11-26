"""Robust PID trace analyzer
Reads a PID trace CSV (t,agent,err_deg,r_des_deg,derr_deg,P_deg,I_deg,D_deg,raw_deg,rud_deg)
and tolerates lines with extra columns (it parses at least the first 10 fields).
Prints mean/max stats and a list of top error rows.
"""
import os
import math
import statistics
from emergent.ship_abm.config import SHIP_PHYSICS


def parse_trace(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # Need at least 10 fields (t,agent,err_deg,r_des_deg,derr_deg,P_deg,I_deg,D_deg,raw_deg,rud_deg)
            if len(parts) < 10:
                # skip malformed short lines
                continue
            try:
                t = float(parts[0])
                agent = int(float(parts[1]))
                err_deg = float(parts[2])
                r_des_deg = float(parts[3])
                derr_deg = float(parts[4])
                P_deg = float(parts[5])
                I_deg = float(parts[6])
                D_deg = float(parts[7])
                raw_deg = float(parts[8])
                rud_deg = float(parts[9])
            except Exception:
                # skip lines that can't be parsed as numbers
                continue
            rows.append({
                't': t,
                'agent': agent,
                'err_deg': err_deg,
                'r_des_deg': r_des_deg,
                'raw_deg': raw_deg,
                'rud_deg': rud_deg,
            })
    return rows


def summarize(rows):
    errs = [r['err_deg'] for r in rows]
    raws = [r['raw_deg'] for r in rows]
    ruds = [r['rud_deg'] for r in rows]
    ts = [r['t'] for r in rows]
    n = len(rows)
    if n == 0:
        print('No valid rows found')
        return
    mean_err = statistics.mean(errs)
    mean_abs_err = statistics.mean([abs(x) for x in errs])
    max_abs_err = max(abs(x) for x in errs)
    frac_gt_90 = sum(1 for x in errs if abs(x) > 90) / n
    mean_abs_raw = statistics.mean([abs(x) for x in raws])
    mean_abs_rud = statistics.mean([abs(x) for x in ruds])
    max_abs_rud = max(abs(x) for x in ruds)

    max_rudder_deg = math.degrees(SHIP_PHYSICS.get('max_rudder', math.radians(20)))
    sat_frac = sum(1 for x in ruds if abs(x) >= max_rudder_deg * 0.999) / n

    print('Trace:', TRACE_PATH)
    print(f'rows={n}')
    print(f'mean_err_deg={mean_err:.3f}, mean_abs_err_deg={mean_abs_err:.3f}, max_abs_err_deg={max_abs_err:.3f}')
    print(f'frac_err_gt_90={frac_gt_90:.3%}')
    print(f'mean_abs_raw_deg={mean_abs_raw:.3f}, mean_abs_rud_deg={mean_abs_rud:.3f}, max_abs_rud_deg={max_abs_rud:.3f}')
    print(f'max_rudder_deg_cfg={max_rudder_deg:.3f}, sat_frac={sat_frac:.3%}')

    # show worst rows
    worst = sorted(rows, key=lambda r: abs(r['err_deg']), reverse=True)[:8]
    print('\nTop errors (t, err_deg, raw_deg, rud_deg, r_des_deg):')
    for r in worst:
        print(f"t={r['t']:.1f}, err={r['err_deg']:.2f}°, raw={r['raw_deg']:.2f}°, rud={r['rud_deg']:.2f}°, r_des={r['r_des_deg']:.2f}°")


if __name__ == '__main__':
    TRACE_PATH = os.path.join(os.path.dirname(__file__), 'pid_trace_exp_waypoint_crosswind.csv')
    if not os.path.exists(TRACE_PATH):
        print('Trace not found:', TRACE_PATH)
        raise SystemExit(1)
    rows = parse_trace(TRACE_PATH)
    summarize(rows)
