"""Inspect extracted spike windows and correlate with PID trace.
Prints a compact summary per spike: spike time, nearest PID rows, psi and hd_cmd around the spike, and position.
"""
import os
import numpy as np
import csv

BASE = os.path.join(os.getcwd(), 'scripts')
PID = os.path.join(BASE, 'pid_trace_exp_waypoint_crosswind.csv')
WIN_DIR = os.path.join(BASE, 'spike_windows')

npz_files = sorted([f for f in os.listdir(WIN_DIR) if f.endswith('.npz')])
if not npz_files:
    print('No spike windows found in', WIN_DIR)
    raise SystemExit(1)

def load_pid_rows(pid_path):
    rows = []
    with open(pid_path, 'r', newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                t = float(row.get('t', row.get('time', 'nan')))
            except Exception:
                continue
            rows.append(row)
    return rows

pid_rows = load_pid_rows(PID)

def find_pid_rows_near(t, window=1.0):
    # return rows s.t. abs(t_row - t) <= window
    out = []
    for row in pid_rows:
        try:
            tr = float(row['t'])
        except Exception:
            continue
        if abs(tr - t) <= window:
            out.append((tr, row))
    out.sort(key=lambda x: x[0])
    return out

for fname in npz_files:
    path = os.path.join(WIN_DIR, fname)
    z = np.load(path)
    t0 = float(z['t0'])
    times = z['times']
    pos = z['pos']
    psi = z['psi']
    hd = z['hd_cmd']

    print('\n=== Window file:', fname)
    print(f'spike t0={t0:.3f}s  window samples={len(times)}')
    # nearest PID rows
    near = find_pid_rows_near(t0, window=1.0)
    print('PID rows near spike:')
    for tr, row in near:
        err = row.get('err_deg','')
        raw = row.get('raw_deg','')
        rud = row.get('rud_deg','')
        hd_cmd = row.get('hd_cmd_deg', row.get('hd_cmd',''))
        psi_deg = row.get('psi_deg','')
        print(f'  t={tr:.3f} err={err} raw={raw} rud={rud} hd_cmd={hd_cmd} psi={psi_deg}')

    # Find index in times closest to t0
    idx = int(np.argmin(np.abs(times - t0))) if len(times) else 0
    print('Window sample at spike idx:', idx, 't_sample=', times[idx] if len(times) else 'N/A')
    print('  pos (first,last):', pos[0].tolist(), '...', pos[-1].tolist())
    print('  psi (deg) around spike:')
    for k in range(max(0, idx-3), min(len(psi), idx+4)):
        print(f'    t={times[k]:.3f} psi={np.degrees(psi[k]):.3f} hd_cmd={np.degrees(hd[k]):.3f}')

print('\nDone')
