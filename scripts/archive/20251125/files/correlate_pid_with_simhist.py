"""Programmatically correlate PID trace rows with sim-history NPZ and write compact CSV of high-error events.

Usage: python scripts/correlate_pid_with_simhist.py <pid_csv> <simhist_npz> [threshold_deg] [out_csv]
"""
import sys
import os
import numpy as np
import pandas as pd


def load_pid(path):
    # reuse tolerant parsing from correlate_pid_state.py
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
    cols10 = ['t','agent','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']
    cols15 = cols10 + ['psi_deg','hd_cmd_deg','r_meas_deg','x_m','y_m']
    start_idx = 0
    if lines and lines[0].split(',')[0].strip().lower() == 't':
        start_idx = 1
    rows = []
    for ln in lines[start_idx:]:
        parts = [p for p in ln.split(',')]
        if len(parts) == 10:
            row = dict(zip(cols10, parts))
        elif len(parts) >= 15:
            row = dict(zip(cols15, parts[:15]))
        else:
            continue
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
    return df


def correlate(pid_csv, sim_npz, thr=160.0, out_csv=None):
    pid = load_pid(pid_csv)
    if pid.empty:
        print('No PID rows loaded')
        return 2
    if 'err_deg' not in pid.columns:
        print('err_deg missing in PID trace')
        return 3


    # allow_pickle=True because some simhist files may contain object arrays
    # (e.g., pos history stored as a list of tuples) — safe for offline analysis.
    sim = np.load(sim_npz, allow_pickle=True)
    keys = list(sim.keys())
    # helper to pick first available candidate
    def pick(*cands):
        for c in cands:
            if c in sim:
                return sim[c]
        return None

    t = pick('t', 't_history', 'time', 'times')
    psi = pick('psi', 'psi_history', 'psi_deg', 'heading')
    hd_cmd = pick('hd_cmd', 'hd_cmd_history', 'hd_cmds', 'hd_cmd_deg')
    pos = pick('pos', 'pos_history', 'positions', 'ship_pos', 'pos_history_xy')

    if t is None or psi is None or hd_cmd is None or pos is None:
        print('Simhistory NPZ missing expected arrays. Keys:', keys)
        return 4

    # build indexable time->index map
    # t may be 1D array of times; use nearest-index matching
    def find_idx(time_val):
        arr = np.asarray(t)
        idx = np.argmin(np.abs(arr - time_val))
        return int(idx)

    mask = pid['err_deg'].abs() >= thr
    events = []
    for _, row in pid.loc[mask].iterrows():
        ti = float(row['t'])
        idx = find_idx(ti)
        # extract pos which may be (N,2) or separate arrays
        x = y = np.nan
        p = pos
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2 and p.shape[0] > idx:
            x = float(p[idx,0]); y = float(p[idx,1])
        else:
            # try separate x/y arrays
            if 'pos_x' in sim and 'pos_y' in sim:
                x = float(sim['pos_x'][idx]); y = float(sim['pos_y'][idx])
            elif 'x' in sim and 'y' in sim:
                x = float(sim['x'][idx]); y = float(sim['y'][idx])

        psi_val = float(np.asarray(psi)[idx]) if idx < np.asarray(psi).shape[0] else np.nan
        hd_val = float(np.asarray(hd_cmd)[idx]) if idx < np.asarray(hd_cmd).shape[0] else np.nan
        rmeas = float(sim['r_meas'][idx]) if 'r_meas' in sim and idx < np.asarray(sim['r_meas']).shape[0] else np.nan

        ev = {
            't': ti,
            'agent': int(row.get('agent', 0)),
            'err_deg': row['err_deg'],
            'raw_deg': row.get('raw_deg', np.nan),
            'rud_deg': row.get('rud_deg', np.nan),
            'psi_deg': psi_val,
            'hd_cmd_deg': hd_val,
            'r_meas_deg_s': rmeas,
            'x_m': x,
            'y_m': y,
        }
        events.append(ev)

    out_df = pd.DataFrame(events)
    if out_csv is None:
        base = os.path.splitext(os.path.basename(pid_csv))[0]
        out_csv = os.path.join(os.path.dirname(pid_csv), f"{base}_correlated.csv")
    out_df.to_csv(out_csv, index=False)
    print(f'Wrote correlated events to: {out_csv} (count={len(out_df)})')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python scripts/correlate_pid_with_simhist.py <pid_csv> <sim_npz> [threshold_deg] [out_csv]')
        sys.exit(2)
    pid_csv = sys.argv[1]
    sim_npz = sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 160.0
    out_csv = sys.argv[4] if len(sys.argv) > 4 else None
    sys.exit(correlate(pid_csv, sim_npz, thr=thr, out_csv=out_csv))
