import sys
import os
import numpy as np
from glob import glob
import h5py
import csv
from pathlib import Path

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return dict(data)


def compute_resultant(cue_keys, payload):
    # Sum all cue vecs; if key missing, skip
    vecs = []
    for k in cue_keys:
        if k in payload:
            v = np.array(payload[k])
            if v.ndim == 1:
                # maybe flattened; try to reshape to (n,2)
                if v.size % 2 == 0:
                    v = v.reshape(-1,2)
            vecs.append(v)
    if not vecs:
        return None
    # broadcast sum
    s = np.zeros_like(vecs[0], dtype=float)
    for v in vecs:
        if v.shape != s.shape:
            # try to coerce
            try:
                v = v.reshape(s.shape)
            except Exception:
                raise RuntimeError(f"Shape mismatch in cue vecs: {v.shape} vs {s.shape}")
        s += v
    return s


def vecs_to_angles(vecs):
    # returns radians in [-pi, pi]
    return np.arctan2(vecs[:,1], vecs[:,0])


def abs_angle_diff(a, b):
    d = np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
    return d


def process_many(inputs, out_dir):
    aggregated_rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_agent_rows = []
    for inp in inputs:
        if not os.path.exists(inp):
            continue
        if inp.endswith('.h5') or inp.endswith('.hdf5'):
            with h5py.File(inp, 'r') as f:
                grp = f.get('steps')
                if grp is None:
                    continue
                for s in grp.keys():
                    try:
                        g = grp[s]
                        payload = {k: g[k][()] for k in g.keys()}
                        # compute candidate keys
                        candidate_keys = sorted([k for k in payload.keys() if ('vec' in k or 'raw' in k or 'cue' in k) and k != 'head_vec'])
                        # infer n_agents
                        n_agents = None
                        if 'head_vec' in payload:
                            hv = np.array(payload['head_vec'])
                            if hv.ndim == 2:
                                n_agents = hv.shape[0]
                        cue_vec_keys = []
                        if n_agents is None:
                            for k in candidate_keys:
                                v = np.array(payload[k])
                                if v.ndim == 2 and v.shape[1] == 2:
                                    n_agents = v.shape[0]
                                    break
                        for k in candidate_keys:
                            v = np.array(payload[k])
                            if n_agents is not None:
                                if v.ndim == 2 and v.shape[0] == n_agents and v.shape[1] == 2:
                                    cue_vec_keys.append(k)
                            else:
                                if v.ndim == 2 and v.shape[1] == 2:
                                    cue_vec_keys.append(k)
                        # compute magnitudes per cue
                        mag_matrix = []
                        cue_list = []
                        for ck in cue_vec_keys:
                            arr = np.array(payload[ck]).astype(float)
                            mags = np.linalg.norm(arr, axis=1)
                            mag_matrix.append(mags)
                            cue_list.append(ck)
                        if mag_matrix:
                            M = np.column_stack(mag_matrix)
                        else:
                            M = None
                        res = compute_resultant(cue_vec_keys, payload)
                        if res is None:
                            continue
                        res_angles = vecs_to_angles(res)
                        head_vec = None
                        if 'head_vec' in payload:
                            head_vec = np.array(payload['head_vec'])
                            if head_vec.ndim == 1 and head_vec.size % 2 == 0:
                                head_vec = head_vec.reshape(-1,2)
                        if head_vec is not None and head_vec.shape == res.shape:
                            head_angles = vecs_to_angles(head_vec)
                            diffs = abs_angle_diff(res_angles, head_angles)
                        else:
                            head_angles = None
                            diffs = np.full(res_angles.shape, np.nan)
                        for i in range(res.shape[0]):
                            ra = float(np.degrees(res_angles[i]))
                            ha = float(np.degrees(head_angles[i])) if head_angles is not None else float('nan')
                            dd = float(np.degrees(diffs[i])) if diffs is not None else float('nan')
                            aggregated_rows.append((os.path.basename(inp), s, int(i), ra, ha, dd))
                            # per-cue contribution fractions
                            if M is not None:
                                row_mags = M[i, :]
                                total = np.linalg.norm(res[i])
                                for j, ck in enumerate(cue_list):
                                    mag = float(row_mags[j])
                                    frac = float(mag / total) if total > 0 else float('nan')
                                    per_agent_rows.append((os.path.basename(inp), s, int(i), ck, mag, frac))
                    except Exception:
                        continue
        elif inp.endswith('.npz'):
            try:
                payload = load_npz(inp)
            except Exception:
                continue
            try:
                base = os.path.basename(inp)
                step = 0
                parts = base.split('_')
                for i,p in enumerate(parts):
                    if p=='step' and i+1 < len(parts):
                        try:
                            step = int(parts[i+1]); break
                        except Exception:
                            continue
                candidate_keys = sorted([k for k in payload.keys() if ('vec' in k or 'raw' in k or 'cue' in k) and k != 'head_vec'])
                cue_vec_keys = [k for k in candidate_keys if np.array(payload[k]).ndim==2 and np.array(payload[k]).shape[1]==2]
                # compute magnitudes per cue for NPZ
                mag_matrix = []
                cue_list = []
                for ck in cue_vec_keys:
                    arr = np.array(payload[ck]).astype(float)
                    mags = np.linalg.norm(arr, axis=1)
                    mag_matrix.append(mags)
                    cue_list.append(ck)
                if mag_matrix:
                    M = np.column_stack(mag_matrix)
                else:
                    M = None
                res = compute_resultant(cue_vec_keys, payload)
                if res is None:
                    continue
                res_angles = vecs_to_angles(res)
                head_vec = None
                if 'head_vec' in payload:
                    head_vec = np.array(payload['head_vec'])
                    if head_vec.ndim == 1 and head_vec.size % 2 == 0:
                        head_vec = head_vec.reshape(-1,2)
                if head_vec is not None and head_vec.shape == res.shape:
                    head_angles = vecs_to_angles(head_vec)
                    diffs = abs_angle_diff(res_angles, head_angles)
                else:
                    head_angles = None
                    diffs = np.full(res_angles.shape, np.nan)
                for i in range(res.shape[0]):
                    ra = float(np.degrees(res_angles[i]))
                    ha = float(np.degrees(head_angles[i])) if head_angles is not None else float('nan')
                    dd = float(np.degrees(diffs[i])) if diffs is not None else float('nan')
                    aggregated_rows.append((os.path.basename(inp), str(step), int(i), ra, ha, dd))
                    if M is not None:
                        row_mags = M[i, :]
                        total = np.linalg.norm(res[i])
                        for j, ck in enumerate(cue_list):
                            mag = float(row_mags[j])
                            frac = float(mag / total) if total > 0 else float('nan')
                            per_agent_rows.append((os.path.basename(inp), str(step), int(i), ck, mag, frac))
            except Exception:
                continue
    agg_path = out_dir / 'aggregated_cue_checks.csv'
    with open(agg_path, 'w', newline='') as af:
        w = csv.writer(af)
        w.writerow(['file','step','agent','resultant_angle_deg','head_angle_deg','abs_diff_deg'])
        for r in aggregated_rows:
            w.writerow(r)
    print('Wrote aggregated CSV to', agg_path)
    # write per-agent per-cue contributions
    contrib_path = out_dir / 'per_agent_cue_contribs.csv'
    with open(contrib_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['file','step','agent','cue','mag','frac_of_resultant'])
        for r in per_agent_rows:
            w.writerow(r)
    print('Wrote per-agent cue contributions to', contrib_path)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', help='Path to .npz or .h5 diagnostics file or directory')
    p.add_argument('--step', type=int, default=0)
    p.add_argument('--out-csv-dir', default='outputs/diagnostics/cue_checks')
    p.add_argument('--multi', action='store_true', help='Process multiple files in a directory or multiple inputs')
    args = p.parse_args()
    target = args.path
    if target is None:
        # pick latest NPZ in forced_rawvecs
        files = sorted(glob('outputs/diagnostics/forced_rawvecs/*.npz'), key=os.path.getmtime)
        if not files:
            print('No authoritative NPZs found in outputs/diagnostics/forced_rawvecs')
            sys.exit(1)
        target = files[-1]
    # multi-mode: process directory or many inputs
    if target is None and args.multi:
        # find HDF5s/NPZs under outputs/diagnostics
        inputs = sorted(glob('outputs/diagnostics/*_diagnostics.h5')) + sorted(glob('outputs/diagnostics/behavior_debug_step_*.npz'))
        process_many(inputs, args.out_csv_dir)
        return
    if args.multi:
        # if target provided and multi, treat target as a directory or glob
        if os.path.isdir(target):
            inputs = sorted(glob(os.path.join(target, '*_diagnostics.h5'))) + sorted(glob(os.path.join(target, 'behavior_debug_step_*.npz')))
        else:
            inputs = [target]
        process_many(inputs, args.out_csv_dir)
        return
    if target is None:
        print('No input provided')
        sys.exit(1)
    # allow .h5 or .npz
    if target.endswith('.npz'):
        print('Using NPZ:', target)
        payload = load_npz(target)
        src_type = 'npz'
    elif target.endswith('.h5') or target.endswith('.hdf5'):
        print('Using HDF5:', target)
        # default to step from args
        step = args.step
        # read /steps/<step>
        with h5py.File(target, 'r') as f:
            grp = f.get('steps')
            if grp is None or str(step) not in grp:
                print(f'steps/{step} not found in HDF5')
                sys.exit(1)
            payload = {}
            g = grp[str(step)]
            for k in g.keys():
                payload[k] = g[k][()]
        src_type = 'h5'
    else:
        print('Unsupported input; provide .npz or .h5')
        sys.exit(1)
    print('All keys in payload:', list(payload.keys()))
    # show shapes / samples for quick inspection
    for k, v in payload.items():
        try:
            arr = np.array(v)
            shape = arr.shape
            sample = arr.flatten()[:8]
        except Exception:
            shape = getattr(v, 'shape', None)
            sample = None
        print(f"  key={k} shape={shape} sample={sample}")
    # heuristics: keys that look like per-cue vecs
    # Prefer keys that are per-agent vectors (shape: (n_agents, 2)) matching head_vec when present.
    candidate_keys = sorted([k for k in payload.keys() if ('vec' in k or 'raw' in k or 'cue' in k) and k != 'head_vec'])
    cue_vec_keys = []
    # determine n_agents from head_vec if available, else infer from first (n,2) candidate
    n_agents = None
    if 'head_vec' in payload:
        try:
            hv = np.array(payload['head_vec'])
            if hv.ndim == 2:
                n_agents = hv.shape[0]
        except Exception:
            n_agents = None
    if n_agents is None:
        for k in candidate_keys:
            v = np.array(payload[k])
            if v.ndim == 2 and v.shape[1] == 2:
                n_agents = v.shape[0]
                break
    for k in candidate_keys:
        try:
            v = np.array(payload[k])
            if n_agents is not None:
                if v.ndim == 2 and v.shape[0] == n_agents and v.shape[1] == 2:
                    cue_vec_keys.append(k)
                elif v.ndim == 1 and v.size == 2:
                    cue_vec_keys.append(k)
                else:
                    # skip per-neighbor or other shapes
                    continue
            else:
                # fallback: accept 2D (n,2) shapes
                if v.ndim == 2 and v.shape[1] == 2:
                    cue_vec_keys.append(k)
        except Exception:
            continue
    print('Candidate cue vec keys (heuristic):', cue_vec_keys)
    if 'head_vec' not in payload:
        print('Warning: head_vec not present in NPZ; cannot compare if missing.')
    head_vec = None
    if 'head_vec' in payload:
        head_vec = np.array(payload['head_vec'])
        if head_vec.ndim == 1 and head_vec.size % 2 == 0:
            head_vec = head_vec.reshape(-1,2)
    resultant = compute_resultant(cue_vec_keys, payload)
    if resultant is None:
        print('No cue vecs found to sum (from candidate keys). Exiting.')
        sys.exit(1)
    # compute angles
    res_angles = vecs_to_angles(resultant)
    out_dir = Path('outputs/diagnostics/cue_checks')
    out_dir.mkdir(parents=True, exist_ok=True)
    # write per-cue CSVs: each CSV rows are agents, columns are x,y, angle_deg
    summary_rows = []
    for k in cue_vec_keys:
        arr = np.array(payload[k])
        if arr.ndim == 1 and arr.size == 2:
            arr = arr.reshape(1,2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        angles = np.degrees(vecs_to_angles(arr))
        csv_path = out_dir / f'{k}_step{str(int(payload.get("step",0)))}.csv'
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['agent_index','vx','vy','angle_deg'])
            for i, (vx, vy) in enumerate(arr.tolist()):
                writer.writerow([i, vx, vy, angles[i]])
        summary_rows.append((k, float(np.nanmean(angles)), float(np.nanmedian(angles)), float(np.nanmax(angles))))

    # write summary CSV
    sum_path = out_dir / f'summary_step{str(int(payload.get("step",0)))}.csv'
    with open(sum_path, 'w', newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(['cue','mean_abs_angle_deg','median_abs_angle_deg','max_abs_angle_deg'])
        for r in summary_rows:
            writer.writerow(r)

    if head_vec is not None and head_vec.shape == resultant.shape:
        head_angles = vecs_to_angles(head_vec)
        diffs = abs_angle_diff(res_angles, head_angles)
        print('n agents:', res_angles.size)
        print('mean abs angle deg:', np.degrees(np.nanmean(diffs)))
        print('median abs angle deg:', np.degrees(np.nanmedian(diffs)))
        print('max abs angle deg:', np.degrees(np.nanmax(diffs)))
    else:
        print('Wrote per-cue CSVs to', out_dir)
        print('Resultant shape:', resultant.shape)
        if head_vec is not None:
            print('Head vec shape:', head_vec.shape)
            print('Cannot compare due to shape mismatch')
        else:
            print('Head vec missing; printing resultant angles (deg)')
            print(np.degrees(res_angles))
    # end

if __name__ == '__main__':
    main()


def process_many(inputs, out_dir):
    aggregated_rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for inp in inputs:
        if not os.path.exists(inp):
            continue
        if inp.endswith('.h5') or inp.endswith('.hdf5'):
            with h5py.File(inp, 'r') as f:
                grp = f.get('steps')
                if grp is None:
                    continue
                for s in grp.keys():
                    try:
                        g = grp[s]
                        payload = {k: g[k][()] for k in g.keys()}
                        # compute candidate keys
                        candidate_keys = sorted([k for k in payload.keys() if ('vec' in k or 'raw' in k or 'cue' in k) and k != 'head_vec'])
                        # infer n_agents
                        n_agents = None
                        if 'head_vec' in payload:
                            hv = np.array(payload['head_vec'])
                            if hv.ndim == 2:
                                n_agents = hv.shape[0]
                        cue_vec_keys = []
                        if n_agents is None:
                            for k in candidate_keys:
                                v = np.array(payload[k])
                                if v.ndim == 2 and v.shape[1] == 2:
                                    n_agents = v.shape[0]
                                    break
                        for k in candidate_keys:
                            v = np.array(payload[k])
                            if n_agents is not None:
                                if v.ndim == 2 and v.shape[0] == n_agents and v.shape[1] == 2:
                                    cue_vec_keys.append(k)
                            else:
                                if v.ndim == 2 and v.shape[1] == 2:
                                    cue_vec_keys.append(k)
                        res = compute_resultant(cue_vec_keys, payload)
                        if res is None:
                            continue
                        res_angles = vecs_to_angles(res)
                        head_vec = None
                        if 'head_vec' in payload:
                            head_vec = np.array(payload['head_vec'])
                            if head_vec.ndim == 1 and head_vec.size % 2 == 0:
                                head_vec = head_vec.reshape(-1,2)
                        if head_vec is not None and head_vec.shape == res.shape:
                            head_angles = vecs_to_angles(head_vec)
                            diffs = abs_angle_diff(res_angles, head_angles)
                        else:
                            head_angles = None
                            diffs = np.full(res_angles.shape, np.nan)
                        for i in range(res.shape[0]):
                            ra = float(np.degrees(res_angles[i]))
                            ha = float(np.degrees(head_angles[i])) if head_angles is not None else float('nan')
                            dd = float(np.degrees(diffs[i])) if diffs is not None else float('nan')
                            aggregated_rows.append((os.path.basename(inp), s, int(i), ra, ha, dd))
                    except Exception:
                        continue
        elif inp.endswith('.npz'):
            try:
                payload = load_npz(inp)
            except Exception:
                continue
            try:
                base = os.path.basename(inp)
                step = 0
                parts = base.split('_')
                for i,p in enumerate(parts):
                    if p=='step' and i+1 < len(parts):
                        try:
                            step = int(parts[i+1]); break
                        except Exception:
                            continue
                candidate_keys = sorted([k for k in payload.keys() if ('vec' in k or 'raw' in k or 'cue' in k) and k != 'head_vec'])
                cue_vec_keys = [k for k in candidate_keys if np.array(payload[k]).ndim==2 and np.array(payload[k]).shape[1]==2]
                res = compute_resultant(cue_vec_keys, payload)
                if res is None:
                    continue
                res_angles = vecs_to_angles(res)
                head_vec = None
                if 'head_vec' in payload:
                    head_vec = np.array(payload['head_vec'])
                    if head_vec.ndim == 1 and head_vec.size % 2 == 0:
                        head_vec = head_vec.reshape(-1,2)
                if head_vec is not None and head_vec.shape == res.shape:
                    head_angles = vecs_to_angles(head_vec)
                    diffs = abs_angle_diff(res_angles, head_angles)
                else:
                    head_angles = None
                    diffs = np.full(res_angles.shape, np.nan)
                for i in range(res.shape[0]):
                    ra = float(np.degrees(res_angles[i]))
                    ha = float(np.degrees(head_angles[i])) if head_angles is not None else float('nan')
                    dd = float(np.degrees(diffs[i])) if diffs is not None else float('nan')
                    aggregated_rows.append((os.path.basename(inp), str(step), int(i), ra, ha, dd))
            except Exception:
                continue
    agg_path = out_dir / 'aggregated_cue_checks.csv'
    with open(agg_path, 'w', newline='') as af:
        w = csv.writer(af)
        w.writerow(['file','step','agent','resultant_angle_deg','head_angle_deg','abs_diff_deg'])
        for r in aggregated_rows:
            w.writerow(r)
    print('Wrote aggregated CSV to', agg_path)
