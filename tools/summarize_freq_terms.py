"""Summarize diagnostics/freq_terms_history_json from the most recent sim DB.

Usage:
    python tools/summarize_freq_terms.py

Outputs:
    - outputs/diagnostics_summary.json
    - outputs/diagnostics_freq_terms_samples.csv
"""
import os
import json
import h5py
import glob
import numpy as np
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'outputs')
PAT = os.path.join(OUT_DIR, 'sim_db_*.h5')
JSON_PAT = os.path.join(OUT_DIR, 'diagnostics_freq_terms_*.json')

def find_latest_db_with_key(key='diagnostics/freq_terms_history_json'):
    files = glob.glob(PAT)
    if not files:
        raise FileNotFoundError('No sim_db_*.h5 found in outputs/')
    files.sort(key=os.path.getmtime, reverse=True)
    for p in files:
        try:
            with h5py.File(p, 'r') as f:
                if key in f:
                    return p
        except Exception:
            # skip files we can't open
            continue
    raise KeyError(f'No sim_db_*.h5 contains key: {key}')


def load_freq_terms(db_path):
    with h5py.File(db_path, 'r') as f:
        key = 'diagnostics/freq_terms_history_json'
        if key not in f:
            raise KeyError(f'Key not found: {key} in {db_path}')
        v = f[key][()]
        # expect a single-element array containing JSON string
        raw = v[0].decode('utf-8') if isinstance(v[0], (bytes, bytearray)) else v[0]
        data = json.loads(raw)
        return data


def summarize(data):
    # data is a list of per-step diagnostic dicts, each with many list-valued fields
    n_steps = len(data)
    summary = {'n_steps': n_steps}
    # we'll aggregate counts of invalid ratios across steps
    total_invalid = 0
    total_count = 0
    denom_values = []
    num_values = []
    hz_raw_nans = 0
    hz_values = []
    example_bad = []

    for step_idx, step in enumerate(data):
        denom = np.array(step.get('denom_si', []), dtype=float)
        num = np.array(step.get('num_si', []), dtype=float)
        hz_raw = np.array(step.get('Hz_raw', []), dtype=float)
        hz = np.array(step.get('Hz', []), dtype=float)
        total_count += denom.size
        # denom <= 0 considered problematic for sqrt
        bad_mask = ~np.isfinite(hz_raw) | (denom <= 0) | (num <= 0)
        total_invalid += int(np.count_nonzero(bad_mask))
        denom_values.extend(denom.tolist())
        num_values.extend(num.tolist())
        hz_raw_nans += int(np.count_nonzero(~np.isfinite(hz_raw)))
        hz_values.extend(hz.tolist())

        # collect up to 5 examples per step where bad_mask is True
        if np.any(bad_mask):
            idxs = np.where(bad_mask)[0][:5]
            for i in idxs:
                example_bad.append({
                    'step': step_idx,
                    'agent_index': int(i),
                    'denom_si': float(denom[i]) if i < denom.size else None,
                    'num_si': float(num[i]) if i < num.size else None,
                    'Hz_raw': float(hz_raw[i]) if i < hz_raw.size else None,
                    'Hz': float(hz[i]) if i < hz.size else None,
                    'V_m_s': float(step.get('V_m_s', [None])[i]) if i < len(step.get('V_m_s', [])) else None,
                    'U_m_s': float(step.get('U_m_s', [None])[i]) if i < len(step.get('U_m_s', [])) else None,
                })

    summary['total_agents_observations'] = total_count
    summary['total_invalid'] = total_invalid
    summary['percent_invalid'] = 100.0 * total_invalid / total_count if total_count > 0 else None
    summary['hz_raw_nans'] = hz_raw_nans
    # basic stats for denom and num
    if denom_values:
        arr_d = np.array(denom_values)
        summary['denom_min'] = float(np.nanmin(arr_d))
        summary['denom_5pct'] = float(np.nanpercentile(arr_d, 5))
        summary['denom_median'] = float(np.nanmedian(arr_d))
        summary['denom_95pct'] = float(np.nanpercentile(arr_d, 95))
        summary['denom_max'] = float(np.nanmax(arr_d))
    if num_values:
        arr_n = np.array(num_values)
        summary['num_min'] = float(np.nanmin(arr_n))
        summary['num_median'] = float(np.nanmedian(arr_n))
        summary['num_max'] = float(np.nanmax(arr_n))

    # Hz stats
    if hz_values:
        arr_h = np.array(hz_values)
        summary['Hz_min'] = float(np.nanmin(arr_h))
        summary['Hz_median'] = float(np.nanmedian(arr_h))
        summary['Hz_max'] = float(np.nanmax(arr_h))

    return summary, example_bad


def save_outputs(summary, examples, db_path):
    out_json = os.path.join(OUT_DIR, 'diagnostics_summary.json')
    out_csv = os.path.join(OUT_DIR, 'diagnostics_freq_terms_samples.csv')
    with open(out_json, 'w') as f:
        json.dump({'db': db_path, 'summary': summary}, f, indent=2)
    # save examples to CSV
    if examples:
        keys = sorted(examples[0].keys())
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in examples:
                writer.writerow(r)
    return out_json, out_csv


def main():
    # Prefer an exported JSON diagnostics copy if present
    jfiles = glob.glob(JSON_PAT)
    if jfiles:
        jfiles.sort(key=os.path.getmtime, reverse=True)
        j = jfiles[0]
        print('Using JSON diagnostics copy:', j)
        with open(j, 'r') as f:
            data = json.load(f)
    else:
        try:
            db = find_latest_db_with_key()
        except Exception as e:
            print('No DB with diagnostics key found:', e)
            return
        print('Using DB:', db)
        data = load_freq_terms(db)
    print('Loaded freq_terms_history with', len(data), 'steps')
    summary, examples = summarize(data)
    print('Summary:')
    for k, v in summary.items():
        print(' ', k, ':', v)
    # determine source path for metadata (db or json)
    source_path = locals().get('db', None)
    if source_path is None:
        # use the JSON filename if present
        source_path = j if 'j' in locals() else '<unknown>'
    out_json, out_csv = save_outputs(summary, examples, source_path)
    print('Wrote:', out_json)
    if examples:
        print('Wrote samples CSV:', out_csv)
    else:
        print('No problematic examples collected')

if __name__ == '__main__':
    main()
