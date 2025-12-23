"""Analyze freq probe CSVs and diagnostics JSON to summarize why Hz falls back.

Outputs:
 - outputs/freq_probe_analysis.json
 - outputs/freq_probe_problem_samples.csv
"""
import os
import glob
import json
import csv
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'outputs')
SRC_OUT = os.path.join(ROOT, 'src', 'outputs')

CSV_PAT = os.path.join(SRC_OUT, 'freq_probe_*.csv')
JSON_PAT = os.path.join(OUT_DIR, 'diagnostics_freq_terms_*.json')


def find_latest(paths):
    paths = sorted(paths, key=os.path.getmtime, reverse=True)
    return paths[0] if paths else None


def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)
    finite_hz = df['Hz'].replace([np.inf, -np.inf], np.nan).dropna().shape[0]
    fallback = (df['Hz'] == 2.0).sum()
    num_zero = (df['num_si'] == 0).sum()
    denom_neg = (df['denom_si'] < 0).sum()
    denom_zero = (df['denom_si'] == 0).sum()
    ratio_pos = (df['ratio_si'] > 0).sum()
    return {
        'csv_path': csv_path,
        'rows': int(n),
        'finite_hz': int(finite_hz),
        'fallback_count': int(fallback),
        'num_zero': int(num_zero),
        'denom_negative': int(denom_neg),
        'denom_zero': int(denom_zero),
        'ratio_positive': int(ratio_pos)
    }, df


def analyze_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data is a list of per-step diag dicts
    total = 0
    denom_neg = 0
    num_zero = 0
    hz_nans = 0
    ratio_pos = 0
    for step in data:
        denom = np.array(step.get('denom_si', []), dtype=float)
        num = np.array(step.get('num_si', []), dtype=float)
        hz_raw = np.array(step.get('Hz_raw', []), dtype=float)
        total += denom.size
        denom_neg += int(np.count_nonzero(denom < 0))
        num_zero += int(np.count_nonzero(num == 0))
        hz_nans += int(np.count_nonzero(~np.isfinite(hz_raw)))
        ratio_pos += int(np.count_nonzero(np.array(step.get('ratio_si', []), dtype=float) > 0))
    return {
        'json_path': json_path,
        'total_observations': int(total),
        'denom_negative': int(denom_neg),
        'num_zero': int(num_zero),
        'hz_raw_nans': int(hz_nans),
        'ratio_positive': int(ratio_pos)
    }


def save_outputs(summary, df_samples=None):
    out_json = os.path.join(OUT_DIR, 'freq_probe_analysis.json')
    out_csv = os.path.join(OUT_DIR, 'freq_probe_problem_samples.csv')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    if df_samples is not None and not df_samples.empty:
        df_samples.to_csv(out_csv, index=False)
        return out_json, out_csv
    return out_json, None


def main():
    csvs = glob.glob(CSV_PAT)
    jsons = glob.glob(JSON_PAT)
    csv_latest = find_latest(csvs)
    json_latest = find_latest(jsons)
    report = {'csv_found': bool(csv_latest), 'json_found': bool(json_latest)}
    df = None
    if csv_latest:
        csv_summary, df = analyze_csv(csv_latest)
        report['csv_summary'] = csv_summary
    if json_latest:
        json_summary = analyze_json(json_latest)
        report['json_summary'] = json_summary

    # produce sample problematic rows from CSV: where denom_si <= 0 or num_si == 0
    df_samples = None
    if df is not None:
        mask = (df['denom_si'] <= 0) | (df['num_si'] == 0) | (~np.isfinite(df['Hz_raw']))
        df_samples = df[mask].head(200)

    out_json, out_csv = save_outputs(report, df_samples)
    print('Wrote analysis:', out_json)
    if out_csv:
        print('Wrote sample CSV:', out_csv)

if __name__ == '__main__':
    main()
