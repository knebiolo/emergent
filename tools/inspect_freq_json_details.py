"""Inspect diagnostics_freq_terms JSON and print per-step detail examples."""
import glob
import json
import os
import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
PAT = os.path.join(OUT, 'diagnostics_freq_terms_*.json')

def find_latest():
    files = glob.glob(PAT)
    if not files:
        raise FileNotFoundError('No diagnostics JSON found')
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    p = find_latest()
    print('Inspecting', p)
    with open(p, 'r') as f:
        data = json.load(f)
    for si, step in enumerate(data):
        denom = np.array(step.get('denom_si', []), dtype=float)
        num = np.array(step.get('num_si', []), dtype=float)
        ratio = np.array(step.get('ratio_si', []), dtype=float)
        Hz_raw = np.array(step.get('Hz_raw', []), dtype=float)
        counts = step.get('counts', {})
        print(f'-- step {si}: N_total={counts.get("N_total", len(denom))} N_safe={counts.get("N_safe_ratio",0)} N_invalid={counts.get("N_invalid_ratio",0)}')
        print('   denom min/max:', np.nanmin(denom) if denom.size else None, np.nanmax(denom) if denom.size else None)
        print('   num min/max:', np.nanmin(num) if num.size else None, np.nanmax(num) if num.size else None)
        print('   ratio pos count:', int(np.count_nonzero(ratio > 0)))
        # print up to 3 examples where ratio > 0
        pos_idxs = np.where(ratio > 0)[0]
        if pos_idxs.size:
            print('   Examples ratio>0:')
            for i in pos_idxs[:3]:
                print('     idx', int(i), 'num', float(num[i]), 'denom', float(denom[i]), 'ratio', float(ratio[i]), 'Hz_raw', float(Hz_raw[i]) if np.isfinite(Hz_raw[i]) else 'nan')
        zero_num_idxs = np.where(num == 0)[0]
        if zero_num_idxs.size:
            print('   Examples num==0 (up to 3):')
            for i in zero_num_idxs[:3]:
                print('     idx', int(i), 'U_m_s', step.get('U_m_s', [None])[i], 'V_m_s', step.get('V_m_s', [None])[i], 'denom', step.get('denom_si', [None])[i])

if __name__ == '__main__':
    main()
