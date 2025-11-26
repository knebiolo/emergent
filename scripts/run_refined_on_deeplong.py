"""Run the existing refined parametric fitter specifically on deep_long PRBS files.
This uses runpy to load the analyzer from `prbs_parametric_refined.py` so we don't
duplicate fitting code.
"""
import glob
import runpy
from pathlib import Path
import pandas as pd

OUT_DIR = Path('figs') / 'prbs_id'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_deeplong():
    return sorted(glob.glob('id_deep_long_*prbs_*.csv'))


def main():
    mod = runpy.run_path('scripts/prbs_parametric_refined.py')
    analyze = mod.get('analyze')
    if analyze is None:
        raise RuntimeError('Could not load analyze() from prbs_parametric_refined.py')

    files = find_deeplong()
    if not files:
        print('No deep_long PRBS files found')
        return

    rows = []
    figs = []
    for f in files:
        print('Analyzing', f)
        s, fs = analyze(f)
        rows.append(s)
        figs.extend(fs)

    out = OUT_DIR / 'prbs_parametric_refined_deeplong_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Wrote', out)
    for p in figs:
        print(' ', p)


if __name__ == '__main__':
    main()
