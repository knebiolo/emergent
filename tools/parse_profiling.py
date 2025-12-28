import re
from pathlib import Path
import csv

PROF_DIR = Path('outputs/profiling')

def parse_wallclock(text):
    m = re.search(r'wallclock=([0-9\.]+)s', text)
    return float(m.group(1)) if m else None

def top_funcs(text, n=2):
    lines = text.splitlines()
    funcs = []
    for ln in lines:
        m = re.match(r"\s*\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(.*):\d+\((.*)\)", ln)
        if m:
            fpath, fn = m.groups()
            funcs.append(f"{Path(fpath).name}:{fn}")
        if len(funcs) >= n:
            break
    return funcs

def main():
    rows = []
    for p in sorted(PROF_DIR.glob('hotspot_profile_*.txt')):
        txt = p.read_text(encoding='utf-8')
        wc = parse_wallclock(txt)
        tops = top_funcs(txt, n=3)
        rows.append({'file': p.name, 'wallclock': wc, 'top1': tops[0] if len(tops)>0 else '', 'top2': tops[1] if len(tops)>1 else '', 'top3': tops[2] if len(tops)>2 else ''})

    out = PROF_DIR / 'summary.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file','wallclock','top1','top2','top3'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'Wrote summary to {out}')

if __name__ == '__main__':
    main()
