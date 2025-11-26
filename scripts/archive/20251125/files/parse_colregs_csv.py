import csv
import glob
import os
from collections import defaultdict

logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
logs_dir = os.path.abspath(logs_dir)
pattern = os.path.join(logs_dir, 'colregs_repro_*.csv')
files = glob.glob(pattern)
if not files:
    print('No colregs_repro CSV files found in', logs_dir)
    raise SystemExit(1)

files.sort(key=os.path.getmtime, reverse=True)
fn = files[0]
print('Using CSV:', fn)

rows = []
with open(fn, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # normalize types
        r['step'] = int(r.get('step', 0))
        r['t'] = float(r.get('t', 0.0))
        r['agent'] = int(r.get('agent', 0))
        for k in ['x','y','psi_deg','commanded_rpm']:
            try:
                r[k] = float(r.get(k, 0.0))
            except:
                r[k] = float('nan')
        # flagged_give_way might be 'True'/'False' or '1'/'0'
        fg = r.get('flagged_give_way', '')
        if fg.lower() in ('true','1','t','yes'):
            r['flagged_give_way'] = True
        else:
            r['flagged_give_way'] = False
        rows.append(r)

print('Total rows:', len(rows))

# overall stats
cmd_zero = [r for r in rows if r.get('commanded_rpm') == 0.0]
print('Rows with commanded_rpm == 0:', len(cmd_zero))

fg_rows = [r for r in rows if r.get('flagged_give_way')]
print('Rows with flagged_give_way True:', len(fg_rows))

# per-agent stats
agents = defaultdict(lambda: {'rows':0,'cmd_zero':0,'fg':0,'first_fg':None,'first_cmd_zero':None})
for i,r in enumerate(rows):
    a = r['agent']
    agents[a]['rows'] += 1
    if r.get('commanded_rpm') == 0.0:
        agents[a]['cmd_zero'] += 1
        if agents[a]['first_cmd_zero'] is None:
            agents[a]['first_cmd_zero'] = i
    if r.get('flagged_give_way'):
        agents[a]['fg'] += 1
        if agents[a]['first_fg'] is None:
            agents[a]['first_fg'] = i

print('\nPer-agent summary:')
for a in sorted(agents.keys()):
    s = agents[a]
    print(f" agent={a}: rows={s['rows']} flagged={s['fg']} cmd_zero={s['cmd_zero']} first_fg_idx={s['first_fg']} first_cmd_zero_idx={s['first_cmd_zero']}")

# show sample rows around first fg and first cmd_zero per agent
print('\nSample rows around first events (±5 rows):')
for a in sorted(agents.keys()):
    s = agents[a]
    if s['first_fg'] is not None:
        idx = s['first_fg']
        lo = max(0, idx-5)
        hi = min(len(rows)-1, idx+5)
        print(f"\nagent {a} first flagged_give_way at row index {idx} (showing rows {lo}-{hi}):")
        for j in range(lo, hi+1):
            r = rows[j]
            if r['agent']==a:
                print(j, r)
    else:
        print(f"\nagent {a} had no flagged_give_way rows")

    if s['first_cmd_zero'] is not None:
        idx = s['first_cmd_zero']
        lo = max(0, idx-5)
        hi = min(len(rows)-1, idx+5)
        print(f"\nagent {a} first commanded_rpm==0 at row index {idx} (showing rows {lo}-{hi}):")
        for j in range(lo, hi+1):
            r = rows[j]
            if r['agent']==a:
                print(j, r)
    else:
        print(f"agent {a} had no commanded_rpm==0 rows")

print('\nDone.')
