import glob
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
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
        try:
            r['step'] = int(r.get('step', 0))
            r['t'] = float(r.get('t', 0.0))
            r['agent'] = int(r.get('agent', 0))
            r['x'] = float(r.get('x', 0.0))
            r['y'] = float(r.get('y', 0.0))
            r['psi_deg'] = float(r.get('psi_deg', 0.0))
            r['commanded_rpm'] = float(r.get('commanded_rpm', 0.0))
        except Exception:
            continue
        fg = r.get('flagged_give_way', '')
        if fg.lower() in ('true','1','t','yes'):
            r['flagged_give_way'] = True
        else:
            r['flagged_give_way'] = False
        rows.append(r)

# group by agent
agents = defaultdict(list)
for r in rows:
    agents[r['agent']].append(r)

fig, ax = plt.subplots(figsize=(10,10))
colors = ['tab:blue','tab:orange','tab:green','tab:red']

first_flags = {}

for a, recs in sorted(agents.items()):
    xs = [r['x'] for r in recs]
    ys = [r['y'] for r in recs]
    ax.plot(xs, ys, '-', color=colors[a % len(colors)], label=f'agent {a}')

    # highlight flagged segments
    seg_x = []
    seg_y = []
    had_first = False
    for r in recs:
        if r['flagged_give_way']:
            seg_x.append(r['x'])
            seg_y.append(r['y'])
            if not had_first:
                first_flags[a] = r
                had_first = True
        else:
            if seg_x:
                ax.plot(seg_x, seg_y, linewidth=4, color='k', alpha=0.6)
                seg_x = []
                seg_y = []
    if seg_x:
        ax.plot(seg_x, seg_y, linewidth=4, color='k', alpha=0.6)

# mark first-flag points
for a, r in first_flags.items():
    ax.plot(r['x'], r['y'], 'o', color='magenta', markersize=8, label=f'agent {a} first_flag t={r["t"]:.1f}s')

ax.set_xlabel('x (projected meters)')
ax.set_ylabel('y (projected meters)')
ax.set_title('Reproduced tracks — flagged_give_way highlighted (thick black)')
ax.legend(loc='best')
ax.axis('equal')

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figs'))
os.makedirs(out_dir, exist_ok=True)
out_fn = os.path.join(out_dir, 'colregs_tracks_flagged.png')
fig.savefig(out_fn, dpi=200, bbox_inches='tight')
print('Saved plot to', out_fn)

# Print a short textual summary
print('\nSummary:')
print(' Total rows:', len(rows))
print(' Agents found:', sorted(agents.keys()))
for a, recs in sorted(agents.items()):
    fg_count = sum(1 for r in recs if r['flagged_give_way'])
    print(f' agent {a}: rows={len(recs)} flagged_give_way={fg_count}')
    if a in first_flags:
        r = first_flags[a]
        print(f"  first flagged at t={r['t']:.1f}s step={r['step']} psi={r['psi_deg']:.3f} commanded_rpm={r['commanded_rpm']:.3f}")

print('\nDone.')
