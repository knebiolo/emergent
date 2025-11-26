import glob
import os
import csv
import math
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

# Group by time
time_dict = defaultdict(dict)
for r in rows:
    time_dict[r['step']][r['agent']] = r

steps = sorted(time_dict.keys())
# build time series arrays
ts = []
xs = defaultdict(list)
ys = defaultdict(list)
psis = defaultdict(list)
rpms = defaultdict(list)
fgs = defaultdict(list)

for step in steps:
    rowpair = time_dict[step]
    # expect two agents
    if 0 in rowpair and 1 in rowpair:
        r0 = rowpair[0]
        r1 = rowpair[1]
        ts.append(r0['t'])
        xs[0].append(r0['x']); ys[0].append(r0['y']); psis[0].append(r0['psi_deg']); rpms[0].append(r0['commanded_rpm']); fgs[0].append(r0['flagged_give_way'])
        xs[1].append(r1['x']); ys[1].append(r1['y']); psis[1].append(r1['psi_deg']); rpms[1].append(r1['commanded_rpm']); fgs[1].append(r1['flagged_give_way'])

# compute inter-ship distance
dists = []
for i in range(len(ts)):
    dx = xs[0][i] - xs[1][i]
    dy = ys[0][i] - ys[1][i]
    d = math.hypot(dx, dy)
    dists.append(d)

min_d = min(dists)
min_idx = dists.index(min_d)
min_t = ts[min_idx]
min_step = steps[min_idx]

print(f'Total time steps: {len(ts)}, final t={ts[-1]:.1f}s')
print(f'Min inter-ship distance: {min_d:.2f} m at step {min_step} t={min_t:.2f}s')

# check whether x or y crossed (simple sign-change test on along-track)
# compute vector between start positions to end positions for each agent
start0 = (xs[0][0], ys[0][0]); end0 = (xs[0][-1], ys[0][-1])
start1 = (xs[1][0], ys[1][0]); end1 = (xs[1][-1], ys[1][-1])

# simple bounding boxes overlap test
def bbox(pstart, pend):
    xs_ = [pstart[0], pend[0]]
    ys_ = [pstart[1], pend[1]]
    return (min(xs_), max(xs_), min(ys_), max(ys_))

b0 = bbox(start0, end0)
b1 = bbox(start1, end1)

def bbox_overlap(bA, bB):
    ax0, ax1, ay0, ay1 = bA
    bx0, bx1, by0, by1 = bB
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

cross_bbox = bbox_overlap(b0, b1)
print('Bounding boxes overlap (trajectories cross area):', cross_bbox)

# plot distance vs time
fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(ts, dists, '-k')
ax1.axvline(min_t, color='red', linestyle='--', label=f'min d={min_d:.1f}m t={min_t:.1f}s')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('inter-ship distance (m)')
ax1.legend()

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figs'))
os.makedirs(out_dir, exist_ok=True)
plot1 = os.path.join(out_dir, 'colregs_distance_time.png')
fig1.savefig(plot1, dpi=200, bbox_inches='tight')
print('Saved distance plot to', plot1)

# plot headings and rpms vs time for both agents, zoom around encounter ±30s
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(ts, psis[0], '-', color='tab:blue', label='psi agent0 (deg)')
ax2.plot(ts, psis[1], '-', color='tab:orange', label='psi agent1 (deg)')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('heading (deg)')
ax2.legend(loc='upper left')

ax3 = ax2.twinx()
ax3.plot(ts, rpms[0], '--', color='tab:blue', alpha=0.6, label='rpm agent0')
ax3.plot(ts, rpms[1], '--', color='tab:orange', alpha=0.6, label='rpm agent1')
ax3.set_ylabel('commanded_rpm')
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines+lines2, labels+labels2, loc='best')

# zoom window
tlo = max(0, min_t-30)
thi = min(ts[-1], min_t+30)
ax2.set_xlim(tlo, thi)

plot2 = os.path.join(out_dir, 'colregs_headings_rpms.png')
fig2.savefig(plot2, dpi=200, bbox_inches='tight')
print('Saved heading/rpm plot to', plot2)

# also print sample rows around min distance
lo = max(0, min_idx-5)
hi = min(len(ts)-1, min_idx+5)
print('\nSample rows around min distance:')
for i in range(lo, hi+1):
    print(f'step {steps[i]} t={ts[i]:.1f}s d={dists[i]:.2f}m agent0 psi={psis[0][i]:.3f} rpm={rpms[0][i]:.3f} agent1 psi={psis[1][i]:.3f} rpm={rpms[1][i]:.3f} fg0={fgs[0][i]} fg1={fgs[1][i]}')

print('\nDone.')
