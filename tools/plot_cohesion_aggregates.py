"""Compute per-agent aggregates from behavior_cohesion_alignment_timeseries.csv and create plots.

Outputs:
 - outputs/diagnostics/behavior_cohesion_per_agent.csv
 - outputs/diagnostics/plots/cohesion_mean_per_agent.png
 - outputs/diagnostics/plots/cohesion_abs_angle_hist.png
 - outputs/diagnostics/plots/cohesion_abs_angle_boxplot.png
"""
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join('outputs', 'diagnostics')
PLOTDIR = os.path.join(OUTDIR, 'plots')
os.makedirs(PLOTDIR, exist_ok=True)

CSV = os.path.join(OUTDIR, 'behavior_cohesion_alignment_timeseries.csv')
if not os.path.exists(CSV):
    raise RuntimeError('Timeseries CSV not found: ' + CSV)

# read CSV: step, agent, cohesion_vx, cohesion_vy, heading_radians
per_agent = {}
all_diffs = []
with open(CSV, 'r', newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        try:
            step = int(row['step'])
            agent = int(row['agent'])
            vx = float(row['cohesion_vx'])
            vy = float(row['cohesion_vy'])
            heading = float(row['heading_radians']) if row['heading_radians'] not in ('', 'nan', 'None') else float('nan')
        except Exception:
            continue
        # skip NaN cohesion
        if math.isnan(vx) or math.isnan(vy):
            continue
        # if heading is NaN skip
        if math.isnan(heading):
            continue
        # compute abs angle diff
        ha = math.atan2(vy, vx)
        diff = abs((ha - heading + math.pi) % (2 * math.pi) - math.pi)
        deg = math.degrees(diff)
        per_agent.setdefault(agent, []).append(deg)
        all_diffs.append(deg)

# per-agent aggregates
agents = sorted(per_agent.keys())
rows = []
means = []
medians = []
maxs = []
for a in agents:
    arr = np.array(per_agent[a])
    m = float(np.mean(arr)) if arr.size > 0 else float('nan')
    med = float(np.median(arr)) if arr.size > 0 else float('nan')
    mx = float(np.max(arr)) if arr.size > 0 else float('nan')
    rows.append((a, m, med, mx, int(arr.size)))
    means.append(m)
    medians.append(med)
    maxs.append(mx)

# write per-agent CSV
out_csv = os.path.join(OUTDIR, 'behavior_cohesion_per_agent.csv')
with open(out_csv, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['agent', 'mean_abs_angle_deg', 'median_abs_angle_deg', 'max_abs_angle_deg', 'n_observations'])
    for r in rows:
        w.writerow(r)
print('Wrote per-agent aggregates:', out_csv)

# Plot mean per agent bar chart
plt.figure(figsize=(10, 4))
plt.bar(agents, means)
plt.xlabel('Agent')
plt.ylabel('Mean abs angle (deg)')
plt.title('Mean absolute angle between cohesion vector and heading per agent')
plt.tight_layout()
barfile = os.path.join(PLOTDIR, 'cohesion_mean_per_agent.png')
plt.savefig(barfile, dpi=150)
plt.close()
print('Wrote plot:', barfile)

# Histogram of all diffs
plt.figure(figsize=(6,4))
plt.hist(all_diffs, bins=40)
plt.xlabel('Absolute angle diff (deg)')
plt.ylabel('Count')
plt.title('Histogram of absolute angle diffs (cohesion vs heading)')
plt.tight_layout()
histfile = os.path.join(PLOTDIR, 'cohesion_abs_angle_hist.png')
plt.savefig(histfile, dpi=150)
plt.close()
print('Wrote plot:', histfile)

# Boxplot
plt.figure(figsize=(6,4))
plt.boxplot(all_diffs, vert=False)
plt.xlabel('Absolute angle diff (deg)')
plt.title('Boxplot of absolute angle diffs')
plt.tight_layout()
boxfile = os.path.join(PLOTDIR, 'cohesion_abs_angle_boxplot.png')
plt.savefig(boxfile, dpi=150)
plt.close()
print('Wrote plot:', boxfile)
