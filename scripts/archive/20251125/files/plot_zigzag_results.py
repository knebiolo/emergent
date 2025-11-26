"""Generate polished plots for zig-zag sweep results

Reads:
 - zigzag_kf_sweep_summary.csv
 - per-run trajectory CSVs referenced in the summary

Writes PNGs to figs/zigzag/
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'figs' / 'zigzag'
FIGS.mkdir(parents=True, exist_ok=True)
SUMMARY = ROOT / 'zigzag_kf_sweep_summary.csv'

if not SUMMARY.exists():
    print('No summary CSV found at', SUMMARY)
    raise SystemExit(1)

df = pd.read_csv(SUMMARY)

# 1) RMSE vs Kf per wind (line plot)
try:
    plt.style.use('seaborn-darkgrid')
except Exception:
    plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8,4))
for wind in sorted(df['wind_speed'].unique()):
    sub = df[df['wind_speed'] == wind]
    ax.plot(sub['Kf_gain'], sub['rmse_cross_m'], marker='o', label=f'wind={wind} m/s')
ax.set_xlabel('Kf (feed-forward)')
ax.set_ylabel('RMSE to polyline (m)')
ax.set_title('Zig-zag RMSE vs Kf')
ax.legend()
fig.tight_layout()
out = FIGS / 'rmse_vs_kf.png'
fig.savefig(out, dpi=200)
print('Wrote', out)

# 2) Rudder saturation fraction bar chart
fig, ax = plt.subplots(figsize=(8,4))
x_labels = [f"w{w}_Kf{kf}" for w,kf in zip(df['wind_speed'], df['Kf_gain'])]
ax.bar(range(len(df)), df['rudder_saturation_frac'], color='C1')
ax.set_xticks(range(len(df)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel('Rudder saturation fraction')
ax.set_title('Rudder Saturation across runs')
fig.tight_layout()
out = FIGS / 'rudder_saturation.png'
fig.savefig(out, dpi=200)
print('Wrote', out)

# 3) Overlay trajectories for best runs
best = df.nsmallest(6, 'composite_cost')
fig, ax = plt.subplots(figsize=(8,8))
for i, row in best.iterrows():
    try:
        traj = pd.read_csv(row['traj_csv'])
        ax.plot(traj['x_m'], traj['y_m'], lw=1.2)
    except Exception as e:
        print('Could not read', row['traj_csv'], e)
# draw prototypical waypoint polyline using first traj start
if len(best) > 0:
    try:
        sample = pd.read_csv(best.iloc[0]['traj_csv'])
        x0 = sample.iloc[0]['x_m']
        y0 = sample.iloc[0]['y_m']
        leg_length = 200.0
        zig_legs = 6
        zig_amp = 30.0
        way = [(x0, y0)]
        dir = 1
        for j in range(1, zig_legs+1):
            way.append((x0 + j * leg_length, y0 + dir * zig_amp))
            dir *= -1
        wx = [p[0] for p in way]
        wy = [p[1] for p in way]
        ax.plot(wx, wy, 'k--', lw=1.5, label='waypoints')
    except Exception:
        pass
ax.set_title('Top trajectories overlay')
fig.tight_layout()
out = FIGS / 'top_trajectories_overlay.png'
fig.savefig(out, dpi=200)
print('Wrote', out)

# 4) Time-series for one representative run (worst and best)
for label, selector in [('best', df.nsmallest(1, 'composite_cost')), ('worst', df.nlargest(1, 'composite_cost'))]:
    if selector.shape[0] == 0:
        continue
    row = selector.iloc[0]
    try:
        traj = pd.read_csv(row['traj_csv'])
        t = traj['t']
        fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
        axes[0].plot(t, traj['dist_to_poly_m']); axes[0].set_ylabel('dist to poly (m)')
        axes[1].plot(t, traj['psi_deg'] - traj['hd_cmd_deg']); axes[1].set_ylabel('heading err (deg)')
        axes[2].plot(t, traj['rudder_deg']); axes[2].set_ylabel('rudder (deg)')
        axes[2].set_xlabel('time (s)')
        fig.suptitle(f"Time series ({label}) Kp={row['Kp']}, Ki={row['Ki']}, Kf={row['Kf_gain']} wind={row['wind_speed']}")
        fig.tight_layout(rect=[0,0,1,0.96])
        out = FIGS / f'time_series_{label}.png'
        fig.savefig(out, dpi=200)
        print('Wrote', out)
    except Exception as e:
        print('Failed time series for', label, e)

print('Plotting complete')
