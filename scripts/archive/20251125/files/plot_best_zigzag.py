import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent
summary_csv = root / 'zigzag_search_summary.csv'
if not summary_csv.exists():
    raise SystemExit(f"Summary CSV not found: {summary_csv}")

df = pd.read_csv(summary_csv)
# Prefer rmse_cross_m (smaller better), fallback to final_cross_m
if 'rmse_cross_m' in df.columns and df['rmse_cross_m'].notna().any():
    key = 'rmse_cross_m'
elif 'final_cross_m' in df.columns and df['final_cross_m'].notna().any():
    key = 'final_cross_m'
else:
    key = None

if key is not None:
    # pick minimal
    df_valid = df[df[key].notna()].copy()
    if df_valid.empty:
        idx = 0
    else:
        idx = df_valid[key].idxmin()
else:
    idx = 0

row = df.loc[idx]
traj_path = Path(row['traj_csv'])
# If path is relative, resolve against repo root
if not traj_path.exists():
    traj_path = (root / traj_path).resolve()

if not traj_path.exists():
    raise SystemExit(f"Trajectory CSV not found: {traj_path}")

traj = pd.read_csv(traj_path)
# Expect columns: t,x_m,y_m,psi_deg,hd_cmd_deg,dist_to_poly_m,rudder_deg
# Prepare output dir
out_dir = root / 'figs' / 'zigzag'
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / f"best_zigzag_{traj_path.stem}.png"

# Plot trajectory: use matplotlib built-in style only to avoid seaborn dependency
plt.style.use('classic')
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,1,1)
sc = ax1.scatter(traj['x_m'], traj['y_m'], c=traj['t'], cmap='viridis', s=6)
ax1.plot(traj['x_m'], traj['y_m'], lw=0.8, color='k', alpha=0.6)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title(f"Zig-zag trajectory: {traj_path.name}\nparams: Kp={row.get('Kp')}, Ki={row.get('Ki')}, Kd={row.get('Kd')}")
cb = fig.colorbar(sc, ax=ax1, label='time (s)')

# Mark start and end
ax1.scatter([traj['x_m'].iloc[0]], [traj['y_m'].iloc[0]], c='green', s=50, label='start')
ax1.scatter([traj['x_m'].iloc[-1]], [traj['y_m'].iloc[-1]], c='red', s=50, label='end')
ax1.legend()

# Rudder vs time
ax2 = fig.add_subplot(2,1,2)
ax2.plot(traj['t'], traj['rudder_deg'], color='tab:blue')
ax2.set_xlabel('t (s)')
ax2.set_ylabel('rudder (deg)')
ax2.grid(True)

plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(out_png)
