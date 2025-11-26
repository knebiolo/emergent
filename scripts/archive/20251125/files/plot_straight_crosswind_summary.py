"""Plot straight-line crosswind PID trace and print numeric summary.
Reads: scripts/pid_trace_straight_wind.csv
Writes: sweep_results/figs/straight_crosswind.png
Prints a short numeric summary to stdout.
"""
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

IN = 'scripts/pid_trace_straight_wind.csv'
OUT_DIR = 'sweep_results/figs'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)
df0 = df[df['agent'] == 0].reset_index(drop=True)

t = df0['t']
psi = df0['psi_deg']
hd = df0['hd_cmd_deg']
err = df0['err_deg']
raw = df0['raw_deg']
rud = df0['rud_deg']

# metrics
max_err = float(err.abs().max())
mean_err = float(err.abs().mean())
frac_err_gt_30 = float((err.abs() > 30.0).sum()) / len(err)
max_rud = float(rud.abs().max())
frac_rud_sat = float((rud.abs() >= (max_rud - 1e-6)).sum()) / len(rud)

# figure
fig, axs = plt.subplots(2,2, figsize=(12,8))
axs = axs.ravel()
axs[0].plot(t, hd, label='hd_cmd_deg', color='C0')
axs[0].plot(t, psi, label='psi_deg', color='C1', alpha=0.8)
axs[0].set_title('Desired vs Actual Heading')
axs[0].set_ylabel('deg')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, err, color='C2')
axs[1].axhline(0, color='k', lw=0.5)
axs[1].set_title('Heading error (deg)')
axs[1].set_ylabel('deg')
axs[1].grid(True)

axs[2].plot(t, raw, '--', label='raw_cmd_deg', color='C3')
axs[2].plot(t, rud, label='applied_rudder_deg', color='C4')
axs[2].set_title('Rudder command vs applied')
axs[2].set_ylabel('deg')
axs[2].grid(True)
axs[2].legend()

# track subplot: small inset-like plot (x,y vs t available in pid trace x_m,y_m)
try:
    axs[3].plot(df0['x_m'], df0['y_m'], '-k')
    axs[3].set_title('Planimetric track (m)')
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].grid(True)
except Exception:
    axs[3].text(0.1, 0.5, 'No pos data', transform=axs[3].transAxes)
    axs[3].set_title('Track')

for ax in axs:
    ax.set_xlabel('t (s)')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'straight_crosswind.png')
plt.savefig(out_path)
plt.close()

print('Wrote:', out_path)
print('Metrics:')
print(f"  max_err_deg = {max_err:.2f}")
print(f"  mean_err_deg = {mean_err:.2f}")
print(f"  frac_time_err>|30°| = {frac_err_gt_30:.3f}")
print(f"  max_rud_deg = {max_rud:.2f}")
print(f"  frac_time_rudder_at_max = {frac_rud_sat:.3f}")
