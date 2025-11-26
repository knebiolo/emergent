"""Plot cleaned straight-line crosswind PID trace (nearest-angle error) and print numeric summary.
Reads (prefer): sweep_results/pid_trace_straight_wind_smoothed.csv
Fallback: scripts/pid_trace_straight_wind.csv
Writes: sweep_results/figs/straight_crosswind_cleaned.png
Prints a short numeric summary to stdout.
"""
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

PREFER = 'sweep_results/pid_trace_straight_wind_smoothed.csv'
FALLBACK = 'scripts/pid_trace_straight_wind.csv'
OUT_DIR = 'sweep_results/figs'
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(PREFER):
    IN = PREFER
else:
    IN = FALLBACK

print('Reading:', IN)

def wrap_deg_to_signed(a_deg):
    """Wrap angle(s) in degrees to [-180, 180).
    Accepts scalar or pandas Series / numpy array.
    """
    return ((a_deg + 180.0) % 360.0) - 180.0


df = pd.read_csv(IN)
df0 = df[df['agent'] == 0].reset_index(drop=True)

t = df0['t']
psi = df0['psi_deg']
hd = df0['hd_cmd_deg']
raw = df0.get('raw_deg', pd.Series([float('nan')] * len(df0)))
rud = df0['rud_deg']

# compute nearest-angle error between commanded heading and actual heading
hd_wrapped = wrap_deg_to_signed(hd)
psi_wrapped = wrap_deg_to_signed(psi)
err_clean = wrap_deg_to_signed(hd_wrapped - psi_wrapped)

# metrics (on absolute cleaned error)
max_err = float(err_clean.abs().max())
mean_err = float(err_clean.abs().mean())
frac_err_gt_30 = float((err_clean.abs() > 30.0).sum()) / len(err_clean)
max_rud = float(rud.abs().max())
frac_rud_sat = float((rud.abs() >= (max_rud - 1e-6)).sum()) / len(rud)

# figure
fig, axs = plt.subplots(3,1, figsize=(10,9), sharex=True)
axs[0].plot(t, hd_wrapped, label='hd_cmd_deg (wrapped)', color='C0')
axs[0].plot(t, psi_wrapped, label='psi_deg (wrapped)', color='C1', alpha=0.9)
axs[0].set_title('Desired vs Actual Heading (wrapped)')
axs[0].set_ylabel('deg')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, err_clean, color='C2')
axs[1].axhline(0, color='k', lw=0.5)
axs[1].set_title('Heading error (nearest-angle, deg)')
axs[1].set_ylabel('deg')
axs[1].grid(True)

axs[2].plot(t, raw, '--', label='raw_cmd_deg', color='C3')
axs[2].plot(t, rud, label='applied_rudder_deg', color='C4')
axs[2].set_title('Rudder (deg)')
axs[2].set_ylabel('deg')
axs[2].grid(True)
axs[2].legend()

for ax in axs:
    ax.set_xlabel('t (s)')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'straight_crosswind_cleaned.png')
plt.savefig(out_path)
plt.close()

print('Wrote:', out_path)
print('Metrics (nearest-angle):')
print(f"  max_err_deg = {max_err:.2f}")
print(f"  mean_err_deg = {mean_err:.2f}")
print(f"  frac_time_err>|30°| = {frac_err_gt_30:.3f}")
print(f"  max_rud_deg = {max_rud:.2f}")
print(f"  frac_time_rudder_at_max = {frac_rud_sat:.3f}")
