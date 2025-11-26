"""Analyze the most recent headless zigzag/straight-run CSV and produce diagnostics.
Saves PNGs under figs/diagnostics/ and prints numeric metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import math

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'headless_zigzag_crosswind_Kp0.400_Ki0.010_Kd0.120_w1.5_dr0.5.csv'
OUT = ROOT / 'figs' / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise SystemExit(f"CSV not found: {CSV}")

df = pd.read_csv(CSV)
# time
t = df['t'].values
if len(t) < 2:
    raise SystemExit('Too few samples')
dt = np.median(np.diff(t))

# positions
x = df['x_m'].values
y = df['y_m'].values
# heading and commanded
psi = np.radians(df['psi_deg'].values)  # rad
hd_cmd = np.radians(df['hd_cmd_deg'].values)
# rudder
cmd_rud = df['cmd_rudder_deg'].values
applied = df['applied_rudder_deg'].values

# 1) planned centerline (straight from start->end)
x0, y0 = x[0], y[0]
x1, y1 = x[-1], y[-1]
line_vec = np.array([x1 - x0, y1 - y0])
line_len = np.hypot(line_vec[0], line_vec[1])
if line_len == 0:
    line_unit = np.array([1.0,0.0])
else:
    line_unit = line_vec / line_len
# cross-track error: vector from start to each point, project orthogonal
rel = np.vstack([x - x0, y - y0]).T
proj_along = rel.dot(line_unit)
proj_point = np.outer(proj_along, line_unit) + np.array([x0,y0])
cross_vec = rel - (proj_point - np.array([x0,y0]))
cross_track = np.sign(np.cross(np.append(line_unit,0), np.vstack([line_unit, np.zeros(len(line_unit))]).T[0]))  # placeholder
# simpler: compute perpendicular distance using area formula
cross_track = ( (x1 - x0)*(y0 - y) - (x0 - x)*(y1 - y0) ) / (line_len + 1e-12)

# metrics
rmse_cross = np.sqrt(np.mean(cross_track**2))
max_cross = np.max(np.abs(cross_track))

# heading error between psi and hd_cmd (wrap)
err = ( (psi - hd_cmd + np.pi) % (2*np.pi) ) - np.pi
err_deg = np.degrees(err)
rmse_heading = np.sqrt(np.mean(err_deg**2))
max_heading_err = np.max(np.abs(err_deg))

# yaw-rate (deg/s)
r_deg_s = np.gradient(np.degrees(psi), dt)
r_std = np.nanstd(r_deg_s)

# PSD of heading error
f, Pxx = signal.welch(err_deg - np.mean(err_deg), fs=1.0/dt, nperseg=min(256, len(err_deg)))
# dominant frequency
idx = np.argmax(Pxx)
dom_freq = f[idx]

# overshoot proxy: count peaks in heading error magnitude > threshold
peaks, props = signal.find_peaks(np.abs(err_deg), height=2.0)  # >2 deg
peak_count = len(peaks)
peak_heights = props['peak_heights'] if 'peak_heights' in props else []

# print summary
print('Diagnostic summary for', CSV)
print(f'  samples: {len(t)}, dt={dt:.3f}s')
print(f'  cross-track RMSE = {rmse_cross:.2f} m, max = {max_cross:.2f} m')
print(f'  heading RMSE = {rmse_heading:.2f} deg, max err = {max_heading_err:.2f} deg')
print(f'  yaw-rate std = {r_std:.3f} deg/s, dominant freq in heading error = {dom_freq:.3f} Hz')
print(f'  heading-error peaks (>2deg): {peak_count}')

# plots
plt.figure(figsize=(10,4))
plt.plot(t, np.degrees(hd_cmd), label='hd_cmd (deg)')
plt.plot(t, np.degrees(psi), label='psi_actual (deg)')
plt.plot(t, err_deg, label='heading error (deg)')
plt.xlabel('t (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / 'heading_vs_cmd.png', dpi=150)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(t, cross_track, label='cross-track (m)')
plt.xlabel('t (s)')
plt.ylabel('cross-track (m)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / 'cross_track.png', dpi=150)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(t, r_deg_s, label='yaw-rate deg/s')
plt.xlabel('t (s)')
plt.ylabel('r (deg/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / 'yaw_rate.png', dpi=150)
plt.close()

plt.figure(figsize=(6,4))
plt.semilogy(f, Pxx)
plt.xlabel('Hz')
plt.ylabel('PSD (deg^2/Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / 'heading_err_psd.png', dpi=150)
plt.close()

# rudder traces
plt.figure(figsize=(10,4))
plt.plot(t, cmd_rud, label='cmd rudder (deg)')
plt.plot(t, applied, label='applied rudder (deg)')
plt.xlabel('t (s)')
plt.ylabel('rudder (deg)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'rudder_traces.png', dpi=150)
plt.close()

# write numeric summary
with open(OUT / 'diagnostic_summary.txt', 'w') as fh:
    fh.write('Diagnostic summary for ' + str(CSV) + '\n')
    fh.write(f'samples: {len(t)}, dt={dt:.3f}s\n')
    fh.write(f'cross-track RMSE = {rmse_cross:.2f} m, max = {max_cross:.2f} m\n')
    fh.write(f'heading RMSE = {rmse_heading:.2f} deg, max err = {max_heading_err:.2f} deg\n')
    fh.write(f'yaw-rate std = {r_std:.3f} deg/s, dom freq = {dom_freq:.3f} Hz\n')
    fh.write(f'heading-error peaks (>2deg): {peak_count}\n')

print('Wrote diagnostics to', OUT)
