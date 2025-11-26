"""Validate identified continuous model by simulating the vehicle response to the
recorded zig-zag applied rudder and comparing yaw-rate and heading.

Outputs go to `figs/validation/`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
# load fixed-tau refined summary
fixed_csv = ROOT / 'figs' / 'prbs_id' / 'prbs_parametric_refined_fixedtau_summary.csv'
if not fixed_csv.exists():
    raise SystemExit('Fixed-tau refined summary missing: ' + str(fixed_csv))
fixed = pd.read_csv(fixed_csv)
# pick the deep_long row (if multiple, pick U=3.0 or first)
row = fixed.iloc[0]
K = float(row['K'])
wn = float(row['wn'])
zeta = float(row['zeta'])
tau_a = float(row['tau_a_fixed'])

# load best zigzag trajectory (reuse earlier script logic)
zig_summary = ROOT / 'zigzag_search_summary.csv'
if not zig_summary.exists():
    raise SystemExit('Zigzag summary missing: ' + str(zig_summary))
zdf = pd.read_csv(zig_summary)
# prefer rmse_cross_m
if 'rmse_cross_m' in zdf.columns and zdf['rmse_cross_m'].notna().any():
    key = 'rmse_cross_m'
elif 'final_cross_m' in zdf.columns and zdf['final_cross_m'].notna().any():
    key = 'final_cross_m'
else:
    key = None
if key:
    idx = zdf[key].idxmin()
else:
    idx = 0
best = zdf.loc[idx]
traj_path = Path(best['traj_csv'])
if not traj_path.exists():
    traj_path = (ROOT / traj_path).resolve()
if not traj_path.exists():
    raise SystemExit('Trajectory file missing: ' + str(traj_path))
traj = pd.read_csv(traj_path)

# expecting columns: t, psi_deg, r_deg_s, u_m_s, v_m_s, cmd_rudder_deg, applied_rudder_deg, ...
if 'applied_rudder_deg' in traj.columns:
    u_deg = traj['applied_rudder_deg'].values
else:
    u_deg = traj['rudder_deg'].values if 'rudder_deg' in traj.columns else traj['cmd_rudder_deg'].values
u = np.deg2rad(u_deg)

t = traj['t'].values
dt = np.median(np.diff(t))

# simulate model: cascade actuator (1st order) and ship 2nd order
# build continuous TF
num_ship = [K * (wn**2)]
den_ship = [1.0, 2.0*zeta*wn, wn**2]
num_act = [1.0]
den_act = [tau_a, 1.0]
num_tot = np.polymul(num_act, num_ship)
den_tot = np.polymul(den_act, den_ship)
# discretize via ZOH at dt
sysd = signal.cont2discrete((num_tot, den_tot), dt, method='zoh')
b = sysd[0].flatten()
a = sysd[1].flatten()

# simulate yaw-rate output in deg/s
y_sim = signal.lfilter(b, a, u)
y_sim_deg = np.degrees(y_sim)

# compute heading by integrating yaw-rate
psi_sim = np.unwrap(np.radians(traj['psi_deg'].values[0:1]))
# simpler: integrate y_sim (rad/s) to get psi increment
psi_sim = np.zeros_like(y_sim)
psi_sim[0] = np.radians(traj['psi_deg'].values[0])
for i in range(1, len(psi_sim)):
    psi_sim[i] = psi_sim[i-1] + y_sim[i-1] * dt
psi_sim_deg = np.degrees(psi_sim)

# measured: prefer recorded yaw-rate if present, otherwise compute from heading
if 'r_deg_s' in traj.columns:
    r_meas_deg = traj['r_deg_s'].values
else:
    # compute yaw-rate by differentiating unwrapped heading
    if 'psi_deg' in traj.columns:
        psi_meas_deg = traj['psi_deg'].values
        psi_rad = np.unwrap(np.radians(psi_meas_deg))
        r_rad_s = np.gradient(psi_rad, dt)
        r_meas_deg = np.degrees(r_rad_s)
    else:
        raise SystemExit('Trajectory missing both r_deg_s and psi_deg')
# ensure psi_meas_deg exists for plotting/metrics
if 'psi_meas_deg' not in locals():
    psi_meas_deg = traj['psi_deg'].values

# metrics
def metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    var = np.var(y_true)
    r2 = 1.0 - np.var(y_true - y_pred) / (var + 1e-12)
    return rmse, r2

rmse_r, r2_r = metrics(r_meas_deg, y_sim_deg)
rmse_psi, r2_psi = metrics(psi_meas_deg, psi_sim_deg)

OUT = ROOT / 'figs' / 'validation'
OUT.mkdir(parents=True, exist_ok=True)
# save metrics
pd.DataFrame([{
    'traj': str(traj_path.name), 'K': K, 'wn': wn, 'zeta': zeta, 'tau_a': tau_a,
    'rmse_r_deg_s': rmse_r, 'r2_r': r2_r, 'rmse_psi_deg': rmse_psi, 'r2_psi': r2_psi
}]).to_csv(OUT / 'validation_zigzag_summary.csv', index=False)

# plots
fig1, ax = plt.subplots(figsize=(8,3))
ax.plot(t, r_meas_deg, label='meas r')
ax.plot(t, y_sim_deg, '--', label='sim r')
ax.set_xlabel('t [s]'); ax.set_ylabel('r [deg/s]'); ax.legend(); fig1.tight_layout()
fig1.savefig(OUT / 'zigzag_r_compare.png')
plt.close(fig1)

fig2, ax = plt.subplots(figsize=(8,3))
ax.plot(t, psi_meas_deg, label='meas psi')
ax.plot(t, psi_sim_deg, '--', label='sim psi')
ax.set_xlabel('t [s]'); ax.set_ylabel('psi [deg]'); ax.legend(); fig2.tight_layout()
fig2.savefig(OUT / 'zigzag_psi_compare.png')
plt.close(fig2)

print('Validation metrics:')
print(' rmse_r_deg_s:', rmse_r, ' r2_r:', r2_r)
print(' rmse_psi_deg:', rmse_psi, ' r2_psi:', r2_psi)
print('Plots and summary written to', OUT)
