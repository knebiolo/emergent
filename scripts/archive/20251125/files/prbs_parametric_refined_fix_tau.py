"""Refined parametric fit with fixed actuator tau (tau_a).
Reads latest PRBS files and uses median tau_a from servo runs (if available)
then fits only K, wn, zeta.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize

OUT_DIR = Path('figs') / 'prbs_id'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def latest_prbs(prefix_filter='id_deep_long'):
    files = glob.glob('id*U*_prbs_*.csv')
    by_U = {}
    for f in files:
        if prefix_filter and prefix_filter not in f:
            continue
        name = Path(f).name
        parts = name.split('_')
        Upart = None
        for p in parts:
            if p.startswith('U'):
                Upart = p
                break
        if Upart is None:
            continue
        mtime = Path(f).stat().st_mtime
        if Upart not in by_U or mtime > by_U[Upart][0]:
            by_U[Upart] = (mtime, f)
    latest = {U: v[1] for U, v in by_U.items()}
    return latest


def compute_etfe(u, y, fs, nperseg=256):
    f, Puu = signal.welch(u, fs=fs, nperseg=nperseg)
    f, Pyy = signal.welch(y, fs=fs, nperseg=nperseg)
    f, Puy = signal.csd(u, y, fs=fs, nperseg=nperseg)
    ETFE = Puy / Puu
    coh = np.abs(Puy)**2 / (Puu * Pyy)
    return f, ETFE, coh


def model_frf(f_hz, K, wn, zeta, tau_a):
    w = 2.0 * np.pi * f_hz
    s = 1j * w
    G_act = 1.0 / (tau_a * s + 1.0)
    G_ship = (K * (wn**2)) / (s**2 + 2.0*zeta*wn*s + wn**2)
    return G_act * G_ship


def fit_fixed_tau(f, ETFE, coh, dt, tau_a, init=None):
    mask = (f > 0) & (f < 0.5/dt)
    f_fit = f[mask]
    H_meas = ETFE[mask]
    W = coh[mask]
    W = np.clip(W, 0.01, 1.0)

    if init is None:
        K0 = 0.8
        wn0 = 0.05
        zeta0 = 0.1
    else:
        K0, wn0, zeta0 = init

    def obj(x):
        K, wn, zeta = x
        if wn <= 0 or zeta <= 0:
            return 1e9
        H = model_frf(f_fit, K, wn, zeta, tau_a)
        err = (H - H_meas)
        return np.sum(W * (np.abs(err)**2))

    x0 = [K0, wn0, zeta0]
    bounds = [(-10, 10), (1e-4, 1.0), (1e-3, 5.0)]
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter':500})
    return res


def simulate_model_time(K, wn, zeta, tau_a, u, dt):
    num_ship = [K * (wn**2)]
    den_ship = [1.0, 2.0*zeta*wn, wn**2]
    num_act = [1.0]
    den_act = [tau_a, 1.0]
    num_tot = np.polymul(num_act, num_ship)
    den_tot = np.polymul(den_act, den_ship)
    sysd = signal.cont2discrete((num_tot, den_tot), dt, method='zoh')
    b = sysd[0].flatten()
    a = sysd[1].flatten()
    y = signal.lfilter(b, a, u)
    return y


def estimate_tau_from_servo_summary():
    p = OUT_DIR / 'prbs_parametric_refined_summary.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # take rows that contain 'servo' in file name
    ser = df[df['file'].str.contains('servo')]
    if ser.empty:
        return None
    return float(ser['tau_a'].median())


def analyze_one(path, tau_a_fixed):
    df = pd.read_csv(path)
    t = df['t'].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    r = df['r_deg_s'].values
    if 'applied_rudder_deg' in df.columns:
        u = df['applied_rudder_deg'].values
    else:
        u = df['cmd_rudder_deg'].values
    u_d = signal.detrend(u)
    r_d = signal.detrend(r)
    f, ETFE, coh = compute_etfe(u_d, r_d, fs, nperseg=1024)

    res = fit_fixed_tau(f, ETFE, coh, dt, tau_a_fixed, init=[0.8, 0.02, 0.1])
    if not res.success:
        res = fit_fixed_tau(f, ETFE, coh, dt, tau_a_fixed)
    K, wn, zeta = res.x
    y_sim = simulate_model_time(K, wn, zeta, tau_a_fixed, u, dt)
    rmse = float(np.sqrt(np.mean((r - y_sim)**2)))
    varr = float(np.var(r))
    r2 = 1.0 - float(np.var(r - y_sim)) / (varr + 1e-12)

    base = Path(path).stem
    f_plot = f[f>0]
    H_meas = ETFE[f>0]
    H_model = model_frf(f_plot, K, wn, zeta, tau_a_fixed)

    fig1, ax = plt.subplots(2,1, figsize=(8,6))
    ax[0].semilogx(f_plot, 20*np.log10(np.abs(H_meas)), label='ETFE')
    ax[0].semilogx(f_plot, 20*np.log10(np.abs(H_model)), label='Model')
    ax[0].set_ylabel('Mag [dB]')
    ang_meas = np.angle(H_meas, deg=True)
    ang_mod = np.angle(H_model, deg=True)
    ax[1].semilogx(f_plot, ang_meas, label='ETFE')
    ax[1].semilogx(f_plot, ang_mod, label='Model')
    ax[1].set_ylabel('Phase [deg]'); ax[1].set_xlabel('Frequency [Hz]')
    ax[0].legend()
    fig1.tight_layout()
    fig1_path = OUT_DIR / (base + '_fixedtau_frf.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, r, label='meas')
    ax.plot(t, y_sim, '--', label='model')
    ax.set_xlabel('t [s]'); ax.set_ylabel('r [deg/s]')
    ax.legend(); fig2.tight_layout()
    fig2_path = OUT_DIR / (base + '_fixedtau_time.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)

    summary = {'file': str(path), 'K': float(K), 'wn': float(wn), 'zeta': float(zeta), 'tau_a_fixed': float(tau_a_fixed), 'rmse': rmse, 'r2': r2}
    return summary, [fig1_path, fig2_path]


def main():
    tau_est = estimate_tau_from_servo_summary()
    if tau_est is None:
        print('No servo summary found; using tau_a=1.0')
        tau_est = 1.0
    else:
        print('Using estimated tau_a from servo runs:', tau_est)

    latest = latest_prbs(prefix_filter='id_deep_long')
    rows = []
    figs = []
    for U, path in latest.items():
        print('Fitting deep_long for', U, path)
        s, fs = analyze_one(path, tau_est)
        rows.append(s); figs.extend(fs)
    out = OUT_DIR / 'prbs_parametric_refined_fixedtau_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Wrote', out)
    for f in figs:
        print(' ', f)

if __name__ == '__main__':
    main()
