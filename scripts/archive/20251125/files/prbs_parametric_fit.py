"""Parametric continuous-time fit for PRBS ID data

Fits a 2nd-order continuous-time transfer function
    G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
optionally with input delay (approximated as integer sample shift), by
minimizing time-domain RMSE between measured yaw-rate and simulated output
when driving with the recorded applied rudder.

Produces FRF overlay, time-domain fit and residual plots, and a CSV summary.
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

def load_latest_per_speed():
    files = sorted(glob.glob('id_U*_prbs_*.csv'))
    by_U = {}
    for f in files:
        name = Path(f).name
        parts = name.split('_')
        if len(parts) < 2:
            continue
        Upart = parts[1]
        if not Upart.startswith('U'):
            continue
        by_U.setdefault(Upart, []).append(f)
    latest = {}
    for U, flist in by_U.items():
        latest[U] = sorted(flist)[-1]
    return latest

def simulate_ct_second_order(K, wn, zeta, u, dt, delay_samples=0):
    # continuous-time TF: num = [K*wn^2], den=[1, 2*zeta*wn, wn^2]
    num = [K * (wn**2)]
    den = [1.0, 2.0*zeta*wn, wn**2]
    # discretize via ZOH
    sysd = signal.cont2discrete((num, den), dt, method='zoh')
    b = sysd[0].flatten()
    a = sysd[1].flatten()
    # apply delay by shifting input
    if delay_samples > 0:
        u_shift = np.concatenate([np.zeros(delay_samples), u[:-delay_samples]])
    else:
        u_shift = u
    # simulate using lfilter: a[0] should be 1.0
    try:
        y = signal.lfilter(b, a, u_shift)
    except Exception:
        # fallback: simple convolution if lfilter fails
        y = np.convolve(u_shift, b)[:len(u_shift)]
    return y

def objective(params, u, r_meas, dt):
    K, wn, zeta, delay = params
    # enforce positivity
    if wn <= 0 or zeta <= 0:
        return 1e6
    delay_samples = int(round(max(0.0, delay) / dt))
    r_sim = simulate_ct_second_order(K, wn, zeta, u, dt, delay_samples=delay_samples)
    # compute RMSE over all
    err = r_meas - r_sim
    return np.sqrt(np.mean(err**2))

def fit_params(u, r, dt, init=None):
    # initial guess
    if init is None:
        # estimate DC gain from low-frequency ETFE approx
        f, Puu = signal.welch(u, fs=1.0/dt, nperseg=256)
        f, Pyy = signal.welch(r, fs=1.0/dt, nperseg=256)
        # crude initial K: ratio of variances
        K0 = (np.std(r) / (np.std(u) + 1e-12)) if np.std(u)>0 else 0.1
        wn0 = 0.5
        zeta0 = 0.3
        delay0 = 0.0
    else:
        K0, wn0, zeta0, delay0 = init

    x0 = [K0, wn0, zeta0, delay0]
    bounds = [(-10,10), (0.01, 10.0), (0.01, 5.0), (0.0, 2.0)]
    res = minimize(objective, x0, args=(u, r, dt), bounds=bounds, method='L-BFGS-B', options={'maxiter':200})
    return res

def analyze_file(csv_path):
    df = pd.read_csv(csv_path)
    t = df['t'].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    r = df['r_deg_s'].values
    if 'applied_rudder_deg' in df.columns:
        u = df['applied_rudder_deg'].values
    else:
        u = df['cmd_rudder_deg'].values

    # detrend input slightly
    u = signal.detrend(u)

    # initial fit with simple guess
    res = fit_params(u, r, dt)
    if not res.success:
        # try alternative init
        res = fit_params(u, r, dt, init=[0.1, 0.3, 0.5, 0.0])

    K, wn, zeta, delay = res.x
    delay_samples = int(round(delay / dt))
    r_sim = simulate_ct_second_order(K, wn, zeta, u, dt, delay_samples=delay_samples)

    # frequency response of continuous model
    num = [K * (wn**2)]
    den = [1.0, 2.0*zeta*wn, wn**2]
    # frequency vector (rad/s) up to Nyquist for sampling dt
    w = np.linspace(0.0, np.pi / dt, 512)
    w, h = signal.freqresp(signal.TransferFunction(num, den), w)
    f_hz = w / (2.0 * np.pi)

    # ETFE of data
    f_etfe, Puu = signal.welch(u, fs=fs, nperseg=256)
    f_etfe, Pyy = signal.welch(r, fs=fs, nperseg=256)
    f_etfe, Pxy = signal.csd(u, r, fs=fs, nperseg=256)
    ETFE = Pxy / Puu

    base = Path(csv_path).stem
    # FRF overlay
    fig1, ax = plt.subplots(figsize=(8,3))
    ax.semilogx(f_etfe, 20*np.log10(np.abs(ETFE)+1e-12), label='ETFE (data)')
    # interpolate model h to f_etfe
    h_interp = np.interp(f_etfe, f_hz, 20*np.log10(np.abs(h)+1e-12))
    ax.semilogx(f_etfe, h_interp, label='Model FRF (mag)')
    ax.set_ylabel('Mag [dB]')
    ax.set_title(f'FRF overlay {base}')
    ax.legend()
    fig1.tight_layout()
    fig1_path = OUT_DIR / (base + '_frf_overlay.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # time-domain overlay
    fig2, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, r, label='r measured')
    ax.plot(t, r_sim, '--', label='r model')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('r [deg/s]')
    ax.set_title(f'Time-domain fit {base}')
    ax.legend()
    fig2.tight_layout()
    fig2_path = OUT_DIR / (base + '_time_fit.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # residuals
    resids = r - r_sim
    fig3, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, resids)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('residual [deg/s]')
    ax.set_title(f'Residuals {base}')
    fig3.tight_layout()
    fig3_path = OUT_DIR / (base + '_residuals.png')
    fig3.savefig(fig3_path)
    plt.close(fig3)

    # metrics
    rmse = float(np.sqrt(np.mean((r - r_sim)**2)))
    varr = float(np.var(r))
    r2 = 1.0 - float(np.var(r - r_sim)) / (varr + 1e-12)

    summary = {
        'file': str(csv_path),
        'K': float(K),
        'wn': float(wn),
        'zeta': float(zeta),
        'delay_s': float(delay),
        'rmse_r_deg_s': rmse,
        'r2_r': r2
    }
    return summary, [fig1_path, fig2_path, fig3_path]

def main():
    latest = load_latest_per_speed()
    rows = []
    figs_all = []
    for U, path in latest.items():
        print('Fitting parametric model for', U, path)
        summary, figs = analyze_file(path)
        rows.append(summary)
        figs_all.extend(figs)
    out_csv = OUT_DIR / 'prbs_parametric_summary.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('Wrote summary:', out_csv)
    for f in figs_all:
        print(' ', f)

if __name__ == '__main__':
    main()
