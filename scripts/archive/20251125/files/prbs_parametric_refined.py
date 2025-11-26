"""Refined parametric fit (actuator + ship) using frequency-domain weighted LS

Model structure:
  G_act(s) = 1 / (tau_a*s + 1)  (actuator first-order)
  G_ship(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
  G_total(s) = G_act(s) * G_ship(s)

Fit performed by minimizing weighted complex error between model FRF and ETFE
with weights derived from coherence. Then validate in time-domain and save plots.
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

def latest_prbs():
    files = glob.glob('id*U*_prbs_*.csv')
    by_U = {}
    for f in files:
        name = Path(f).name
        parts = name.split('_')
        # find token that starts with 'U' (supports id_long_U... and id_U...)
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
    f = f
    # coherence
    coh = np.abs(Puy)**2 / (Puu * Pyy)
    return f, ETFE, coh

def model_frf(f_hz, K, wn, zeta, tau_a):
    # f_hz in Hz; convert to rad/s
    w = 2.0 * np.pi * f_hz
    s = 1j * w
    G_act = 1.0 / (tau_a * s + 1.0)
    G_ship = (K * (wn**2)) / (s**2 + 2.0*zeta*wn*s + wn**2)
    return G_act * G_ship

def fit_frequency_domain(f, ETFE, coh, dt, init=None):
    # fit parameters [K, wn, zeta, tau_a] using weighted LS on complex FRF
    fs = 1.0 / dt
    # restrict to f < fs/2
    mask = (f > 0) & (f < fs/2)
    f_fit = f[mask]
    H_meas = ETFE[mask]
    W = coh[mask]
    # weight floor to avoid zero
    W = np.clip(W, 0.01, 1.0)

    if init is None:
        K0 = 0.8
        wn0 = 0.02
        zeta0 = 0.1
        tau0 = 1.0
    else:
        K0, wn0, zeta0, tau0 = init

    def obj(x):
        K, wn, zeta, tau = x
        if wn <= 0 or zeta <= 0 or tau < 0:
            return 1e6
        H = model_frf(f_fit, K, wn, zeta, tau)
        err = (H - H_meas)
        # complex weighted sum of squares
        val = np.sum(W * (np.abs(err)**2))
        return val

    x0 = [K0, wn0, zeta0, tau0]
    bounds = [(-5, 5), (1e-4, 1.0), (1e-3, 5.0), (0.0, 10.0)]
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter':300})
    return res

def simulate_model_time(K, wn, zeta, tau_a, u, dt, delay_samples=0):
    # Build continuous TF and discretize via ZOH
    num_ship = [K * (wn**2)]
    den_ship = [1.0, 2.0*zeta*wn, wn**2]
    # actuator TF
    num_act = [1.0]
    den_act = [tau_a, 1.0]
    # cascade: conv numerators/denominators
    num_tot = np.polymul(num_act, num_ship)
    den_tot = np.polymul(den_act, den_ship)
    sysd = signal.cont2discrete((num_tot, den_tot), dt, method='zoh')
    b = sysd[0].flatten()
    a = sysd[1].flatten()
    if delay_samples > 0:
        u_shift = np.concatenate([np.zeros(delay_samples), u[:-delay_samples]])
    else:
        u_shift = u
    y = signal.lfilter(b, a, u_shift)
    return y

def analyze(csv_path):
    df = pd.read_csv(csv_path)
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
    f, ETFE, coh = compute_etfe(u_d, r_d, fs, nperseg=256)

    res = fit_frequency_domain(f, ETFE, coh, dt)
    if not res.success:
        res = fit_frequency_domain(f, ETFE, coh, dt, init=[0.8, 0.02, 0.1, 1.0])

    K, wn, zeta, tau = res.x
    # simulate in time-domain
    y_sim = simulate_model_time(K, wn, zeta, tau, u, dt)

    # Metrics
    rmse = float(np.sqrt(np.mean((r - y_sim)**2)))
    varr = float(np.var(r))
    r2 = 1.0 - float(np.var(r - y_sim)) / (varr + 1e-12)

    base = Path(csv_path).stem
    # FRF overlay
    f_plot = f[f>0]
    H_meas = ETFE[f>0]
    H_model = model_frf(f_plot, K, wn, zeta, tau)
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
    fig1_path = OUT_DIR / (base + '_refined_frf.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # time-domain
    fig2, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, r, label='meas')
    ax.plot(t, y_sim, '--', label='model')
    ax.set_xlabel('t [s]'); ax.set_ylabel('r [deg/s]')
    ax.legend(); fig2.tight_layout()
    fig2_path = OUT_DIR / (base + '_refined_time.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # residual
    fig3, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, r - y_sim)
    ax.set_xlabel('t [s]'); ax.set_ylabel('residual')
    fig3.tight_layout(); fig3_path = OUT_DIR / (base + '_refined_resid.png')
    fig3.savefig(fig3_path)
    plt.close(fig3)

    summary = {
        'file': str(csv_path), 'K': float(K), 'wn': float(wn), 'zeta': float(zeta), 'tau_a': float(tau), 'rmse': rmse, 'r2': r2
    }
    return summary, [fig1_path, fig2_path, fig3_path]

def main():
    latest = latest_prbs()
    rows = []
    figs = []
    for U, path in latest.items():
        print('Refined fit for', U, path)
        s, fs = analyze(path)
        rows.append(s); figs.extend(fs)
    out = OUT_DIR / 'prbs_parametric_refined_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Wrote', out)
    for f in figs:
        print(' ', f)

if __name__ == '__main__':
    main()
