"""PRBS analysis and simple ARX identification

Reads PRBS CSV telemetry files produced by `scripts/id_prbs.py`, computes
Welch PSD/coherence and ETFE for rudder -> yaw-rate, fits a discrete-time
ARX(2,2,nk=1) model (yaw-rate in deg/s, rudder in deg), and saves plots
and a summary CSV of fitted parameters.

Usage:
    python scripts/prbs_analysis_and_id.py
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

OUT_DIR = Path('figs') / 'prbs_id'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_per_speed():
    # match both 'id_U...' and 'id_long_U...' variants and pick the newest file per U
    files = glob.glob('id*U*_prbs_*.csv')
    by_U = {}
    for f in files:
        name = Path(f).name
        parts = name.split('_')
        # find the token that starts with 'U' (supports id_long_U... and id_U...)
        Upart = None
        for p in parts:
            if p.startswith('U'):
                Upart = p
                break
        if Upart is None:
            continue
        # choose newest by file modification time
        mtime = Path(f).stat().st_mtime
        if Upart not in by_U or mtime > by_U[Upart][0]:
            by_U[Upart] = (mtime, f)
    latest = {U: v[1] for U, v in by_U.items()}
    # debug: print matched files and selection
    print('Matched PRBS files:')
    for f in files:
        print('  ', f)
    print('Selected latest per U:')
    for U, f in latest.items():
        print('  ', U, '->', f)
    return latest

def compute_etfe_and_coherence(t,u,y,fs, nperseg=256):
    # u: input, y: output (1D arrays)
    f, Pxx = signal.welch(u, fs=fs, nperseg=nperseg)
    f, Pyy = signal.welch(y, fs=fs, nperseg=nperseg)
    f, Pxy = signal.csd(u, y, fs=fs, nperseg=nperseg)
    # ETFE (u->y) = Pyu / Puu  (we computed Pxy = Suy)
    ETFE = Pxy / Pxx
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)
    return f, ETFE, coh, Pxx, Pyy, Pxy

def fit_arx(y, u, na=2, nb=2, nk=1):
    # y, u are 1D arrays sampled at uniform dt
    N = len(y)
    d = max(na, nb + nk - 1)
    rows = []
    ys = []
    for k in range(d, N):
        phi = []
        # - past outputs
        for i in range(1, na+1):
            phi.append(-y[k-i])
        # past inputs with delay nk
        for j in range(nk, nk+nb):
            phi.append(u[k-j])
        rows.append(phi)
        ys.append(y[k])
    Phi = np.vstack(rows)
    Y = np.array(ys)
    # least squares
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    # split params
    a = theta[:na]
    b = theta[na:na+nb]
    return a, b, nk

def simulate_arx(u, a, b, nk, y0=None):
    na = len(a); nb = len(b)
    N = len(u)
    y = np.zeros(N)
    if y0 is not None:
        y[:len(y0)] = y0
    for k in range( max(na, nb+nk-1), N):
        val = 0.0
        for i in range(1, na+1):
            val += -a[i-1] * y[k-i]
        for j in range(nk, nk+nb):
            val += b[j-nk] * u[k-j]
        y[k] = val
    return y

def analyze_file(csv_path):
    df = pd.read_csv(csv_path)
    t = df['t'].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    psi = df['psi_deg'].values
    r = df['r_deg_s'].values
    # prefer applied rudder if available else cmd_rudder_deg
    if 'applied_rudder_deg' in df.columns:
        u = df['applied_rudder_deg'].values
    else:
        u = df['cmd_rudder_deg'].values

    # detrend signals for spectral estimates
    u_d = signal.detrend(u)
    r_d = signal.detrend(r)

    f, ETFE, coh, Pxx, Pyy, Pxy = compute_etfe_and_coherence(t, u_d, r_d, fs, nperseg=256)

    # Plot coherence and ETFE magnitude/phase
    base = Path(csv_path).stem
    fig1, ax = plt.subplots(figsize=(8,3))
    ax.semilogx(f, coh)
    ax.set_ylim([0,1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_title(f'Coherence (rudder -> yaw-rate) {base}')
    fig1.tight_layout()
    fig1_path = OUT_DIR / (base + '_coherence.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)

    fig2, ax = plt.subplots(2,1, figsize=(8,6))
    ax[0].semilogx(f, 20*np.log10(np.abs(ETFE)))
    ax[0].set_ylabel('Mag [dB]')
    ang = np.angle(ETFE, deg=True)
    ax[1].semilogx(f, ang)
    ax[1].set_ylabel('Phase [deg]')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[0].set_title(f'ETFE (rudder -> yaw-rate) {base}')
    fig2.tight_layout()
    fig2_path = OUT_DIR / (base + '_etfe.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # Fit ARX model
    a,b,nk = fit_arx(r, u, na=2, nb=2, nk=1)
    r_hat = simulate_arx(u, a, b, nk)

    # time-domain plot
    fig3, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, r, label='r (measured)')
    ax.plot(t, r_hat, label='r (ARX pred)', linestyle='--')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('r [deg/s]')
    ax.set_title(f'Time-domain: measured vs ARX {base}')
    ax.legend()
    fig3.tight_layout()
    fig3_path = OUT_DIR / (base + '_time_arx.png')
    fig3.savefig(fig3_path)
    plt.close(fig3)

    # metrics
    rmse = np.sqrt(np.mean((r - r_hat)**2))
    varr = np.var(r)
    r2 = 1 - np.var(r - r_hat) / (varr + 1e-12)

    summary = {
        'file': str(csv_path),
        'dt': float(dt),
        'na': 2,
        'nb': 2,
        'nk': nk,
        'a1': float(a[0]),
        'a2': float(a[1]),
        'b1': float(b[0]),
        'b2': float(b[1]),
        'rmse_r_deg_s': float(rmse),
        'r2_r': float(r2),
        'mean_coherence': float(np.nanmean(coh[(f>0) & (f<fs/2)]))
    }

    return summary, [fig1_path, fig2_path, fig3_path]

def main():
    latest = load_latest_per_speed()
    rows = []
    files_done = []
    for U, path in latest.items():
        print('Analyzing', U, path)
        summary, figs = analyze_file(path)
        rows.append(summary)
        files_done.extend(figs)
    # write summary CSV
    out_csv = OUT_DIR / 'prbs_id_summary.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('Wrote summary:', out_csv)
    print('Wrote figures:')
    for f in files_done:
        print(' ', f)

if __name__ == '__main__':
    main()
