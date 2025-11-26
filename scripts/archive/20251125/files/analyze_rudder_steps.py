import glob
import pandas as pd
import numpy as np
import math

def analyze_file(path):
    df = pd.read_csv(path)
    t = df['t'].values
    cmd = df['cmd_rudder_deg'].values
    applied = df['applied_rudder_deg'].values
    psi = df['psi_deg'].values
    T = t[-1]
    # steady-state estimate: mean of last 20% of samples
    n_tail = max(3, int(len(t)*0.2))
    steady = np.mean(applied[-n_tail:])
    # sign-aware magnitude
    sign = np.sign(steady) if steady!=0 else 1.0
    target_mag = abs(steady)
    # time to reach 63% of steady (first crossing)
    thresh = 0.632 * target_mag
    t63 = None
    for ti, a in zip(t, np.abs(applied)):
        if a >= thresh:
            t63 = ti
            break
    # peak applied rudder magnitude and time
    peak_idx = np.argmax(np.abs(applied))
    peak = applied[peak_idx]
    t_peak = t[peak_idx]
    # heading change from start to end
    delta_psi = psi[-1] - psi[0]
    return {
        'file': path,
        'T': T,
        'steady_deg': steady,
        'peak_deg': peak,
        't_peak_s': t_peak,
        't63_s': t63,
        'delta_psi_deg': delta_psi,
        'n_rows': len(df)
    }

def main():
    files = sorted(glob.glob('scripts/rudder_step_*.csv'))
    results = []
    for f in files:
        results.append(analyze_file(f))
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('scripts/rudder_step_summary.csv', index=False)
    print('\nWrote scripts/rudder_step_summary.csv')

if __name__ == '__main__':
    main()
