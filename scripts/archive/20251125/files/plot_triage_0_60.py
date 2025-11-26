import pandas as pd
import matplotlib.pyplot as plt
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
res_dir = os.path.join(base, 'sweep_results')
fig_dir = os.path.join(res_dir, 'figs')
os.makedirs(fig_dir, exist_ok=True)

files = [
    ('pid_trace_osc_keepKp_lowerKd_dtau2.0_full.csv', 'deadreckon_on'),
    ('pid_trace_osc_keepKp_lowerKd_dtau2.0_full_nodead.csv', 'deadreckon_off'),
]

for fname, tag in files:
    path = os.path.join(res_dir, fname)
    print('Reading', path)
    df = pd.read_csv(path)
    df60 = df[df['t'] <= 60.0]

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.plot(df60['t'], df60['hd_cmd_deg'], label='hd_cmd_deg', color='C0')
    ax.plot(df60['t'], df60['psi_deg'], label='psi_deg', color='C1')
    ax.plot(df60['t'], df60['err_deg'], label='err_deg', color='C2')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('deg / deg-ref')
    ax.set_title(f'triage 0-60s {tag}')
    ax.legend(loc='upper right')
    plt.grid(True)
    out_png = os.path.join(fig_dir, f'triage_0_60_{tag}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

    # actuator & raw plot
    plt.figure(figsize=(10,4))
    plt.plot(df60['t'], df60['raw_deg'], label='raw_deg')
    plt.plot(df60['t'], df60['rud_deg'], label='rud_deg')
    plt.xlabel('t (s)')
    plt.ylabel('deg')
    plt.title(f'actuator 0-60s {tag}')
    plt.legend()
    plt.grid(True)
    out_png2 = os.path.join(fig_dir, f'actuator_0_60_{tag}.png')
    plt.savefig(out_png2, dpi=150)
    plt.close()

print('Done')
