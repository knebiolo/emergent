"""Plot PID trace CSV produced by simulation-level tracing.

Usage: run inside project root. Generates PID trace plots (P/I/D/raw/rud vs time)
and saves PNGs to the workspace.
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_trace(csv_path, out_dir=None, agent=None):
    df = pd.read_csv(csv_path)
    if agent is not None:
        df = df[df['agent'] == agent]
    if df.empty:
        print('No rows to plot for', csv_path)
        return None
    t = df['t']
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, df['err_deg'], label='err_deg')
    ax[0].plot(t, df['derr_deg'], label='derr_deg')
    ax[0].set_ylabel('deg / deg/s')
    ax[0].legend()

    ax[1].plot(t, df['P_deg'], label='P')
    ax[1].plot(t, df['I_deg'], label='I')
    ax[1].plot(t, df['D_deg'], label='D')
    ax[1].set_ylabel('deg')
    ax[1].legend()

    ax[2].plot(t, df['raw_deg'], label='raw (P+I+D+FF)')
    ax[2].plot(t, df['rud_deg'], label='applied rudder')
    ax[2].set_ylabel('deg')
    ax[2].set_xlabel('time (s)')
    ax[2].legend()

    fn = os.path.basename(csv_path).replace('.csv', '')
    if agent is not None:
        fn += f'_agent{agent}'
    out_dir = out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fn + '.png')
    fig.suptitle(fn)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    print('Saved', out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='PID trace CSV path')
    parser.add_argument('--agent', type=int, help='Agent id to plot (optional)')
    parser.add_argument('--out', help='Output directory for PNG')
    args = parser.parse_args()
    plot_trace(args.csv, out_dir=args.out, agent=args.agent)


if __name__ == '__main__':
    main()
