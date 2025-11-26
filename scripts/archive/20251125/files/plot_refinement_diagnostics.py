import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap_deg(d):
    return (d + 180) % 360 - 180


def heading_error_deg(psi, cmd):
    # signed error psi - cmd normalized to [-180,180]
    return wrap_deg(psi - cmd)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def read_traj(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return None


def plot_run(df, outdir, run_label):
    t = df['t']
    x = df['x_m']
    y = df['y_m']
    psi = df['psi_deg']
    hd_cmd = df['hd_cmd_deg']
    cross = df['cross_track_m'] if 'cross_track_m' in df.columns else None
    rudder = df['rudder_deg'] if 'rudder_deg' in df.columns else None

    he = heading_error_deg(psi.values, hd_cmd.values)

    ensure_dir(outdir)

    # Plan view
    plt.figure(figsize=(6,6))
    plt.plot(x - x.iloc[0], y - y.iloc[0], '-k', lw=1)
    plt.scatter([0], [0], c='g', s=50, label='start')
    plt.scatter([x.iloc[-1]-x.iloc[0]], [y.iloc[-1]-y.iloc[0]], c='r', s=50, label='end')
    plt.axis('equal')
    plt.xlabel('x - x0 (m)')
    plt.ylabel('y - y0 (m)')
    plt.title(f'Plan view: {run_label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{run_label}_plan.png'))
    plt.close()

    # Cross-track vs time
    if cross is not None:
        plt.figure(figsize=(6,3))
        plt.plot(t, cross, '-b')
        plt.xlabel('t (s)')
        plt.ylabel('cross-track (m)')
        plt.title(f'Cross-track: {run_label}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{run_label}_cross_vs_time.png'))
        plt.close()

    # Heading error vs time
    plt.figure(figsize=(6,3))
    plt.plot(t, he, '-m')
    plt.xlabel('t (s)')
    plt.ylabel('heading error (deg)')
    plt.title(f'Heading error: {run_label}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{run_label}_heading_err_vs_time.png'))
    plt.close()

    # Rudder vs time
    if rudder is not None:
        plt.figure(figsize=(6,3))
        plt.plot(t, rudder, '-r')
        plt.xlabel('t (s)')
        plt.ylabel('rudder (deg)')
        plt.title(f'Rudder: {run_label}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{run_label}_rudder_vs_time.png'))
        plt.close()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    workspace = root
    ref_csv = os.path.join(workspace, 'refinement_summary.csv')
    gust_csv = os.path.join(workspace, 'gust_robustness_summary.csv')

    out_root = os.path.join(root, 'figs', 'refinement')
    ensure_dir(out_root)

    if not os.path.exists(ref_csv):
        print('refinement_summary.csv not found at', ref_csv)
        return

    ref = pd.read_csv(ref_csv)

    # choose top-3 by composite_cost (lowest), and the worst 1 (highest)
    ref_sorted = ref.sort_values('composite_cost')
    top3 = ref_sorted.head(3)
    worst1 = ref_sorted.tail(1)

    picks = []
    for _, row in top3.iterrows():
        picks.append((int(row.get('parent_candidate_index', -1)), row['traj_csv'], row))
    for _, row in worst1.iterrows():
        picks.append((int(row.get('parent_candidate_index', -1)), row['traj_csv'], row))

    # analyze gusts: group by parent_candidate_index and compute mean rmse_cross_m
    if os.path.exists(gust_csv):
        gust = pd.read_csv(gust_csv)
        grp = gust.groupby('parent_candidate_index')['rmse_cross_m'].agg(['mean','std','count']).reset_index()
        grp_sorted = grp.sort_values('mean')
        # pick best robust candidate
        best_robust = grp_sorted.head(1)
        for _, r in best_robust.iterrows():
            idx = int(r['parent_candidate_index'])
            # find any matching traj in ref
            match = ref[ref['parent_candidate_index'] == idx]
            if len(match) > 0:
                picks.append((idx, match.iloc[0]['traj_csv'], match.iloc[0]))

    # unique picks
    unique_paths = []
    final_picks = []
    for idx, path, row in picks:
        if path in unique_paths:
            continue
        unique_paths.append(path)
        final_picks.append((idx, path, row))

    print(f"Will produce plots for {len(final_picks)} runs")

    for idx, path, row in final_picks:
        run_label = f'cand{idx}_Kp{row.get("Kp")}_Ki{row.get("Ki")}_Kd{row.get("Kd")}_dr{row.get("dead_reck_sens")}'
        safe_label = run_label.replace(' ', '_').replace('/', '_')
        outdir = os.path.join(out_root, safe_label)
        print('Processing', path, '->', outdir)
        df = read_traj(path)
        if df is None:
            print('  missing trajectory file, skipping')
            continue
        plot_run(df, outdir, safe_label)

    # create a short summary figure of refinement composite_cost distribution
    plt.figure(figsize=(6,3))
    plt.hist(ref['composite_cost'].dropna(), bins=30)
    plt.xlabel('composite_cost')
    plt.ylabel('count')
    plt.title('Composite cost distribution (refinement)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, 'composite_cost_hist.png'))
    plt.close()

    if os.path.exists(gust_csv):
        gust = pd.read_csv(gust_csv)
        # boxplot of rmse_cross_m by parent_candidate_index (only small set)
        small = gust.groupby('parent_candidate_index')['rmse_cross_m'].apply(list)
        keys = list(small.index.astype(str))
        vals = list(small.values)
        plt.figure(figsize=(8,4))
        plt.boxplot(vals, labels=keys, showfliers=False)
        plt.xlabel('parent_candidate_index')
        plt.ylabel('rmse_cross_m')
        plt.title('Gust trial RMSE by candidate')
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, 'gust_rmse_boxplot.png'))
        plt.close()

    print('Plots written to', out_root)


if __name__ == '__main__':
    main()
