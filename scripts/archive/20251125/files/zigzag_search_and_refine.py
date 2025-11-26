"""Driver: zig-zag search and local refinement

- Runs a moderate grid sweep over Kp, Ki, Kf, max_rudder, and wind speeds.
- Selects top candidates by composite_cost.
- Runs a local refinement grid around the top candidates and performs Monte-Carlo gust trials.
- Writes summaries and generates plots under figs/zigzag/ and figs/search/.

This script imports the existing `run_one` from `scripts/zigzag_kf_sweep.py` by loading it as a module.
"""
import os
import csv
import math
import numpy as np
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
FIGS = ROOT / 'figs' / 'zigzag'
FIGS.mkdir(parents=True, exist_ok=True)

# load run_one from scripts/zigzag_kf_sweep.py
spec = importlib.util.spec_from_file_location('zigzag_kf_sweep', str(SCRIPTS / 'zigzag_kf_sweep.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_one = getattr(mod, 'run_one')


def random_constant_wind_fn(base_speed, gust_amp):
    # returns a wind_fn(lons,lats,when) which yields a random constant vector for this trial
    ang = np.random.uniform(0, 2 * math.pi)
    mag = float(base_speed + np.random.uniform(-gust_amp, gust_amp))
    wx = mag * math.cos(ang)
    wy = mag * math.sin(ang)
    def wind_fn(lons, lats, when):
        N = int(np.atleast_1d(lons).size)
        return np.tile(np.array([[wx, wy]]), (N, 1))
    return wind_fn


def run_search():
    # moderate grid
    Kp_vals = [0.4, 0.5, 0.6]
    Ki_vals = [0.01, 0.05]
    Kf_vals = [0.0, 0.002, 0.01, 0.02]
    max_rudder_vals = [20.0, 25.0]
    wind_speeds = [0.5, 1.5]

    results = []
    total = len(Kp_vals) * len(Ki_vals) * len(Kf_vals) * len(max_rudder_vals) * len(wind_speeds)
    i = 0
    for wind in wind_speeds:
        for mr in max_rudder_vals:
            for Kf in Kf_vals:
                for Kp in Kp_vals:
                    for Ki in Ki_vals:
                        i += 1
                        print(f"[SEARCH {i}/{total}] wind={wind}, mr={mr}, Kf={Kf}, Kp={Kp}, Ki={Ki}")
                        res = run_one(wind_speed=wind, dead_reck_sens=0.5, Kf_gain=Kf,
                                      Kp=Kp, Ki=Ki, Kd=0.12,
                                      max_rudder_deg=mr,
                                      zig_legs=6, leg_length=200.0, zig_amp=30.0,
                                      dt=0.5, T=240.0, verbose=False)
                        results.append(res)
    # write summary
    df = pd.DataFrame(results)
    out = ROOT / 'zigzag_search_summary.csv'
    df.to_csv(out, index=False)
    print('[SEARCH] wrote', out)
    return df


def local_refine(df, top_n=3, gust_trials=10):
    # pick top by composite_cost
    df_sorted = df.sort_values('composite_cost', na_position='last')
    top = df_sorted.head(top_n)
    all_ref = []
    for idx, row in top.iterrows():
        base_Kp = float(row['Kp'])
        base_Ki = float(row['Ki'])
        base_Kf = float(row['Kf_gain'])
        mr = float(row['max_rudder_deg'])
        wind = float(row['wind_speed'])
        print(f"[REFINE] candidate Kp={base_Kp}, Ki={base_Ki}, Kf={base_Kf}, mr={mr}, wind={wind}")
        Kp_factors = [0.85, 1.0, 1.15]
        Ki_factors = [0.5, 1.0, 1.5]
        grid = []
        for kp_fac in Kp_factors:
            for ki_fac in Ki_factors:
                kp = base_Kp * kp_fac
                ki = base_Ki * ki_fac
                # evaluate nominal performance
                res_nom = run_one(wind_speed=wind, dead_reck_sens=0.5, Kf_gain=base_Kf,
                                  Kp=kp, Ki=ki, Kd=0.12,
                                  max_rudder_deg=mr,
                                  zig_legs=6, leg_length=200.0, zig_amp=30.0,
                                  dt=0.5, T=240.0, verbose=False)
                # run gust Monte Carlo trials: constant random wind perturbations
                rmse_trials = []
                for t in range(gust_trials):
                    wind_fn = random_constant_wind_fn(base_speed=wind, gust_amp=1.5)
                    res_t = run_one(wind_speed=wind, dead_reck_sens=0.5, Kf_gain=base_Kf,
                                    Kp=kp, Ki=ki, Kd=0.12,
                                    max_rudder_deg=mr,
                                    zig_legs=6, leg_length=200.0, zig_amp=30.0,
                                    dt=0.5, T=240.0, verbose=False, wind_fn_override=wind_fn)
                    rmse_trials.append(res_t['rmse_cross_m'])
                rmse_trials = np.array(rmse_trials, dtype=float)
                grid.append({
                    'parent_idx': idx,
                    'base_Kp': base_Kp,
                    'base_Ki': base_Ki,
                    'Kp': kp,
                    'Ki': ki,
                    'Kf': base_Kf,
                    'max_rudder_deg': mr,
                    'wind_speed': wind,
                    'nominal_rmse': res_nom['rmse_cross_m'],
                    'nominal_composite': res_nom['composite_cost'],
                    'gust_mean_rmse': float(np.nanmean(rmse_trials)),
                    'gust_std_rmse': float(np.nanstd(rmse_trials)),
                    'gust_worst_rmse': float(np.nanmax(rmse_trials)),
                    'traj_csv': res_nom['traj_csv']
                })
        all_ref.extend(grid)
    df_ref = pd.DataFrame(all_ref)
    out = ROOT / 'zigzag_refinement_summary.csv'
    df_ref.to_csv(out, index=False)
    print('[REFINE] wrote', out)
    return df_ref


def make_plots(df_search, df_ref):
    # simple bar plot: mean rmse vs Kf per wind
    p = FIGS / 'search_rmse_by_kf.png'
    fig, ax = plt.subplots(figsize=(8,4))
    gb = df_search.groupby(['wind_speed','Kf_gain'])['rmse_cross_m'].mean().reset_index()
    for wind in sorted(gb['wind_speed'].unique()):
        sub = gb[gb['wind_speed'] == wind]
        ax.plot(sub['Kf_gain'], sub['rmse_cross_m'], marker='o', label=f'wind={wind} m/s')
    ax.set_xlabel('Kf')
    ax.set_ylabel('mean RMSE (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(p)
    print('[PLOT] wrote', p)

    # top trajectories overlay
    top_rows = df_search.nsmallest(6, 'composite_cost')
    fig, ax = plt.subplots(figsize=(8,8))
    for _, r in top_rows.iterrows():
        try:
            traj = pd.read_csv(r['traj_csv'])
            ax.plot(traj['x_m'], traj['y_m'], label=f"Kp={r['Kp']:.2f},Ki={r['Ki']:.3f},Kf={r['Kf_gain']}")
        except Exception:
            pass
    # draw polyline used in zigzag (centered) — reuse one of the traj files to reconstruct waypoint poly
    if len(top_rows) > 0:
        try:
            sample = pd.read_csv(top_rows.iloc[0]['traj_csv'])
            x0 = sample.iloc[0]['x_m']
            y0 = sample.iloc[0]['y_m']
            leg_length = 200.0
            zig_legs = 6
            zig_amp = 30.0
            way = [(x0, y0)]
            dir = 1
            for i in range(1, zig_legs+1):
                way.append((x0 + i * leg_length, y0 + dir * zig_amp))
                dir *= -1
            wx = [p[0] for p in way]
            wy = [p[1] for p in way]
            ax.plot(wx, wy, 'k--', lw=1.5, label='waypoints')
        except Exception:
            pass
    ax.set_title('Top trajectories (overlay)')
    ax.legend()
    fig.savefig(FIGS / 'top_trajectories.png')
    print('[PLOT] wrote', FIGS / 'top_trajectories.png')

    # refinement heatmaps per parent
    for parent_idx, g in df_ref.groupby('parent_idx'):
        pivot = g.pivot(index='Ki', columns='Kp', values='gust_mean_rmse')
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(pivot.values, origin='lower', cmap='viridis', aspect='auto')
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels([f"{c:.3f}" for c in pivot.index])
        ax.set_xlabel('Kp'); ax.set_ylabel('Ki')
        fig.colorbar(im, ax=ax, label='gust mean RMSE (m)')
        fig.tight_layout()
        outp = FIGS / f'refine_parent_{parent_idx}_heatmap.png'
        fig.savefig(outp)
        print('[PLOT] wrote', outp)


if __name__ == '__main__':
    df_search = run_search()
    df_ref = local_refine(df_search, top_n=3, gust_trials=10)
    make_plots(df_search, df_ref)
    print('Done')
