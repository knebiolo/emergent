import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from pyproj import Transformer

from pathlib import Path as _P

def plot_dali(output_dir: str, config_path: str, out_png: str = None):
    cfg = None
    import json
    with open(config_path, 'r') as fh:
        cfg = json.load(fh)

    out = Path(output_dir)
    files = sorted(out.glob('run_*_trajectory.csv'))
    if not files:
        raise SystemExit(f'No trajectory CSVs found in {out}')

    port = cfg['simulation']['port_name']
    # instantiate a small simulation object to get the sim CRS
    try:
        from emergent.ship_abm.simulation_core import simulation
        sim = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=False)
        crs = sim.crs_utm
    except Exception:
        # fallback: approximate UTM zone from first waypoint lon
        wp = cfg['route']['waypoints_lonlat'][0][0]
        lon = float(wp[0])
        utm_zone = int((lon + 180) // 6) + 1
        crs = f"EPSG:{32600 + utm_zone}"

    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

    # Bridge coordinates
    bridge = cfg.get('incident_data', {}).get('collision_coordinates', None)
    if bridge:
        bx, by = transformer.transform(bridge['lon'], bridge['lat'])
    else:
        bx = by = None

    # waypoints
    wps = cfg.get('route', {}).get('waypoints_lonlat', [])
    wp_xy = []
    for wp in wps:
        try:
            x, y = transformer.transform(wp[0][0], wp[0][1])
            gx, gy = transformer.transform(wp[-1][0], wp[-1][1])
            wp_xy.append(((x, y), (gx, gy)))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.get_cmap('tab10')
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            ax.plot(df['x_m'], df['y_m'], '-', color=colors(i % 10), label=f.name)
            ax.plot(df['x_m'].iloc[0], df['y_m'].iloc[0], 'o', color=colors(i % 10))
            ax.plot(df['x_m'].iloc[-1], df['y_m'].iloc[-1], 's', color=colors(i % 10))
        except Exception as e:
            print('Failed to plot', f, e)

    # plot waypoints
    for (s, g) in wp_xy:
        ax.plot([s[0]], [s[1]], marker='*', color='green', markersize=10)
        ax.plot([g[0]], [g[1]], marker='D', color='orange', markersize=8)
        ax.plot([s[0], g[0]], [s[1], g[1]], '--', color='gray')

    if bx is not None and by is not None:
        ax.plot([bx], [by], 'x', color='red', markersize=12, label='Bridge incident')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Dali scenario trajectories')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()

    if out_png is None:
        out_png = out / 'dali_trajectories.png'
    else:
        out_png = Path(out_png)
    fig.savefig(out_png, dpi=200)
    print('Wrote', out_png)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', default='outputs/dali_scenario')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    plot_dali(args.output_dir, args.config, args.out)
