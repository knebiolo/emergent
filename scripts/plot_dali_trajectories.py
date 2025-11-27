import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer
import numpy as np

OUT = Path('outputs/dali_scenario')
OUT.mkdir(parents=True, exist_ok=True)

def load_trajectories(outdir=OUT):
    files = sorted(outdir.glob('run_*_trajectory.csv'))
    trajs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            trajs.append((f.name, df))
        except Exception:
            pass
    return trajs

def plot(trajs, config_path='scenarios/dali_bridge_collision_config.json', out_png=None):
    import json
    from pathlib import Path as _P
    import codecs as _codecs
    cfg = None
    try:
        raw = _P(config_path).read_bytes()
        if raw.startswith(b"\xff\xfe") or raw.startswith(_codecs.BOM_UTF16_LE):
            text = raw.decode('utf-16')
        elif raw.startswith(b"\xfe\xff") or raw.startswith(_codecs.BOM_UTF16_BE):
            text = raw.decode('utf-16')
        elif raw.startswith(_codecs.BOM_UTF8) or raw.startswith(b"\xef\xbb\xbf"):
            text = raw.decode('utf-8-sig')
        else:
            try:
                text = raw.decode('utf-8')
            except Exception:
                text = raw.decode('latin-1')
        cfg = json.loads(text)
    except Exception as e:
        print('Could not read config at', config_path, ':', e)
        cfg = None
    # extract waypoints and incident (lon, lat)
    if cfg is not None:
        try:
            wps = cfg['route']['waypoints_lonlat'][0]
            start_lon, start_lat = wps[0]
        except Exception:
            start_lon = start_lat = None
        try:
            bridge_lon = cfg['incident_data']['collision_coordinates']['lon']
            bridge_lat = cfg['incident_data']['collision_coordinates']['lat']
        except Exception:
            bridge_lon = bridge_lat = None
    else:
        start_lon = start_lat = bridge_lon = bridge_lat = None

    # If config missing, try to infer approximate start from first trajectory file (UTM coords -> leave None)

    # Use UTM projection of the simulation to get consistent meters coords
    # If we have config, compute UTM transformer from midpoint lon; otherwise plot raw UTM from files
    if start_lon is None or bridge_lon is None:
        # fallback: just compute bounds from trajectories and plot in meters
        fig, ax = plt.subplots(figsize=(8,10))
        ax.add_patch(plt.Rectangle((0,0), 1, 1, color='#e6f3ff', zorder=0))
        for name, df in trajs:
            try:
                xs = df['x_m'].values
                ys = df['y_m'].values
                ax.plot(xs, ys, alpha=0.8, linewidth=1)
            except Exception:
                pass
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('MV Dali Trajectories (UTM meters)')
        out_png = OUT / 'dali_trajectories.png'
        plt.savefig(out_png, dpi=200)
        print('Wrote', out_png)
        return

    midlon = (start_lon + bridge_lon) / 2.0
    utm_zone = int((midlon + 180) // 6) + 1
    utm_epsg = 32600 + utm_zone
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}', always_xy=True)
    sx, sy = transformer.transform(start_lon, start_lat)
    bx, by = transformer.transform(bridge_lon, bridge_lat)

    fig, ax = plt.subplots(figsize=(8,10))

    # compute bounds from trajs
    all_x = np.hstack([df['x_m'].values for _, df in trajs if 'x_m' in df.columns]) if trajs else np.array([sx, bx])
    all_y = np.hstack([df['y_m'].values for _, df in trajs if 'y_m' in df.columns]) if trajs else np.array([sy, by])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    padx = max((xmax - xmin) * 0.05, 100.0)
    pady = max((ymax - ymin) * 0.05, 100.0)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)

    # Pale-blue water background
    ax.add_patch(plt.Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2*padx, (ymax - ymin) + 2*pady, color='#e6f3ff', zorder=0))

    # Try to instantiate sim and load ENC (best-effort)
    sim_enc = None
    try:
        from emergent.ship_abm.simulation_core import simulation
        sim_enc = simulation(port_name=cfg['simulation']['port_name'], dt=1.0, T=1.0, n_agents=0, load_enc=True)
    except Exception:
        sim_enc = None

    # Draw ENC land polygons if present
    try:
        enc = sim_enc.enc_data if sim_enc is not None else None
        if enc and 'LNDARE' in enc:
            lnds = enc.get('LNDARE')
            if isinstance(lnds, list):
                for gdf in lnds:
                    for geom in getattr(gdf, 'geometry', []):
                        try:
                            if hasattr(geom, 'exterior'):
                                xs, ys = geom.exterior.xy
                            else:
                                xs, ys = geom.xy
                            ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=1)
                        except Exception:
                            pass
            else:
                for geom in getattr(lnds, 'geometry', []):
                    try:
                        if hasattr(geom, 'exterior'):
                            xs, ys = geom.exterior.xy
                        else:
                            xs, ys = geom.xy
                        ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=1)
                    except Exception:
                        pass
    except Exception:
        pass

    # Trajectories
    colors = plt.cm.get_cmap('tab10')
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            ax.plot(df['x_m'], df['y_m'], '-', alpha=0.9, linewidth=1.2, color=colors(i % 10), zorder=5)
            ax.plot(df['x_m'].iloc[0], df['y_m'].iloc[0], 'o', color=colors(i % 10), zorder=6)
            ax.plot(df['x_m'].iloc[-1], df['y_m'].iloc[-1], 's', color=colors(i % 10), zorder=6)
        except Exception as e:
            print('Failed to plot', f, e)

    # plot start and bridge
    ax.scatter([sx], [sy], c='green', s=100, label='Start', zorder=7)
    # Bridge: try to find separate bridge shapefile in outputs or use config point
    bridge_drawn = False
    try:
        # look for bridge files in OUT
        bfiles = list(_P('outputs/dali_scenario').glob('bridge.*')) + list(_P('outputs/dali_scenario').glob('bridge_*.*'))
        if bfiles:
            import geopandas as gpd
            for bf in bfiles:
                try:
                    g = gpd.read_file(bf)
                    g = g.to_crs(f'EPSG:{utm_epsg}')
                    g.plot(ax=ax, facecolor='#bba977', edgecolor='#8c6d3f', zorder=2)
                    bridge_drawn = True
                except Exception:
                    pass
    except Exception:
        pass
    if not bridge_drawn:
        ax.scatter([bx], [by], c='red', s=80, marker='x', label='Bridge (incident)', zorder=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_title('MV Dali Trajectories (case runs)')
    ax.legend()
    fig.tight_layout()
    if out_png is None:
        out_png = OUT / 'dali_trajectories.png'
    plt.savefig(out_png, dpi=200)
    print('Wrote', out_png)

if __name__ == '__main__':
    trajs = load_trajectories()
    if not trajs:
        print('No trajectory files found in', OUT)
    else:
        plot(trajs)
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
    from pathlib import Path as _P
    import codecs as _codecs
    try:
        raw = _P(config_path).read_bytes()
        if raw.startswith(b"\xff\xfe") or raw.startswith(_codecs.BOM_UTF16_LE):
            text = raw.decode('utf-16')
        elif raw.startswith(b"\xfe\xff") or raw.startswith(_codecs.BOM_UTF16_BE):
            text = raw.decode('utf-16')
        elif raw.startswith(_codecs.BOM_UTF8) or raw.startswith(b"\xef\xbb\xbf"):
            text = raw.decode('utf-8-sig')
        else:
            try:
                text = raw.decode('utf-8')
            except Exception:
                text = raw.decode('latin-1')
        cfg = json.loads(text)
    except Exception as e:
        print('Could not read config at', config_path, ':', e)
        cfg = None

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

    # Pale-blue water background
    try:
        xmin = min([min(pd.read_csv(f)['x_m']) for f in files])
        xmax = max([max(pd.read_csv(f)['x_m']) for f in files])
        ymin = min([min(pd.read_csv(f)['y_m']) for f in files])
        ymax = max([max(pd.read_csv(f)['y_m']) for f in files])
    except Exception:
        xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy if 'sim' in locals() else (0,0,1,1)

    padx = (xmax - xmin) * 0.05 if xmax > xmin else 1000
    pady = (ymax - ymin) * 0.05 if ymax > ymin else 1000
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.add_patch(plt.Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2*padx, (ymax - ymin) + 2*pady, color='#e6f3ff', zorder=0))

    # Draw ENC land polygons (LNDARE) if available
    try:
        from emergent.ship_abm.simulation_core import simulation
        # instantiate sim to get enc_data if possible (non-destructive)
        sim = simulation(port_name=cfg['simulation']['port_name'], dt=1.0, T=1.0, n_agents=0, load_enc=False)
        enc = None
        try:
            # reuse sim.load_enc_features to populate enc_data if ENC files are available
            sim.load_enc_features(None, verbose=False)
            enc = sim.enc_data
        except Exception:
            enc = getattr(sim, 'enc_data', None)
        if enc and 'LNDARE' in enc:
            # enc['LNDARE'] may be a list of GeoDataFrames
            lnds = enc.get('LNDARE')
            if isinstance(lnds, list):
                for gdf in lnds:
                    try:
                        for geom in gdf.geometry:
                            xs, ys = geom.exterior.xy if hasattr(geom, 'exterior') else (geom.xy[0], geom.xy[1])
                            ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=1)
                    except Exception:
                        pass
            else:
                try:
                    for geom in lnds.geometry:
                        xs, ys = geom.exterior.xy if hasattr(geom, 'exterior') else (geom.xy[0], geom.xy[1])
                        ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=1)
                except Exception:
                    pass
    except Exception:
        # if ENC cannot be loaded, skip basemap overlay
        pass

    colors = plt.cm.get_cmap('tab10')
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            ax.plot(df['x_m'], df['y_m'], '-', color=colors(i % 10), label=f.name, zorder=5)
            ax.plot(df['x_m'].iloc[0], df['y_m'].iloc[0], 'o', color=colors(i % 10), zorder=6)
            ax.plot(df['x_m'].iloc[-1], df['y_m'].iloc[-1], 's', color=colors(i % 10), zorder=6)
        except Exception as e:
            print('Failed to plot', f, e)

    # plot waypoints
    for (s, g) in wp_xy:
        ax.plot([s[0]], [s[1]], marker='*', color='green', markersize=10, zorder=7)
        ax.plot([g[0]], [g[1]], marker='D', color='orange', markersize=8, zorder=7)
        ax.plot([s[0], g[0]], [s[1], g[1]], '--', color='gray', zorder=6)

    if bx is not None and by is not None:
        ax.plot([bx], [by], 'x', color='red', markersize=12, label='Bridge incident', zorder=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Dali scenario trajectories')
    ax.grid(True, zorder=9)
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
