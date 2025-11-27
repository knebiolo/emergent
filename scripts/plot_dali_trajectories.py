from pathlib import Path
import json
import codecs
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer

OUT = Path('outputs/dali_scenario')
OUT.mkdir(parents=True, exist_ok=True)


def read_json_bom(path: Path):
    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(b"\xff\xfe"):
        return json.loads(raw.decode('utf-16'))
    if raw.startswith(codecs.BOM_UTF16_BE) or raw.startswith(b"\xfe\xff"):
        return json.loads(raw.decode('utf-16'))
    if raw.startswith(codecs.BOM_UTF8) or raw.startswith(b"\xef\xbb\xbf"):
        return json.loads(raw.decode('utf-8-sig'))
    try:
        return json.loads(raw.decode('utf-8'))
    except Exception:
        return json.loads(raw.decode('latin-1'))


def load_trajectories(output_dir: Path):
    files = sorted(output_dir.glob('run_*_trajectory.csv'))
    trajs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            trajs.append((f.name, df))
        except Exception:
            pass
    return trajs


def plot_dali(output_dir: str, config_path: str, out_png: str = None):
    out = Path(output_dir)
    trajs = load_trajectories(out)
    if not trajs:
        raise SystemExit(f'No trajectory CSVs found in {out}')

    cfg = None
    try:
        cfg = read_json_bom(Path(config_path))
    except Exception as e:
        print('Warning: could not read config; proceeding with UTM-only plot:', e)

    # determine UTM/crs
    crs = None
    transformer = None
    sx = sy = bx = by = None
    if cfg is not None:
        try:
            port = cfg['simulation']['port_name']
            from emergent.ship_abm.simulation_core import simulation
            sim = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=False)
            crs = sim.crs_utm
        except Exception:
            # fallback from lon
            try:
                lon = float(cfg['route']['waypoints_lonlat'][0][0][0])
                utm_zone = int((lon + 180) // 6) + 1
                crs = f'EPSG:{32600 + utm_zone}'
            except Exception:
                crs = None
        if crs is not None:
            transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
            try:
                start = cfg['route']['waypoints_lonlat'][0][0]
                sx, sy = transformer.transform(float(start[0]), float(start[1]))
            except Exception:
                sx = sy = None
            try:
                bridge = cfg['incident_data']['collision_coordinates']
                bx, by = transformer.transform(float(bridge['lon']), float(bridge['lat']))
            except Exception:
                bx = by = None

    # compute bounds from trajectories
    all_x = np.hstack([df['x_m'].values for _, df in trajs if 'x_m' in df.columns])
    all_y = np.hstack([df['y_m'].values for _, df in trajs if 'y_m' in df.columns])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    padx = max((xmax - xmin) * 0.05, 100.0)
    pady = max((ymax - ymin) * 0.05, 100.0)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)

    # pale-blue water background (draw first with slight transparency)
    water_rect = plt.Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2*padx, (ymax - ymin) + 2*pady,
                               facecolor='#e6f3ff', edgecolor='none', alpha=0.9, zorder=0)
    ax.add_patch(water_rect)

    # best-effort ENC overlay (robust handling)
    enc_drawn = False
    def iter_geom_xy(geom):
        # return list of (xs, ys) sequences for Polygon/MultiPolygon/LineString
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
            if geom is None:
                return []
            if geom.geom_type == 'Polygon':
                try:
                    xs, ys = geom.exterior.xy
                    return [(xs, ys)]
                except Exception:
                    return []
            if geom.geom_type == 'MultiPolygon':
                seqs = []
                for p in geom.geoms:
                    try:
                        xs, ys = p.exterior.xy
                        seqs.append((xs, ys))
                    except Exception:
                        pass
                return seqs
            if geom.geom_type in ('LineString', 'LinearRing'):
                try:
                    xs, ys = geom.xy
                    return [(xs, ys)]
                except Exception:
                    return []
            if geom.geom_type == 'MultiLineString':
                seqs = []
                for l in geom.geoms:
                    try:
                        xs, ys = l.xy
                        seqs.append((xs, ys))
                    except Exception:
                        pass
                return seqs
        except Exception:
            return []

    if cfg is not None and crs is not None:
        try:
            from emergent.ship_abm.simulation_core import simulation
            sim_enc = simulation(port_name=cfg['simulation']['port_name'], dt=1.0, T=1.0, n_agents=0, load_enc=True)
            enc = getattr(sim_enc, 'enc_data', {})
            if enc:
                # compute combined ENC bounds for diagnostics and potential extent expansion
                try:
                    enc_bounds = None
                    for layer_gdf in enc.values():
                        if layer_gdf is None:
                            continue
                        if isinstance(layer_gdf, list):
                            for gdf in layer_gdf:
                                if getattr(gdf, 'empty', True):
                                    continue
                                b = gdf.total_bounds
                                if enc_bounds is None:
                                    enc_bounds = b.copy()
                                else:
                                    enc_bounds[0] = min(enc_bounds[0], b[0])
                                    enc_bounds[1] = min(enc_bounds[1], b[1])
                                    enc_bounds[2] = max(enc_bounds[2], b[2])
                                    enc_bounds[3] = max(enc_bounds[3], b[3])
                        else:
                            gdf = layer_gdf
                            if getattr(gdf, 'empty', True):
                                continue
                            b = gdf.total_bounds
                            if enc_bounds is None:
                                enc_bounds = b.copy()
                            else:
                                enc_bounds[0] = min(enc_bounds[0], b[0])
                                enc_bounds[1] = min(enc_bounds[1], b[1])
                                enc_bounds[2] = max(enc_bounds[2], b[2])
                                enc_bounds[3] = max(enc_bounds[3], b[3])
                    if enc_bounds is not None:
                        print('ENC combined bbox (UTM):', tuple(enc_bounds))
                        traj_bb = (xmin, ymin, xmax, ymax)
                        enc_bb = tuple(enc_bounds)
                        # quick intersection
                        inter = not (enc_bb[2] < traj_bb[0] or traj_bb[2] < enc_bb[0] or enc_bb[3] < traj_bb[1] or traj_bb[3] < enc_bb[1])
                        print('ENC intersects trajectories:', inter)
                        # if ENC doesn't intersect but is nearby, optionally expand extent to include small overlap
                        if not inter:
                            # expand plot extent to include both (helps visualizing nearby ENC)
                            xmin = min(xmin, enc_bb[0])
                            ymin = min(ymin, enc_bb[1])
                            xmax = max(xmax, enc_bb[2])
                            ymax = max(ymax, enc_bb[3])
                            padx = max((xmax - xmin) * 0.05, 100.0)
                            pady = max((ymax - ymin) * 0.05, 100.0)
                            ax.set_xlim(xmin - padx, xmax + padx)
                            ax.set_ylim(ymin - pady, ymax + pady)
                except Exception as _:
                    pass
                # diagnostic: print available layers and counts
                try:
                    layers = {k: (len(v) if isinstance(v, list) else (0 if v is None else 1)) for k, v in enc.items()}
                    print('ENC layers present:', layers)
                except Exception:
                    pass
                # draw land
                lnds = enc.get('LNDARE')
                if lnds is not None:
                    if isinstance(lnds, list):
                        for gdf in lnds:
                            for geom in getattr(gdf, 'geometry', []):
                                for xs, ys in iter_geom_xy(geom):
                                    try:
                                        ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=3)
                                    except Exception:
                                        pass
                    else:
                        for geom in getattr(lnds, 'geometry', []):
                            for xs, ys in iter_geom_xy(geom):
                                try:
                                    ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=1)
                                except Exception:
                                    pass
                # draw shoreline
                coal = enc.get('COALNE')
                if coal is not None:
                    try:
                        if isinstance(coal, list):
                            for gdf in coal:
                                for geom in getattr(gdf, 'geometry', []):
                                    for xs, ys in iter_geom_xy(geom):
                                        try:
                                            ax.plot(xs, ys, color='#666666', linewidth=0.8, zorder=4)
                                        except Exception:
                                            pass
                        else:
                            for geom in getattr(coal, 'geometry', []):
                                for xs, ys in iter_geom_xy(geom):
                                    try:
                                        ax.plot(xs, ys, color='#888888', linewidth=0.6, zorder=2)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                enc_drawn = True
        except Exception as e:
            print('ENC load/draw failed:', e)

    # draw trajectories
    colors = plt.cm.get_cmap('tab10')
    for i, (name, df) in enumerate(trajs):
        try:
            ax.plot(df['x_m'], df['y_m'], '-', color=colors(i % 10), linewidth=1.2, zorder=5)
            ax.plot(df['x_m'].iloc[0], df['y_m'].iloc[0], 'o', color=colors(i % 10), zorder=6)
            ax.plot(df['x_m'].iloc[-1], df['y_m'].iloc[-1], 's', color=colors(i % 10), zorder=6)
        except Exception:
            pass

    # draw start and bridge
    if sx is not None and sy is not None:
        ax.scatter([sx], [sy], c='green', s=100, label='Start', zorder=7)
    if bx is not None and by is not None:
        ax.scatter([bx], [by], c='red', marker='x', s=80, label='Bridge', zorder=8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('MV Dali Trajectories')
    ax.legend(loc='best')
    fig.tight_layout()

    if out_png is None:
        out_png = out / 'dali_trajectories_with_enc.png' if enc_drawn else out / 'dali_trajectories.png'
    else:
        out_png = Path(out_png)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print('Wrote', out_png)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', default=str(OUT))
    p.add_argument('--out', default=None)
    args = p.parse_args()
    plot_dali(args.output_dir, args.config, args.out)
