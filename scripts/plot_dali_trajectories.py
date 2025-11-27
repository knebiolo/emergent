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

    # We'll add the pale-blue water background after any ENC-driven extent expansion.

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
        # fallback: ensure function always returns a list
        return []

    if cfg is not None and crs is not None:
        try:
            from emergent.ship_abm.simulation_core import simulation
            sim_enc = simulation(port_name=cfg['simulation']['port_name'], dt=1.0, T=1.0, n_agents=0, load_enc=True)
            enc = getattr(sim_enc, 'enc_data', {})
            # If a local FSK geojson exists, load and patch into enc overlay so it is visible
            try:
                from pathlib import Path as _Path
                import geopandas as _gpd
                fsk_path = _Path('data/ship_abm/fsk_bridge.geojson')
                if fsk_path.exists():
                    try:
                        fsk_gdf = _gpd.read_file(str(fsk_path))
                        # ensure CRS matches sim
                        if getattr(fsk_gdf, 'crs', None) is None:
                            fsk_gdf.set_crs(sim_enc.crs_utm, inplace=True)
                        elif fsk_gdf.crs != sim_enc.crs_utm:
                            fsk_gdf = fsk_gdf.to_crs(sim_enc.crs_utm)
                        # add to enc under 'FSK_BRIDGE' for plotting
                        enc.setdefault('FSK_BRIDGE', []).append(fsk_gdf)
                        print('Loaded local FSK geojson and added to enc overlay')
                    except Exception as _e:
                        print('Failed to load local FSK geojson:', _e)
            except Exception:
                pass
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
                        # Compute union bbox of trajectories and ENC
                        union_minx = min(xmin, enc_bb[0])
                        union_miny = min(ymin, enc_bb[1])
                        union_maxx = max(xmax, enc_bb[2])
                        union_maxy = max(ymax, enc_bb[3])
                        traj_w = (xmax - xmin) if (xmax - xmin) > 0 else 1.0
                        traj_h = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                        union_w = union_maxx - union_minx
                        union_h = union_maxy - union_miny
                        # If the ENC covers a much larger area than the trajectories, avoid zooming out to whole ENC.
                        # Instead, expand the trajectory bbox by a reasonable buffer so ENC nearby is visible but not the entire ENC world.
                        if union_w > traj_w * 2.5 or union_h > traj_h * 2.5:
                            padx = max(traj_w * 0.25, 1000.0)
                            pady = max(traj_h * 0.25, 1000.0)
                            ax.set_xlim(xmin - padx, xmax + padx)
                            ax.set_ylim(ymin - pady, ymax + pady)
                        else:
                            xmin, ymin, xmax, ymax = union_minx, union_miny, union_maxx, union_maxy
                            padx = max((xmax - xmin) * 0.12, 500.0)
                            pady = max((ymax - ymin) * 0.12, 500.0)
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
                        for gi, gdf in enumerate(lnds):
                            for j, geom in enumerate(getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []):
                                try:
                                    seqs = iter_geom_xy(geom)
                                    if not seqs:
                                        continue
                                    for xs, ys in seqs:
                                        try:
                                            ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', alpha=0.95, zorder=3)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    print(f'Failed to draw LNDARE geom {gi}:{j}:', e)
                    else:
                        for geom in getattr(lnds, 'geometry', []) if getattr(lnds, 'geometry', None) is not None else []:
                            if geom is None:
                                continue
                            for xs, ys in iter_geom_xy(geom):
                                try:
                                    ax.fill(xs, ys, facecolor='#d9d0b8', edgecolor='#bba977', zorder=3)
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
                                            ax.plot(xs, ys, color='#444444', linewidth=1.4, zorder=4)
                                        except Exception:
                                            pass
                        else:
                            for j, geom in enumerate(getattr(coal, 'geometry', []) if getattr(coal, 'geometry', None) is not None else []):
                                try:
                                    if geom is None:
                                        continue
                                    seqs = iter_geom_xy(geom)
                                    if not seqs:
                                        continue
                                    for xs, ys in seqs:
                                        try:
                                            ax.plot(xs, ys, color='#666666', linewidth=0.8, zorder=4)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    print(f'Failed to draw COALNE geom {j}:', e)
                    except Exception:
                        pass
                enc_drawn = True
                # Draw local FSK bridge polygon and abutments if present
                fsk_layers = enc.get('FSK_BRIDGE')
                if fsk_layers:
                    try:
                        for gdf in fsk_layers:
                            for geom in getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []:
                                seqs = iter_geom_xy(geom)
                                if not seqs:
                                    continue
                                for xs, ys in seqs:
                                    try:
                                        ax.fill(xs, ys, facecolor='#ffefc6', edgecolor='#cc9900', linewidth=1.0, zorder=5)
                                        # mark abutments: take polygon first and third vertex as approximations
                                        try:
                                            vx = list(xs)
                                            vy = list(ys)
                                            if len(vx) >= 4:
                                                ab1 = (vx[0], vy[0])
                                                ab2 = (vx[2], vy[2])
                                            else:
                                                ab1 = (vx[0], vy[0])
                                                ab2 = (vx[-1], vy[-1])
                                            ax.plot([ab1[0], ab2[0]], [ab1[1], ab2[1]], 'x', color='magenta', markersize=10, zorder=6)
                                            print('FSK abutments (UTM):', ab1, ab2)
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                    except Exception:
                        pass
        except Exception as e:
            print('ENC load/draw failed:', e)

    # Now that extents may have changed due to ENC, draw the pale-blue water background
    water_rect = plt.Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2*padx, (ymax - ymin) + 2*pady,
                               facecolor='#e6f3ff', edgecolor='none', alpha=0.9, zorder=0)
    ax.add_patch(water_rect)

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

    # load authoritative abutments if present
    try:
        import geopandas as gpd
        auth_path = Path('data/ship_abm/fsk_abutments_authoritative.geojson')
        if auth_path.exists():
            gdf = gpd.read_file(str(auth_path))
            if getattr(gdf, 'crs', None) is None:
                # assume already in sim UTM or EPSG:32618
                try:
                    gdf.set_crs(crs, inplace=True)
                except Exception:
                    gdf.set_crs('EPSG:32618', inplace=True)
            elif crs is not None and gdf.crs != crs:
                try:
                    gdf = gdf.to_crs(crs)
                except Exception:
                    pass
            for i, geom in enumerate(getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []):
                try:
                    if geom is None:
                        continue
                    x, y = geom.x, geom.y
                    ax.scatter([x], [y], marker='*', c='blue', s=140, zorder=9, label='Authoritative Abutment' if i == 0 else None)
                    # annotate with name if present
                    name = gdf.iloc[i].get('name', None) if len(gdf) > i else None
                    if name:
                        ax.text(x, y, name, color='blue', fontsize=8, zorder=10)
                except Exception:
                    pass
    except Exception:
        pass

    # load refined abutments (local PCA/snapped) and plot as magenta X
    try:
        ref_path = Path('data/ship_abm/fsk_abutments_refined.geojson')
        if ref_path.exists():
            ref_gdf = gpd.read_file(str(ref_path))
            if getattr(ref_gdf, 'crs', None) is None:
                try:
                    ref_gdf.set_crs(crs, inplace=True)
                except Exception:
                    ref_gdf.set_crs('EPSG:32618', inplace=True)
            elif crs is not None and ref_gdf.crs != crs:
                try:
                    ref_gdf = ref_gdf.to_crs(crs)
                except Exception:
                    pass
            for i, geom in enumerate(getattr(ref_gdf, 'geometry', []) if getattr(ref_gdf, 'geometry', None) is not None else []):
                try:
                    if geom is None:
                        continue
                    x, y = geom.x, geom.y
                    ax.plot([x], [y], marker='x', color='magenta', markersize=10, mew=2, zorder=9, label='Refined Abutment' if i == 0 else None)
                    name = ref_gdf.iloc[i].get('name', None) if len(ref_gdf) > i else None
                    if name:
                        ax.text(x, y, name, color='magenta', fontsize=8, zorder=10)
                except Exception:
                    pass
    except Exception:
        pass

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
