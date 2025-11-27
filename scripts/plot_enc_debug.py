from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import codecs
import pandas as pd

OUT = Path('outputs/dali_scenario')
OUT.mkdir(parents=True, exist_ok=True)
CFG = Path('scenarios/dali_bridge_collision_config.json')


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


def iter_geom_xy(geom):
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
    # default: ensure we always return a list (never None)
    return []


if __name__ == '__main__':
    cfg = read_json_bom(CFG)
    port = cfg['simulation']['port_name']
    from emergent.ship_abm.simulation_core import simulation
    sim_enc = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=True)
    enc = getattr(sim_enc, 'enc_data', {})

    # Trajectory bbox
    files = sorted(OUT.glob('run_*_trajectory.csv'))
    xs = []
    ys = []
    for f in files:
        try:
            df = pd.read_csv(f)
            xs.extend(df['x_m'].tolist())
            ys.extend(df['y_m'].tolist())
        except Exception:
            pass
    if not xs:
        print('No trajectory data; run the scenario first')
        raise SystemExit(1)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine ENC combined bbox so we can zoom out to include land
    enc_bounds = None
    try:
        for layer in ('LNDARE', 'COALNE', 'BRIDGE'):
            objs = enc.get(layer)
            if objs is None:
                continue
            if isinstance(objs, list):
                for gdf in objs:
                    if getattr(gdf, 'empty', True):
                        continue
                    b = gdf.total_bounds
                    if enc_bounds is None:
                        enc_bounds = list(b)
                    else:
                        enc_bounds[0] = min(enc_bounds[0], b[0])
                        enc_bounds[1] = min(enc_bounds[1], b[1])
                        enc_bounds[2] = max(enc_bounds[2], b[2])
                        enc_bounds[3] = max(enc_bounds[3], b[3])
            else:
                gdf = objs
                if getattr(gdf, 'empty', True):
                    continue
                b = gdf.total_bounds
                if enc_bounds is None:
                    enc_bounds = list(b)
                else:
                    enc_bounds[0] = min(enc_bounds[0], b[0])
                    enc_bounds[1] = min(enc_bounds[1], b[1])
                    enc_bounds[2] = max(enc_bounds[2], b[2])
                    enc_bounds[3] = max(enc_bounds[3], b[3])
    except Exception:
        enc_bounds = None

    # compute final plot extent: include both trajectories and ENC if available
    plot_minx, plot_miny, plot_maxx, plot_maxy = xmin, ymin, xmax, ymax
    if enc_bounds is not None:
        plot_minx = min(plot_minx, enc_bounds[0])
        plot_miny = min(plot_miny, enc_bounds[1])
        plot_maxx = max(plot_maxx, enc_bounds[2])
        plot_maxy = max(plot_maxy, enc_bounds[3])

    # add padding (wider pad to zoom out more for context)
    padx = max((plot_maxx - plot_minx) * 0.25, 2000)
    pady = max((plot_maxy - plot_miny) * 0.25, 2000)
    ax.set_xlim(plot_minx - padx, plot_maxx + padx)
    ax.set_ylim(plot_miny - pady, plot_maxy + pady)

    # Pale-blue water background behind everything
    water = plt.Rectangle((plot_minx - padx, plot_miny - pady), (plot_maxx - plot_minx) + 2*padx, (plot_maxy - plot_miny) + 2*pady,
                          facecolor='#e6f3ff', edgecolor='none', zorder=0)
    ax.add_patch(water)

    # Draw ENC layers with strong styles
    # LNDARE fill (land)
    lnds = enc.get('LNDARE')
    if lnds is not None:
        if isinstance(lnds, list):
            for gdf in lnds:
                for geom in getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []:
                    for xs_, ys_ in iter_geom_xy(geom) or []:
                        try:
                            ax.fill(xs_, ys_, facecolor='#d9d0b8', edgecolor='#8f7a4f', linewidth=0.6, zorder=3)
                        except Exception:
                            pass
        else:
                for geom in getattr(lnds, 'geometry', []) if getattr(lnds, 'geometry', None) is not None else []:
                    for xs_, ys_ in iter_geom_xy(geom) or []:
                        try:
                            ax.fill(xs_, ys_, facecolor='#d9d0b8', edgecolor='#8f7a4f', linewidth=0.6, zorder=3)
                        except Exception:
                            pass

    # COALNE - coastline lines
    coal = enc.get('COALNE')
    if coal is not None:
        if isinstance(coal, list):
            for gdf in coal:
                for geom in getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []:
                    for xs_, ys_ in iter_geom_xy(geom) or []:
                        try:
                            ax.plot(xs_, ys_, color='#333333', linewidth=1.6, zorder=4)
                        except Exception:
                            pass
        else:
            for geom in getattr(coal, 'geometry', []) if getattr(coal, 'geometry', None) is not None else []:
                    for xs_, ys_ in iter_geom_xy(geom) or []:
                        try:
                            ax.plot(xs_, ys_, color='#333333', linewidth=1.6, zorder=4)
                        except Exception:
                            pass

    # BRIDGE - highlight in red
    br = enc.get('BRIDGE')
    if br is not None:
        if isinstance(br, list):
            for gdf in br:
                for geom in getattr(gdf, 'geometry', []) if getattr(gdf, 'geometry', None) is not None else []:
                    for xs_, ys_ in iter_geom_xy(geom) or []:
                        try:
                            ax.plot(xs_, ys_, color='red', linewidth=2.0, zorder=6)
                        except Exception:
                            pass
        else:
            for geom in getattr(br, 'geometry', []) if getattr(br, 'geometry', None) is not None else []:
                for xs_, ys_ in iter_geom_xy(geom) or []:
                    try:
                        ax.plot(xs_, ys_, color='red', linewidth=2.0, zorder=6)
                    except Exception:
                        pass

    # Plot trajectory bbox for reference
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='blue', linewidth=1.2, zorder=4)
    ax.add_patch(rect)

    # Plot trajectories (thin, on top)
    colors = plt.cm.get_cmap('tab10')
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            ax.plot(df['x_m'], df['y_m'], '-', color=colors(i % 10), linewidth=1.0, zorder=5)
        except Exception:
            pass

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin - 100, xmax + 100)
    ax.set_ylim(ymin - 100, ymax + 100)
    ax.set_title('ENC debug: LNDARE/COALNE/BRIDGE and trajectory bbox')
    outp = OUT / 'dali_enc_debug.png'
    fig.savefig(outp, dpi=200, bbox_inches='tight')
    print('Wrote', outp)
    # Also save a wider overview image for extra zoomed-out context
    try:
        outp2 = OUT / 'dali_enc_debug_wide.png'
        fig.savefig(outp2, dpi=150, bbox_inches='tight')
        print('Wrote', outp2)
    except Exception:
        pass
