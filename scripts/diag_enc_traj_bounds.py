from pathlib import Path
import json
import codecs
import pandas as pd
from pyproj import Transformer

OUT = Path('outputs/dali_scenario')
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


if __name__ == '__main__':
    import sys
    cfg = None
    try:
        cfg = read_json_bom(CFG)
    except Exception as e:
        print('Failed to read config:', e)
        sys.exit(1)

    port = cfg['simulation']['port_name']
    from emergent.ship_abm.simulation_core import simulation
    sim = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=True)
    enc = getattr(sim, 'enc_data', {})

    # trajectory bounds
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
        print('No trajectory data found in', OUT)
        sys.exit(1)
    txmin, txmax = min(xs), max(xs)
    tymin, tymax = min(ys), max(ys)
    print('Trajectory bbox (UTM):', txmin, tymin, txmax, tymax)

    # ENC layer bboxes
    def gdf_bbox(gdf):
        try:
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            return tuple(bounds.tolist())
        except Exception:
            return None

    for layer in ('LNDARE', 'COALNE', 'BRIDGE'):
        objs = enc.get(layer)
        if objs is None:
            print(layer, 'not present')
            continue
        if isinstance(objs, list):
            for i, gdf in enumerate(objs):
                bb = gdf_bbox(gdf)
                print(f'{layer}[{i}] bbox:', bb)
        else:
            bb = gdf_bbox(objs)
            print(f'{layer} bbox:', bb)

    # quick intersection check
    def intersects(bb1, bb2):
        if bb1 is None or bb2 is None:
            return False
        a_minx, a_miny, a_maxx, a_maxy = bb1
        b_minx, b_miny, b_maxx, b_maxy = bb2
        return not (a_maxx < b_minx or b_maxx < a_minx or a_maxy < b_miny or b_maxy < a_miny)

    traj_bb = (txmin, tymin, txmax, tymax)
    for layer in ('LNDARE', 'COALNE', 'BRIDGE'):
        objs = enc.get(layer)
        if objs is None:
            continue
        if isinstance(objs, list):
            for i, gdf in enumerate(objs):
                bb = gdf_bbox(gdf)
                print(layer, i, 'intersects traj:', intersects(traj_bb, bb))
        else:
            bb = gdf_bbox(objs)
            print(layer, 'intersects traj:', intersects(traj_bb, bb))
