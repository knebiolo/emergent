import json
from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Transformer

OUT = Path('outputs/dali_scenario')
CFG = Path('scenarios/dali_bridge_collision_config.json')

def main():
    # robust JSON load: try to load, otherwise fall back to known coords
    try:
        txt = open(CFG, 'rb').read()
        try:
            txt = txt.decode('utf-8-sig')
        except Exception:
            try:
                txt = txt.decode('utf-8')
            except Exception:
                txt = txt.decode('latin-1')
        cfg = json.loads(txt.strip())
    except Exception:
        cfg = None

    # Known fallback for the MV Dali scenario (if config unreadable)
    if cfg is None:
        print('Warning: could not parse config; using fallback coords for MV Dali')
        start_lon, start_lat = -76.534072, 39.219800
        bridge_lon, bridge_lat = -76.5297, 39.2156
    else:
        wps = cfg.get('route', {}).get('waypoints_lonlat', [])
        if not wps:
            print('No waypoints found in config')
            return
        start_lon, start_lat = wps[0][0]
        bridge = cfg.get('incident_data', {}).get('collision_coordinates', {})
        bridge_lon = bridge.get('lon')
        bridge_lat = bridge.get('lat')

    # choose UTM zone from midpoint lon
    midlon = (start_lon + (bridge_lon if bridge_lon is not None else start_lon)) / 2.0
    utm_zone = int((midlon + 180) // 6) + 1
    utm_epsg = 32600 + utm_zone
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}', always_xy=True)
    sx, sy = tr.transform(start_lon, start_lat)
    bx = by = None
    if bridge_lon is not None and bridge_lat is not None:
        bx, by = tr.transform(bridge_lon, bridge_lat)

    print(f"Config start (lon,lat)=({start_lon},{start_lat}) -> UTM=({sx:.3f},{sy:.3f}) EPSG:{utm_epsg}")
    if bx is not None:
        print(f"Bridge (lon,lat)=({bridge_lon},{bridge_lat}) -> UTM=({bx:.3f},{by:.3f})")

    files = sorted(OUT.glob('run_*_trajectory.csv'))
    if not files:
        print('No trajectory files found in', OUT)
        return

    print('\nRun starts:')
    for f in files:
        try:
            df = pd.read_csv(f)
            row = df.iloc[0]
            x0 = float(row['x_m'])
            y0 = float(row['y_m'])
            d_start = np.hypot(x0 - sx, y0 - sy)
            d_bridge = np.hypot(x0 - bx, y0 - by) if bx is not None else None
            print(f"{f.name}: start_utm=({x0:.3f},{y0:.3f}), dist_to_config_start={d_start:.1f} m", end='')
            if d_bridge is not None:
                print(f", dist_to_bridge={d_bridge:.1f} m")
            else:
                print()
        except Exception as e:
            print(f"{f.name}: ERROR reading file: {e}")

if __name__ == '__main__':
    main()
