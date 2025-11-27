from pathlib import Path
import json
import codecs

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


if __name__ == '__main__':
    import sys
    try:
        cfg = read_json_bom(CFG)
    except Exception as e:
        print('Failed to read config:', e)
        sys.exit(1)

    port = cfg['simulation']['port_name']
    from emergent.ship_abm.simulation_core import simulation
    sim = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=True)
    enc = getattr(sim, 'enc_data', {})

    # collect layer bboxes
    preview = {'trajectory_bbox': None, 'enc_bboxes': {}, 'sample_geoms': {}}
    # trajectory bbox
    files = sorted(OUT.glob('run_*_trajectory.csv'))
    xs = []
    ys = []
    for f in files:
        try:
            import pandas as pd
            df = pd.read_csv(f)
            xs.extend(df['x_m'].tolist())
            ys.extend(df['y_m'].tolist())
        except Exception:
            pass
    if xs:
        preview['trajectory_bbox'] = [min(xs), min(ys), max(xs), max(ys)]

    # collect bboxes and simplified sample coordinates (first geometry per layer)
    for layer in ('LNDARE', 'COALNE', 'BRIDGE'):
        objs = enc.get(layer)
        if objs is None:
            preview['enc_bboxes'][layer] = None
            preview['sample_geoms'][layer] = None
            continue
        # support list-of-gdfs or single gdf
        try:
            if isinstance(objs, list):
                bboxes = []
                samples = []
                for gdf in objs:
                    if getattr(gdf, 'empty', True):
                        continue
                    bboxes.append(list(gdf.total_bounds))
                    geom = next((g for g in getattr(gdf, 'geometry', []) if g is not None), None)
                    if geom is not None:
                        # extract exterior coords for Polygons, or coords for lines
                        try:
                            coords = []
                            if hasattr(geom, 'exterior'):
                                coords = list(map(list, zip(*geom.exterior.xy)))
                            elif hasattr(geom, 'xy'):
                                coords = list(map(list, zip(*geom.xy)))
                            samples.append(coords[:100])
                        except Exception:
                            samples.append(None)
                preview['enc_bboxes'][layer] = bboxes
                preview['sample_geoms'][layer] = samples
            else:
                gdf = objs
                if getattr(gdf, 'empty', True):
                    preview['enc_bboxes'][layer] = None
                    preview['sample_geoms'][layer] = None
                else:
                    preview['enc_bboxes'][layer] = list(gdf.total_bounds)
                    geom = next((g for g in getattr(gdf, 'geometry', []) if g is not None), None)
                    if geom is None:
                        preview['sample_geoms'][layer] = None
                    else:
                        try:
                            if hasattr(geom, 'exterior'):
                                coords = list(map(list, zip(*geom.exterior.xy)))
                            elif hasattr(geom, 'xy'):
                                coords = list(map(list, zip(*geom.xy)))
                            else:
                                coords = None
                            preview['sample_geoms'][layer] = coords[:100] if coords is not None else None
                        except Exception:
                            preview['sample_geoms'][layer] = None
        except Exception as e:
            preview['enc_bboxes'][layer] = f'error: {e}'
            preview['sample_geoms'][layer] = None

    outp = OUT / 'enc_preview.json'
    outp.write_text(json.dumps(preview, indent=2))
    print('Wrote', outp)
