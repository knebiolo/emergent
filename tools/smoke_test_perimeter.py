"""
Smoke test: instantiate simulation with HECRAS plan, verify perimeter attributes,
call update mapping and verify agent wet flags. Saves a small JSON report.
"""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from time import sleep

hecras_default = os.path.join(ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
if not os.path.exists(hecras_default):
    print('HECRAS plan not found:', hecras_default)
    sys.exit(2)

from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation

# create start polygon convex hull if not present
start_poly = os.path.join(ROOT, 'data', 'start_polygon.geojson')
if not os.path.exists(start_poly):
    try:
        import h5py
        import numpy as np
        from shapely.geometry import MultiPoint, mapping
        with h5py.File(hecras_default, 'r') as hdf:
            coords_all = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
        hull = MultiPoint([tuple(p) for p in coords_all]).convex_hull
        import json
        os.makedirs(os.path.dirname(start_poly), exist_ok=True)
        with open(start_poly, 'w') as fh:
            json.dump({'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(hull),'properties':{}}]}, fh)
    except Exception as e:
        print('Failed to create start polygon:', e)

# instantiate sim with minimal agents
sim = simulation(ROOT, 'test_model', 'EPSG:4326', 'Nushagak River', 10.0, start_poly, None,
                 env_files=None, fish_length=None, num_timesteps=181, num_agents=10,
                 use_gpu=False, pid_tuning=False, hecras_plan_path=hecras_default,
                 hecras_fields=None, hecras_k=8, use_hecras=True, hecras_write_rasters=False)

report = {}
report['perimeter_points_count'] = int(getattr(sim, 'perimeter_points', None).shape[0]) if getattr(sim, 'perimeter_points', None) is not None else 0
report['perimeter_polygon_present'] = bool(getattr(sim, 'perimeter_polygon', None) is not None)
report['wetted_mask_count'] = int(np_sum := int(sim.wetted_mask.sum()) if getattr(sim, 'wetted_mask', None) is not None else 0)

# check initial agent wet flags if available
try:
    sim.update_hecras_mapping_for_current_positions()
    # flatten wet if exists
    wet = getattr(sim, 'wet', None)
    if wet is not None:
        report['agent_wet_count'] = int((wet.flatten()==1).sum())
    else:
        report['agent_wet_count'] = None
except Exception as e:
    report['update_mapping_error'] = str(e)

# Save report
out_dir = os.path.join(ROOT, 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'smoke_test_perimeter.json')
with open(out_path, 'w') as fh:
    json.dump(report, fh, indent=2)

print('Smoke test report written to', out_path)
print(report)
