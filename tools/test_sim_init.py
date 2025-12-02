import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    try:
        from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation
    except Exception:
        traceback.print_exc()
        return 2

    hecras_plan = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    # Auto-detect a start polygon similar to the viewer logic
    start_dir = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location')
    start_polygon = None
    if os.path.exists(start_dir):
        for fname in os.listdir(start_dir):
            if fname.lower().startswith('start_loc_river_right') and fname.lower().endswith('.shp'):
                start_polygon = os.path.join(start_dir, fname)
                break
    if start_polygon is None:
        legacy = os.path.join(start_dir, 'river_right.shp')
        if os.path.exists(legacy):
            start_polygon = legacy
    # Fallback: pick any .shp in starting_location
    if start_polygon is None and os.path.exists(start_dir):
        for fname in os.listdir(start_dir):
            if fname.lower().endswith('.shp'):
                start_polygon = os.path.join(start_dir, fname)
                break

    longitudinal = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'nuyakuk_centerline.shp')

    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training_test'),
        'model_name': 'test_sim',
        'crs': 'EPSG:32605',
        'basin': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Bathymetry.shp'),
        'water_temp': 10.0,
        'start_polygon': start_polygon,
        'longitudinal_profile': longitudinal,
        'env_files': None,
        'num_agents': 10,
        'fish_length': 450,
        'hecras_plan_path': hecras_plan,
        'use_hecras': True,
    }

    try:
        sim = simulation(**config)
        # mark visual_mode to avoid heavy IO
        try:
            sim.visual_mode = True
        except Exception:
            pass
        print('simulation init OK')
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
