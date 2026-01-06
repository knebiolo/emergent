"""Run isolated cue headless tests and save diagnostics.

This harness runs the current raster-based simulation with controlled behavioral
weights for each cue and saves outputs into `outputs/diagnostics/`.

Usage:
  python tools/run_isolated_cue_tests.py rheotaxis cohesion alignment collision low_speed
"""
import os
import sys
from datetime import datetime

import numpy as np

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.rl_training import BehavioralWeights

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs', 'diagnostics')
os.makedirs(OUT, exist_ok=True)


BEHAVIOR_KEYS = ['rheotaxis', 'alignment', 'cohesion', 'low_speed', 'wave_drag', 'refugia', 'border', 'shallow', 'avoid', 'collision']

# Target exploratory weights for each isolated cue test; tests will write a full
# weights JSON where all keys are present and only the target receives the
# non-zero exploratory value.
TARGET_WEIGHTS = {
    'rheotaxis': 5000.0,
    'cohesion': 10000.0,
    'alignment': 8000.0,
    'collision': 20000.0,
    'low_speed': 3000.0,
    'avoid': 15000.0,
    'wave_drag': 4000.0,
    'refugia': 7000.0,
    'border': 12000.0,
    'shallow': 12000.0,
}

CUE_TO_FIELD = {
    'rheotaxis': 'rheotaxis_weight',
    'alignment': 'alignment_weight',
    'cohesion': 'cohesion_weight',
    'collision': 'collision_weight',
    'low_speed': 'low_speed_weight',
    'avoid': 'avoid_weight',
    'wave_drag': 'wave_drag_weight',
    'refugia': 'refugia_weight',
    'border': 'border_cue_weight',
    'shallow': 'shallow_weight',
}

def discover_env_files(base_dir):
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def run_cue(cue, start_shp=None, nagents=50, nsteps=20, seed=42):
    now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    model_name = f'cue_test_{cue}_{now}'
    np.random.seed(int(seed))

    base_dir = os.path.join(ROOT, 'data', 'salmon_abm')
    env_files = discover_env_files(base_dir)
    if not env_files:
        print(f"[ERR] No environment files found in {base_dir}")
        return 2

    start_polygon = start_shp if (start_shp and os.path.exists(start_shp)) else None
    db_path = os.path.join(OUT, f'{model_name}.h5')

    sim = simulation(
        model_dir=OUT,
        model_name=model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_polygon,
        env_files=env_files,
        longitudinal_profile=None,
        num_timesteps=int(nsteps),
        num_agents=int(nagents),
        db_path=db_path,
        output_write_mode='full',
        output_write_backend='sync',
    )

    # zero all cue weights, then set only the target cue
    weights_dict = BehavioralWeights().to_dict()
    for field_name in CUE_TO_FIELD.values():
        weights_dict[field_name] = 0.0
    if cue not in CUE_TO_FIELD:
        raise ValueError(f"Unknown cue '{cue}'. Expected one of: {sorted(CUE_TO_FIELD)}")
    weights_dict[CUE_TO_FIELD[cue]] = float(TARGET_WEIGHTS.get(cue, 0.0))
    sim.load_behavioral_weights(weights_dict=weights_dict)

    sim.debug_behavior = True
    sim.initialize_headings_from_db()

    print(f"Running cue test: cue={cue} agents={nagents} steps={nsteps} -> {db_path}")
    sim.run(dt=1.0)
    sim.close()
    return 0


def main(args):
    cues = args or BEHAVIOR_KEYS
    # starting polygons
    river_right = os.path.join(ROOT, 'data', 'salmon_abm', 'start_loc_river_right.shp')
    near_shore = os.path.join(ROOT, 'data', 'salmon_abm', 'near_shore.shp')
    results = {}
    for cue in cues:
        if cue in ('border', 'shallow'):
            start_shp = near_shore
        else:
            start_shp = river_right
        # use 1000 agents to force interaction as requested
        rc = run_cue(cue, start_shp=start_shp, nagents=1000, nsteps=20, seed=42)
        results[cue] = rc
    print('\nResults:')
    for k, v in results.items():
        print(k, '=>', 'OK' if v == 0 else f'FAILED({v})')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
