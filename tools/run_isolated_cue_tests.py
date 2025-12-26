"""Run isolated cue headless tests and save diagnostics.

This harness runs the headless runner with controlled behavior weights for
each cue, collects the output HDF5/NPZ/CSV files into `outputs/diagnostics/`
and runs the existing summary tools.

Usage:
  python tools/run_isolated_cue_tests.py rheotaxis cohesion alignment collision low_speed
"""
import os
import sys
import subprocess
import shlex
from datetime import datetime

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
}


def build_weight_arg(wmap):
    # build comma-separated key:val pairs for CLI if supported
    return ','.join([f'{k}:{v}' for k, v in wmap.items()])


def run_cue(cue, start_shp=None, nagents=50, nsteps=20, seed=42):
    now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    model_name = f'cue_test_{cue}_{now}'
    outdir = OUT
    # build a full weight dict where non-target cues are zeroed
    weights = {k: 0.0 for k in BEHAVIOR_KEYS}
    weights.update({cue: TARGET_WEIGHTS.get(cue, 0.0)})
    # write temporary weights file as JSON
    import json, tempfile
    tf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
    json.dump(weights, tf)
    tf.flush()
    tf.close()
    weight_file = tf.name
    cmd = [
        sys.executable, 'tools/run_nuyakuk_headless.py',
        '--nagents', str(nagents),
        '--nsteps', str(nsteps),
        '--model-name', model_name,
        '--out', outdir,
        '--debug-behavior',
        '--test-weights-file', weight_file,
    ]
    if start_shp:
        cmd += ['--start-polygon', start_shp]
    print('Running:', ' '.join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main(args):
    cues = args or list(WEIGHT_MAPS.keys())
    start_shp = os.path.join(ROOT, 'data', 'salmon_abm', 'start_loc_river_right.shp')
    results = {}
    for cue in cues:
        rc = run_cue(cue, start_shp=start_shp)
        results[cue] = rc
    print('\nResults:')
    for k, v in results.items():
        print(k, '=>', 'OK' if v == 0 else f'FAILED({v})')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
