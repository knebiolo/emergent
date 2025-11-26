"""
Small parameter sweep driver for ADVANCED_CONTROLLER['backcalc_beta'] and
SHIP_PHYSICS['Ndelta'] scaling. Runs short headless simulations and collects PID traces
and diagnostics using existing compare_pid_diagnostics.py utilities.

This script runs locally and writes CSV traces into scripts/sweep_traces/.
"""
import os
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / 'sweep_traces'
SWEEP_DIR.mkdir(exist_ok=True)

# grid
backcalc_vals = [0.08, 0.12, 0.16]
ndelta_scales = [0.7, 1.0, 1.2]

# helper to run headless sim: uses existing script run_rosario_2_headless_tuned.py
# which writes a PID trace to traces/ or scripts/ depending on config.
# We'll run each configuration by setting environment variables to influence
# config at runtime (monkeypatch via import time isn't available here), so
# as a pragmatic approach we'll copy config.py to a temp file, patch values,
# run the headless script, and restore the original.

CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'src' / 'emergent' / 'ship_abm' / 'config.py'
# Read with explicit utf-8 and replace errors to avoid decoding issues on Windows
ORIG = CONFIG_PATH.read_text(encoding='utf-8', errors='replace')

runs = []
for b, s in itertools.product(backcalc_vals, ndelta_scales):
    name = f"b{b:.2f}_n{int(s*100)}"
    runs.append((b, s, name))

print(f"Running {len(runs)} configurations")

for b, s, name in runs:
    print(f"\n=== RUN {name} : backcalc={b}, ndelta_scale={s} ===")
    # patch config content
    patched = ORIG.replace("\n    \"backcalc_beta\": 0.16,\n", f"\n    \"backcalc_beta\": {b},\n    \"raw_cap_deg\": 90.0,\n")
    # also patch Ndelta value by locating the line with 'Ndelta':
    import re
    patched = re.sub(r"'Ndelta':\s*[^,]+,", f"'Ndelta': {133286909.215551 * s},", patched)
    # write to temp file
    tmp = CONFIG_PATH.with_suffix('.tmp')
    tmp.write_text(patched, encoding='utf-8', errors='replace')
    # swap files
    bak = CONFIG_PATH.with_suffix('.bak')
    CONFIG_PATH.rename(bak)
    # write the patched file atomically
    tmp.rename(CONFIG_PATH)
    try:
        # run the headless script; assumed to write a pid trace at scripts/pid_trace_tuned.csv
        script = ROOT / 'run_rosario_2_headless_tuned.py'
        if not script.exists():
            # fallback: try existing run script in scripts/
            script = ROOT / 'run_ship.py'
        cmd = [sys.executable, str(script), '--no-gui']
        print('Running', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        # move generated trace into sweep_dir with unique name
        # try known outputs
        candidates = [ROOT / 'pid_trace_tuned.csv', ROOT / 'pid_trace.csv', ROOT.parent / 'traces' / 'rosario_2_agent_pid_trace_tuned.csv']
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            print('Warning: no pid trace produced for run', name)
        else:
            dest = SWEEP_DIR / f"pid_{name}.csv"
            shutil.copy(found, dest)
            print('Saved trace to', dest)
    except subprocess.CalledProcessError as e:
        print('Run failed', e)
    finally:
        # restore original config
        CONFIG_PATH.unlink()
        bak.rename(CONFIG_PATH)

print('\nSweep complete. Traces in', SWEEP_DIR)
