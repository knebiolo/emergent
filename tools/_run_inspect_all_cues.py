import glob
import subprocess

files = glob.glob('outputs/diagnostics/cue_test_*_diagnostics.h5')
if not files:
    print('No cue_test HDF5 files found.')
    raise SystemExit(1)

for f in files:
    print('Inspecting', f)
    subprocess.run(['python', 'tools/inspect_h5_cues.py', f, '--step', '0', '--outdir', 'outputs/diagnostics/cue_checks'])

print('Done')
