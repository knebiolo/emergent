import glob, os, numpy as np
files_all = sorted(glob.glob('outputs/diagnostics/behavior_debug_step_*.npz'), key=os.path.getmtime)
if not files_all:
    print('no behavior npz files found')
    raise SystemExit(1)
# prefer non-alignment variants (those without the '_alignment' suffix)
files = [f for f in files_all if not f.endswith('_alignment.npz')]
if not files:
    files = files_all
latest = files[-1]
print('latest behavior npz:', latest)
arr = np.load(latest)
print('keys:', list(arr.keys()))
for k in arr.keys():
    v = arr[k]
    print(k, getattr(v, 'shape', None), type(v))
# print small samples for rheotaxis_vec and head_vec if present
for key in ('rheotaxis_vec','head_vec'):
    if key in arr:
        a = arr[key]
        print('\n', key, 'sample (first rows):')
        print(a[:5])
