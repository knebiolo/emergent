import glob, os, numpy as np
files = glob.glob('outputs/diagnostics/behavior_debug_step_*.npz')
if not files:
    raise SystemExit('No behavior NPZs found')
f = max(files, key=os.path.getmtime)
print('Inspecting', f)
d = np.load(f)
print('keys:', list(d.files))
shapes = {k: d[k].shape for k in d.files}
print('shapes:', shapes)
for k in ['neighbor_counts', 'neighbors_concat', 'neighbor_mean_distance', 'neighbor_any_within_2bl', 'neighbor_headings', 'neighbor_rel_heading', 'neighbors_owner']:
    if k in d:
        print('\n==', k, 'shape', d[k].shape)
        try:
            print(d[k])
        except Exception:
            print('could not print', k)
