#!/usr/bin/env python3
import sys, os
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else 'outputs/diagnostics/behavior_debug_step_0_1766678511.npz'
if not os.path.exists(p):
    print('missing:', p)
    sys.exit(1)

try:
    d = np.load(p, allow_pickle=True)
except Exception as e:
    print('failed to load NPZ:', e)
    sys.exit(2)

print('Loaded:', p)
print('keys:', list(d.files))
for k in d.files:
    a = d[k]
    print(f"- {k}: shape={getattr(a,'shape',None)} dtype={getattr(a,'dtype',None)} size={getattr(a,'size',0)}")
    if getattr(a, 'size', 0) > 0:
        try:
            flat = a.flatten()
            sample = flat[:10].tolist()
            print('  sample:', sample)
        except Exception:
            print('  sample: <could not flatten>')

# extra insights for head_vec and neighbor diagnostics
if 'head_vec' in d:
    try:
        hv = np.asarray(d['head_vec']).astype(float)
        if hv.ndim == 2:
            norms = np.linalg.norm(hv, axis=1)
            print('head_vec: mean_norm=', float(np.nanmean(norms)), 'max_norm=', float(np.nanmax(norms)))
            print('head_vec first rows:', hv[:10].tolist())
    except Exception as e:
        print('head_vec analysis failed:', e)

if 'neighbor_counts' in d:
    try:
        nc = np.asarray(d['neighbor_counts'])
        print('neighbor_counts: mean=', float(np.nanmean(nc)), 'max=', int(np.nanmax(nc)))
        print('neighbor_counts sample:', nc[:20].tolist())
    except Exception as e:
        print('neighbor_counts analysis failed:', e)

print('Done')
