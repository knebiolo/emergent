import os, numpy as np, math, glob

npz_files = sorted(glob.glob('outputs/diagnostics/behavior_debug_step_*.npz'))
if not npz_files:
    print('No NPZs found')
    raise SystemExit(1)

def ang_between(v1, v2):
    a1 = np.degrees(np.arctan2(v1[:,1], v1[:,0]))
    a2 = np.degrees(np.arctan2(v2[:,1], v2[:,0]))
    d = a2 - a1
    d = (d + 180) % 360 - 180
    return d

for f in npz_files[:6]:
    d = np.load(f)
    head = d['head_vec'] if 'head_vec' in d.files else None
    coh = d['cohesion_vec'] if 'cohesion_vec' in d.files else None
    if head is None or coh is None:
        print(f, 'missing fields:', d.files)
        continue
    angs = ang_between(coh, head)
    print(os.path.basename(f), 'mean_abs_angle_deg=', np.mean(np.abs(angs)), 'max_abs=', np.max(np.abs(angs)))
