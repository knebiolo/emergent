#!/usr/bin/env python3
import glob, os, numpy as np
OUTDIR='outputs/diagnostics'

npzs = sorted(glob.glob(os.path.join(OUTDIR, '*.npz')))
if not npzs:
    print('No npz debug files in', OUTDIR)
    raise SystemExit(0)

print('Top 10 most recent NPZs:')
for p in npzs[-10:]:
    print(' -', p, 'size=', os.path.getsize(p))

# prefer the most recent behavior_debug_step NPZ if present
step_npzs = [p for p in npzs if 'behavior_debug_step_' in os.path.basename(p)]
raw_force = [p for p in npzs if 'rawvecs_FORCE' in os.path.basename(p) or 'behavior_debug_rawvecs_' in os.path.basename(p)]

if step_npzs:
    p = step_npzs[-1]
    print('\nInspecting latest behavior_debug_step NPZ:', p)
    try:
        d = np.load(p, allow_pickle=True)
        print(' keys:', d.files)
        for k in ('head_vec','raw_headings_neighbors','headings_neighbors_used','neighbor_counts','neighbors_concat'):
            if k in d:
                print(' ', k, 'shape=', getattr(d[k], 'shape', None), 'dtype=', getattr(d[k], 'dtype', None))
            else:
                print(' ', k, 'MISSING')
    except Exception as e:
        print('Failed to load', p, e)

if raw_force:
    print('\nInspecting forced rawvec NPZs:')
    for p in raw_force[-5:]:
        try:
            d = np.load(p, allow_pickle=True)
            print(' -', p, 'size=', os.path.getsize(p), ' keys=', d.files)
        except Exception as e:
            print(' -', p, 'failed to open:', e)
else:
    print('\nNo forced rawvec NPZs found (rawvecs_FORCE or behavior_debug_rawvecs_*)')

# show the latest headless h5
h5s = sorted(glob.glob(os.path.join(OUTDIR, '*_headless.h5')))
if h5s:
    print('\nLatest headless h5:', h5s[-1], 'size=', os.path.getsize(h5s[-1]))
else:
    print('\nNo headless h5 found')
