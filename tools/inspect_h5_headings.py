#!/usr/bin/env python3
import glob, os, sys
try:
    import h5py
    import numpy as np
except Exception as e:
    print('Missing dependency:', e)
    raise

trace='outputs/diagnostics/cue_test_cohesion_quick_trace.csv'
if len(sys.argv)>1:
    trace=sys.argv[1]

base=os.path.basename(trace).replace('_trace.csv','')
pattern='outputs/diagnostics/*_headless.h5'
files=[f for f in sorted(glob.glob(pattern)) if base in os.path.basename(f)]
if not files:
    files=sorted(glob.glob(pattern))
    if not files:
        print('No h5 files found matching', pattern)
        sys.exit(1)
    print('No exact match; showing latest 3 h5 files:')
    for f in files[-3:]:
        print('  ', os.path.basename(f), ' size=', os.path.getsize(f))
    f=files[-1]
else:
    print('Found matching h5 files:')
    for f0 in files:
        print('  ', os.path.basename(f0), ' size=', os.path.getsize(f0))
    f=files[-1]

print('\nUsing h5:', os.path.abspath(f))
with h5py.File(f,'r') as h:
    print('root keys:', list(h.keys()))
    if 'agent_data' in h:
        ad=h['agent_data']
        print('agent_data keys:', list(ad.keys()))
        if 'heading' in ad:
            arr=ad['heading'][:]
            a=np.array(arr)
            print('agent_data/heading dtype=', a.dtype, ' shape=', a.shape)
            # print small sample
            if a.ndim==2:
                r=min(5,a.shape[0])
                c=min(5,a.shape[1])
                print('sample [rows x cols]:')
                print(a[:r,:c])
            else:
                print('sample:', a[:min(10,a.size)])
        else:
            print('agent_data/heading not present')
    if 'heading' in h:
        hh=h['heading'][:]
        print('top-level heading dtype=', hh.dtype, ' shape=', hh.shape)
        print('top-level heading sample:', hh[:min(10, len(hh))])

print('\nDone')
