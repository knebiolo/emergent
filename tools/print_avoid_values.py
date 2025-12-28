import h5py, sys
fpath = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/avoid_single_agent_debug_h5_diagnostics.h5'
try:
    with h5py.File(fpath,'r') as f:
        steps = list(f.get('steps',{}).keys())
        print('steps:', steps)
        if '0' in [k.split('/')[-1] for k in steps] or '0' in [k for k in steps]:
            # prefer full path
            try:
                av = f['steps/0/avoid_vec'][()]
                am = f['steps/0/avoid_mag'][()]
                print('avoid_vec:', av)
                print('avoid_mag:', am)
            except Exception as e:
                print('Could not read avoid datasets:', e)
        else:
            # attempt direct path
            try:
                av = f['steps/0/avoid_vec'][()]
                am = f['steps/0/avoid_mag'][()]
                print('avoid_vec:', av)
                print('avoid_mag:', am)
            except Exception as e:
                print('No step 0 datasets present or read error:', e)
except Exception as e:
    print('Failed to open', fpath, '->', e)
    sys.exit(1)
