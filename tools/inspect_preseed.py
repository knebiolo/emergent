import h5py, sys
p = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/preseed_memory_at_falls.h5'
try:
    with h5py.File(p,'r') as f:
        print('Top keys:', list(f.keys()))
        print('File attrs:', dict(f.attrs))
        if 'memory' in f:
            for k in f['memory']:
                ds = f['memory'][k]
                arr = ds[()]
                print('memory', k, 'shape', arr.shape, 'min', arr.min(), 'max', arr.max(), 'unique=', len(set(arr.flatten().tolist())))
        else:
            print('No memory group found')
        if 'mental_map_transform' in f.attrs:
            print('mental_map_transform', f.attrs['mental_map_transform'])
except Exception as e:
    print('failed to open', p, e)
    sys.exit(1)
