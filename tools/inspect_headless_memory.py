import h5py, sys
fpath = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/avoid_single_agent_debug_h5_headless.h5'
try:
    with h5py.File(fpath,'r') as f:
        print('Top keys:', list(f.keys()))
        def print_ds(name,obj):
            if isinstance(obj,h5py.Dataset):
                print('Dataset', name, 'shape', obj.shape)
        f.visititems(print_ds)
        # print mental_map_transform attribute if present
        attrs = getattr(f,'attrs',None)
        if attrs is not None:
            if 'mental_map_transform' in attrs:
                print('mental_map_transform attr:', attrs['mental_map_transform'])
        # try group memory
        if 'memory' in f:
            for k in f['memory']:
                ds = f['memory'][k]
                print('memory',k,'shape', ds.shape, 'min/max', ds[:].min(), ds[:].max())
except Exception as e:
    print('failed to open', fpath, e)
    sys.exit(1)
