import h5py, sys
fpath = 'outputs/diagnostics/debug_already_been_here.h5'
try:
    with h5py.File(fpath, 'r') as f:
        print('Groups:', list(f.keys()))
        for gname in f:
            print('Group:', gname, 'datasets:', list(f[gname].keys()))
            for dname in f[gname].keys():
                ds = f[gname][dname]
                print('Dataset', gname+'/'+dname, 'shape', ds.shape)
                try:
                    print(ds[()])
                except Exception as e:
                    print('Error reading dataset', e)
except Exception as e:
    print('Failed to read', fpath, '->', e)
    sys.exit(1)
