import h5py, numpy as np, sys
p = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/import_test_headless.h5'
print('Inspecting', p)
with h5py.File(p,'r') as f:
    for k in f.keys():
        print('Top key:', k)
    for key in ['environment/depth','environment/vel_x','environment/vel_y','environment/vel_mag','environment/vel_dir','environment/x_coords','environment/y_coords']:
        if key in f:
            arr = f[key][()]
            print(key, 'shape=', arr.shape, 'min=', np.nanmin(arr), 'max=', np.nanmax(arr))
        else:
            print('missing', key)
