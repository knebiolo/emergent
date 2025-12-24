import glob, os, h5py, numpy as np
files = glob.glob(os.path.join('outputs','sim_db_*.h5'))
if not files:
    print('no sim db files')
    raise SystemExit(1)
latest = max(files, key=os.path.getmtime)
print('Inspecting', latest)
with h5py.File(latest,'r') as f:
    for key in ['environment/depth','environment/vel_x','environment/vel_y','environment/vel_mag','environment/vel_dir','x_coords','y_coords']:
        if key in f:
            arr = f[key][()]
            print(key, 'shape=', arr.shape, 'dtype=', arr.dtype)
            if arr.size>0 and arr.ndim>=2:
                print('  sample rows first col:', arr[:3,0].tolist())
                print('  sample last rows first col:', arr[-3:,0].tolist())
                # check monotonicity along rows
                try:
                    r0 = arr[0,:]
                    rlast = arr[-1,:]
                    print('  row0 first 5:', r0[:5].tolist())
                    print('  rowlast first 5:', rlast[:5].tolist())
                    inc = (r0[0] < rlast[0])
                    print('  row0[0] < rowlast[0] ?', inc)
                except Exception:
                    pass
        else:
            print('  missing', key)
