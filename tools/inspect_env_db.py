import sys
import h5py
import numpy as np

def inspect(db_path):
    with h5py.File(db_path,'r') as f:
        if 'environment' not in f:
            print('No environment group in DB')
            return
        print('environment datasets:')
        for k in sorted(f['environment'].keys()):
            try:
                arr = f['environment/'+k][:]
                arr = arr.astype(float)
                print(f' - {k}: shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}, mean={np.nanmean(arr):.6g}, unique_vals={np.unique(arr).size if arr.size<100000 else "(large)"}')
            except Exception as e:
                print(f' - {k}: ERROR {e}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/inspect_env_db.py /path/to/db.h5')
        sys.exit(2)
    inspect(sys.argv[1])
