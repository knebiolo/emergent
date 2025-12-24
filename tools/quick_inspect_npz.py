import numpy as np, sys
if len(sys.argv) < 2:
    print('usage: quick_inspect_npz.py <npz_path>'); sys.exit(2)
npz = np.load(sys.argv[1])
print('keys:', npz.files)
for k in npz.files:
    a = npz[k]
    try:
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        mean = float(np.nanmean(a))
        print(k, 'shape', a.shape, 'min', mn, 'max', mx, 'mean', mean)
    except Exception as e:
        print(k, 'shape', getattr(a,'shape',None), 'error', str(e))
