import numpy as np
import glob, os

def summarize_npz(path):
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print('Failed to load', path, e)
        return
    print('File:', path)
    for k in npz.files:
        arr = np.asarray(npz[k])
        print(' -', k, 'shape=', arr.shape)
        try:
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim>1 else arr
            # print first 5 agent entries
            sample = arr[:5] if arr.shape[0]>=5 else arr
            print('   sample:', sample)
            if arr.dtype.kind in 'fi':
                print('   stats: min=', float(np.nanmin(arr)), 'max=', float(np.nanmax(arr)), 'mean=', float(np.nanmean(arr)))
        except Exception as e:
            print('   could not summarize', e)

if __name__=='__main__':
    files = sorted(glob.glob(os.path.join('outputs','diagnostics','behavior_debug_step_*.npz')))
    if not files:
        print('No behavior debug NPZs found in outputs/diagnostics')
    else:
        summarize_npz(files[-1])
