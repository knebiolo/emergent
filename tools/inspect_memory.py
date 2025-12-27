import h5py
import numpy as np

def inspect(path, label):
    print('---', label, path)
    try:
        with h5py.File(path, 'r') as f:
            if 'memory' in f:
                keys = sorted(list(f['memory'].keys()))
                print('memory keys count:', len(keys))
                for k in keys[:5]:
                    d = np.array(f[f'memory/{k}'])
                    print(k, d.shape, 'min/max:', np.nanmin(d), np.nanmax(d))
            else:
                print('no memory group')
    except Exception as e:
        print('failed to open', path, e)

if __name__ == '__main__':
    pre = 'outputs/diagnostics/preseed_memory_at_falls.h5'
    avoid = 'outputs/diagnostics/avoid_test_at_falls_1000x20_headless.h5'
    inspect(pre, 'preseed')
    inspect(avoid, 'avoid_headless')
