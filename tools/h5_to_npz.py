import sys
import os
import h5py
import numpy as np

def export(h5_path, step, out_npz=None):
    if out_npz is None:
        base = os.path.splitext(os.path.basename(h5_path))[0]
        out_npz = os.path.join(os.path.dirname(h5_path), f'{base}_step_{step}.npz')
    with h5py.File(h5_path, 'r') as f:
        grp = f.get(f'steps/{int(step)}')
        if grp is None:
            raise ValueError('Step not found in HDF5: ' + str(step))
        payload = {}
        for k in grp.keys():
            try:
                payload[k] = np.array(grp[k])
            except Exception:
                try:
                    payload[k] = np.array(grp[k][()])
                except Exception:
                    payload[k] = None
        np.savez_compressed(out_npz, **payload)
    return out_npz

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python tools/h5_to_npz.py <h5_path> <step> [out_npz]')
        sys.exit(1)
    h5_path = sys.argv[1]
    step = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    out_npz = export(h5_path, step, out)
    print('Wrote', out_npz)