import os
import h5py
import numpy as np

class HDF5DiagnosticsWriter:
    def __init__(self, path):
        self.path = path
        self.file = None

    def open(self, mode='a'):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # use swmr disabled for simplicity; opens in append/create mode
        self.file = h5py.File(self.path, mode)

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass

    def write_step(self, step, payload: dict):
        if self.file is None:
            raise RuntimeError('Diagnostics file not open')
        grp_name = f'steps/{int(step)}'
        if grp_name in self.file:
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        # write datasets for arrays in payload; overwrite if exists
        for k, v in payload.items():
            try:
                arr = np.asarray(v)
                if k in grp:
                    del grp[k]
                # for scalars create scalar dataset
                if arr.ndim == 0:
                    grp.create_dataset(k, data=arr)
                else:
                    grp.create_dataset(k, data=arr, compression='gzip')
            except Exception:
                # best-effort: store string repr
                try:
                    s = str(v)
                    if k in grp:
                        del grp[k]
                    grp.create_dataset(k, data=np.string_(s))
                except Exception:
                    pass
        # flush to disk to minimize window where data is missing
        try:
            self.file.flush()
            os.fsync(self.file.id.fileno())
        except Exception:
            try:
                self.file.flush()
            except Exception:
                pass
