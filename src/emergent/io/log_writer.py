import os
import numpy as np


class LogWriter:
    """Fast per-timestep logger that writes binary .npz files.

    Usage:
        lw = LogWriter(output_dir)
        lw.append(timestep_index, {'X': X_array, 'Y': Y_array, 'vel_x': velx, ...})
        lw.close()

    The writer creates files named `step_{t:06d}.npz` and an index `index.txt`.
    This is intentionally simple and optimized for write speed.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.index_path = os.path.join(self.out_dir, 'index.txt')
        # open index file for append (line-buffered)
        self._index_f = open(self.index_path, 'a', buffering=1)

    def append(self, t, arrays_dict):
        """Write arrays_dict to a single `.npz` for timestep `t`.

        arrays_dict: mapping string -> numpy array (must be serializable by numpy.savez)
        """
        fn = os.path.join(self.out_dir, f'step_{t:06d}.npz')
        # Use plain savez (uncompressed) for max speed
        try:
            np.savez(fn, **arrays_dict)
            self._index_f.write(f'{t},{os.path.basename(fn)}\n')
        except Exception:
            # best-effort: ignore write failures to not crash simulation
            pass

    def close(self):
        try:
            self._index_f.close()
        except Exception:
            pass
