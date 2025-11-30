import os
import numpy as np

class MemmapLogWriter:
    """Very fast append-only writer using numpy.memmap.

    Usage:
      w = MemmapLogWriter(out_dir, var_shapes, dtype=np.float32)
      w.append(t, {'X': array_of_len_N, 'Y': ...})
      w.close()

    Implementation notes:
    - `var_shapes` is a dict mapping variable name -> (num_agents, num_timesteps)
    - memmap files are stored as `<out_dir>/<var>.npy` using numpy.lib.format.write_array
      via np.lib.format.open_memmap for safe memmap creation.
    - append() writes into the memmap slice at index `t`.
    """

    def __init__(self, out_dir, var_shapes, dtype=np.float32):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.dtype = dtype
        self.var_shapes = var_shapes
        self._memmaps = {}
        # create memmaps for each variable
        for name, shape in var_shapes.items():
            path = os.path.join(out_dir, f"{name}.npy")
            # use numpy.lib.format.open_memmap to create an array file with header
            mm = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            self._memmaps[name] = mm

    def append(self, t, arrays):
        """Write arrays for timestep t. arrays is a dict name->1D array of length num_agents."""
        for name, arr in arrays.items():
            if name not in self._memmaps:
                # skip unknown fields
                continue
            mm = self._memmaps[name]
            try:
                # ensure arr is 1D and matches num_agents
                a = np.asarray(arr, dtype=self.dtype)
                if a.ndim == 0:
                    # scalar: broadcast
                    a = np.full((mm.shape[0],), a, dtype=self.dtype)
                elif a.ndim > 1:
                    a = a.ravel()[:mm.shape[0]]
                mm[:, t] = a
            except Exception:
                # best-effort write without crashing the sim; ignore field on error
                continue

    def close(self):
        # flush memmaps by deleting references
        self._memmaps.clear()
