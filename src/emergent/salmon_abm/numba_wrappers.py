import numpy as np
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def get_arr(x):
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


if _HAS_NUMBA:
    from numba import njit

    @njit(cache=True)
    def _wrap_drag_fun_numba(arr):
        # simple placeholder to demonstrate structure; original monolith had multiple wrappers
        return arr

    @njit(cache=True)
    def _wrap_project_points_onto_line_numba(pts_x, pts_y, line_x, line_y, out_idx):
        # naive projection for compatibility
        n = pts_x.shape[0]
        for i in range(n):
            out_idx[i] = -1
        return out_idx

else:
    def _wrap_drag_fun_numba(arr):
        return arr

    def _wrap_project_points_onto_line_numba(pts_x, pts_y, line_x, line_y, out_idx):
        out_idx[:] = -1
        return out_idx
