import numpy as np
from typing import Tuple


def standardize_shape(arr: np.ndarray, target_shape: Tuple[int, int] = (5, 5), fill_value=np.nan) -> np.ndarray:
    """Return an array of `target_shape` containing `arr` in the top-left corner.

    If `arr.shape` already equals `target_shape`, return `arr` unchanged.
    """
    arr = np.asarray(arr)
    if arr.shape == target_shape:
        return arr
    out = np.full(target_shape, fill_value, dtype=arr.dtype if arr.size else float)
    rows = min(arr.shape[0], target_shape[0]) if arr.size else 0
    cols = min(arr.shape[1], target_shape[1]) if arr.size else 0
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return out
