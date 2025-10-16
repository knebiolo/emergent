"""Small utilities for angle wrapping and canonical heading differences.

Keep these pure-Python and dependency-free so tests can import them
without pulling heavy geospatial dependencies.
"""
import math
import numpy as np


def wrap_rad(x: float) -> float:
    """Wrap radians to [-pi, pi).

    Accepts scalars or numpy arrays; returns same-shaped output.
    """
    x_arr = np.asarray(x)
    return (x_arr + math.pi) % (2 * math.pi) - math.pi


def wrap_deg(x: float) -> float:
    """Wrap degrees to [-180, 180).

    Accepts scalars or numpy arrays; returns same-shaped output.
    """
    x_arr = np.asarray(x)
    return (x_arr + 180.0) % 360.0 - 180.0


def heading_diff_rad(hd: float, psi: float) -> np.ndarray:
    """Compute canonical heading error (hd - psi) wrapped to [-pi, pi).

    Works with scalars or array-like hd/psi. Returns numpy array.
    """
    hd_a = np.asarray(hd)
    psi_a = np.asarray(psi)
    # broadcast to common shape
    try:
        hd_b, psi_b = np.broadcast_arrays(hd_a, psi_a)
    except Exception:
        hd_b = np.asarray(hd_a)
        psi_b = np.asarray(psi_a)
    return wrap_rad(hd_b - psi_b)


def heading_diff_deg(hd_deg: float, psi_deg: float) -> np.ndarray:
    """Heading difference in degrees, wrapped to [-180, 180).

    Returns numpy array.
    """
    return wrap_deg(np.asarray(hd_deg) - np.asarray(psi_deg))
