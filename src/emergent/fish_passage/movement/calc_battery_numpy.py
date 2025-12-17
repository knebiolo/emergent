"""NumPy fallback for battery delta calculation."""
import numpy as np

def calc_battery(speeds, efforts, dt=1.0):
    """Return battery delta (positive means consumed).

    speeds: (N,)
    efforts: (N,) (dimensionless effort factor)
    dt: float
    """
    speeds = np.asarray(speeds, dtype=float)
    efforts = np.asarray(efforts, dtype=float)
    # simple model: consumption = effort * (a + b * speed^2) * dt
    a = 0.01
    b = 0.005
    return efforts * (a + b * speeds**2) * dt
