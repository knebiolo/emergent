"""NumPy fallback for combined drag and battery calculations."""
import numpy as np

def drag_and_battery(positions, speeds, headings, env_fields=None, dt=1.0):
    """Compute simple drag force opposing velocity and battery consumption proportional to speed.

    Returns (force_vectors: (N,2), battery_updates: (N,))
    """
    speeds = np.asarray(speeds, dtype=float)
    headings = np.asarray(headings, dtype=float)
    # drag magnitude ~ k * speed^2, take k=0.1 for deterministic behaviour
    k = 0.1
    drag_mag = k * speeds**2
    forces = np.vstack((-np.cos(headings)*drag_mag, -np.sin(headings)*drag_mag)).T
    # battery consumption proportional to work ~ speed * drag_mag * dt
    battery_updates = (speeds * drag_mag) * dt
    return forces, battery_updates
