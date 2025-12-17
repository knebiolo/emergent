"""NumPy fallback for swim core integrator (deterministic, testable)."""
import numpy as np

def swim_core(positions, headings, speeds, env_forces=None, dt=1.0):
    """Advance positions by heading and speed; apply simple env_forces as additive velocity.

    positions: (N,2)
    headings: (N,) radians
    speeds: (N,)
    env_forces: (N,2) optional
    dt: float

    Returns (new_positions, new_speeds)
    """
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    dx = np.vstack((np.cos(headings)*speeds, np.sin(headings)*speeds)).T * dt
    if env_forces is not None:
        dx = dx + np.asarray(env_forces, dtype=float) * dt
    new_positions = positions + dx
    new_speeds = speeds.copy()
    return new_positions, new_speeds
