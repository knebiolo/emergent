"""NumPy fallback for merged battery update kernel."""
import numpy as np

def merged_battery(state_arrays, dt=1.0):
    """Simplified merged battery update.

    state_arrays: dict-like with keys 'battery', 'effort', 'speed' or a tuple in that order
    returns updated battery array
    """
    if isinstance(state_arrays, dict):
        battery = np.asarray(state_arrays.get('battery', []), dtype=float)
        effort = np.asarray(state_arrays.get('effort', []), dtype=float)
        speed = np.asarray(state_arrays.get('speed', []), dtype=float)
    else:
        battery, effort, speed = state_arrays
        battery = np.asarray(battery, dtype=float)
        effort = np.asarray(effort, dtype=float)
        speed = np.asarray(speed, dtype=float)

    # reuse calc_battery-like rule
    a = 0.01
    b = 0.005
    consumption = effort * (a + b * speed**2) * dt
    return battery - consumption
