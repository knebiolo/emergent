"""NumPy fallback for bout distance calculations."""
import numpy as np

def bout_distance(speeds, bout_params=None):
    speeds = np.asarray(speeds, dtype=float)
    # trivial: distance = speed * mean_bout_duration; mean_bout_duration default=1.0
    mean_dur = 1.0
    if bout_params and 'mean_duration' in bout_params:
        mean_dur = float(bout_params['mean_duration'])
    return speeds * mean_dur
