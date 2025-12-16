import numpy as np
from emergent.salmon_abm import sockeye

print('Testing _wrap_merged_battery_numba...')
try:
    battery = np.array([0.5, 0.2, 1.0])
    per_rec = np.array([0.0, 0.1, 0.0])
    ttf = np.array([10.0, 5.0, 2.0])
    mask_sustained = np.array([False, True, False])
    dt = 1.0
    out = sockeye._wrap_merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt)
    print('merged_battery out:', out)
except Exception as e:
    print('merged_battery wrapper raised:', repr(e))

print('\nTesting _wrap_project_points_onto_line_numba...')
try:
    points = np.array([[0.0,0.0],[1.0,1.0],[2.0,0.0]])
    line_start = np.array([0.0,0.0])
    line_end = np.array([2.0,0.0])
    proj = sockeye._wrap_project_points_onto_line_numba(points, line_start, line_end)
    print('projection result:', proj)
except Exception as e:
    print('projection wrapper raised:', repr(e))

print('\nTesting _wrap_time_to_fatigue_numba...')
try:
    swim_speeds = np.array([0.5, 1.0, 2.0])
    mask_prolonged = np.array([True, False, True])
    mask_sprint = np.array([False, True, False])
    a_p = 0.0
    b_p = 0.1
    a_s = 0.0
    b_s = 0.2
    ttf = sockeye._wrap_time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s)
    print('ttf:', ttf)
except Exception as e:
    print('time_to_fatigue wrapper raised:', repr(e))

print('\nTesting _wrap_bout_distance_numba...')
try:
    prev_X = np.array([0.0, 1.0, 2.0])
    X = np.array([0.1, 1.2, 1.8])
    prev_Y = np.array([0.0, 0.0, 0.0])
    Y = np.array([0.0, 0.5, 0.0])
    d = sockeye._wrap_bout_distance_numba(prev_X, X, prev_Y, Y)
    print('bout distances:', d)
except Exception as e:
    print('bout_distance wrapper raised:', repr(e))
