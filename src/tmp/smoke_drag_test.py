import sys
sys.path.insert(0, 'src')
import numpy as np
from emergent.salmon_abm.drags import compute_drags
from emergent.salmon_abm.numba_wrappers import _assess_fatigue_core, _merged_battery_numba

n = 10
fx = np.linspace(0.5, 1.5, n)
fy = np.linspace(0.2, 1.2, n)
wx = np.zeros(n)
wy = np.zeros(n)
mask = np.ones(n, dtype=bool)
density = 1.0
surface_areas = np.ones(n) * 10.0
drag_coeffs = np.ones(n) * 0.5
wave_drag = np.ones(n)
swim_behav = np.zeros(n, dtype=int)

print('Running compute_drags...')
dr = compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
print('drags shape:', dr.shape)
print('drags sample:', dr[:3])

print('\nRunning _assess_fatigue_core...')
sog = np.ones(n) * 0.8
heading = np.zeros(n)
x_vel = np.zeros(n)
y_vel = np.zeros(n)
max_s_U = np.ones(n) * 0.5
max_p_U = np.ones(n) * 1.2
battery = np.ones(n) * 0.8
swim_speeds_buf = np.zeros((n, 5))
res = _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf)
print('assess_fatigue_core results types/shapes:')
for r in res:
    try:
        print(type(r), getattr(r, 'shape', None))
    except Exception:
        print(type(r))

print('\nRunning _merged_battery_numba...')
per_rec = np.ones(n) * 0.01
ttf = np.ones(n) * 10.0
mask_sustained = np.zeros(n, dtype=bool)
dt = 0.1
b = _merged_battery_numba(battery.copy(), per_rec, ttf, mask_sustained, dt)
print('battery updated sample:', b[:3])
