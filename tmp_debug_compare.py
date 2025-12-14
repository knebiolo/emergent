import numpy as np
from emergent.fish_passage import merged_kernels
from emergent.salmon_abm import sockeye

rng = np.random.RandomState(2)
n = 16
sog = np.abs(rng.randn(n))
heading = rng.randn(n)
x_vel = rng.randn(n) * 0.2
y_vel = rng.randn(n) * 0.2
mask = rng.rand(n) > 0.1
density = 1.0
surface_areas = np.abs(rng.randn(n)) + 0.1
drag_coeffs = np.abs(rng.randn(n)) + 0.1
wave_drag = np.abs(rng.randn(n)) + 0.1
swim_behav = rng.randint(0,5,size=n)
max_s_U = np.full(n, 1.0)
max_p_U = np.full(n, 2.0)
battery = np.clip(np.abs(rng.randn(n)), 0.0, 1.0)
per_rec = np.abs(rng.randn(n)) * 0.1
ttf = np.abs(rng.randn(n)) + 0.1
dt = 0.5
swim_speeds_buf = np.zeros((n, 4))

out_new = merged_kernels.drag_and_battery(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery.copy(), per_rec, ttf, dt, True, swim_speeds_buf)
out_old = sockeye._drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery.copy(), per_rec, ttf, dt, True)

ss_new, bl_new, prolonged_new, sprint_new, sustained_new, drags_new, batt_new = out_new
ss_old, bl_old, prolonged_old, sprint_old, sustained_old, drags_old, batt_old = out_old

print('index | batt_old | batt_new | per_rec | ttf | battery_init')
for i in range(n):
    if not np.isclose(batt_old[i], batt_new[i], rtol=1e-12, atol=1e-12):
        print(i, batt_old[i], batt_new[i], per_rec[i], ttf[i], battery[i])

print('\nDetailed diff for all indices:')
for i in range(n):
    print(i, batt_old[i], batt_new[i], batt_old[i]-batt_new[i])
