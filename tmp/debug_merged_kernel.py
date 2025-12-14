import numpy as np
from emergent.fish_passage import merged_kernels
from emergent.salmon_abm import sockeye
rng = np.random.RandomState(1)
n = 20
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
battery = np.ones(n)
swim_speeds_buf = np.zeros((n, 4))

print('mask:', mask)
out_new = merged_kernels.merged_swim_drag_fatigue(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf)
out_old = sockeye._merged_swim_drag_fatigue_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf)
labels = ['swim_speeds','bl_s','prolonged','sprint','sustained','drags']
for name,a,b in zip(labels,out_new,out_old):
    print('---', name, '---')
    print('new:', a)
    print('old:', b)
    if isinstance(a, np.ndarray):
        if a.dtype==bool:
            neq = np.where(a != b)[0]
        else:
            neq = np.where(~np.isclose(a,b))[0]
        print('mismatch indices:', neq.tolist())
