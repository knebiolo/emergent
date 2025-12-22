# Quick verification script comparing solve_for_f.py algebra to movement.frequency implementation
import numpy as np
from scipy.interpolate import UnivariateSpline

# Re-implement the algebraic frequency from scripts/archive/.../solve_for_f.py as a function
def frequency_algebra(U_cm_s, L_cm, D_erg_s):
    # constants
    rho = 1.0
    theta = 32.0

    # Webb table fits (copied)
    length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
    speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.]) / 100.0
    amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4]) / 100.0
    wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
    edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.]) / 100.0

    amplitude = UnivariateSpline(length_dat, amp_dat, k=2, ext=0)
    wave = UnivariateSpline(speed_dat, wave_dat, k=1, ext=0)
    trail = UnivariateSpline(length_dat, edge_dat, k=1, ext=0)

    A = amplitude(L_cm)
    V = wave(U_cm_s/100.0) if (U_cm_s>1) else wave(U_cm_s)
    B = trail(L_cm)

    # copied expression from solve_for_f: sqrt(D*V**2*cos(theta)/(A**2*B**2*U*pi**3*rho*(U - V)*(...)))
    # Note: this expression in the original script used some hard-coded coefficients; we'll copy them
    denom_coeffs = (-0.062518880701972*U_cm_s - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)
    numerator = D_erg_s * V**2 * np.cos(np.radians(theta))
    denom = (A**2 * B**2 * U_cm_s * np.pi**3 * rho * (U_cm_s - V) * denom_coeffs)

    val = None
    if denom>0 and numerator>0:
        val = np.sqrt(numerator/denom)
    return val

# Movement implementation re-creation (simplified numeric) matching movement.frequency logic
from emergent.salmon_abm.movement import movement as Movement

# Create a fake simulation object with required attributes
class FakeSim:
    def __init__(self, n, length_mm):
        self.length = np.full(n, length_mm) # mm
        self.x_vel = np.zeros(n)
        self.y_vel = np.zeros(n)
        self.ideal_sog = np.full(n, 0.2) # m/s
        self.heading = np.zeros(n)
        self.max_s_U = np.full(n, 0.3)
        self.wave_drag = np.ones(n)
        self.water_temp = 10
        self.swim_behav = np.zeros(n, dtype=int)
        self.is_stuck = np.zeros(n, dtype=bool)
        self.Hz = np.zeros(n)
        self.prev_Hz = np.zeros(n)
        self.X = np.zeros(n)
        self.prev_X = np.zeros(n)
        self.Y = np.zeros(n)
        self.prev_Y = np.zeros(n)
    def drag_coeff(self, reynolds):
        # delegate to movement.drag_coeff implementation for consistency
        from emergent.salmon_abm.movement import movement as _M
        return _M.drag_coeff(_M, reynolds)

# We'll compare for a single representative agent: length 20cm (200mm), U=0.5 m/s -> 50 cm/s
U_m_s = 0.5
U_cm_s = U_m_s * 100.0
L_cm = 20.0
D_erg_s = 1e5  # arbitrary

alg_f = frequency_algebra(U_cm_s, L_cm, D_erg_s)
print('Algebra freq (solved):', alg_f)

# Now use movement.frequency
sim = FakeSim(1, length_mm=200.)
mov = Movement(sim)
# set fish velocities so that swim_speeds_cms ~ 50
sim.ideal_sog = np.array([U_m_s])
# compute ideal drag using movement.ideal_drag_fun so algebra uses same D
ideal_drags = mov.ideal_drag_fun(fish_velocities=np.stack((sim.ideal_sog * np.cos(sim.heading), sim.ideal_sog * np.sin(sim.heading)), axis=-1))
drag_force_N = np.linalg.norm(ideal_drags, axis=-1)[0]
swim_speed_m_s = U_m_s
drags_erg_s = drag_force_N * swim_speed_m_s * 1e7

alg_f_from_D = frequency_algebra(U_cm_s, L_cm, drags_erg_s)
print('Computed drag_force_N:', drag_force_N)
print('Derived D (erg/s):', drags_erg_s)
print('Algebra freq with same D:', alg_f_from_D)

# now call movement.frequency to compute Hz stored in sim
mov.frequency(mask=np.array([True]), t=1, dt=1.0)
print('movement.Hz:', sim.Hz)

print('\nDone')
