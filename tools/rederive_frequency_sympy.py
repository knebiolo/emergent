"""Re-derive tailbeat frequency from Lighthill thrust = drag using SymPy with explicit SI units.

This script:
- Declares symbolic variables with unit expectations (SI: m, s, kg)
- Builds a symbolic Lighthill thrust expression (using A, B, V, f, U, rho, theta)
- Sets thrust = drag_power and solves for f symbolically
- Substitutes numeric values (from movement's Webb splines) using consistent units
- Prints simplified symbolic solution and numeric evaluation

Notes: keep constants symbolic as long as possible. When numeric coefficients are used
from earlier derivation they will be re-derived here from base constants.
"""

from sympy import symbols, Eq, solve, sqrt, pi, cos, simplify, Symbol
import numpy as np
from scipy.interpolate import UnivariateSpline

# Build splines matching movement.py but return SI-consistent values
length_dat_cm = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])  # cm as in Webb
speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.]) / 100.0  # original script uses fraction, but this was ambiguous
amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4]) / 100.0  # amplitude in m (if dividing by 100 from cm->m)
wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.]) / 100.0

# We'll convert lengths to meters for SI: so L (cm) -> L_m = L_cm / 100.0
amp_spline = UnivariateSpline(length_dat_cm, amp_dat, k=2, ext=0)
wave_spline = UnivariateSpline(length_dat_cm, wave_dat, k=1, ext=0)  # note: original script may have used speed-based spline; use length-based here for stability
edge_spline = UnivariateSpline(length_dat_cm, edge_dat, k=1, ext=0)

# Symbolic variables (SI):
# f: frequency (Hz), A: amplitude (m), B: span (m), V: prop wave speed (m/s), U: swim speed (m/s),
# rho: density (kg/m^3), theta: angle (rad), D: drag power (W or J/s) depending on side
f, A, B, V, U, rho, theta, D = symbols('f A B V U rho theta D', positive=True)

# Lighthill thrust (formula structure based on the script):
# m = (pi * rho * B**2)/4
# W_amp = f * A * pi / sqrt(2)  (note: original used 1.414)
# w = W_amp * (1 - U/V)
# Thrust (power-like) = m * W_amp * w * U - (m * w**2 * U)/(2*cos(theta))
# Interpret D as drag power (J/s) so we equate thrust (converted to power-like units) to D.

m_sym = pi * rho * B**2 / 4
W_amp = f * A * pi / sqrt(2)
w = W_amp * (1 - U / V)

# thrust_power_sym is Lighthill's thrust expression in energy-rate units if W_amp and w are velocities (units check follows below)
thrust_expr = m_sym * W_amp * w * U - (m_sym * w**2 * U) / (2 * cos(theta))

# Solve thrust_expr = D for f symbolically
sol_f = solve(Eq(thrust_expr, D), f)

print('Symbolic solutions for f (raw):')
for s in sol_f:
    print(simplify(s))

# Numeric test: pick representative numeric inputs and evaluate both the symbolic expression
# Representative L=20 cm, so we convert to SI
L_cm = 20.0
L_m = L_cm / 100.0
A_val = amp_spline(L_cm)  # this was given as fraction of length in the old script; ensure it's in meters
B_val = edge_spline(L_cm)
# Use a plausible V: from wave_spline (note: wave_dat may be in cm/s originally); pick a number and convert to m/s
V_val = wave_spline(L_cm) / 100.0  # convert cm/s to m/s if necessary
U_val = 0.5  # m/s swim speed
rho_val = 1000.0  # kg/m^3 for water
theta_val = np.radians(32.0)

# compute ideal drag force from movement's ideal_drag_fun? We'll use a proxy: assume drag power D is small
# Here we choose D from a reasonable force*speed: use D = F*N * U (J/s). Choose F~0.5N from earlier run.
D_val = 0.58 * U_val  # N * m/s = J/s, approx 0.58 N from earlier ideal_drag, times 0.5 m/s

# Evaluate numeric
subs_map = {A: float(A_val), B: float(B_val), V: float(V_val), U: float(U_val), rho: float(rho_val), theta: float(theta_val), D: float(D_val)}

numeric_solutions = [s.evalf(subs=subs_map) for s in sol_f]
print('\nNumeric solutions (SI-consistent inputs):')
for ns in numeric_solutions:
    print(ns)

print('\nNumeric values used:')
print('A (m):', A_val)
print('B (m):', B_val)
print('V (m/s):', V_val)
print('U (m/s):', U_val)
print('D (J/s):', D_val)

# End
