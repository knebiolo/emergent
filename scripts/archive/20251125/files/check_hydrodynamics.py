"""Quick hydrodynamics sanity checker for SHIP_PHYSICS.

- Prints key coefficients with units and recommended ranges.
- Runs a small rudder torque -> yaw acceleration test using the ship model dynamics.

Usage:
    python scripts/check_hydrodynamics.py
"""
import numpy as np
import importlib
import math

cfg = importlib.import_module('emergent.ship_abm.config')
SHIP = cfg.SHIP_PHYSICS

checks = []

def warn(msg):
    checks.append(('WARN', msg))

def info(msg):
    checks.append(('INFO', msg))

# Basic structural checks
required = ['length','beam','draft','m','Izz','Ydelta','Ndelta','max_rudder','max_rudder_rate']
for k in required:
    if k not in SHIP:
        warn(f"Missing key: {k}")

# Print core properties
print("--- SHIP_PHYSICS summary ---")
for k in ['length','beam','draft','m','Izz','Ydelta','Ndelta','linear_damping','quad_damping','max_rudder','max_rudder_rate']:
    v = SHIP.get(k, None)
    print(f"{k:16s}: {v}")

# Quick magnitude checks
if SHIP['m'] <= 0 or math.isnan(SHIP['m']):
    warn('Non-positive/NaN mass')
if SHIP['Izz'] <= 0 or math.isnan(SHIP['Izz']):
    warn('Non-positive/NaN Izz')

# Rudder effectiveness signs and magnitudes
Ydelta = SHIP.get('Ydelta', 0.0)
Ndelta = SHIP.get('Ndelta', 0.0)
if abs(Ydelta) < 1e5:
    warn('Ydelta unusually small (expect ~1e7)')
if abs(Ndelta) < 1e7:
    warn('Ndelta unusually small (expect ~1e8+)')

# Rudder limits
if SHIP['max_rudder'] <= 0 or math.isnan(SHIP['max_rudder']):
    warn('max_rudder non-positive or NaN')
if SHIP['max_rudder_rate'] <= 0 or math.isnan(SHIP['max_rudder_rate']):
    warn('max_rudder_rate non-positive or NaN')

# Run a small dynamics test using the ship model
print('\n--- Dynamic rudder-step test ---')
ship_mod = importlib.import_module('emergent.ship_abm.ship_model')
# minimal initial state for single ship
state0 = np.zeros((4,1))
pos0 = np.zeros((2,1))
psi0 = np.zeros(1)
goals = np.zeros((2,1))
s = ship_mod.ship(state0, pos0, psi0, goals)

# ensure initial body rates are zero
u0, v0, p0, r0 = np.zeros(4)
state = np.array([u0, v0, p0, r0])

# apply a nominal rudder (50% of max) and compute yaw accel via dynamics()
rudder = np.array([0.5 * s.max_rudder])
prop_thrust = np.zeros(1)
drag_force = np.zeros(1)
wind_force = np.zeros((2,1))
current_force = np.zeros((2,1))

u_dot, v_dot, p_dot, r_dot = s.dynamics(state.reshape(4,1), prop_thrust, drag_force, wind_force, current_force, rudder)

print(f"Applied rudder (deg): {np.degrees(rudder[0]):.3f}")
print(f"Yaw acceleration r_dot (rad/s^2): {r_dot[0]:.6e}")
print(f"Yaw acceleration deg/s^2: {np.degrees(r_dot[0]):.6e}")

# Simple heuristic: if yaw accel per deg of rudder is extremely large, flag
yaw_per_deg = (np.degrees(r_dot[0]) / (np.degrees(rudder[0]) + 1e-12))
print(f"Yaw accel per deg rudder (deg/s^2 per deg): {yaw_per_deg:.6e}")
if abs(yaw_per_deg) > 5.0:
    warn('High yaw acceleration per deg of rudder (>5 deg/s^2 per deg) — hydrodynamic gains may be too large')

# Summarize checks
print('\n--- Checks summary ---')
for lvl,msg in checks:
    print(f"[{lvl}] {msg}")

if not checks:
    print('No issues detected (quick checks).')
