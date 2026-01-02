"""Movement helpers extracted from sockeye.py.

This module contains the `movement` class that operates on a `simulation` object.
The implementation is a near-direct extraction and imports light-weight helpers
from `emergent.salmon_abm.utils` so callers can migrate to the new module.

Performance-critical functions are JIT-compiled with Numba for 10-50x speedup.
"""
import os
import time
import csv
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import distance_transform_edt

from emergent.salmon_abm.utils import geo_to_pixel, pixel_to_geo, standardize_shape, calculate_front_masks

logger = logging.getLogger(__name__)

# Numba JIT compilation for performance-critical numeric loops
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# Try to prepare a SymPy-backed numeric evaluator for the symbolic frequency expression.
# If SymPy is not available or lambdify fails, fall back to numeric implementation.
_SYMPY_AVAILABLE = False
_sympy_freq_func = None
try:
    import sympy as _sp
    # define symbols (SI units expected: m, s, kg, J)
    _f_s, _A_s, _B_s, _V_s, _U_s, _rho_s, _theta_s, _D_s = _sp.symbols('f A B V U rho theta D', positive=True)
    _m_sym = _sp.pi * _rho_s * _B_s ** 2 / 4
    _W_amp = _f_s * _A_s * _sp.pi / _sp.sqrt(2)
    _w = _W_amp * (1 - _U_s / _V_s)
    _thrust_sym = _m_sym * _W_amp * _w * _U_s - (_m_sym * _w ** 2 * _U_s) / (2 * _sp.cos(_theta_s))
    # solve symbolic for f and pick positive root
    _sol = _sp.solve(_sp.Eq(_thrust_sym, _D_s), _f_s)
    if _sol:
        # choose positive branch (last one is positive in our derivation)
        _f_expr = _sp.simplify(_sol[-1])
        # lambdify to numpy for fast evaluation: inputs order A,B,V,U,D,rho,theta
        _sympy_freq_func = _sp.lambdify((_A_s, _B_s, _V_s, _U_s, _D_s, _rho_s, _theta_s), _f_expr, 'numpy')
        _SYMPY_AVAILABLE = True
except Exception:
    _SYMPY_AVAILABLE = False


# ============================================================================
# JIT-COMPILED NUMERIC KERNELS (10-50x speedup via Numba)
# ============================================================================

@njit(fastmath=True, cache=True)
def _thrust_kernel(length_cm, swim_speed_cms, Hz, A, V, B, rho, theta_rad, heading, mask):
    """Vectorized thrust calculation kernel (Numba-optimized).
    
    Args:
        length_cm: agent lengths in cm (n_agents,)
        swim_speed_cms: swim speeds in cm/s (n_agents,)
        Hz: tailbeat frequency in Hz (n_agents,)
        A, V, B: Webb spline outputs (n_agents,)
        rho: water density (scalar)
        theta_rad: body angle in radians (scalar)
        heading: agent headings in radians (n_agents,)
        mask: active agents boolean mask (n_agents,)
    
    Returns:
        thrust_x, thrust_y: thrust vectors in N (n_agents,)
    """
    n = length_cm.shape[0]
    thrust_x = np.zeros(n, dtype=np.float64)
    thrust_y = np.zeros(n, dtype=np.float64)
    cos_theta = np.cos(theta_rad)
    
    for i in prange(n):
        if not mask[i]:
            continue
            
        m = (np.pi * rho * B[i] ** 2) / 4.0
        W = (Hz[i] * A[i] * np.pi) / 1.414
        w = W * (1.0 - swim_speed_cms[i] / V[i])
        
        # Thrust in erg/s
        thrust_erg_s = m * W * w * swim_speed_cms[i] - (m * w ** 2 * swim_speed_cms[i]) / (2.0 * cos_theta)
        # Convert to N
        thrust_Nm = thrust_erg_s / 10000000.0
        thrust_N = thrust_Nm / (length_cm[i] / 100.0)  # length to meters
        
        # Vectorize by heading
        thrust_x[i] = thrust_N * np.cos(heading[i])
        thrust_y[i] = thrust_N * np.sin(heading[i])
    
    return thrust_x, thrust_y


@njit(fastmath=True, cache=True)
def _drag_kernel(fish_vel_x, fish_vel_y, water_vel_x, water_vel_y, length_m, 
                 surface_areas, drag_coeffs, density_kg_m3, wave_drag, mask, tired_mask):
    """Vectorized drag calculation kernel (Numba-optimized).
    
    Returns:
        drag_x, drag_y: drag force vectors in N (n_agents,)
    """
    n = fish_vel_x.shape[0]
    drag_x = np.zeros(n, dtype=np.float64)
    drag_y = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        if not mask[i]:
            continue
        
        # Reduce water velocity for tired fish
        wx = water_vel_x[i] * (0.2 if tired_mask[i] else 1.0)
        wy = water_vel_y[i] * (0.2 if tired_mask[i] else 1.0)
        
        # Relative velocity
        rel_vx = fish_vel_x[i] - wx
        rel_vy = fish_vel_y[i] - wy
        rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
        
        if rel_speed_sq < 1e-12:
            continue
        
        rel_speed = np.sqrt(rel_speed_sq)
        unit_x = rel_vx / rel_speed
        unit_y = rel_vy / rel_speed
        
        # Drag force magnitude
        drag_mag = -0.5 * density_kg_m3 * (surface_areas[i] / 10000.0) * drag_coeffs[i] * rel_speed_sq * wave_drag[i]
        
        # Cap at max_drag for tired fish
        if tired_mask[i] and abs(drag_mag) > 5.0:
            drag_mag = -5.0 if drag_mag < 0 else 5.0
        
        drag_x[i] = drag_mag * unit_x
        drag_y[i] = drag_mag * unit_y
    
    return drag_x, drag_y


class movement():
    def __init__(self, simulation_object):
        self.simulation = simulation_object

    # Webb/Empirical spline data (shared between thrust and frequency)
    LENGTH_DAT = np.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
    SPEED_DAT = np.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
    AMP_DAT = np.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
    WAVE_DAT = np.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
    EDGE_DAT = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])

    # cached splines
    _SPLINES = None

    @classmethod
    def _get_webb_splines(cls):
        if cls._SPLINES is None:
            A_spline = UnivariateSpline(cls.LENGTH_DAT, cls.AMP_DAT, k=2, ext=0)
            V_spline = UnivariateSpline(cls.SPEED_DAT, cls.WAVE_DAT, k=1, ext=0)
            B_spline = UnivariateSpline(cls.LENGTH_DAT, cls.EDGE_DAT, k=1, ext=0)
            cls._SPLINES = (A_spline, V_spline, B_spline)
        return cls._SPLINES

    def find_z(self):
        """
        Calculate the z-coordinate for an agent based on its depth and body depth.
        """
        # Optimized: direct indexing instead of chained np.where
        target_depth = self.simulation.body_depth * 3 / 100.
        shallow_mask = self.simulation.depth < target_depth
        
        self.simulation.z = target_depth.copy() if hasattr(target_depth, 'copy') else np.full_like(self.simulation.depth, target_depth)
        
        # Handle shallow depths (add too_shallow offset)
        too_shallow = np.asarray(self.simulation.too_shallow)
        if too_shallow.ndim == 0:
            # Scalar too_shallow
            self.simulation.z[shallow_mask] = self.simulation.depth[shallow_mask] + float(too_shallow)
        else:
            # Array too_shallow
            self.simulation.z[shallow_mask] = self.simulation.depth[shallow_mask] + too_shallow[shallow_mask]
        
        # Clamp negative values to 0
        self.simulation.z[self.simulation.z < 0] = 0

    def thrust_fun(self, mask, t, dt, fish_velocities=None):
        rho = 1.0  # density of freshwater
        theta = 32.  # theta that produces cos(theta) = 0.85
        length_cm = self.simulation.length / 1000 * 100.

        # Calculate swim speed
        water_vel = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
        if fish_velocities is None:
            if t == 0:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
            else:
                fish_x_vel = (self.simulation.X - self.simulation.prev_X) / dt
                fish_y_vel = (self.simulation.Y - self.simulation.prev_Y) / dt
                fish_velocities = np.stack((fish_x_vel, fish_y_vel)).T

        ideal_swim_speed = np.linalg.norm(fish_velocities - water_vel, axis=-1)
        swim_speed_cms = ideal_swim_speed * 100.

        # Interpolation (cached) using Webb empirical data
        A_spline, V_spline, B_spline = self._get_webb_splines()
        A = A_spline(length_cm)
        V = V_spline(swim_speed_cms)
        B = B_spline(length_cm)

        # Use JIT-compiled kernel for thrust calculation (10-20x faster)
        if _NUMBA_AVAILABLE:
            thrust_x, thrust_y = _thrust_kernel(
                np.ascontiguousarray(length_cm, dtype=np.float64),
                np.ascontiguousarray(swim_speed_cms, dtype=np.float64),
                np.ascontiguousarray(self.simulation.Hz, dtype=np.float64),
                np.ascontiguousarray(A, dtype=np.float64),
                np.ascontiguousarray(V, dtype=np.float64),
                np.ascontiguousarray(B, dtype=np.float64),
                float(rho),
                float(np.radians(theta)),
                np.ascontiguousarray(self.simulation.heading, dtype=np.float64),
                np.ascontiguousarray(mask, dtype=np.bool_)
            )
            thrust = np.stack((thrust_x, thrust_y), axis=1).astype(np.float32)
        else:
            # Fallback: vectorized NumPy implementation
            m = (np.pi * rho * B ** 2) / 4.
            W = (self.simulation.Hz * A * np.pi) / 1.414
            w = W * (1 - swim_speed_cms / V)
            thrust_erg_s = m * W * w * swim_speed_cms - (m * w ** 2 * swim_speed_cms) / (2. * np.cos(np.radians(theta)))
            thrust_Nm = thrust_erg_s / 10000000.
            thrust_N = thrust_Nm / (self.simulation.length / 1000.)
            thrust = np.where(mask[:, np.newaxis], np.stack((thrust_N * np.cos(self.simulation.heading),
                                     thrust_N * np.sin(self.simulation.heading)), axis=1), 0.0)

        self.simulation.thrust = thrust

        # optional debug print for thrust internals
        try:
            if getattr(self.simulation, 'debug_freq', False):
                n_dbg = min(5, thrust.shape[0])
                logger.debug('THRUST debug: thrust[:5]= %s', thrust[:n_dbg])
        except Exception:
            pass

    def frequency(self, mask, t, dt, fish_velocities=None, use_sympy=False):
        rho = 1.0
        theta = 32.
        lengths_cm = self.simulation.length / 10

        water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
        alternate = True

        if fish_velocities is None:
            if t == 0:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
            else:
                fish_x_vel = (self.simulation.X - self.simulation.prev_X) / dt
                fish_y_vel = (self.simulation.Y - self.simulation.prev_Y) / dt
                fish_velocities = np.stack((fish_x_vel, fish_y_vel)).T
                alternate = False

        swim_speeds_cms = np.linalg.norm(fish_velocities - water_velocities, axis=-1) * 100 + 0.00001

        # Interpolation (cached) using Webb empirical data
        A_spline, V_spline, B_spline = self._get_webb_splines()
        A = A_spline(lengths_cm)
        V = V_spline(swim_speeds_cms)
        B = B_spline(lengths_cm)

        # compute ideal drag vector for numeric D (N)
        ideal_drag = self.ideal_drag_fun(fish_velocities=fish_velocities)
        drag_force_N = np.linalg.norm(ideal_drag, axis=-1)
        swim_speed_m_s = swim_speeds_cms / 100.0
        # power in J/s (W) = N * m/s
        drags_J_s = np.where(mask, drag_force_N * swim_speed_m_s, 0.0)
        # legacy code used erg/s in places; keep drags_erg_s for backward compatibility
        drags_erg_s = drags_J_s * 1e7

        # baseline minimum Hz for certain swim behaviors (per original code)
        min_Hz = np.interp(self.simulation.length, [450, 7.5], [690, 2.])

        # Safe calculation of Hz: guard denominators and negative/near-zero values
        # numerator and denominator for the square-root expression
        # compute numerator and denominator in SI units (vectorized)
        # Use J/s (W) for power: drags_J_s is already J/s
        theta_rad = np.radians(theta)

        # convert spline outputs to SI-consistent units
        # A spline: amplitude values in the archived derivation are given as fraction/percent -> convert to meters
        # Empirically the Webb data in previous scripts divided amp_dat by 100 -> use the same convention
        A_m = A / 100.0
        # B (trailing edge span) historically provided as percent -> convert to meters
        B_m = B / 100.0
        # V from spline historically in cm/s -> convert to m/s
        V_m_s = V / 100.0
        # swim speed in m/s
        U_m_s = swim_speeds_cms / 100.0
        # density: rho was given as 1.0 (g/cm^3) in legacy code; convert to kg/m^3
        rho_si = rho * 1000.0

        # numerator (SI): D (J/s) * V^2 * cos(theta)
        num_si = drags_J_s * (V_m_s ** 2) * np.cos(theta_rad)

        # denominator (SI): A^2 * B^2 * U * pi^3 * rho * (U - V) * (U + 2*V*cos(theta) - V)
        small_val = 1e-20
        term1_si = (A_m ** 2) * (B_m ** 2) * U_m_s * (np.pi ** 3) * rho_si
        # Use (V - U) so denominator is positive when body wave speed V > swim speed U
        term2_si = (V_m_s - U_m_s)
        term3_si = (U_m_s + 2.0 * V_m_s * np.cos(theta_rad) - V_m_s)
        denom_si = term1_si * term2_si * term3_si

        # safe ratio and compute Hz in SI
        # compute ratio with guarded division to avoid warnings and infinities
        ratio = np.full_like(num_si, np.nan, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            mask_valid_denom = (denom_si != 0) & np.isfinite(denom_si)
            ratio = np.where(mask_valid_denom, num_si / denom_si, np.nan)

        # only accept positive ratios above a very small threshold
        safe_ratio = (ratio > small_val)
        Hz_raw = np.where(safe_ratio, np.sqrt(ratio), np.full_like(ratio, np.nan, dtype=float))

        # If requested, and sympy is available, compute Hz using the symbolic lambdified function
        if use_sympy and _SYMPY_AVAILABLE:
            try:
                # SymPy lambda expects A (m), B (m), V (m/s), U (m/s), D (J/s), rho (kg/m^3), theta (rad)
                # movement uses lengths in mm for self.simulation.length, convert appropriately
                A_m = A / 100.0 if np.any(A > 1.0) else A  # in case A is in cm-like units, but splines usually gave m
                B_m = B
                V_m_s = V / 100.0 if np.nanmax(V) > 10 else V  # V in movement was derived sometimes in cm/s; make safe
                U_m_s = swim_speeds_cms / 100.0
                D_J_s = drags_J_s
                rho_si = rho * 1000.0 if np.nanmax(rho) < 10 else rho  # sustain rho scaling if needed
                theta_rad = np.radians(theta)
                # call lambdified sympy function; it supports vectorized numpy inputs
                Hz_sym = _sympy_freq_func(A_m, B_m, V_m_s, U_m_s, D_J_s, rho_si, theta_rad)
                # ensure shape and finite values
                Hz_sym = np.where(np.isfinite(Hz_sym), Hz_sym, np.nan)
                Hz_raw = np.where(mask, Hz_sym, Hz_raw)
            except Exception:
                # if anything fails, leave Hz_raw as computed numerically
                pass

        # where swim behavior indicates minimum Hz, set to min_Hz
        Hz = Hz_raw.copy()
        Hz[self.simulation.swim_behav == 3] = min_Hz[self.simulation.swim_behav == 3]

        # if the numerator (power) is essentially zero, there is no thrust requirement -> no tailbeat
        # do not override minimum-Hz behavior (swim_behav == 3)
        zero_power_mask = (num_si <= (1e-12)) & (self.simulation.swim_behav != 3)
        Hz[zero_power_mask] = 0.0

        # stuck agents have zero Hz
        Hz[self.simulation.is_stuck] = 0.0

        # replace NaNs and infinities with a conservative value (min_Hz)
        invalid_mask = ~np.isfinite(Hz)
        Hz[invalid_mask] = min_Hz[invalid_mask]

        # finally, clip to a biologically plausible range (0.0 - 20.0 Hz)
        Hz = np.clip(Hz, 0.0, 20.0)

        # store prev and current Hz
        self.simulation.prev_Hz = getattr(self.simulation, 'Hz', np.zeros_like(Hz))
        self.simulation.Hz = Hz

        # Optional instrumentation for debugging frequency internals.
        # If the simulation has attribute `debug_freq` set to True, store
        # a compact diagnostics dict for the first few agents in `simulation.freq_debug`
        # and a per-step JSON-serializable history in `simulation.freq_terms_history`.
        try:
            if getattr(self.simulation, 'debug_freq', False):
                # number of agents in mask
                N_total = int(mask.sum())
                # snapshot first few agents for readability
                Nsnap = min(10, N_total)
                # build a compact, JSON-serializable diagnostics dict
                diag = {
                    'term1_si': np.asarray(term1_si[:Nsnap]).astype(float).tolist(),
                    'term2_si': np.asarray(term2_si[:Nsnap]).astype(float).tolist(),
                    'term3_si': np.asarray(term3_si[:Nsnap]).astype(float).tolist(),
                    'num_si': np.asarray(num_si[:Nsnap]).astype(float).tolist(),
                    'denom_si': np.asarray(denom_si[:Nsnap]).astype(float).tolist(),
                    'ratio_si': np.asarray(ratio[:Nsnap]).astype(float).tolist(),
                    'Hz_raw': np.asarray(Hz_raw[:Nsnap]).astype(float).tolist(),
                    'Hz': np.asarray(Hz[:Nsnap]).astype(float).tolist(),
                    'A_m': np.asarray(A_m[:Nsnap]).astype(float).tolist(),
                    'B_m': np.asarray(B_m[:Nsnap]).astype(float).tolist(),
                    'V_m_s': np.asarray(V_m_s[:Nsnap]).astype(float).tolist(),
                    'U_m_s': np.asarray(U_m_s[:Nsnap]).astype(float).tolist(),
                    'safe_ratio': np.asarray(safe_ratio[:Nsnap]).astype(bool).tolist(),
                    'counts': {
                        'N_total': N_total,
                        'N_safe_ratio': int(np.sum(safe_ratio)),
                        'N_invalid_ratio': int(N_total - int(np.sum(safe_ratio)))
                    }
                }

                # legacy single-step compact debug for quick inspection
                self.simulation.freq_debug = diag

                # maintain a short rolling history of these per-step diagnostics
                if getattr(self.simulation, 'freq_terms_history', None) is None:
                    self.simulation.freq_terms_history = [diag]
                else:
                    self.simulation.freq_terms_history.append(diag)
                    # cap history length to keep memory bounded
                    if len(self.simulation.freq_terms_history) > 200:
                        self.simulation.freq_terms_history.pop(0)
                    # optional probe: write CSV row(s) with raw Webb/term values for first few agents
                    try:
                        if getattr(self.simulation, 'debug_freq_probe', False):
                            probe_dir = os.path.join(os.path.dirname(__file__), '../../outputs')
                            probe_dir = os.path.abspath(probe_dir)
                            if not os.path.exists(probe_dir):
                                os.makedirs(probe_dir, exist_ok=True)
                            fname = os.path.join(probe_dir, f'freq_probe_{int(time.time())}.csv')
                            Np = min(10, N_total)
                            with open(fname, 'w', newline='') as cf:
                                writer = csv.writer(cf)
                                # header
                                writer.writerow(['step', 'agent_idx', 'A_m', 'B_m', 'V_m_s', 'U_m_s', 'term1_si', 'term2_si', 'term3_si', 'num_si', 'denom_si', 'ratio_si', 'Hz_raw', 'Hz'])
                                for ai in range(Np):
                                    writer.writerow([
                                        int(getattr(self.simulation, 'current_step', 0)),
                                        ai,
                                        float(A_m[ai]) if ai < len(A_m) else None,
                                        float(B_m[ai]) if ai < len(B_m) else None,
                                        float(V_m_s[ai]) if ai < len(V_m_s) else None,
                                        float(U_m_s[ai]) if ai < len(U_m_s) else None,
                                        float(term1_si[ai]) if ai < len(term1_si) else None,
                                        float(term2_si[ai]) if ai < len(term2_si) else None,
                                        float(term3_si[ai]) if ai < len(term3_si) else None,
                                        float(num_si[ai]) if ai < len(num_si) else None,
                                        float(denom_si[ai]) if ai < len(denom_si) else None,
                                        float(ratio[ai]) if ai < len(ratio) else None,
                                        float(Hz_raw[ai]) if ai < len(Hz_raw) else None,
                                        float(Hz[ai]) if ai < len(Hz) else None,
                                    ])
                            logger.info('Wrote probe CSV: %s', fname)
                    except Exception:
                        pass
        except Exception:
            # never raise from instrumentation
            pass

    def kin_visc(self, temp):
        kin_temp = np.array([0.01, 10., 20., 25., 30., 40., 50., 60., 70., 80.,
                             90., 100., 110., 120., 140., 160., 180., 200.,
                             220., 240., 260., 280., 300., 320., 340., 360.])
        kin_visc = np.array([0.00000179180, 0.00000130650, 0.00000100350,
                             0.00000089270, 0.00000080070, 0.00000065790,
                             0.00000055310, 0.00000047400, 0.00000041270,
                             0.00000036430, 0.00000032550, 0.00000029380,
                             0.00000026770, 0.00000024600, 0.00000021230,
                             0.00000018780, 0.00000016950, 0.00000015560,
                             0.00000014490, 0.00000013650, 0.00000012990,
                             0.00000012470, 0.00000012060, 0.00000011740,
                             0.00000011520, 0.00000011430])
        f_kinvisc = np.interp(temp, kin_temp, kin_visc)
        return f_kinvisc

    def calc_surface_area(self):
        """Estimate surface area from length using legacy power-law fit.

        Returns array shaped (n_agents,) matching `self.simulation.length`.
        """
        a = -0.143
        b = 1.881
        # legacy code used length in cm within the log; convert from mm->cm if length stored as mm
        # Here self.simulation.length is in mm in this codebase; convert to cm
        length_cm = (self.simulation.length / 1000.0) * 100.0
        surface_areas = 10 ** (a + b * np.log10(length_cm))
        return surface_areas

    def drag_coeff(self, reynolds):
        """Return drag coefficient interpolated from empirical Reynolds/drag table.

        Accepts scalar or array `reynolds` and returns same-shaped array.
        """
        reynolds_data = np.array([2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5])
        drag_data = np.array([0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
        return np.interp(reynolds, reynolds_data, drag_data)

    def wat_dens(self, temp):
        dens_temp = np.array([0.1, 1., 4., 10., 15., 20., 25., 30., 35., 40.,
                              45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
                              95., 100., 110., 120., 140., 160., 180., 200.,
                              220., 240., 260., 280., 300., 320., 340., 360., 373.946])
        density = np.array([0.9998495, 0.9999017, 0.9999749, 0.9997, 0.9991026,
                            0.9982067, 0.997047, 0.9956488, 0.9940326, 0.9922152,
                            0.99021, 0.98804, 0.98569, 0.9832, 0.98055, 0.97776,
                            0.97484, 0.97179, 0.96861, 0.96531, 0.96189, 0.95835,
                            0.95095, 0.94311, 0.92613, 0.90745, 0.887, 0.86466,
                            0.84022, 0.81337, 0.78363, 0.75028, 0.71214, 0.66709,
                            0.61067, 0.52759, 0.322])
        f_density = np.interp(temp, dens_temp, density)
        return f_density

    def drag_fun(self, mask, t, dt, fish_velocities=None):
        tired_mask = (self.simulation.swim_behav == 3)

        if fish_velocities is None:
            if t == 0:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
            else:
                fish_x_vel = (self.simulation.X - self.simulation.prev_X) / dt
                fish_y_vel = (self.simulation.Y - self.simulation.prev_Y) / dt
                fish_velocities = np.stack((fish_x_vel, fish_y_vel)).T

        fish_vel_x = fish_velocities[:, 0]
        fish_vel_y = fish_velocities[:, 1]
        water_vel_x = self.simulation.x_vel
        water_vel_y = self.simulation.y_vel

        viscosity = self.kin_visc(self.simulation.water_temp)
        density = self.wat_dens(self.simulation.water_temp)
        density_kg_m3 = density * 1000.0

        length_m = self.simulation.length / 1000.
        water_velocities = np.stack((water_vel_x, water_vel_y), axis=-1)
        reynolds_numbers = np.linalg.norm(water_velocities, axis=-1) * length_m / viscosity

        a = -0.143
        b = 1.881
        surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))

        drag_coeffs = self.drag_coeff(reynolds_numbers)

        # Use JIT-compiled kernel for drag calculation (10-20x faster)
        if _NUMBA_AVAILABLE:
            drag_x, drag_y = _drag_kernel(
                np.ascontiguousarray(fish_vel_x, dtype=np.float64),
                np.ascontiguousarray(fish_vel_y, dtype=np.float64),
                np.ascontiguousarray(water_vel_x, dtype=np.float64),
                np.ascontiguousarray(water_vel_y, dtype=np.float64),
                np.ascontiguousarray(length_m, dtype=np.float64),
                np.ascontiguousarray(surface_areas, dtype=np.float64),
                np.ascontiguousarray(drag_coeffs, dtype=np.float64),
                float(density_kg_m3),
                np.ascontiguousarray(self.simulation.wave_drag, dtype=np.float64),
                np.ascontiguousarray(mask, dtype=np.bool_),
                np.ascontiguousarray(tired_mask, dtype=np.bool_)
            )
            drags = np.stack((drag_x, drag_y), axis=1).astype(np.float32)
        else:
            # Fallback: original vectorized implementation
            water_velocities_adj = water_velocities * np.where(tired_mask[:, np.newaxis], 0.2, 1.0)
            fish_speeds = np.linalg.norm(fish_velocities, axis=-1)
            fish_speeds[fish_speeds == 0.0] = 0.0001
            fish_velocities_safe = fish_velocities.copy()
            fish_velocities_safe[fish_speeds == 0.0] = [0.0001, 0.0001]

            relative_velocities = fish_velocities_safe - water_velocities_adj
            relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1) ** 2

            rel_norms = np.linalg.norm(relative_velocities, axis=1)
            rel_norms_safe = np.where(rel_norms == 0, 1.0, rel_norms)
            unit_relative_vector = np.nan_to_num(relative_velocities / rel_norms_safe[:, np.newaxis])

            drags = np.where(mask[:, np.newaxis],
                             -0.5 * (density * 1000) * (surface_areas[:, np.newaxis] / 100 ** 2) \
                                           * drag_coeffs[:, np.newaxis] * relative_speeds_squared[:, np.newaxis] \
                                               * unit_relative_vector * self.simulation.wave_drag[:, np.newaxis], 0)

            max_drag_magnitude = 5.0
            drag_magnitudes = np.linalg.norm(drags, axis=1)
            excessive_drag_indices = np.logical_and(self.simulation.swim_behav == 3, drag_magnitudes > max_drag_magnitude)
            drags[excessive_drag_indices] = (drags[excessive_drag_indices].T * (max_drag_magnitude / drag_magnitudes[excessive_drag_indices])).T

        self.simulation.drag = drags

    def ideal_drag_fun(self, fish_velocities=None):
        water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)

        if fish_velocities is None:
            fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                        self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)

        ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)

        # `max_s_U` is stored in body-lengths/second (BL/s) but `ideal_swim_speeds`
        # is in m/s. Convert per-agent sustainable speeds into m/s using length.
        try:
            length_m = np.asarray(self.simulation.length, dtype=float) / 1000.0
        except Exception:
            length_m = np.ones_like(ideal_swim_speeds, dtype=float)
        length_m = np.where(length_m <= 0, 1.0, length_m)

        try:
            max_s_bl_s = np.asarray(self.simulation.max_s_U, dtype=float).reshape((-1,))
        except Exception:
            max_s_bl_s = np.full_like(ideal_swim_speeds, 2.77, dtype=float)
        if max_s_bl_s.size != ideal_swim_speeds.size:
            try:
                fill = float(np.nanmedian(max_s_bl_s))
            except Exception:
                fill = 2.77
            max_s_bl_s = np.full_like(ideal_swim_speeds, fill, dtype=float)

        max_s_fatigued_bl_s = max_s_bl_s
        try:
            msf = getattr(self.simulation, 'max_s_U_fatigued', None)
            if msf is not None:
                max_s_fatigued_bl_s = np.asarray(msf, dtype=float).reshape((-1,))
                if max_s_fatigued_bl_s.size != ideal_swim_speeds.size:
                    try:
                        fill = float(np.nanmedian(max_s_fatigued_bl_s))
                    except Exception:
                        fill = float(np.nanmedian(max_s_bl_s))
                    max_s_fatigued_bl_s = np.full_like(ideal_swim_speeds, fill, dtype=float)
        except Exception:
            max_s_fatigued_bl_s = max_s_bl_s

        max_s_m_s = max_s_bl_s * length_m
        max_s_fatigued_m_s = max_s_fatigued_bl_s * length_m

        refugia_mask = (self.simulation.swim_behav == 2) & (ideal_swim_speeds > max_s_m_s)
        holding_mask = (self.simulation.swim_behav == 3) & (ideal_swim_speeds > max_s_fatigued_m_s)
        too_fast = refugia_mask | holding_mask
        
        # Optimized: direct indexing instead of np.where for max_allowed
        max_allowed = max_s_m_s.copy()
        max_allowed[self.simulation.swim_behav == 3] = max_s_fatigued_m_s[self.simulation.swim_behav == 3]

        # ensure proper broadcasting: shape max_allowed as (n,1) so division
        # yields (n,1) and multiplies correctly with fish_velocities (n,2).
        denom = ideal_swim_speeds[:, np.newaxis]
        ratio = np.divide(max_allowed[:, np.newaxis], denom, out=np.ones_like(denom), where=denom != 0)
        
        # Only apply speed capping if there are any too_fast agents
        # (avoids broadcasting error when too_fast mask is empty)
        if np.any(too_fast):
            fish_velocities[too_fast] = ratio[too_fast] * fish_velocities[too_fast]

        self.simulation.max_practical_sog = fish_velocities

        viscosity = self.kin_visc(self.simulation.water_temp)
        density = self.wat_dens(self.simulation.water_temp)

        length_m = self.simulation.length / 1000.
        reynolds_numbers = np.linalg.norm(water_velocities, axis=-1) * length_m / viscosity

        a = -0.143
        b = 1.881
        surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))

        drag_coeffs = self.simulation.drag_coeff(reynolds_numbers)

        relative_velocities = self.simulation.max_practical_sog - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1) ** 2
        max_prac_norms = np.linalg.norm(self.simulation.max_practical_sog, axis=1)
        max_prac_norms_safe = np.where(max_prac_norms == 0, 1.0, max_prac_norms)
        unit_max_practical_sog = self.simulation.max_practical_sog / max_prac_norms_safe[:, np.newaxis]

        ideal_drags = -0.5 * (density * 1000) * (surface_areas[:, np.newaxis] / 100 ** 2) * drag_coeffs[:, np.newaxis] * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog * self.simulation.wave_drag[:, np.newaxis]

        return ideal_drags

    def swim(self, t, dt, pid_controller, mask):
        tired_mask = (self.simulation.swim_behav == 3)
        if t == 0:
            # If simulation provided `initial_fish_vel`, use it so agents start moving
            init_fv = getattr(self.simulation, 'initial_fish_vel', None)
            if init_fv is not None:
                try:
                    fish_vel_0 = np.array(init_fv, dtype=float)
                except Exception:
                    fish_vel_0_x = self.simulation.sog * np.cos(self.simulation.heading)
                    fish_vel_0_y = self.simulation.sog * np.sin(self.simulation.heading)
                    fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
                    fish_vel_0[~mask] = 0.0
            else:
                fish_vel_0_x = self.simulation.sog * np.cos(self.simulation.heading)
                fish_vel_0_y = self.simulation.sog * np.sin(self.simulation.heading)
                fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
                fish_vel_0[~mask] = 0.0
        else:
            fish_vel_0_x = (self.simulation.X - self.simulation.prev_X) / dt
            fish_vel_0_y = (self.simulation.Y - self.simulation.prev_Y) / dt
            fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T

        # Optimized: compute ideal velocity components directly
        ideal_vel_x = self.simulation.ideal_sog * np.cos(self.simulation.heading)
        ideal_vel_y = self.simulation.ideal_sog * np.sin(self.simulation.heading)
        ideal_vel = np.stack((ideal_vel_x, ideal_vel_y)).T
        ideal_vel[~mask] = 0.0

        surge_ini = self.simulation.thrust + self.simulation.drag
        acc_ini = np.round(surge_ini / self.simulation.weight[:, np.newaxis], 2)

        fish_vel_1_ini = fish_vel_0 + acc_ini * dt

        # Optimized: direct indexing for error calculation
        error = np.round(ideal_vel - fish_vel_1_ini, 12)
        error[~mask] = 0.0

        self.simulation.error = error
        # Mark dead if NaN
        self.simulation.dead[np.isnan(error[:, 0])] = 1

        if self.simulation.pid_tuning:
            pass
        else:
            k_p, k_i, k_d = pid_controller.PID_func(np.sqrt(np.power(self.simulation.x_vel, 2) + np.power(self.simulation.y_vel, 2)), self.simulation.length)
            pid_controller.k_p = np.array([1.])
            pid_controller.k_i = np.array([0.])
            pid_controller.k_d = np.array([0.])

        pid_adjustment = pid_controller.update(error, dt, None)
        self.simulation.integral = pid_controller.integral
        self.simulation.pid_adjustment = pid_adjustment

        # Optimized: compute fish_vel_1 then selectively disable PID for tired fish
        fish_vel_1 = fish_vel_0 + acc_ini * dt + pid_adjustment
        fish_vel_1[tired_mask] = fish_vel_0[tired_mask] + acc_ini[tired_mask] * dt
        
        # PHYSICS FIX: Fatigued fish (swim_behav==3) with insufficient thrust must fall back
        # Check if surge produces negative acceleration in forward direction (thrust < drag)
        fatigued_mask = (self.simulation.swim_behav == 3)
        heading_vec = np.column_stack((np.cos(self.simulation.heading), np.sin(self.simulation.heading)))
        
        # Project acceleration onto heading direction
        acc_forward = np.sum(acc_ini * heading_vec, axis=1)
        
        # Fatigued fish with negative forward acceleration cannot maintain position - they fall back
        fatigued_falling_back = fatigued_mask & (acc_forward < -0.01)  # threshold to avoid numerical noise
        
        # For falling back fish, disable PID entirely and let physics take over
        # They drift backward due to drag > thrust
        if np.any(fatigued_falling_back):
            fish_vel_1[fatigued_falling_back] = fish_vel_0[fatigued_falling_back] + acc_ini[fatigued_falling_back] * dt
            # Store fallback state for diagnostics
            if not hasattr(self.simulation, 'fish_falling_back'):
                self.simulation.fish_falling_back = np.zeros(self.simulation.num_agents, dtype=bool)
            self.simulation.fish_falling_back[:] = fatigued_falling_back
        
        # Zero velocity for dead fish
        fish_vel_1[self.simulation.dead == 1] = 0.0

        # return displacement (dx, dy) over this timestep
        dxdy = fish_vel_1 * dt

        # optional movement debug: dump per-step arrays (enqueue to diagnostics when possible)
        try:
            if getattr(self.simulation, 'debug_movement', False):
                outdir = getattr(self.simulation, 'model_dir', None) or './outputs'
                try:
                    ideal_drag = None
                    try:
                        ideal_drag = self.ideal_drag_fun()
                    except Exception:
                        ideal_drag = getattr(self.simulation, 'drag', None)

                    payload = dict(
                        dxdy=dxdy,
                        thrust=getattr(self.simulation, 'thrust', None),
                        drag=getattr(self.simulation, 'drag', None),
                        ideal_drag=ideal_drag,
                        Hz=getattr(self.simulation, 'Hz', None),
                        max_practical_sog=getattr(self.simulation, 'max_practical_sog', None),
                        X=self.simulation.X,
                        Y=self.simulation.Y,
                        heading=self.simulation.heading,
                    )
                    beh = getattr(self.simulation, 'behavior', None)
                    if beh is not None and hasattr(beh, '_safe_npz_dump'):
                        beh._safe_npz_dump(outdir, f'move_debug_step_{int(getattr(self.simulation, "current_step", t))}', payload)
                    else:
                        import os, time
                        os.makedirs(outdir, exist_ok=True)
                        fname = os.path.join(outdir, f'move_debug_step_{int(getattr(self.simulation, "current_step", t))}_{int(time.time())}.npz')
                        np.savez_compressed(fname, **payload)
                except Exception:
                    pass
        except Exception:
            pass

        # Apply mask to final displacement
        dxdy[~mask] = 0.0
        return dxdy

    def jump(self, t, g, mask):
        # Optimized: compute for all, then mask at end
        jump_angles = np.random.choice([np.radians(45), np.radians(60)], size=self.simulation.ucrit.shape)
        time_airborne = (2 * self.simulation.ucrit * np.sin(jump_angles)) / g
        displacement = self.simulation.ucrit * time_airborne * np.cos(jump_angles)
        dx = displacement * np.cos(self.simulation.heading)
        dy = displacement * np.sin(self.simulation.heading)
        dxdy = np.stack((dx, dy)).T
        
        # Check if any jump would land in nodata region (dry land)
        if np.any(mask):
            # Calculate landing positions
            landing_x = self.simulation.X + dx
            landing_y = self.simulation.Y + dy
            
            # Check if landing positions are valid (sample depth at landing spot)
            depth_ds = np.asarray(self.simulation._get_env('environment/depth', default=np.zeros((1, 1))), dtype=float)
            if depth_ds.ndim == 2 and depth_ds.size > 1:
                transform = getattr(self.simulation, 'depth_rast_transform', None)
                if transform is not None:
                    # Convert landing positions to raster coordinates
                    from affine import Affine
                    if isinstance(transform, Affine):
                        inv_transform = ~transform
                        landing_cols, landing_rows = inv_transform * (landing_x[mask], landing_y[mask])
                    else:
                        inv_transform = ~Affine.from_gdal(*transform)
                        landing_cols, landing_rows = inv_transform * (landing_x[mask], landing_y[mask])
                    
                    landing_rows = np.asarray(landing_rows, dtype=np.int32)
                    landing_cols = np.asarray(landing_cols, dtype=np.int32)
                    H, W = depth_ds.shape
                    
                    # Check for out-of-bounds or nodata landing spots
                    valid_landing = (landing_rows >= 0) & (landing_cols >= 0) & (landing_rows < H) & (landing_cols < W)
                    landing_rows_clipped = np.clip(landing_rows, 0, H - 1)
                    landing_cols_clipped = np.clip(landing_cols, 0, W - 1)
                    landing_depth = depth_ds[landing_rows_clipped, landing_cols_clipped]
                    
                    # Nodata in depth raster is typically |depth| > 9990
                    nodata_landing = np.abs(landing_depth) > 9990
                    invalid_landing = ~valid_landing | nodata_landing
                    
                    if np.any(invalid_landing):
                        # Fish jumped onto DRY LAND! Mark them for flopping behavior
                        jumping_agents = np.where(mask)[0]
                        landed_on_dry = jumping_agents[invalid_landing]
                        
                        # Mark as on_land and record landing time
                        self.simulation.on_land[landed_on_dry] = True
                        self.simulation.time_landed[landed_on_dry] = np.where(
                            np.isinf(self.simulation.time_landed[landed_on_dry]),
                            t,  # First time landing
                            self.simulation.time_landed[landed_on_dry]  # Keep original landing time
                        )
                        
                        # Initialize random flop heading for newly landed fish
                        newly_landed = landed_on_dry[np.isinf(self.simulation.time_landed[landed_on_dry])]
                        if len(newly_landed) > 0:
                            self.simulation.flop_heading[newly_landed] = np.random.uniform(0, 2*np.pi, len(newly_landed))
                        
                        print(f"WARNING: {len(landed_on_dry)} fish jumped onto DRY LAND at t={t}. They will flop around for {self.simulation.max_flop_time}s trying to find water.")
        
        # Apply mask and update time of jump
        dxdy[~mask] = 0.0
        self.simulation.time_of_jump[mask] = t
        
        return dxdy

    def flop(self, t, dt, mask):
        """
        Fish flopping on dry land - random movement trying to get back to water.
        Fish change flop direction every ~1 second (randomly).
        Returns displacement for flopping fish.
        """
        # Flop distance is small - fish thrashing around ~0.5 body lengths per second
        flop_speed = 0.5 * self.simulation.length  # m/s
        
        # Change flop heading randomly (roughly every 1 second)
        # Probability of changing direction this timestep
        change_heading_prob = dt / 1.0  # Average 1 second between direction changes
        should_change = np.random.random(self.simulation.num_agents) < change_heading_prob
        change_and_flopping = should_change & mask
        
        if np.any(change_and_flopping):
            self.simulation.flop_heading[change_and_flopping] = np.random.uniform(
                0, 2*np.pi, np.sum(change_and_flopping)
            )
        
        # Calculate flop displacement
        dx = flop_speed * dt * np.cos(self.simulation.flop_heading)
        dy = flop_speed * dt * np.sin(self.simulation.flop_heading)
        dxdy = np.column_stack((dx, dy))
        
        # Only return displacement for flopping fish
        dxdy[~mask] = 0.0
        
        return dxdy
