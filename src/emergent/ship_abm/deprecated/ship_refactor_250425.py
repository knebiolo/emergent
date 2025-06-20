# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:07:19 2025

@author: Kevin.Nebiolo
"""
from collections import OrderedDict
from itertools import product
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button            # ← NEW
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path as mplPath
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch
import os
import xml.etree.ElementTree as ET
import requests, zipfile, io
import geopandas as gpd
from shapely.geometry import Point, box, LineString, MultiPolygon
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon as ShapelyPolygon
import pandas as pd
from shapely.ops import nearest_points, unary_union
import networkx as nx
import fiona
import urllib3
import warnings
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import cKDTree
import time
from pyproj import Transformer
import shutil
import tempfile
import sys

# suppress SSL warnings, reduce noisy logging, and ignore non-critical warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('fiona').setLevel(logging.ERROR)
logging.getLogger('geopandas').setLevel(logging.ERROR)
logging.getLogger('shapely').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

log = logging.getLogger("emergent")
if not log.handlers:                                    # avoid dupes in Spyder
    h = logging.StreamHandler(sys.stdout)               # console only
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    h.setLevel(logging.INFO)                            # default level
    log.addHandler(h)
log.setLevel(logging.INFO)

# Configure logging
log_file_path = os.path.join(os.getcwd(), 'simulation_debug.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file_path)]  # Removed StreamHandler to avoid flush errors in certain environments
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
TILE_METRES = 2000          #   2 km × 2 km tiles
LRU_CAP     = 32             #   max live tiles (tune to GPU RAM)

def _bbox_to_tiles(xmin, ymin, xmax, ymax, size=TILE_METRES):
    """Return all (ix, iy) touching this bbox."""
    eps = 1e-6 * size         # tiny buffer to avoid numerical jitter
    ix0 = int(np.floor((xmin - eps) / size))
    ix1 = int(np.floor((xmax + eps) / size))
    iy0 = int(np.floor((ymin - eps) / size))
    iy1 = int(np.floor((ymax + eps) / size))
    return list(product(range(ix0, ix1 + 1), range(iy0, iy1 + 1)))

def pre_tile(enc_series: gpd.GeoSeries, tol: float = 10.0):

    """
    Pre-chunk ENC geometries onto tile grid.
    Returns dict {tile_id: GeoSeries}.
    """
    enc = enc_series.simplify(tol, preserve_topology=True)
    tile_map = {}
    for geom in enc.geometry:
        if geom.is_empty:
            continue
        for tid in _bbox_to_tiles(*geom.bounds):
            tile_map.setdefault(tid, []).append(geom)
    # collapse lists → GeoSeries + STRtree for hit-testing
    for tid, geoms in tile_map.items():
        gs   = gpd.GeoSeries(geoms, crs=enc_series.crs)
        tree = STRtree(gs.values)            # ← build explicit spatial index
        tile_map[tid] = dict(gs=gs, tree=tree)
    return tile_map


# ─────────────────────────────────────────────────────────────────────────────
class TileCache:
    """LRU cache that stores already-drawn PathCollections per tile."""
    def __init__(self, ax, tile_dict):
        self.ax   = ax
        self.tmap = tile_dict          # output of pre_tile
        self.art  = OrderedDict()      # {tile_id: PathCollection}

    def _draw_tile(self, tid):
        gs = self.tmap[tid]["gs"]
        # add the tile’s PatchCollection and capture the handle
        before = set(self.ax.collections)

        gs.plot(ax=self.ax, facecolor='#f2e9dc', edgecolor='none',
                linewidth=0.0, zorder=0, antialiased=False)
        # the new collection(s) are whatever appeared since “before”
        new_artists = list(set(self.ax.collections) - before)
        if not new_artists:
            return                      # nothing drawn (empty geo)
        pc = new_artists[0]            # first (only) PatchCollection
        self.art[tid] = pc
        self.art.move_to_end(tid)
        # evict LRU
        while len(self.art) > LRU_CAP:
            old_tid, old_pc = self.art.popitem(last=False)
            if old_pc in self.ax.collections:   # only remove if still live
                try:
                    old_pc.remove()
                except ValueError:
                    pass               # already gone → ignore
                    
    def ensure_visible(self, view_bbox):
        xmin, xmax = view_bbox.xmin, view_bbox.xmax
        ymin, ymax = view_bbox.ymin, view_bbox.ymax
        needed = set(_bbox_to_tiles(xmin, ymin, xmax, ymax))
        changed = False
        for tid in needed.difference(self.art.keys()):
            if tid in self.tmap:
                self._draw_tile(tid)
                changed = True
        return changed 
        # optionally remove off-screen early (LRU handles it lazily)

def north_south_current(state, t, speed=0.4):
    """
    Constant current flowing due south (–y in simulation coords).
    Parameters
    ----------
    state : ndarray (ignored, but lets us match the sim signature)
    t     : float   (ignored)
    speed : float   magnitude in m/s
    Returns
    -------
    ndarray (2, n) – [[u_current],[v_current]]
    """
    n = state.shape[1]
    # u = 0  ,  v = –speed  (all vessels)
    return np.tile(np.array([[0.0], [-speed]]), (1, n))


def playful_wind(state, t,
                 base=5.0,
                 gust_amp=3.0,
                 gust_period=120.0):
    """
    Very light-weight wind model:
      • speed  = base ± gust_amp·sin(2πt/period)
      • dir    = slowly-rotating global angle
    Good enough for testing controller wiring without worrying
    about realism (swap in WRF/etc. later).
    """
    n = state.shape[1]
    speed = base + gust_amp * np.sin(2 * np.pi * t / gust_period)
    theta = 0.25 * t * np.pi / 180.0           # ~0.25° s-¹ rotation
    wx, wy = speed * np.cos(theta), speed * np.sin(theta)
    return np.tile(np.array([[wx], [wy]]), (1, n))

# --- Vectorized 4-DOF Ship Model with Full FossenShip Methods ---
class ship:
    def __init__(self, *args, **kwargs):
        """
        Overloaded constructor:
        - Legacy: Ship(params: dict, n: int)
        - Spawn: Ship(state0: np.ndarray, pos0: np.ndarray, psi0: np.ndarray, goals_arr: np.ndarray, params: dict)
        """
        # Legacy initialization from parameters
        if len(args) >= 2 and isinstance(args[0], dict):
            params, n = args[0], args[1]
            self.n = n
            self._init_from_params(params, n)
            # state, pos, psi, goals will be set by simulation as needed
        # Spawn initialization from state arrays + explicit params
        elif len(args) >= 3 and 'goals_arr' in kwargs:
            state0, pos0, psi0 = args[0], args[1], args[2]
            goals_arr = kwargs['goals_arr']
            params = kwargs.get('params')
            if params is None or not isinstance(params, dict):
                raise TypeError("Spawn constructor requires a 'params' dict to initialize geometry")
            # infer fleet size
            self.n = state0.shape[-1] if hasattr(state0, 'ndim') and state0.ndim > 1 else 1
            # shared geometry and dynamics setup
            self._init_from_params(params, self.n)
            # assign initial state
            self.state = state0
            self.pos   = pos0
            self.psi   = psi0
            self.goals = goals_arr
        else:
            raise TypeError(
                "Invalid Ship constructor.\n"
                "Use Ship(params: dict, n: int) or\n"
                "    Ship(state0, pos0, psi0, goals_arr=..., params=...)."
            )
        
        # --- control dead-zone and predictive horizon parameters ---
        # trim_band_deg: dead-zone half-angle in degrees
        self.trim_band_deg = params.get('trim_band_deg', 3.0)
        # lead_time: prediction horizon for proactive dead-band
        self.lead_time      = params.get('lead_time', 0.0)
        # hysteresis state for predictive dead-band per agent
        # False = rudder currently engaged; True = in dead-zone off state
        self._deadzone_off_state = np.zeros(self.n, dtype=bool)
        
         # ───── extra aero/hydro constants ─────
        self.rho_air   = params.get('rho_air', 1.225)           # kg·m⁻³
        self.A_air     = params.get('A_air',
                                   self.length * self.beam * .7)  # proj. area above WL
        self.Cd_air    = params.get('Cd_air', 1.0)              # bluff-body
        self.Cd_water  = params.get('Cd_water', self.drag_coeff)

    def cut_power(self, idx: int):
        """
        Instantly sets RPM and desired speed to zero for vessel *idx*.
        """
        # Support both scalar and vector storage just in case.
        if np.ndim(self.commanded_rpm) == 0:        # single-vessel scalar
            self.commanded_rpm = 0.0
            self.desired_speed = 0.0
            self.integral      = 0.0
            self.prev_rudder   = 0.0
        else:                                       # normal vector case
            self.commanded_rpm[idx] = 0.0
            self.desired_speed[idx] = 0.0
            self.integral[idx]      = 0.0
            self.prev_rudder[idx]   = 0.0
        
        # log for debugging
        log.warning(f"Power cut to vessel {idx}")        # ASCII → avoids CP1252 gripe

    def _init_from_params(self, params: dict, n: int):
        """
        Shared setup for geometry, mass, hydrodynamics, and controllers.
        """
        # Vessel geometry
        self.length = params['length']
        self.beam   = params['beam']
        self.draft  = params['draft']
        
        # drag coefficient
        self.drag_coeff = 0.002

        # Block coefficient & approximate wetted area
        self.C_B = params.get('C_B', 0.80)
        self.A = ((self.length * self.beam)
                  + 2 * (self.beam * self.draft)
                  + 2 * (self.length * self.draft)) * self.C_B

        # Propeller diameter heuristic
        self.D = params.get('prop_diameter', self.beam / 4.0)

        # Hydrodynamic derivatives
        hydro_keys = ['Xu','Xv','Xp','Xr','Yu','Yv','Yp','Yr',
                      'Ku','Kv','Kp','Kr','Nu','Nv','Np','Nr']
        for key in hydro_keys:
            setattr(self, key, np.full(n, params.get(key, 0.0)))

        # Mass & inertia
        self.m   = params['m']
        self.Ixx = params['Ixx']
        self.Izz = params['Izz']
        self.xG  = params['xG']
        self.zG  = params['zG']
        H = np.zeros((4, 4))
        H[0,0] = self.m
        H[1,1], H[1,2], H[1,3] = self.m, -self.m * self.zG,  self.m * self.xG
        H[2,1], H[2,2]        = -self.m * self.zG, self.Ixx
        H[3,1], H[3,3]        =  self.m * self.xG, self.Izz
        self.H_inv = np.linalg.inv(H)

        # Rudder effectiveness & limits
        self.Ydelta = np.full(n, params.get('Ydelta', 0.0))
        self.Kdelta = np.full(n, params.get('Kdelta', 0.0))
        self.Ndelta = np.full(n, params.get('Ndelta', 1.3e6))
        self.max_rudder = np.radians(params.get('max_rudder_deg', 35.0))
        self.max_rudder_rate = np.radians(params.get('rudder_rate_deg_per_s', 10.0))
        self.smoothed_rudder = np.zeros(n)
        self.prev_rudder = np.zeros(n)
        self.rudder_tau = params.get('rudder_tau', 1.0)

        # Damping coefficients
        self.linear_damping = np.full((3, n), params.get('linear_damping', 0.0))
        self.quad_damping   = np.full((3, n), params.get('quad_damping',   0.0))

        # PID controller gains
        self.Kp = params.get('Kp_gain', 1.0)
        self.Ki = params.get('Ki_gain', 0.0)
        self.Kd = params.get('Kd_gain', 0.0)
        self.integral = np.zeros(n)
        self.prev_error = np.zeros(n)

        # Speed & propulsion settings
        init_sp = params.get('initial_speed', 0.0)
        self.desired_speed = np.full(n, params.get('desired_speed', init_sp))
        self.cruise_speed  = self.desired_speed.copy()
        self.max_speed     = np.full(n, params.get('max_speed', self.desired_speed * 1.2))
        self.rho = params.get('rho', 1025.0)
        self.K_T = params.get('K_T', 0.5)
        self.max_rpm = params.get('max_rpm', 90.0)
        self.max_rpm_rate = params.get('max_rpm_rate', 5.0)
        rpm0 = self.speed_to_rpm(init_sp)
        self.commanded_rpm = np.full(n, rpm0, dtype=float)
        self.prev_rpm = np.copy(self.commanded_rpm)
        self.n_prop = 2

        # Collision radii
        self.radii = np.full(n, params.get('radii', 10.0))

    def speed_to_rpm(self, u):
        """
        Compute the propeller RPM that exactly balances drag at speed u.
        Requires physical parameters rho, D, and K_T to be > 0.
        """
        # required thrust to hold speed: T_req = -drag(u)
        T_req = -self.drag(u)
        T_req = np.maximum(T_req, 0.0)
        # invert T_prop = rho * n^2 * D^4 * K_T
        denom = self.rho * (self.D ** 4) * self.K_T
        if denom <= 0.0:
            raise ValueError("rho, D, and K_T must all be > 0 to compute speed_to_rpm")
        # compute revolutions per second, then convert to RPM
        n = np.sqrt(T_req / denom)
        rpm = n * 60.0
# =============================================================================
#         print("DBG speed_to_rpm:", 
#               "U=", u, 
#               "T_req=", T_req, "denom=", denom)
# =============================================================================
        #return rpm
        return np.clip(rpm, 0.0, self.max_rpm)
        
    def estimate_mass(self, lengths):
        return 8.0 * lengths**2.2
    
    def _compute_prop_diameter(self):
        """
        D = 0.2 * L, but capped at 0.9 * B to ensure it fits under the hull.
        """
        raw_D = 0.2 * self.length
        max_D = 0.33 * self.beam
        return min(raw_D, max_D)

    def thrust(self, rpm):
        """
        Classic momentum-theory thrust:
          T = rho * n^2 * D^4 * K_T,
        where n = revolutions per second.
        """
        n = np.clip(rpm, 0, self.max_rpm) / 60.0
        T_single = self.K_T * self.rho * n**2 * self.D**4
        total_T  = self.n_prop * T_single
        T = np.where(rpm == 0,0.0, total_T)
        return T

    def drag(self, u):
        """
        Compute quadratic drag force in newtons:
          F_d = -½ * ρ * C_d(u) * A * u²
        where
          C_d(u) = self.drag_coeff * (1 + 0.8*|u|)
          A      = vessel’s wetted area (self.A)
        """

        # angle-dependent drag coefficient
        C_d = self.drag_coeff * (1 + 0.8 * np.abs(u))
        # quadratic drag force (N)
        return -0.5 * self.rho * C_d * self.A * u**2

    def environmental_forces(self, wind_vec, current_vec, u, v, psi):
        """
        Convert *velocity* fields (wind & current) into body-frame forces.

        Parameters
        ----------
        wind_vec, current_vec : (2, n) arrays of env. velocity in Earth frame
        u, v : (n,) surge/sway speeds (body frame)
        psi  : (n,) heading of each vessel (rad)
        Returns
        -------
        wind_force, current_force : (2, n) forces in SURGE/SWAY (body) axes
        """
        # --- ship velocity in Earth frame ---------------------------------
        u_e = u*np.cos(psi) - v*np.sin(psi)
        v_e = u*np.sin(psi) + v*np.cos(psi)

        def _force(rel_vec, rho, Cd, A):
            mag   = np.hypot(rel_vec[0], rel_vec[1])
            mag   = np.where(mag < 1e-6, 1e-6, mag)          # avoid /0
            dir_x = rel_vec[0] / mag
            dir_y = rel_vec[1] / mag
            F     = 0.5 * rho * Cd * A * mag**2              # scalar
            Fx_e, Fy_e = F*dir_x, F*dir_y                    # Earth frame
            # rotate to body frame (surge = +Xfwd, sway = +Yport)
            X_b =  Fx_e*np.cos(psi) + Fy_e*np.sin(psi)
            Y_b = -Fx_e*np.sin(psi) + Fy_e*np.cos(psi)
            return np.vstack([X_b, Y_b])

        rel_wind    = wind_vec    - np.vstack([u_e, v_e])
        rel_current = current_vec - np.vstack([u_e, v_e])

        wind_force    = _force(rel_wind,    self.rho_air, self.Cd_air,   self.A_air)
        current_force = _force(rel_current, self.rho,     self.Cd_water, self.A)
        return wind_force, current_force      # shape (2, n) each
    
    def coriolis_matrix(self, u, v):
        c2 = -self.m * v; c3 = self.m * u; c1 = np.zeros(self.n)
        return np.vstack([c1, c2, c3])

    def quadratic_damping(self, v, r):
        return np.vstack([np.zeros(self.n), 500*np.abs(v)*v, 1e6*np.abs(r)*r])

    def compute_rudder_torque(self, rudder_angle, lengths, u):
        return rudder_angle * lengths * np.maximum(u,0)

    def compute_attractive_force(self, positions, goals):
        dirs  = goals - positions
        norms = np.linalg.norm(dirs, axis=0)
        safe  = norms.copy()
        safe[safe<1e-6] = 1e-6   # avoid zero‐division
        uv    = dirs / safe
        return 500 * uv

    def compute_desired(self, goals, x, y, u, v, r, psi):
        """
        Compute desired heading & speed toward waypoint, reusing
        compute_attractive_force for the direction vector.
        """
        # — if no explicit goals given, default to the next route waypoint
        if goals is None and hasattr(self, 'wpts') and len(self.wpts) > 1:
            # wpts is a list of numpy [x,y]; take the second entry as our single goal
            goals = [ tuple(self.wpts[1]) ]

        # — stack x,y into shape (2, n)
        positions = np.vstack([x, y])

        # — normalize goals into a (2×n) NumPy array
        if isinstance(goals, list):
            coord_list = [ (g.x, g.y) if hasattr(g, 'x') else tuple(g) for g in goals ]
            goals_arr = np.array(coord_list).T        # shape = (2, n)
        else:
            goals_arr = np.atleast_2d(goals)
            if goals_arr.shape[0] != 2:
                goals_arr = goals_arr.T

        # get the attraction vector (2,n)
        attract = self.compute_attractive_force(positions, goals_arr)
        # heading toward the goal is the arctan2 of that vector
        hd = np.arctan2(attract[1], attract[0])
        # desired speed stays whatever you’ve set in self.desired_speed
        sp = np.array(self.desired_speed, copy=True)
        return hd, sp

    # === Updated PID Controller ===
    def pid_control(self, psi, hd, dt):
        """
        P-I-D with gain scheduling, anti-windup, dead-band, and rate-limit.
        """
        # 1) heading error normalized to [-π, π]
        err = (hd - psi + np.pi) % (2*np.pi) - np.pi

        # 1.1) small dead-band: ignore tiny errors < 0.5°
        db_rad = np.deg2rad(0.5)
        mask_db = np.abs(err) < db_rad
        err[mask_db] = 0.0

        # 1.2) gain-schedule: Kp ramps from 50% at 5° → 100% at 15°
        medium = np.deg2rad(5.0)
        large  = np.deg2rad(15.0)
        fac = np.clip((np.abs(err) - medium) / (large - medium), 0.0, 1.0)
        Kp_eff = self.Kp * (0.5 + 0.5 * fac)

        # 2) derivative term
        derr = (err - self.prev_error) / dt
        derr[mask_db] = 0.0  # prevent spikes when exiting dead-band

        # 3) compute raw command
        raw = Kp_eff * err + self.Ki * self.integral + self.Kd * derr
        # --- predictive dead-band: if in lead_time seconds we'll be within trim_band, zero rudder ---
        trim_rad   = np.radians(self.trim_band_deg)
        err_pred   = err + derr * self.lead_time
        release_m  = np.abs(err_pred) < trim_rad
        raw[release_m] = 0
        
        # hysteresis around dead-zone
        inner = np.radians(self.trim_band_deg)
        outer = np.radians(self.trim_band_deg + 1.0)  # 1° hysteresis
        staying_off = np.abs(err_pred) < inner
        coming_on    = np.abs(err_pred) > outer
        # --- predictive dead-band with hysteresis update ---
        # load previous off-state (boolean array)
        state_off = self._deadzone_off_state
        # predict future error
        err_pred = err + derr * self.lead_time
        # dead-zone radii
        trim_rad = np.radians(self.trim_band_deg)
        inner    = trim_rad
        outer    = np.radians(self.trim_band_deg + 1.0)  # 1° hysteresis
        # determine stay-off vs. re-engage
        staying_off = np.abs(err_pred) < inner
        coming_on   = np.abs(err_pred) > outer
        # compute new off-state and ensure it's boolean
        new_off = np.where(staying_off, True,
                   np.where(coming_on, False, state_off))
        new_off = new_off.astype(bool)
        # apply dead-zone: zero out rudder where off-state is True
        raw[new_off] = 0
        # persist off-state for next call
        self._deadzone_off_state = new_off
        
        # 4) clamp to rudder limits
        des = np.clip(raw, -self.max_rudder, self.max_rudder)

        # 5) anti-windup: integrate only when not saturated
        windup_mask = np.abs(des) < self.max_rudder
        self.integral[windup_mask] += err[windup_mask] * dt

        # 6) rate-limit: max change ±(max_rudder_rate·dt)
        max_delta = self.max_rudder_rate * dt
        rud = np.clip(
            des,
            self.prev_rudder - max_delta,
            self.prev_rudder + max_delta
        )

        # 7) low-pass filter the commanded rudder
        # alpha = dt / (self.rudder_tau + dt)
        # self.smoothed_rudder += alpha * (rud - self.smoothed_rudder)
        self.smoothed_rudder = rud
        
        # 8) update state
        self.prev_error  = err
        # update prev_error only outside the dead-band
        # self.prev_error[~mask_db] = err[~mask_db]
        self.prev_rudder = rud

        return self.smoothed_rudder

    def colregs(self, positions, nu, psi, current_rpm):
        """
        Simplified COLREGS for n vessels:
          - “Unlock” first when the contact moves outside your avoidance cone
          - Overtake / head‐on / crossing maneuvers next
          - Stand‐on otherwise
    
        Returns:
          head  : np.ndarray (n,)  → desired heading
          rpm   : np.ndarray (n,)  → commanded RPM toward scenario‐optimal speed
          role  : list of str      → 'give_way', 'stand_on', or 'neutral'
        """
        # ───── TUNE THESE ────────────────────────────────────
        safe_dist  = 2000.0     # begin avoidance if another ship within this (m)
        clear_dist =  500.0     # for future‑position clearance (unused)
        unlock_ang = np.radians(15)  # if |bearing| > 15°, go back to neutral
        #───────────────────────────────────────────────────────

        n         = self.n
        head      = np.zeros(n)
        # start from current cruise speeds, to be overridden per scenario
        speed_des = self.desired_speed.copy()
        role      = ['neutral'] * n

        for i in range(n):
            pd    = positions[:, i]
            psi_i = psi[i]
            hd    = psi_i         # default: hold heading
            rl    = role[i]       # default: neutral

            for j in range(n):
                if i == j:
                    continue
                delta = positions[:, j] - pd
                dist  = np.linalg.norm(delta)
                if dist > safe_dist:
                    continue

                bearing = ((np.arctan2(delta[1], delta[0])
                           - psi_i + np.pi) % (2*np.pi) - np.pi)

                # CROSSING FROM STARBOARD (Rule 15)
                if 0 < bearing < np.pi/2:
                    rl = 'give_way'
                    hd = (psi_i - 2*np.pi/3) % (2*np.pi)
                    sf = max(0.3, min(1.0, dist/safe_dist))
                    speed_des[i] = self.desired_speed[i] * sf

                # HEAD‑ON (Rule 14)
                elif abs(bearing) < np.radians(10):
                    rl = 'give_way'
                    hd = (psi_i - np.pi/3) % (2*np.pi)
                    speed_des[i] = 5. #max(self.desired_speed[i] * 0.3, 2.0)

                # OVERTAKING (Rule 13 variant)
                elif -np.pi/8 < bearing < np.pi/8:
                    rl = 'give_way'
                    hd = (psi_i + np.pi/18) % (2*np.pi)
                    speed_des[i] = min(self.desired_speed[i] * 1.2, self.max_speed[i])

                # STAND‑ON (Rule 17)
                else:
                    rl = 'stand_on'
                    # retains default speed_des

            # neutral or stand‑on → resume cruise
            if rl in ['neutral', 'stand_on']:
                speed_des[i] = self.desired_speed[i]

            head[i] = hd
            role[i] = rl

        # single conversion: desired speeds → RPM
        rpm = self.speed_to_rpm(speed_des)
        
        # DEBUG
        #print(f"[COLREGS QC] roles = {role}")

        return head, speed_des, rpm, role

    def reset_behavior(self):
        """
        Reset vessel behavior when its role changes.
        """
        self.current_speed = self.desired_speed
        self.current_heading = self.default_heading
        log.debug(f"[Vessel {self.ID}] Behavior reset to default values.")

    def step(self, state, rpms, goals, wind, current, dt):
        # state: [x;y;u;v;p;r;phi;psi]
        x,y,u,v,p,r,phi,psi = state
        # COLREGS & control
        desired_heading, desired_speed = self.compute_desired(goals, x, y, u, v, r, psi)
        rudder = self.pid_control(psi, desired_heading, dt)
        rpms = self.adjust_rpm(rpms, desired_speed, u, dt)
        # Propulsion + drag + environment
        thrust = self.calculate_thrust(rpms)
        drag = self.compute_drag(u)
        env = self.environmental_forces(wind, current)
        # Rudder torque
        rudder_tau = self.compute_rudder_torque(rudder, state)
        # Hydrodynamic
        Xh = self.Xu*u + self.Xv*v + self.Xp*p + self.Xr*r
        Yh = self.Yu*u + self.Yv*v + self.Yp*p + self.Yr*r
        Kh = self.Ku*u + self.Kv*v + self.Kp*p + self.Kr*r
        Nh = self.Nu*u + self.Nv*v + self.Np*p + self.Nr*r
        tau = np.vstack([Xh + thrust + drag + env[0],
                         Yh + env[1],
                         Kh,
                         Nh + rudder_tau])
        # Coriolis/centripetal
        C0 = self.m*(v*r + self.xG*r**2 - self.zG*p*r)
        C1 = -self.m*u*r
        C2 = self.m*self.zG*u*r
        C3 = -self.m*self.xG*u*r
        C = np.vstack([C0,C1,C2,C3])
        acc = self.H_inv.dot(tau + C)
        u += acc[0]*dt; v += acc[1]*dt; p += acc[2]*dt; r += acc[3]*dt
        # Kinematics
        phi += p*dt
        psi += r*np.cos(phi)*dt
        x += (u*np.cos(psi) - v*np.sin(psi))*dt
        y += (u*np.sin(psi) + v*np.cos(psi))*dt
        return np.vstack([x,y,u,v,p,r,phi,psi]), rpms
    
    def dynamics(self, state, prop_thrust, drag_force, wind_force, current_force, rudder_angle):
        """
        Vectorized 4-DOF dynamics:
        state: (4,n) = [u, v, p, r]
        prop_thrust: (n,)
        drag_force:  (n,)
        wind_force:  (2,n)
        current_force:(2,n)
        rudder_angle:(n,)
        Returns: u_dot, v_dot, p_dot, r_dot (each (n,))
        """
        # Surge and sway forces
        X = prop_thrust + drag_force + wind_force[0] + current_force[0]
        Y =                 wind_force[1] + current_force[1]
        # Moments from rudder
        K = self.Kdelta * rudder_angle
        N = self.Ndelta * rudder_angle
        # Damping (linear + quadratic) applied only to u, v, r
        u, v, p, r = state
        state_uvr = state[[0,1,3], :]  # u, v, r only
        # lin = self.linear_damping * state_uvr
        # quad = self.quad_damping * state_uvr * np.abs(state_uvr)
        lin = np.zeros_like(state_uvr)
        quad = np.zeros_like(state_uvr)
        # --- DEBUG: break down all the force terms ---
        #print(f"[DYN-DBG] prop_thrust = {prop_thrust}")
        # print(f"[DYN-DBG] drag_force = {drag_force}")
        # print(f"[DYN-DBG] wind_force_x = {wind_force[0]}")
        # print(f"[DYN-DBG] current_force_x = {current_force[0]}")
        X = prop_thrust + drag_force + wind_force[0] + current_force[0]
        # print(f"[DYN-DBG] X (sum)    = {X}")
        # print(f"[DYN-DBG] lin_damp   = {lin[0]}")
        # print(f"[DYN-DBG] quad_damp  = {quad[0]}")
        
        surge_tau = X - lin[0] - quad[0]
        # print(f"[DYN-DBG] Tau_surge  = {surge_tau}")
        
        # Optional: show the inverse‐mass term
        # print(f"[DYN-DBG] H_inv[0,0] = {self.H_inv[0,0]}")

        # Then acc = H_inv @ Tau …

        # Net generalized forces
        Tau = np.vstack([
            X - lin[0] - quad[0],  # surge
            Y - lin[1] - quad[1],  # sway
            K,                     # roll
            N                      # yaw
        ])  # shape (4,n)
        # Accelerations
        acc = self.H_inv @ Tau  # (4,n)
        u_dot, v_dot, p_dot, r_dot = acc
        return u_dot, v_dot, p_dot, r_dot

class simulation:
    def __init__(
        self,
        params,
        wind_fn,
        current_fn,
        tide_fn,
        minx=None,
        maxx=None,
        miny=None,
        maxy=None,
        enc_catalog_url=None,
        dt=0.1,
        T=100,
        obs_threshold=None,
        obs_gain=None,
        n_agents=None,
        spawn_speed=5.0,
        spawn_waypoints=20,
        coast_simplify_tol = 50.0,
        light_bg = True,
        verbose=False
    ):
        """
        Initialize simulation with ENC data and optionally spawn agents.

        Parameters:
        ----------
        wind_fn, current_fn, tide_fn : callables
            Environmental forcings (x, y, t) -> vector.
        minx, maxx, miny, maxy : float
            Domain bounding box in lon/lat.
        enc_catalog_url : str
            Path or URL to NOAA ENC catalog XML.
        dt : float
            Time step (s).
        T : float
            Total simulation duration (s).
        obs_threshold : float, optional
            Distance threshold for obstacle avoidance (m).
        obs_gain : float, optional
            Gain for obstacle repulsion.
        n_agents : int, optional
            Number of agents to spawn immediately.
        spawn_speed : float
            Initial surge speed for spawned agents (m/s).
        spawn_waypoints : int
            Number of waypoints per agent route.
        """
        # Store parameters & environment functions
        self.params = params
        self.wind_fn = wind_fn
        self.current_fn = current_fn
        self.tide_fn = tide_fn
        self.dt = dt
        self.t = 0
        self.steps = int(T / dt)
        self.n = n_agents
        self.in_deadband = np.zeros(self.n, dtype=bool)    # Are we currently in the dead-band?
        self.entry_sign  = np.zeros(self.n, dtype=float)   # Which side (±1) we locked in on entry
        self.hyst_done   = np.zeros(self.n, dtype=bool)    # Have we already exited once?
        self.prev_rudder = np.zeros(n_agents, dtype=float)
        self.prev_sign   = np.zeros(self.n, dtype=float)  # last step’s sign(err_vec)
        self.prev_abs_err  = np.full(self.n, np.inf)
        self.tile_dict = {}   # always defined (ENC or not)
        self.tcache    = None # will hold TileCache only for ENC runs
        self.light_bg = light_bg                 # ← new flag
        self.coast_simplify_tol = coast_simplify_tol
     
        # Base polygon from ship scale
        self.L = getattr(ship, 'length', 400.0)
        self.B = getattr(ship, 'beam', 60.0)
        self._ship_base = np.array([
            [-self.L/2, -self.B/2],
            [-self.L/2,  self.B/2],
            [ self.L/4,  self.B/2],
            [ self.L/2,  0.0      ],
            [ self.L/4, -self.B/2]
        ])

        # Determine or default domain bounds
        if None in (minx, maxx, miny, maxy):
            if n_agents:
                # default 2km square around origin
                minx, maxx = -2000.0, 2000.0
                miny, maxy = -2000.0, 2000.0
                log.warning("No domain bounds provided; using [-2km,2km]×[-2km,2km].")
            else:
                raise ValueError("Must provide domain bounds or n_agents>0 for default domain.")
        self.bounds = (minx, miny, maxx, maxy)

        # Compute UTM CRS from domain center
        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2
        utm_zone = int((midx + 180) // 6) + 1
        utm_epsg = 32600 + utm_zone  # Northern hemisphere
        self.crs_utm = f"EPSG:{utm_epsg}"
        
        # enable follow‐ship zoom (meters from ship center)
        self.dynamic_zoom = False
        self.zoom = 5000    # e.g. ±2 km view radius

        # Initialize matplotlib figure in true UTM (meters)
        # ── SPLIT FIGURE INTO TWO PANELS (MAP + CONTROL/STATUS) ─────────────────────
        plt.ion()
        self.fig = plt.figure(figsize=(6, 4), dpi = 300, constrained_layout = True)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08)

        # Left panel: UTM map
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_map.set_aspect('equal', adjustable='box')
        # Alias for backward compatibility (so old code referring to self.ax still works)
        self.ax = self.ax_map

        # Right panel: controls/status (no ticks, no frame)
        self.ax_panel = self.fig.add_subplot(gs[0, 1])
        self.ax_panel.set_xticks([]);  self.ax_panel.set_yticks([])
        self.ax_panel.set_frame_on(False)
        
        # # ── WIND‐VANE ARROW (in self.ax_panel) ────────────────────────────────────
        # from matplotlib.patches import FancyArrowPatch, Circle
        # # Use exactly one “panel‐tail” so that creation+update match
        # self._panel_tail = (0.5, 0.85)
        # self.wind_arrow = FancyArrowPatch(
        #     posA=self._panel_tail,
        #     posB=self._panel_tail,        # start collapsed at the tail
        #     arrowstyle='-|>',
        #     mutation_scale=15,
        #     linewidth=2.5,
        #     color='purple',
        #     transform=self.ax_panel.transAxes,
        #     zorder=5,
        #     animated=True
        # )
        # self.ax_panel.add_patch(self.wind_arrow)

        # # ── HEADER TEXT (time / Wind / Current / Tide) ───────────────────────────
        # self.header_text = self.ax_panel.text(
        #     0.02, 0.95,
        #     "",                           # updated once per frame
        #     transform=self.ax_panel.transAxes,
        #     fontsize=10,
        #     va='top',
        #     animated=True
        # )

        # ── LOG LINES (stack of up to N lines, bottom half of panel) ───────────────
        self.log_lines = []                    # store most‐recent messages
        self.max_log_lines = 5
        self.log_text_artists = []
        for i in range(self.max_log_lines):
            txt = self.ax_panel.text(
                0.02, 0.60 - i * 0.05,           # stagger 5% down each line
                "",
                transform=self.ax_panel.transAxes,
                fontsize=8,
                color='black',
                va='top',
                animated=True
            )
            self.log_text_artists.append(txt)

        # Reproject lon/lat bbox into UTM and set axes accordingly
        ll_box = box(minx, miny, maxx, maxy)
        utm_box = (
            gpd.GeoDataFrame(geometry=[ll_box], crs="EPSG:4326")
            .to_crs(self.crs_utm)
        )
        x0, y0, x1, y1 = utm_box.total_bounds
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.tick_params(labelsize=8)
        for tl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            tl.set_fontfamily('serif')
        # Store UTM extents for interactive routing
        self.minx, self.maxx = x0, x1
        self.miny, self.maxy = y0, y1

        # Initialize trajectory lines
        self.traj = [
            self.ax.plot([], [], linestyle='--', linewidth=1, color='orange', zorder=6)[0]
            for _ in range(n_agents)
        ]
        
        # Save spawn parameters for programmatic entrance
        self.spawn_speed = spawn_speed
        self.spawn_waypoints = spawn_waypoints
        
        # Obstacle avoidance parameters
        domain_width = maxx - minx
        self._coast_paths = None
        self.collision_tol_area = 1.0       
        domain_width = maxx - minx
        self.obs_threshold = obs_threshold or (0.1 * domain_width)
        self.obs_gain = obs_gain or 1e5
        # PRM sampling: how many random free‐space nodes to add per agent
        self.prm_samples = params.get('prm_samples', 100)


        # Placeholders for agent state and plotting elements
        self.pos = None  # 2×n array in meters
        self.psi = None  # n array of headings
        self.state = None  # 4×n state array
        self.goals = None  # 2×n goal positions in meters
        self.patches = []  # ship patches
        self.traj = []     # trajectory lines
        self.texts = []    # text labels
        self.history = []  # position histories
        self.rudder_lines = []
        self.danger_cones = []
        self.heading_arrows = []
 
        # 1) If ENC URL provided, delegate to helper
        if enc_catalog_url:
            self.load_enc_features(enc_catalog_url, verbose=verbose)
            
            # Build one composite GeoSeries for tiling (land ⊕ shoal areas)
            land_gdf  = self.enc_data.get("LNDARE")
            #shoal_gdf = self.enc_data.get("DEPARE")
            coast_geo = []
            if land_gdf is not None and not land_gdf.empty:
                coast_geo.extend(land_gdf.geometry)
            if not self.light_bg:
                shoal_gdf = self.enc_data.get("DEPARE")
                if shoal_gdf is not None and not shoal_gdf.empty:
                    coast_geo.extend(shoal_gdf.geometry)
            coast_series = gpd.GeoSeries(coast_geo, crs=self.crs_utm)
        
            self.tile_dict = pre_tile(coast_series, tol=10.0)   # once
            self.tcache    = TileCache(self.ax, self.tile_dict)
            # Draw nautical-chart background once
            self._draw_background()            

            # Set plot limits based on input bbox reprojected to UTM
            bbox_ll = box(minx, miny, maxx, maxy)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_ll], crs='EPSG:4326')
            bbox_utm = bbox_gdf.to_crs(self.crs_utm)
            xmin_u, ymin_u, xmax_u, ymax_u = bbox_utm.total_bounds
            self.ax.set_xlim(xmin_u, xmax_u)
            self.ax.set_ylim(ymin_u, ymax_u)

            # Spawn agents after UTM setup
            if n_agents:
                self.ship = ship(self.params, self.n)
                state0, pos0, psi0, goals = self.spawn()

            # Now everything is in meters, matching `self.waterway` and route()
            self.state  = state0.astype(float)
            self.pos    = pos0
            self.psi    = psi0
            self.goals  = goals
            self.prev_psi      = self.psi.copy()

            #--- Setup steering controller state & tuning ---
            # prev_psi for turn-rate calculation
            self.prev_psi = psi0.copy()
            # integral of heading error for I-term
            self.integral_error = np.zeros_like(psi0)
            
            self.tuning = {
                'Kp': params['Kp_gain'],
                'Ki': params['Ki_gain'],
                'Kd': params['Kd_gain'],
                'Kf': params.get('Kf_gain', 1.0),           # feed-forward gain
                'r_rate_max_deg': params.get('r_rate_max_deg', 8.0),
                'I_max_deg': params.get('I_max_deg', 30.0),   # anti-windup limit
                'trim_band_deg': params.get('trim_band_deg', 1.0),
                'lead_time': params.get('lead_time', 20.0),    # prediction horizon (s) - increased to react earlier,    # prediction horizon (s) - reduced for quicker re-engagement,    # prediction horizon (s) - increased for earlier release
                'release_band_deg': params.get('release_band_deg', 5.0), # early release band (°) - widened to back off sooner and reduce hard-over, # early release band (°) - narrowed to engage rudder sooner, # early release band (°) - widen to back off sooner, # early release band
            }
            
            self._last_was_give = np.zeros(self.psi.shape, dtype=bool)

            # Initialize ship patches, lines, and text in UTM coords
            for i in range(self.n):
                rudder_line, = self.ax.plot([], [], 'r', zorder=8)
                self.rudder_lines.append(rudder_line)

                danger_cone, = self.ax.plot([], [], color='orange', lw=1, linestyle='--', zorder=7)
                self.danger_cones.append(danger_cone)

                poly = MplPolygon(
                    self._ship_base,
                    closed=True,
                    facecolor='blue',
                    edgecolor='k',
                    alpha=0.6,
                    zorder=5
                )
                self.ax.add_patch(poly)
                self.patches.append(poly)

                line, = self.ax.plot([], [], '--', linewidth=1)
                self.traj.append(line)

                txt = self.ax.text(self.pos[0, i], self.pos[1, i], '', fontsize=8, zorder=9)
                self.texts.append(txt)

                self.history.append([self.pos[:, i].copy()])
        
        # ── “Kill Power” control, embedded in the RIGHT panel ────────────────────
        bax = inset_axes(
            self.ax_panel,
            width="80%",    # 80% of panel’s width
            height="8%",    # 8% of panel’s height
            loc='lower center'  # centered at the bottom of panel
        )
        self._btn_kill = Button(bax, 'Kill Power', color='#cc4444', hovercolor='#ff6666')
        self._btn_kill.on_clicked(self._cut_random_power)

        # ── WIND‐VANE ARROW (in self.ax_panel) ───────────────────────────
        # TAIL‐POINT in panel‐axes coordinates (fractional [0..1])
        self._panel_tail = (0.5, 0.85)
        self.wind_arrow = FancyArrowPatch(
            posA=self._panel_tail,
            posB=self._panel_tail,        # starts collapsed
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=2.5,
            color='purple',
            transform=self.ax_panel.transAxes,
            zorder=5,
            animated=True
        )
        self.ax_panel.add_patch(self.wind_arrow)

        # ── HEADER TEXT (time / Wind / Current / Tide) ───────────────────
        #     size 12, serif in the RIGHT panel
        self.header_text = self.ax_panel.text(
            0.02, 0.95,
            "",                           # updated each frame
            transform=self.ax_panel.transAxes,
            fontsize=12,
            fontfamily='serif',
            va='top',
            animated=True
        )
        
        # Initialize placeholders for cached backgrounds
        self._bg_cache = None
        self._panel_bg_cache = None

        
    def playful_wind_polar(state, t):
        vec = playful_wind(state, t)          # existing 2×n array
        wx, wy = vec[:, 0]                    # same for every ship
        return {'speed': np.hypot(wx, wy),
                'dir':   np.arctan2(wy, wx)}

    def _cut_random_power(self, *_):
        """
        Callback for the *Kill Power* button.  Randomly selects one
        vessel (or the only vessel) and zeroes its propulsion.
        """
        idx = np.random.randint(self.n) if self.n > 1 else 0
        self.ship.cut_power(idx)

        log.info(f"[EMERGENT] Kill-switch → vessel {idx} (power cut at t={self.t:.1f}s)")

        # prepend a new log line (no change to header_text here)
        msg = f"Vessel {idx} lost power at t={self.t:.1f}s"
        self.log_lines.insert(0, msg)
        self.log_lines = self.log_lines[: self.max_log_lines]
        for j, txt in enumerate(self.log_text_artists):
            txt.set_text(self.log_lines[j] if j < len(self.log_lines) else "")
            txt.set_fontsize(12)
            txt.set_fontfamily('serif')

    def _draw_background(self):
        """
        Draws nautical-chart style background:
        - Water fill (pale blue)
        - Land fill (sandy tone)
        - Coastline lines
        - Depth contours & soundings (if available)
        - Gridlines and axis styling
        """
        # ─── Water base ───────────────────────────────
        domain = box(self.minx, self.miny, self.maxx, self.maxy)
        self.ax.add_patch(
            MplPolygon(np.array(domain.exterior.coords),
                       closed=True, facecolor='#cde6f7',
                       edgecolor='none', zorder=0)
        )

        # ─── Land areas (LNDARE) ──────────────────────
        land = self.enc_data.get('LNDARE')
        if land is not None and not land.empty:
            land_polys = []
            for geom in land.geometry:
                if isinstance(geom, (ShapelyPolygon, MultiPolygon)):
                    land_polys.extend(getattr(geom, 'geoms', [geom]))
            patches = [
                MplPolygon(np.array(poly.exterior.coords), closed=True)
                for poly in land_polys
            ]
            land_coll = PatchCollection(
                patches,
                facecolor='#f2e9dc',
                edgecolor='#c4b090',
                linewidths=0.5,
                zorder=1
            )
            self.ax.add_collection(land_coll)

        # ─── Shoal/depth areas (DEPARE) ───────────────
        if not self.light_bg:
            # ─── Shoal/depth areas (DEPARE) ──────────────
            shoals = self.enc_data.get('DEPARE')
            if shoals is not None and not shoals.empty:
                shoal_polys = []
                for geom in shoals.geometry:
                    if isinstance(geom, (ShapelyPolygon, MultiPolygon)):
                        shoal_polys.extend(getattr(geom, 'geoms', [geom]))
                shoal_coll = PatchCollection(
                    [
                        MplPolygon(np.array(poly.exterior.coords), closed=True)
                        for poly in shoal_polys
                    ],
                    facecolor='#b3d9ff',
                    edgecolor='none',
                    alpha=0.5,
                    zorder=1.5
                )
                self.ax.add_collection(shoal_coll)

            # ─── Depth contours (DEPVAL) ────────────────
            contours = self.enc_data.get('DEPVAL')
            if contours is not None and not contours.empty:
                for geom in contours.geometry:
                    lines = (
                        [geom] if isinstance(geom, LineString)
                        else getattr(geom, 'geoms', [])
                    )
                    for line in lines:
                        x, y = line.xy
                        self.ax.plot(
                            x, y,
                            linestyle='--',
                            linewidth=0.5,
                            color='dodgerblue',
                            zorder=3
                        )
                    
        # # 3) Coastline styling
        # coast_lines = [np.array(geom.coords)
        #                for geom in self.waterway.geometry
        #                if isinstance(geom, LineString)]
        # coast_coll = LineCollection(coast_lines,
        #                             colors='#003366',
        #                             linewidths=1.2,
        #                             alpha=0.9,
        #                             zorder=2)
        # self.ax.add_collection(coast_coll)
        # 4) Grid and axes styling
        self.ax.set_facecolor('#e0f8ff')
        self.ax.grid(True, which='major', color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax.tick_params(which='both', color='gray', labelcolor='gray', length=4)

    def _refresh_bg_cache(self):
        """Rasterise current axes into an RGBA buffer & enable blitting."""
        self.fig.canvas.draw()
        self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self._blit_enabled = True

    def load_enc_catalog(self, xml_path_or_url, verify_ssl=False, verbose=False):
        """
        Read ENC catalog (XML) via URL or file into ElementTree.
        """
        if verbose:
            print(f"Fetching ENC catalog from {xml_path_or_url}")
        if xml_path_or_url.startswith(('http://', 'https://')):
            r = requests.get(xml_path_or_url, verify=verify_ssl)
            r.raise_for_status()
            data = r.content
        else:
            data = Path(xml_path_or_url).read_bytes()

        self._enc_tree = ET.fromstring(data)
        if verbose:
            print(f"Parsed XML root tag: {self._enc_tree.tag}")

    def find_enc_cells(self, bounds, region_name = None, resolution='best', verbose=False):
        """bounds: (minx, miny, maxx, maxy) in lon/lat.  
           resolution: 'best', 'medium', or 'low'     
        Return list of the highest-resolution ENC cells whose bounding
        boxes overlap `bounds` or match `region_name`.
        """
        if self._enc_tree is None:
            raise RuntimeError("ENC catalog not loaded; call load_enc_catalog() first")

        ns = {
            'gmd': 'http://www.isotc211.org/2005/gmd',
            'gco': 'http://www.isotc211.org/2005/gco',
            'gml': 'http://www.opengis.net/gml/3.2',
            'xlink': 'http://www.w3.org/1999/xlink'
        }
        ET.register_namespace('gmd', ns['gmd'])
        ET.register_namespace('gml', ns['gml'])

        if verbose:
            print(f"Searching for DS_DataSet entries overlapping {bounds or region_name}")
        datasets = self._enc_tree.findall('.//gmd:DS_DataSet', namespaces=ns)

        hits = []
        for idx, ds in enumerate(datasets):
            # collect envelope coords
            coords = []
            for pos in ds.findall('.//gml:pos', namespaces=ns):
                parts = pos.text.split()
                if len(parts) >= 2:
                    lat, lon = map(float, parts[:2])
                    coords.append((lon, lat))
            if not coords:
                if verbose:
                    print(f"[#{idx}] no coordinates, skipping")
                continue
            xs, ys = zip(*coords)
            env = (min(xs), min(ys), max(xs), max(ys))
            if verbose:
                print(f"[#{idx}] envelope={env}")

            # bounds filter
            if bounds and isinstance(bounds, (list, tuple)):
                if not box(*bounds).intersects(box(*env)):
                    if verbose:
                        print(f"[#{idx}] does NOT overlap bounds, skipping")
                    continue

            # region_name filter
            if region_name:
                title = ds.findtext('.//gmd:title//gco:CharacterString', namespaces=ns) or ''
                if region_name.lower() not in title.lower():
                    if verbose:
                        print(f"[#{idx}] title '{title}' mismatch, skipping")
                    continue

            url = ds.findtext('.//gmd:transferOptions//gmd:CI_OnlineResource//gmd:URL', namespaces=ns)
            fid = ds.findtext('.//gmd:fileIdentifier//gco:CharacterString', namespaces=ns)
            cell_id = fid.strip() if fid else Path(url or '').stem or None
            # extract scale code (3rd char) if present
            scale = None
            if cell_id and len(cell_id) > 2 and cell_id[2].isdigit():
                scale = int(cell_id[2])

            hits.append({'cell_id': cell_id, 'url': url, 'bbox': env, 'scale': scale})

        # if no hits, return empty
        if not hits:
            if verbose:
                print("• No ENC cells found")
            return []

        # select highest resolution (max scale) among hits
        valid_scales = [h['scale'] for h in hits if h.get('scale') is not None]
        if not valid_scales:
            if verbose:
                print("No scale codes found; returning all hits")
            return hits

        # pick target scale based on desired resolution
        scales_sorted = sorted(set(valid_scales))
        if resolution == 'best':
            target_scale = scales_sorted[-1]
        elif resolution == 'medium':
            target_scale = scales_sorted[len(scales_sorted) // 2]
        elif resolution == 'low':
            target_scale = scales_sorted[0]
        else:
            raise ValueError(f"Unknown resolution '{resolution}'. Use 'best', 'medium', or 'low'.")

        # filter hits to just that scale
        hits = [h for h in hits if h.get('scale') == target_scale]
        if verbose:
            print(f"Selected {len(hits)} cells at {resolution} resolution (scale code {target_scale})")

        return [
            {
                'cell_id': h['cell_id'],
                'url':     h['url'],
                'bbox':    h['bbox']
            }
            for h in hits
        ]
    
    def fetch_and_extract_enc(self, enc_url, extract_to, verbose=False):
        """
        Download ENC .zip from enc_url and unzip into extract_to.
        Returns path containing the .000 S-57 files.
        """
        if verbose:
            print(f"Downloading {enc_url}")
        r = requests.get(enc_url, verify=False)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(extract_to)

        # locate folder with .000 files
        for root_dir, _, files in os.walk(extract_to):
            s57 = [f for f in files if f.lower().endswith('.000')]
            if s57:
                return root_dir
        raise RuntimeError(f"No S-57 .000 files found in {extract_to}")

    def get_obstacle_features(self):
        # all polygons (land/shoal areas)
        poly = self.waterway[self.waterway.geometry.type == 'Polygon']
    
        # plus any linear/point hazards by code
        hazard_codes = ['WRECKS','OBSTRN','MNEPNT','MNEARE','COALNE']
        hazards = self.waterway[
            self.waterway['feature_code'].isin(hazard_codes)
        ]
    
        return pd.concat([poly, hazards], ignore_index=True)

    def load_enc_features(self, enc_catalog_url, verbose=False):
        """
        Download and load every layer from each best-resolution ENC cell.
        Populates self.enc_data (dict of GeoDataFrames) and
        sets self.waterway = self.enc_data['COALNE'] for backwards compatibility.
        """
        # a) fetch catalog & select cells at all resolutions to ensure full coverage
        self.load_enc_catalog(enc_catalog_url, verbose=verbose)
        resolutions = ['best', 'medium']#, 'low']
        cells = []
        seen_ids = set()
        for res in resolutions:
            hits = self.find_enc_cells(bounds=self.bounds,
                                       resolution=res,
                                       verbose=verbose)
            for hit in hits:
                cid = hit.get('cell_id')
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    cells.append(hit)
    
        self.enc_data = {}
    
        for cell in cells:
            if verbose:
                print(f"Fetching ENC cell: {cell['url']}")
            # create a unique temp dir so files from each cell don't clobber each other
            extract_dir = tempfile.mkdtemp(prefix=f"enc_{cell.get('cell_id','')}_" )
            cell_dir = self.fetch_and_extract_enc(
                cell['url'],
                extract_to=extract_dir,
                verbose=verbose
            )
            # only iterate the true S-57 base (.000) files; skip .001 updates
            enc_files = list(Path(cell_dir).glob("*.000"))
            
            for enc_file in enc_files:
                if verbose:
                    print(f"  Opening ENC file: {enc_file.name}")
                for layer in fiona.listlayers(str(enc_file)):
                    if verbose:
                        print(f"    reading layer: {layer}")
                    try:
                        with fiona.open(str(enc_file), layer=layer) as src:
                            gdf = gpd.GeoDataFrame.from_features(src)
                    except Exception as e:
                        if verbose:
                            log.warning(f"Skipped {layer}: {e}")
                        continue

                    if gdf.empty:
                        continue
                    if gdf.crs is None:
                        gdf.set_crs(epsg=4326, inplace=True)

                    gdf = (
                        gdf[gdf.geom_type.isin([
                            "Point","MultiPoint",
                            "LineString","MultiLineString",
                            "Polygon","MultiPolygon"
                        ])]
                        .to_crs(self.crs_utm)
                    )

                    self.enc_data.setdefault(layer, []).append(gdf)
    
        # concatenate each layer
        for layer, gdfs in self.enc_data.items():
            self.enc_data[layer] = gpd.GeoDataFrame(
                pd.concat(gdfs, ignore_index=True),
                crs=self.crs_utm
            )
    
        # backward compatibility for coastline (ensure geometry column)
        self.waterway = self.enc_data.get(
            "COALNE",
            gpd.GeoDataFrame(geometry=[], crs=self.crs_utm)
        )

        # ── NEW: re-project every GeoDataFrame to the sim’s UTM CRS ─────────
        for k, gdf in self.enc_data.items():
            if not gdf.empty and gdf.crs != self.crs_utm:
                self.enc_data[k] = gdf.to_crs(self.crs_utm)

    def spawn_agents_at_entrance(self):
        """
        Programmatic spawn: place each of n agents evenly along the left (min-x) boundary,
        set their initial surge speed, and point goals at the opposite side.

        Returns:
            state0 (4×n ndarray): initial [u, v, p, r] for each ship
            pos0   (2×n ndarray): initial [x, y] for each ship
            psi0   (n,) ndarray: initial heading for each ship
            goals_arr (2×n ndarray): goal [x, y] for each ship
        """
        # Domain limits in UTM (meters)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        n = self.n

        # Evenly space along left edge for spawns
        ys = np.linspace(y0, y1, n)
        xs = np.full(n, x0)
        pos0 = np.vstack((xs, ys))

        # Goals along right edge at same y
        gx = np.full(n, x1)
        gy = ys.copy()
        goals_arr = np.vstack((gx, gy))

        # Initial heading toward each goal
        psi0 = np.arctan2(gy - ys, gx - xs)

        # Initial state: surge = spawn_speed, other components zero
        state0 = np.zeros((4, n), dtype=float)
        state0[0, :] = getattr(self, 'spawn_speed', self.params.get('desired_speed', 0.0))

        # store into self for later updating in run()
        self.state = state0
        self.pos   = pos0
        self.psi   = psi0
        self.goals = goals_arr

        # position each ship‐patch at its spawn point
        # (self.base is your Nx2 polygon template)
        for i, patch in enumerate(self.patches):
            # shift the base polygon by the agent’s (x,y)
            patch.set_xy(self.base + pos0[:, i])

        return state0, pos0, psi0, goals_arr

    def spawn(self):
        """
        Interactive spawn and routing: collect waypoints first, then initialize states.

        - Uses `route()` to let user click all desired waypoints (including start and goal).
        - Sets initial position to first click, heading toward second click, and goal to last click.
        """
        # 1) Collect interactive waypoints
        self.route()

        # 2) Derive state arrays from waypoints
        n = self.n
        # Initialize arrays
        state0 = np.zeros((4, n), dtype=float)
        pos0 = np.zeros((2, n), dtype=float)
        psi0 = np.zeros(n, dtype=float)
        goals_arr = np.zeros((2, n), dtype=float)

        # Set initial surge speed
        spawn_speed = getattr(self, 'spawn_speed', self.params.get('desired_speed', 0.0))
        state0[0, :] = spawn_speed

        # Populate from waypoints
        for idx, wp in enumerate(self.waypoints):
            # First waypoint = start
            p0 = wp[0]
            pos0[:, idx] = p0
            # Heading toward second waypoint if exists, else toward last
            p1 = wp[1] if len(wp) > 1 else wp[-1]
            psi0[idx] = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            # Last waypoint = goal
            p_goal = wp[-1]
            goals_arr[:, idx] = p_goal

        # 3) Plot start/goal markers
        for i in range(n):
            sx, sy = pos0[0, i], pos0[1, i]
            gx, gy = goals_arr[0, i], goals_arr[1, i]
            self.ax.scatter(sx, sy, marker='o', s=50, color='green', zorder=2)
            self.ax.scatter(gx, gy, marker='8', s=50, color='red',   zorder=2)

        # 4) Instantiate ship and assign to simulation state
        self.ship = ship(self.params, self.n)
        self.state  = state0.copy()
        self.pos    = pos0.copy()
        self.psi    = psi0.copy()
        self.goals  = goals_arr.copy()

        return state0, pos0, psi0, goals_arr

    def route(self, *args, **kwargs):
        """
        Interactive route entry using the current plot.
        User clicks all waypoints (including start and goal) directly on the existing axes.
        Finish with Enter or right-click.
        """

        print(f"[ROUTE] Starting interactive routing using current plot for {self.n} agent(s)...")
        # — make sure the window is visible and ENC lines get drawn —
        # re-render the nautical-chart background
        try:
            # bring up the existing figure window
            self.fig.show()
            # clear any old artists, reset limits & aspect
            self.ax.clear()
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.set_xlim(self.minx, self.maxx)
            self.ax.set_ylim(self.miny, self.maxy)
            # draw the full nautical-chart background
            self._draw_background()
        except Exception as e:
            log.error(f"[ROUTE] Failed to draw full background: {e}")
            
        # lock in the original UTM extents/aspect from __init__
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.minx, self.maxx)
        self.ax.set_ylim(self.miny, self.maxy)
        # trigger a non-blocking redraw so the GUI can update
        self.fig.canvas.draw_idle()
        plt.pause(0.1)           # slight pause to ensure GUI refresh

        self.waypoints = []
        self.final_goals = np.zeros((2, self.n), dtype=float)

        for idx in range(self.n):
            print(f"[ROUTE] Agent {idx+1}/{self.n}: click all desired waypoints, finish with Enter/right-click.")
            pts = plt.ginput(n=-1, timeout=0)

            if not pts:
                print(f"[ROUTE] No clicks detected for agent {idx+1}; defaulting to straight line.")
                # use current pos and goal if defined
                if hasattr(self, 'pos') and hasattr(self, 'goals'):
                    pts = [(self.pos[0, idx], self.pos[1, idx]), (self.goals[0, idx], self.goals[1, idx])]
                else:
                    pts = []

            waypoints = [np.array(pt) for pt in pts]
            self.waypoints.append(waypoints)
            if waypoints:
                self.final_goals[:, idx] = waypoints[-1]
            print(f"[ROUTE] Agent {idx+1} waypoints: {len(waypoints)}")

        self.ship.wpts = self.waypoints
        self.ship.short_route = self.waypoints
        # avoid method shadowing
        self.user_routes = list(self.waypoints)
        print("[ROUTE] Interactive routing complete.")


    # def route(self, k=8, simplify_tol=10.0, vertex_stride=3):
    #     """
    #     Generate efficient routes using simplified obstacles, vertex subsampling,
    #     k-NN via KD-tree, and A*, all operating in UTM (meters).

    #     Args:
    #         k (int): number of neighbors per node
    #         simplify_tol (float): tolerance for geometry simplification (meters)
    #         vertex_stride (int): sample every Nth vertex from obstacle boundaries
    #     """
    #     print(f"[ROUTE] Starting route generation for {self.n} agent(s)...")

    #     # 1) Non-ENC fallback: straight-line in UTM meters
    #     if not hasattr(self, 'waterway') or 'geom_simple' not in self.waterway:
    #         print("[ROUTE] No ENC data found; using direct straight-line routes.")
    #         self.waypoints = [
    #             [self.pos[:, i].copy(), self.goals[:, i].copy()]
    #             for i in range(self.n)
    #         ]
      
    #         self.final_goals = self.goals.copy()
    #         self.ship.wpts = self.waypoints
    #         self.ship.short_route = self.waypoints
    #         print("[ROUTE] Completed non-ENC fallback.")
    #         return

    #     # 2) Simplify & buffer obstacles in meters
    #     buf_dist = getattr(self, 'buffer_dist', 1000.0)  # buffer (m)
    #     print("[ROUTE] Simplifying and buffering obstacles...")
    #     print(f"[ROUTE] Shoreline buffer: {buf_dist:.0f} m")

    #     raw = self.waterway['geom_simple']
    #     poly_obs = [geom.simplify(simplify_tol) for geom in raw
    #                 if isinstance(geom, (ShapelyPolygon, MultiPolygon))]
    #     coast_bufs = []
    #     for geom in raw:
    #         if isinstance(geom, LineString):
    #             buf = geom.buffer(buf_dist).simplify(simplify_tol)
    #             coast_bufs.append(buf)
    #     obstacles = poly_obs + coast_bufs
    #     obs_union = unary_union(obstacles)

    #     # Save obstacles for collision checks
    #     self.obstacles = obstacles
    #     print(f"[ROUTE] Total obstacles after buffering: {len(obstacles)}")

    #     tree = STRtree(obstacles) if obstacles else None

    #     # 3) Build waypoints per agent
    #     self.waypoints = []
    #     self.final_goals = self.goals.copy()

    #     for idx in range(self.n):
    #         print(f"[ROUTE] Routing agent {idx+1}/{self.n}...")
    #         start_time = time.time()

    #         A = Point(self.pos[0, idx], self.pos[1, idx])
    #         B = Point(self.goals[0, idx], self.goals[1, idx])

    #         # Corridor filter in meters
    #         corridor = LineString([A, B]).buffer(buf_dist)
    #         relevant = [obs for obs in obstacles if obs.intersects(corridor)]
    #         print(f"[ROUTE]  Relevant obstacles: {len(relevant)}")

    #         # 3a) Sample nodes: endpoints + PRM + obstacle vertices
    #         nodes = [A, B]
    #         rng = np.random.default_rng()
    #         self.prm_samples = getattr(self, 'prm_samples', 100)
    #         minx, miny, maxx, maxy = corridor.bounds
    #         cand = rng.uniform((minx, miny), (maxx, maxy), size=(self.prm_samples, 2))
    #         for x, y in cand:
    #             p = Point(x, y)
    #             if corridor.contains(p) and not obs_union.contains(p):
    #                 nodes.append(p)
    #         for poly in relevant:
    #             coords = list(poly.exterior.coords)[::vertex_stride]
    #             nodes.extend(Point(x, y) for x, y in coords)

    #         # Dedupe and arrayify
    #         unique = list({(pt.x, pt.y): pt for pt in nodes}.values())
    #         pts_arr = np.array([[pt.x, pt.y] for pt in unique])
    #         print(f"[ROUTE]  Nodes count after subsampling: {len(unique)}")

    #         # 3b) Iterative graph + A* attempts
    #         max_tries = 3
    #         path = None
    #         for attempt in range(1, max_tries + 1):
    #             print(f"[ROUTE]  Graph build attempt {attempt} (PRM={self.prm_samples})")
    #             # Build k-NN graph
    #             kd = cKDTree(pts_arr)
    #             nbrs = kd.query(pts_arr, k=min(k+1, len(unique)))[1]
    #             G = nx.Graph()
    #             for i, pt in enumerate(unique):
    #                 G.add_node(i, point=pt)
    #             for i, neigh in enumerate(nbrs):
    #                 for j in neigh[1:]:
    #                     seg = LineString([unique[i], unique[j]])
    #                     if tree and any(seg.intersects(obstacles[o]) for o in tree.query(seg)):
    #                         continue
    #                     d = np.hypot(*(pts_arr[i] - pts_arr[j]))
    #                     G.add_edge(i, j, weight=d)
    #             # Delaunay fallback if no edges
    #             if G.number_of_edges() == 0:
    #                 from scipy.spatial import Delaunay
    #                 import itertools
    #                 tri = Delaunay(pts_arr)
    #                 for simplex in tri.simplices:
    #                     for i, j in itertools.combinations(simplex, 2):
    #                         if G.has_edge(i, j):
    #                             continue
    #                         seg = LineString([unique[i], unique[j]])
    #                         if tree and any(seg.intersects(obstacles[o]) for o in tree.query(seg)):
    #                             continue
    #                         G.add_edge(i, j, weight=np.linalg.norm(pts_arr[i] - pts_arr[j]))
    #             # A* attempt
    #             try:
    #                 path = nx.astar_path(
    #                     G, 0, 1,
    #                     heuristic=lambda u, v: G.nodes[u]['point'].distance(G.nodes[v]['point']),
    #                     weight='weight'
    #                 )
    #                 print(f"[ROUTE]  Path found on attempt {attempt}")
    #                 break
    #             except nx.NetworkXNoPath:
    #                 print(f"[ROUTE]  A* failed on attempt {attempt}")
    #                 self.prm_samples *= 2
    #         # Final fallback
    #         if path is None:
    #             print(f"[ROUTE]  All {max_tries} attempts failed; using safe straight route.")
    #             path = [0, 1]

    #         # 4) Extract waypoints
    #         waypoints = [
    #             np.array((G.nodes[n]['point'].x, G.nodes[n]['point'].y))
    #             for n in path
    #         ]
    #         self.waypoints.append(waypoints)

    #         elapsed = time.time() - start_time
    #         print(f"[ROUTE]  Agent {idx+1} path length: {len(waypoints)} waypoints, time: {elapsed:.2f}s")

    #     # 5) Attach to ship
    #     self.ship.wpts = self.waypoints
    #     self.ship.short_route = self.waypoints
    #     self.route = list(self.waypoints)
    #     print("[ROUTE] Route generation complete.")
        

    def _check_collision(self, t):
        """
        Return True if any two ship patches overlap (collision).
        """
        # Test every pair of ships using freshly‐built hulls
        for i in range(self.n):
            poly_i = self._current_hull_poly(i)
            for j in range(i+1, self.n):
                poly_j = self._current_hull_poly(j)
                if poly_i.intersects(poly_j):
                    inter = poly_i.intersection(poly_j)
                    if inter.area > self.collision_tol_area:
                        log.error(f"Collision detected between Ship{i} and Ship{j} at t={t:.2f}s")
                        return True
        return False

    def _check_allision(self, t):
        """
        Return True if any ship hull polygon intersects land.
        """
        # Build Shapely Polygons for each ship patch
        ship_polys = [ShapelyPolygon(patch.get_xy()) for patch in self.patches]
        
        # Pull the true coastline/land polygons out of your waterway object.
        # Assuming self.waterway is a GeoDataFrame with a 'geometry' column:
        land_geoms = list(self.waterway.geometry)
    
        for i, ship_poly in enumerate(ship_polys):
            for land in land_geoms:
                if ship_poly.intersects(land):
                    log.error(f"Allision detected: Ship{i} with shore at t={t:.2f}s")
                    return True
    
        return False

    def _update_panel(self, t):
        """
        Helper to update and draw the RIGHT panel (wind vane arrow, header text, log lines).
        Must be called inside the main run(...) loop, after map‐side blitting is handled.
        """
        # 1) Compute wind vector (wx, wy) as floats for the first vessel (or dict case)
        wind_raw = self.wind_fn(self.state, t)
        if isinstance(wind_raw, dict):
            wx = float(wind_raw['speed'] * np.cos(wind_raw['dir']))
            wy = float(wind_raw['speed'] * np.sin(wind_raw['dir']))
        else:
            wx = float(wind_raw[0, 0])
            wy = float(wind_raw[1, 0])
            
        # 2) Recompute arrow tip in panel‐axes coords
        theta   = np.arctan2(wy, wx)
        L_panel = 0.10 * (1 + 0.3 * np.hypot(wx, wy))   # 10% base length
        dx_p, dy_p = L_panel * np.cos(theta), L_panel * np.sin(theta)
        tail_x, tail_y = self._panel_tail
        tip_x = tail_x + dx_p
        tip_y = tail_y + dy_p
        # clamp to [0..1] so arrow never wanders off‐screen
        tip_x = max(0.0, min(1.0, tip_x))
        tip_y = max(0.0, min(1.0, tip_y))
        self.wind_arrow.set_positions((tail_x, tail_y), (tip_x, tip_y))

        # 3) Update header text (time, wind speed, current speed, tide)
        current_raw = self.current_fn(self.state, t)
        if isinstance(current_raw, dict):
            cx = float(current_raw['speed'] * np.cos(current_raw['dir']))
            cy = float(current_raw['speed'] * np.sin(current_raw['dir']))
        else:
            cx = float(current_raw[0, 0])
            cy = float(current_raw[1, 0])
        tide_val = float(self.tide_fn(t))
        title_str = (
            f"t = {t:.1f} s   \n"
            f"Wind = {np.hypot(wx, wy):.2f} m/s \n"
            f"Current = {np.hypot(cx, cy):.2f} m/s \n"
            f"Tide = {tide_val:.2f} m"
        )
        self.header_text.set_text(title_str)

        # # 4) Collect RIGHT‐panel artists for blitting (arrow, header, log lines)
        # dynamic_panel = [self.wind_arrow, self.header_text] + self.log_text_artists

        # # 5) Draw each artist in ax_panel
        # for art in dynamic_panel:
        #     self.ax_panel.draw_artist(art)

    def run(self):
        """Advance the simulation through all time steps."""
        total_steps = int(self.steps * self.dt)
        self.stop = False
        for cone in self.danger_cones:
            cone.set_data([], [])
        self._startup_diagnostics(total_steps)
        nu = np.vstack([self.state[0], self.state[1], self.state[3]])

        for step in range(total_steps):
            t = step * self.dt
            self.t = t
            # pick the next dynamic waypoint as current goal
            self._update_goals()

            
            print (f"Time step: {step}")
            t = step * self.dt
            if self.stop:
                print(f"[SIM] Halted at t={t:.1f}s.")
                break
            # ────────────────────────────────────────────────────────────────
            # 1) apply follow-camera *first* and detect movement
            old_xlim, old_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            if self.dynamic_zoom:
                self._apply_zoom()               # may pan / zoom axes
            self._camera_moved = (old_xlim != self.ax.get_xlim()
                                  or old_ylim != self.ax.get_ylim())

            # 2) compute controls & dynamics (unchanged)
            hd, sp, rud = self._compute_controls_and_update(nu, t)

            # 3) integrate dynamics and record history
            self._step_dynamics(hd, sp, rud)
            
            # 4) check for ship–ship collisions
            if self._check_collision(t):
                print(f"[SIM] Collision at t={t:.1f}s, halting.")
                break

            # 5) check for ship–shore allisions
            if self._check_allision(t):
                print(f"[SIM] Allision at t={t:.1f}s, halting.")
                break

            # 6) decide if background cache is stale (after camera could move)
            tiles_changed = (self.tcache.ensure_visible(self.ax.viewLim)
                             if self.tcache else False)
            cache_stale = (
                tiles_changed
                or self._camera_moved
                or not getattr(self, "_blit_enabled", False)
            )
            
            # 7)  draw background exactly once ────────────────────────────────────
            if cache_stale:
                # ── FULL redraw ──
                self.fig.canvas.draw()
                self._bg_cache       = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
                self._panel_bg_cache = self.fig.canvas.copy_from_bbox(self.ax_panel.bbox)
                self._blit_enabled   = True
            else:
                # Restore both map & panel backgrounds from their caches
                self.fig.canvas.restore_region(self._bg_cache)
                self.fig.canvas.restore_region(self._panel_bg_cache)

            # ── Draw map‐side dynamic elements ───────────────────────────────
            self._draw_waypoints()
            self._draw_ships(step)
            self._draw_annotations(hd, sp, rud)

            # ── Draw control‐panel dynamic elements ──────────────────────────
            self._update_panel(t)
            
            # ── Draw panel‐side dynamic artists (all in serif, size 12) ─────────
            # Make sure header_text is serif, size 12
            self.header_text.set_fontsize(12)
            self.header_text.set_fontfamily('serif')

            # Ensure each log line is serif, size 12
            for txt in self.log_text_artists:
                txt.set_fontsize(12)
                txt.set_fontfamily('serif')

            dynamic_panel = [self.wind_arrow, self.header_text] + self.log_text_artists
            for art in dynamic_panel:
                self.ax_panel.draw_artist(art)

            #  • blit the full figure (map + panel)
            if self._blit_enabled:
                self.fig.canvas.blit(self.fig.bbox)
            else:
                self.fig.canvas.draw_idle()

            self.fig.canvas.flush_events()
            plt.pause(self.dt)
            
            self._camera_moved = False

        print(f"[SIM] Completed at t={total_steps*self.dt:.1f}s")
        plt.ioff(); plt.show()

    def _startup_diagnostics(self, total_steps):
        print(f"[SIM] Starting: {self.n} agents, dt={self.dt}, steps={total_steps}")
        print(f"[SIM] Initial pos: {self.pos.T.round(2)}")
        
    def _update_goals(self):
        """Point each agent’s `self.goals` at its first unmet waypoint."""
        #tol = getattr(self, 'wp_tol', self.L * 2.)
        tol = getattr(self, 'wp_tol', self.L)

        for i in range(self.n):
            # if we still have queued waypoints, aim at the first one
            if self.waypoints[i]:
                self.goals[:,i] = self.waypoints[i][0]
                # if we’re close, pop it off
                if np.linalg.norm(self.pos[:,i] - self.goals[:,i]) < tol:
                    self.waypoints[i].pop(0)
            else:
                # no more intermediate wpts: aim at the true end
                self.goals[:,i] = self.route[i][-1]

    def _compute_controls_and_update(self, nu, t):
        # compute desired & COLREGS headings/speeds
        goal_hd, goal_sp = self.ship.compute_desired(
            self.goals, self.pos[0], self.pos[1],
            self.state[0], self.state[1], self.state[3], self.psi
        )
        col_hd, col_sp, _, roles = self.ship.colregs(
            self.pos, nu, self.psi, self.ship.commanded_rpm
        )
        # fuse & PID
        hd, sp = self._fuse_and_pid(goal_hd, goal_sp, col_hd, col_sp)
        rud = self._compute_rudder(hd, roles)
        return hd, sp, rud

    def _fuse_and_pid(self, goal_hd, goal_sp, col_hd, col_sp):
        roles = np.array(self.ship.colregs(self.pos, np.vstack([self.state[0],self.state[1],self.state[3]]), self.psi, self.ship.commanded_rpm)[3])
        is_give = (roles == 'give_way')
        hd = np.where(is_give, col_hd, goal_hd)
        sp = np.where(is_give, col_sp, goal_sp)
        raw_rud = self.ship.pid_control(self.psi, hd, self.dt)
        # … blend override code here …
        return hd, np.minimum(sp, self.ship.max_speed)

    def _compute_rudder(self, psi_ref, roles):
        """
        Advanced helm controller combining PID with feed-forward and micro-trimming:
          - Proportional on heading error
          - Derivative on turn-rate error
          - Integral term with anti-windup
          - Feed-forward on desired turn-rate
          - Micro-trim zeroing for chatter avoidance
          - COLREGS override for give-way situations
        """
        dt = self.dt
    
        # 1) Heading error in [-π, π]
        err = ((psi_ref - self.psi + np.pi) % (2 * np.pi)) - np.pi
    
        # 2) Feed-forward: desired turn rate
        Kf = self.tuning['Kf']                # feed-forward gain
        r_rate_max = np.radians(self.tuning['r_rate_max_deg'])
        r_des = np.clip(Kf * err, -r_rate_max, r_rate_max)
    
        # 3) Measured turn-rate
        r = (self.psi - self.prev_psi) / dt
    
        # 4) PID on turn-rate error
        derr = r_des - r
        # update integral of heading error, with anti-windup limits
        self.integral_error += err * dt
        I_max = np.radians(self.tuning['I_max_deg'])
        self.integral_error = np.clip(self.integral_error, -I_max, I_max)
    
        Kp = self.tuning['Kp']
        Ki = self.tuning['Ki']
        Kd = self.tuning['Kd']
        rud_cmd = Kp * err + Ki * self.integral_error + Kd * derr
    
        # 5) Prediction-based early release: anticipate inertia-driven overshoot
        predicted_err = err + r * self.tuning['lead_time']
        release_rad   = np.radians(self.tuning['release_band_deg'])
        early_release = np.abs(predicted_err) < release_rad
        rud_cmd = np.where(early_release, 0.0, rud_cmd)
    
        # 6) Micro-trim: zero small commands to avoid constant nudging
        trim_rad = np.radians(self.tuning['trim_band_deg'])
        rud_cmd[np.abs(rud_cmd) < trim_rad] = 0.0
    
        # 6) COLREGS override for give-way: hard opposite rudder beyond 30° error
        err_abs = np.abs(err)
        ov_mask = (np.array(roles) == 'give_way') & (err_abs >= np.radians(30))
        rud_cmd = np.where(ov_mask,
                           -np.sign(err) * self.ship.max_rudder,
                           rud_cmd)
    
        # 7) Saturate to max rudder and rate-limit the change
        rud_sat = np.clip(rud_cmd, -self.ship.max_rudder, self.ship.max_rudder)
        max_delta = self.ship.max_rudder_rate * dt
        rud = np.clip(rud_sat,
                      self.prev_rudder - max_delta,
                      self.prev_rudder + max_delta)
    
        # 8) Save state for next cycle
        self.prev_psi = self.psi.copy()
        self.prev_rudder = rud
        self.ship.prev_rudder = rud
    
        return rud

    def _step_dynamics(self, hd, sp, rud):
        # throttle, drag, forces, integrate self.state & self.pos
        thrust = self.ship.thrust(self.ship.speed_to_rpm(sp))
        drag = self.ship.drag(self.state[0])
        wind_raw = self.wind_fn(self.state, self.t)

        # ─── Update wind-vane ------------------------------------------------
        if isinstance(wind_raw, dict):
            wx = float(wind_raw['speed'] * np.cos(wind_raw['dir']))
            wy = float(wind_raw['speed'] * np.sin(wind_raw['dir']))
        else:  # 2×n array → take the first vessel
            wx = float(wind_raw[0, 0])
            wy = float(wind_raw[1, 0])
        if self.t == 0.0:
            print(f"[DBG] t=0 arrow values: wx={wx:.2f}, wy={wy:.2f}")
            
        # Compute wind vector (wx, wy) in m/s (as floats) …
        theta = np.arctan2(wy, wx)

        # Make the arrow length some fraction of the axes size:
        L = 0.08 * (1 + 0.5 * np.hypot(wx, wy))
        dx, dy = L * np.cos(theta), L * np.sin(theta)

        tail_x, tail_y = self._panel_tail
        tip_x, tip_y = tail_x + dx, tail_y + dy

        # Clamp to [0..1] so the arrow never wanders off-screen
        tip_x = max(0.0, min(1.0, tip_x))
        tip_y = max(0.0, min(1.0, tip_y))

       # Finally update the FancyArrowPatch (both coords in axes-fraction)
        self.wind_arrow.set_positions((tail_x, tail_y),
                                      (tip_x, tip_y))

        wind_vec    = self.wind_fn(self.state, self.t)
        current_vec = self.current_fn(self.state, self.t)
        wind, current = self.ship.environmental_forces(
            wind_vec, current_vec,
            self.state[0], self.state[1], self.psi
        )
        
        tide = self.tide_fn(0)
        u_dot,v_dot,p_dot,r_dot = self.ship.dynamics(
            self.state, thrust, drag, wind, current, rud
        )
        # update state & history arrays
        # integrate body-fixed accelerations
        self.state[0] += u_dot * self.dt    # surge acceleration → surge speed
        self.state[1] += v_dot * self.dt    # sway acceleration → sway speed
        self.state[3] += r_dot * self.dt    # yaw rate
        self.psi     += self.state[3] * self.dt

        # transform body-frame velocities into Earth-frame for position update
        for i in range(self.n):
            u_i = self.state[0, i]
            v_i = self.state[1, i]
            psi_i = self.psi[i]
            # body→world rotation
            dx =  u_i * np.cos(psi_i) - v_i * np.sin(psi_i)
            dy =  u_i * np.sin(psi_i) + v_i * np.cos(psi_i)
            self.pos[0, i] += dx * self.dt
            self.pos[1, i] += dy * self.dt
            self.history[i].append(self.pos[:, i].copy())
            
    def _current_hull_poly(self, i):
        """
        Build a Shapely Polygon for ship i at its current pos & heading.
        """
        from shapely.geometry import Polygon
        # Rotation matrix
        c, s = np.cos(self.psi[i]), np.sin(self.psi[i])
        R = np.array([[ c, -s],
                      [ s,  c]])
        # stern offset in body frame
        stern_rel = np.array([-self.ship.length/2, 0.0])
        # world‐frame stern
        stern_glob = R @ stern_rel + self.pos[:, i]
        # apply same shift to every base vertex
        pts = (R @ (self._ship_base - stern_rel).T).T + stern_glob
        return Polygon(pts)

    def _prepare_frame(self):
        self.ax.clear()
        # remove margins
        self.ax.margins(0)

    def _draw_base_layers(self):
        self.ax.add_collection(self._enc_collection)

    def _draw_ships(self, step):
        for p in self.patches:
            self.ax.add_patch(p)
        for line in self.traj + self.rudder_lines + self.danger_cones:
            self.ax.add_line(line)
        for txt in self.texts:
            self.ax.add_artist(txt)
            
    def _draw_waypoints(self):
        """
        Plot each agent’s remaining waypoints with numbered labels.
        """
        # each self.waypoints[i] is a list of np.array([x,y]) for agent i
        for i, wps in enumerate(getattr(self, 'waypoints', [])):
            for j, wp in enumerate(wps):
                x, y = wp
                # marker for waypoint
                self.ax.scatter(x, y, marker='x', s=50, color='magenta', zorder=4)
                # label with its sequence number (font size 8, serif)
                txt = self.ax.text(
                    x, y,
                    str(j+1),
                    color='magenta',
                    fontsize=8,
                    fontfamily='serif',
                    zorder=5,
                    ha='left', va='bottom',
                )

    def _draw_annotations(self, hd, sp, rud):
        # 1) Precompute bow positions so we can index into bows[i] safely
        bows = []
        for k in range(self.n):
            Rk = np.array([[np.cos(self.psi[k]), -np.sin(self.psi[k])],
                           [np.sin(self.psi[k]),  np.cos(self.psi[k])]])
            bow_rel_k = self._ship_base[3]
            bow_glob_k = Rk @ bow_rel_k + self.pos[:, k]
            bows.append(bow_glob_k)
            
        # --- clear old heading arrows so we can re-append without index errors ---
        for arr in getattr(self, 'heading_arrows', []):
            if arr is not None:
                arr.remove()
        self.heading_arrows = []

        for i in range(self.n):
            # Compute rotation & position
            R = np.array([[np.cos(self.psi[i]), -np.sin(self.psi[i])],
                          [np.sin(self.psi[i]),  np.cos(self.psi[i])]])
            stern = R @ np.array([-self.ship.length/2, 0.0]) + self.pos[:, i]
            pts   = (R @ (self._ship_base - np.array([-self.ship.length/2, 0.0])).T).T + stern
            
            # 1) Hull
            self.patches[i].set_xy(pts)
            
            # 2) Trajectory
            traj = np.array(self.history[i])
            self.traj[i].set_data(traj[:,0], traj[:,1])
            # 3) Rudder
            
            rud_end = stern - self.ship.length*0.1 * np.array([
                np.cos(self.psi[i] + np.pi + rud[i] + np.sign(rud[i])*np.pi/2),
                np.sin(self.psi[i] + np.pi + rud[i] + np.sign(rud[i])*np.pi/2)
            ])
            self.rudder_lines[i].set_data([stern[0], rud_end[0]],
                                          [stern[1], rud_end[1]])
            
            # 4) Labels: ensure arrays are length‐n
            cmd_rpms = np.atleast_1d(self.ship.commanded_rpm)
            if cmd_rpms.size == 1:
                cmd_rpms = np.full(self.n, cmd_rpms.item())
            dsp_arr = np.atleast_1d(self.ship.desired_speed)
            if dsp_arr.size == 1:
                dsp_arr = np.full(self.n, dsp_arr.item())

            lbl = (
                f"ID {i}\n"
                f"hd: {(np.degrees(self.psi[i]) % 360):.1f}°\n"
                f"U: {self.state[0,i]:.2f} m/s\n"
                f"Dsp: {dsp_arr[i]:.2f} m/s\n"
                f"Thr%: {(cmd_rpms[i] / self.ship.max_rpm * 100):.0f}%"
            )
            
            self.texts[i].set_text(lbl)
            self.texts[i].set_position(self.pos[:,i] + np.array([self.L*0.02, self.B*0.02]))
            
            # 5) Danger cone
            half = np.radians(30)
            # Danger cone: originate at the bow
            bow = bows[i]
            cone = np.vstack([
                bow,
                bow + 2000 * np.array([np.cos(self.psi[i]+half), np.sin(self.psi[i]+half)]),
                bow + 2000 * np.array([np.cos(self.psi[i]-half), np.sin(self.psi[i]-half)]),
                bow
            ])
            self.danger_cones[i].set_data(cone[:,0], cone[:,1])
            
            # 6) Heading arrow: remove any old arrows and append new one
            for j in range(self.n):
                dx_hd = self.L * 0.2 * np.cos(hd[j])
                dy_hd = self.L * 0.2 * np.sin(hd[j])
                bow = bows[j]
                arr = self.ax.arrow(
                    bow[0], bow[1], dx_hd, dy_hd,
                    head_width=self.L * 0.05, head_length=self.L * 0.1,
                    fc='red', ec='red', length_includes_head=True
                )
                self.heading_arrows.append(arr)
    
    def _apply_zoom(self):
        if self.dynamic_zoom:
            x_c, y_c = self.pos[0,0], self.pos[1,0]
            z = self.zoom
            self.ax.set_xlim(x_c - z, x_c + z)
            self.ax.set_ylim(y_c - z, y_c + z)
            # ensure no extra padding
            self.ax.set_adjustable('box')
            old_xlim, old_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            # ... update limits here ...
            self._camera_moved = (old_xlim != self.ax.get_xlim() or
                                  old_ylim != self.ax.get_ylim())


# Example usage with interactive encounter\ n
if __name__ == '__main__':
    params = {
        # vessel geometry (m)
        'length':400.,
        'beam':60.,
        'draft':10.,
        
        # Rigid-body properties
        'm': 5e7,
        'Ixx': 1e9,
        'Izz': 2e9,
        'xG': 0.0,
        'zG': -5.0,
    
        # Hydrodynamic derivatives
        'Xu': -2.262e-3,
        'Xv': -2.4e-4,
        'Xp': 0.0,
        'Xr': 0.0,
        'Yu': 0.0,
        'Yv': -7.25e-3,
        'Yp': 0.0,
        'Yr': 0.0,
        'Ku': 0.0,
        'Kv': 0.0,
        'Kp': 0.0,
        'Kr': 0.0,
        'Nu': 0.0,
        'Nv': -3e-3,
        'Np': 0.0,
        'Nr': 0.0,
    
        # Rudder effectiveness
        # Boost lateral force so the hull “feels” the rudder earlier:
        'Ydelta': 190e-2,    # up from 1e-2
        # Roll moment (leave if you don’t care about heel):
        'Kdelta': 0.,  
        # Double or triple your yawing “bite”:
        'Ndelta': 60e6,   # up from 1.3e6
    
        # Damping
        'linear_damping': 1e5,
        'quad_damping': 1e4,
    
        # PID gains
        'Kp_gain':      0.09,     # a bit less aggressive P
        'Ki_gain':      0.005,    # unchanged or very small
        'Kd_gain':      4.,     # more damping
        'Kf_gain':      0.09,     # 0.8 modest feed-forward
        'r_rate_max_deg': 6.0,   # slower commanded max turn
        'I_max_deg':    0.05,    # tighter anti-windup
        'lead_time': 8000.0,    # prediction horizon (s) - increased to react earlier,    # prediction horizon (s) - reduced for quicker re-engagement,    # prediction horizon (s) - increased for earlier release
        'trim_band_deg': 1.0,    # wider dead-zone around zero
        'max_rudder': np.radians(35),
        'max_rudder_rate': np.radians(10.),  # allow up to 10°/s change
        
        # Propulsion 
        'initial_speed': 10.,
        'desired_speed': 10.,     
    }
    
n = 1

# initial state: [u, v, ψ̇, ?]
state0 = np.zeros((4, n))
state0[0, :] = 10.0    # 10 m/s initial surge speed

 # ─────────────────────────────────────────────────────────────────────────────
 # Simple environmental forcing utilities+# ---------------------------------------------------------------------------
# current_fn → steady NORTH ➜ SOUTH set in body-fixed meters s-¹
#             y-axis grows “north” in the demo, so southward = −y
# wind_fn    → playful but cheap gust model (global for all agents)
#             fully vectorized, returns shape (2, n_agents)
# ---------------------------------------------------------------------------

wind_fn    = playful_wind
current_fn = north_south_current
tide_fn    = lambda t: 1.5 * np.sin(2 * np.pi / 12.42 * (t / 3600))

# # bounding‐box for Galveston Harbor (lon/lat EPSG:4326)
# minx, maxx = -94.99, -94.70
# miny, maxy =  29.20,  29.50

# Baltimore, MD (City) bounds in EPSG:4326
minx, maxx = -76.60, -76.30 
miny, maxy = 39.19, 39.50

# # Port of Los Angeles & Port of Long Beach (San Pedro Bay) bounds in EPSG:4326
# minx, maxx = -118.290046296, -118.069953704
# miny, maxy =   33.699861117,   33.790046302

# # Oakland/San Francisco Bay (Region) bounds
# minx, maxx = -122.550000, -121.680000
# miny, maxy =   37.360000,   38.300000

# # Seattle, WA (City) bounds
# minx, maxx = -122.459696, -122.224433
# miny, maxy =   47.491911,   47.734061

# # New Orleans downriver to the Mississippi River mouth (EPSG:4326)
# minx, maxx = -89.5, -89.0   # West (upriver New Orleans) to East (river mouth)
# miny, maxy = 28.75, 29.25     # South (delta/mouth) to North (city limits)

# # New York Harbor bounds (EPSG:4326)
# minx, maxx = -74.27, -73.86   # West (Staten Island/Narrows) → East (Queens/LI shore)
# miny, maxy =  40.49,  40.75   # South (Lower Bay)         → North (Upper Bay)


# NOAA ENC catalog
xml_url = 'https://charts.noaa.gov/ENCs/ENCProdCat_19115.xml'

# ── SET UP & RUN SIMULATION ───────────────────────────────────────────────────
sim = simulation(
    params,
    wind_fn,
    current_fn,
    tide_fn,
    minx, maxx, miny, maxy,
    enc_catalog_url=xml_url,
    dt=0.1,
    T=36000,
    n_agents=n,            # spawn 1 vessel at initialization
    spawn_speed=10.0,      # match our state0[0]
    spawn_waypoints=10
)

# This will:
#  • load the ENC catalog & cells covering the bbox
#  • fetch & load waterway shapefiles
#  • simplify geometries for routing
#  • spawn 1 agent at the seaward edge
#  • compute its obstacle‐avoiding route via Route()
#  • initialize sim.pos, sim.state, sim.psi, sim.goals, and plotting patches

sim.run()  # advance the sim for T/dt = 6000 steps