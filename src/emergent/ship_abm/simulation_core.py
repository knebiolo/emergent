"""
simulation.py

This module defines the Simulation class for the Ship Agent-Based Model (ABM) in Emergent.
It orchestrates vessel spawning, routing, collision-avoidance, and time-stepping by leveraging
the Ship class (in ship_model.py) and configuration parameters (in config.py).

Core Responsibilities:
----------------------
1. Domain Setup:
   • Load geographic bounds (lat/lon or UTM) for a user-selected port from SIMULATION_BOUNDS.
   • Initialize wind, current, and tide forcing functions.

2. Propulsion & Control Initialization:
   • Instantiate the Ship model, which handles hydrodynamics, PID-based heading/speed control,
     and advanced tuning (feed-forward, dead-band, anti-windup) as defined in config.py.
   • Set up initial speed, RPM, and propulsion parameters using values from PROPULSION.

3. Agent Spawning & Routing:
   • Use port-specific bounds to spawn N agents at domain entrances.
   • Generate waypoint-based routes for each vessel; store initial positions, headings, and goals.

4. Collision Avoidance:
   • On each timestep, compute pairwise distances among vessels.
   • If another ship enters safe_dist, switch that vessel into avoidance mode.
   • While in avoidance, hold rudder until |bearing| exceeds unlock_ang; then resume normal routing.
   • Constants safe_dist, clear_dist, unlock_ang, etc., are loaded from COLLISION_AVOIDANCE.

5. Main Time-Stepping Loop:
   • For each t in [0, T) with step dt:
       – Update avoidance flags and compute heading goals.
       – Query wind_fn, current_fn, tide_fn to get environmental forcing.
       – Call Ship.step(state, commanded_rpm, goals, wind, current, dt) to advance physics.
       – Update state vector (u, v, p, r, x, y, z, psi), pos, and psi for all vessels.
       – Record new commanded_rpm from the Ship model.
       – (Optionally) draw or log the updated positions for visualization or analysis.

Usage Example:
--------------
    from emergent.ship_abm.simulation import Simulation

    # Define wind, current, and tide functions (e.g., return zero for still conditions)
    wind_fn = lambda state, t: np.zeros((2, n_agents))
    current_fn = lambda state, t: np.zeros((2, n_agents))
    tide_fn = lambda t: 0.0

    # Initialize simulation for Baltimore with 3 vessels, running for 300 s at 0.1 s timesteps
    params = {}  # any additional parameters needed by Ship or spawn logic
    sim = Simulation(
        params=params,
        wind_fn=wind_fn,
        current_fn=current_fn,
        tide_fn=tide_fn,
        port_name="Baltimore",
        dt=0.1,
        T=300.0,
        n_agents=3,
        enc_catalog_url=None,
        verbose=False
    )

    # Run the simulation loop
    sim.run()

    # Access results: sim.state, sim.pos, sim.psi, sim.ship.commanded_rpm, etc.

Notes:
------
– All “magic numbers” (geometry, PID gains, collision thresholds, propulsion constants)
  are centralized in ship_abm/config.py. Modifying any ABM behavior should start there.
– Physics and control laws (Fossen’s 4-DOF dynamics, thrust↔RPM, PID + feed-forward + dead-band)
  are encapsulated in src/emergent/ship_abm/ship_model.py.
– This file only handles high-level orchestration: spawning, collision logic, looping, and IO.
"""


import logging
import numpy as np
import os
import xml.etree.ElementTree as ET
import requests, zipfile, io
import geopandas as gpd
from shapely.geometry import box, LineString, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import pandas as pd
import fiona
import urllib3
import warnings
from pathlib import Path
from typing import Iterable                    # <-- NEW: needed for type-hints
import tempfile
import sys
from emergent.ship_abm.ship_model import ship
from emergent.ship_abm import enctiler
from emergent.ship_abm.config import SHIP_PHYSICS, \
    CONTROLLER_GAINS, \
        ADVANCED_CONTROLLER, \
            PROPULSION, \
                SIMULATION_BOUNDS, \
                    xml_url, COLLISION_AVOIDANCE
from emergent.ship_abm.ais import compute_ais_heatmap
from datetime import date
from emergent.ship_abm.ais import compute_ais_heatmap
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from datetime import datetime, timezone
import datetime as dt
from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
from pyproj import Transformer          # <-- light-weight, pure-python

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
    a = np.tile(np.array([[0.0], [-speed]]), (1, n))
    return a #np.zeros_like(a)

def playful_wind(state, t,
                 base=1.0,
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
    a= np.tile(np.array([[wx], [wy]]), (1, n))
    return a #np.zeros_like(a)

def tide_fn(t):
    return 1.5 * np.sin(2 * np.pi / 12.42 * (t / 3600)) 


class simulation:
    def __init__(
        self,
        port_name,
        dt=0.1,
        T=100,
        n_agents=None,
        ais_dataset_paths: list[str] = None,
        coast_simplify_tol = 50.0,
        light_bg = True,
        verbose=False,
        use_ais=False,
        # ── TEST-MODE ARGS ───────────────────────────────────────────
        test_mode=None,          # e.g. "zigzag" or None
        zigzag_deg=10,           # ± heading deflection (deg)
        zigzag_hold=40,          # hold time per leg (s)
        load_enc = True
    ):
        """
        Initialize simulation with ENC data and optionally spawn agents.

        Parameters:
        ----------
        minx, maxx, miny, maxy : float
            Domain bounding box in lon/lat.
        dt : float
            Time step (s).
        T : float
            Total simulation duration (s).
        n_agents : int, optional
            Number of agents to spawn immediately.

        """
        # Store parameters & environment functions
        self.dt = dt
        self.t = 0
        self.steps = int(T / dt)
        self.use_ais = use_ais
        # Normalize n_agents to integer (allow None -> 0)
        self.n = int(n_agents) if (n_agents is not None) else 0
        self.in_deadband = np.zeros(self.n, dtype=bool)    # Are we currently in the dead-band?
        self.entry_sign  = np.zeros(self.n, dtype=float)   # Which side (±1) we locked in on entry
        self.hyst_done   = np.zeros(self.n, dtype=bool)    # Have we already exited once?
        self.prev_rudder = np.zeros(self.n, dtype=float)
        self.prev_sign   = np.zeros(self.n, dtype=float)  # last step’s sign(err_vec)
        self.prev_abs_err  = np.full(self.n, np.inf)
        self.tile_dict = {}   # always defined (ENC or not)
        self.tcache    = None # will hold TileCache only for ENC runs
        self.light_bg = light_bg                 # ← new flag
        self.coast_simplify_tol = coast_simplify_tol
        self.port_name = port_name
        self.t_history = []          # append t each step
        self.psi_history = []        # append self.psi[0] each step (or full array)
        self.hd_cmd_history = []     # append hd_cmds[0] each step
        # verbosity flag: when False, suppress high-frequency debug prints
        self.verbose = bool(verbose)

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

        bb = SIMULATION_BOUNDS[port_name]
        minx, maxx = bb["minx"], bb["maxx"]
        miny, maxy = bb["miny"], bb["maxy"]    

        # Determine or default domain bounds
        self.bounds = (minx, miny, maxx, maxy)

        # Waypoint tolerance: default to half-ship length for more sensible waypoint popping
        if not hasattr(self, 'wp_tol'):
            self.wp_tol = max(50.0, self.L * 0.5)

        # Collision events log
        self.collision_events = []
        # Allision (ship->shore) events
        self.allision_events = []
        # collision area tolerance (m^2) — small polygon overlaps below this are ignored
        self.collision_tol_area = 1.0

        # Compute UTM CRS from domain center
        # ───────────────────────────────────────────────────────────────
        # 1) CRS setup & transformer (`utm → lon/lat`) – we need this
        #    every timestep so make it once up-front.
        midx = (minx + maxx) / 2
        utm_zone = int((midx + 180) // 6) + 1
        utm_epsg = 32600 + utm_zone  # Northern hemisphere
        self.crs_utm = f"EPSG:{utm_epsg}"
        
        # forward = lon/lat→UTM (already used elsewhere)
        self._utm_to_ll = Transformer.from_crs(
            self.crs_utm, "EPSG:4326", always_xy=True
        )
        
        # enable follow‐ship zoom (meters from ship center)
        self.dynamic_zoom = False
        self.zoom = 5000    # e.g. ±2 km view radius


        # Reproject lon/lat bbox into UTM and set axes accordingly
        ll_box = box(minx, miny, maxx, maxy)
        utm_box = (
            gpd.GeoDataFrame(geometry=[ll_box], crs="EPSG:4326")
            .to_crs(self.crs_utm)
        )
        x0, y0, x1, y1 = utm_box.total_bounds
    
        # Store UTM extents for interactive routing
        self.minx, self.maxx = x0, x1
        self.miny, self.maxy = y0, y1

        # Initialize trajectory lines
        self.traj = []

        # Placeholders for agent state and plotting elements
        self.pos = None  # 2×n array in meters
        self.psi = None  # n array of headings
        self.state = None  # 4×n state array
        self.goals = None  # 2×n goal positions in meters
        self.patches = []  # ship patches
        self.traj = []     # trajectory lines
        self.texts = []    # text labels
        self.rudder_lines = []
        self.danger_cones = []
        self.heading_arrows = []
 
        # 1) Load ENC charts - unless testing
        if load_enc:
            self.load_enc_features(xml_url, verbose=verbose)
        else:
            self.enc_data = {}
            
        if "COALNE" in self.enc_data:
            self.waterway = self.enc_data["COALNE"]
        else:
            from geopandas import GeoDataFrame
            self.waterway = GeoDataFrame(geometry=[], crs=self.crs_utm)
            
        self.ais_coords = np.zeros((0,2), dtype=float)
        if self.use_ais and ais_dataset_paths:
            import pandas as pd
            ais_list = []
            for path in ais_dataset_paths:
                df = pd.read_csv(path, parse_dates=['timestamp'])
                ais_list.append(df)
            ais_df = pd.concat(ais_list, ignore_index=True)
            # convert lon/lat → UTM
            gdf = gpd.GeoDataFrame(
                ais_df,
                geometry=gpd.points_from_xy(ais_df.longitude, ais_df.latitude),
                crs="EPSG:4326"
            ).to_crs(self.crs_utm)
            xs = gdf.geometry.x.values
            ys = gdf.geometry.y.values
            self.ais_coords = np.vstack([xs, ys]).T 

        # Set plot limits based on input bbox reprojected to UTM
        bbox_ll = box(minx, miny, maxx, maxy)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_ll], crs='EPSG:4326')
        bbox_utm = bbox_gdf.to_crs(self.crs_utm)
        xmin_u, ymin_u, xmax_u, ymax_u = bbox_utm.total_bounds
        
        # No agents to spawn → initialize empty state arrays
        self.pos   = np.zeros((2, self.n), dtype=float)
        self.psi   = np.zeros(self.n,          dtype=float)
        self.state = np.zeros((4, self.n),     dtype=float)
        self.goals = np.zeros((2, self.n),     dtype=float)

        #--- Setup steering controller state & tuning ---
        # prev_psi for turn-rate calculation
        self.prev_psi = self.psi.copy()
        # integral of heading error for I-term
        self.integral_error = np.zeros_like(self.psi)
        # runtime rudder inversion detection (per-agent)
        # If the plant (dynamics) responds opposite to applied rudder for
        # several consecutive steps, we flip commanded rudder to correct
        # coordinate-convention mismatches. Counters are conservative.
        self._rudder_inverted = np.zeros(self.n, dtype=bool)
        self._rudder_rev_counter = np.zeros(self.n, dtype=int)
        # lockout timestamp (simulation time) until which further flips are disabled
        self._rudder_rev_lock_until = np.zeros(self.n, dtype=float)
        # store previous raw command (for optional low-pass filtering to avoid chatter)
        self.prev_raw_cmd = np.zeros(self.n, dtype=float)
        # confirm after this many consecutive opposite-sign observations
        # Temporarily disable runtime rudder-inversion auto-correct by
        # setting confirmation to a very large value. This makes the
        # detector inert for short diagnostic runs. Revert after debug.
        self._rudder_rev_confirm = 10**9
        # minimum yaw-rate magnitude (rad/s) to consider for detection (~1°/s)
        self._rudder_rev_threshold = np.radians(1.0)
        
        self.tuning = {
            'Kp': CONTROLLER_GAINS['Kp'],
            'Ki': CONTROLLER_GAINS['Ki'],
            'Kd': CONTROLLER_GAINS['Kd'],
            'Kf': ADVANCED_CONTROLLER['Kf_gain'],           # feed-forward gain
            'r_rate_max_deg': ADVANCED_CONTROLLER['r_rate_max_deg'],
            'I_max_deg': ADVANCED_CONTROLLER['I_max_deg'],   # anti-windup limit
            'trim_band_deg': ADVANCED_CONTROLLER['trim_band_deg'],
            'lead_time': ADVANCED_CONTROLLER['lead_time'],    # prediction horizon (s) - increased to react earlier,    # prediction horizon (s) - reduced for quicker re-engagement,    # prediction horizon (s) - increased for earlier release
            'release_band_deg': ADVANCED_CONTROLLER['release_band_deg'], # early release band (°) - widened to back off sooner and reduce hard-over, # early release band (°) - narrowed to engage rudder sooner, # early release band (°) - widen to back off sooner, # early release band
        }
        
        self._last_was_give = np.zeros(self.psi.shape, dtype=bool)
        self.log_lines = []
        self.max_log_lines = 5
 
        # ── TEST-MODE SETUP ──────────────────────────────────────────
        self.test_mode = test_mode
        if self.test_mode == "zigzag":
            self.zz_delta   = np.radians(zigzag_deg)
            self.zz_hold    = zigzag_hold
            self.zz_next_sw = zigzag_hold           # first switch at t = hold
            self.zz_sign    = 1                     # start with +Δψ
            self.zz_base_psi = 0.0                  # filled in after spawn()
            from emergent.ship_abm.config import PROPULSION
            self.zz_sp_cmd  = PROPULSION.get("desired_speed", 6.0)

        # ── TEST-MODE: TURNING CIRCLE ─────────────────────────────────────────────
        if self.test_mode == "turncircle":
            self.tc_rudder_deg = 20.0  # default rudder angle for test
            self.tc_speed = PROPULSION.get("desired_speed", 6.0)

        # 2) Instantiate the NOAA current sampler for *this* port
        # Deferred loading to avoid Qt threading issue - use load_environmental_forcing() after GUI init
        # Start with dummy functions that return zeros
        def _dummy_current(lon, lat, when):
            n = len(np.atleast_1d(lon))
            return np.zeros((n, 2))
        def _dummy_wind(lon, lat, when):
            n = len(np.atleast_1d(lon))
            return np.zeros((n, 2))
        
        self.current_fn = _dummy_current
        self.wind_fn = _dummy_wind

        # Ensure any environment sampler we expose returns an (N,2) array where N = len(lons)
        def _normalize_env_sampler(raw_sampler):
            import inspect
            def sampler(lons, lats, when):
                import numpy as _np
                N = int(_np.atleast_1d(lons).size)
                # Try calling the sampler with (lon, lat, when)
                try:
                    res = raw_sampler(lons, lats, when)
                except TypeError:
                    # Fallback: sampler may expect (state, t) signature (legacy test funcs)
                    try:
                        res = raw_sampler(getattr(self, 'state', None), getattr(self, 't', None))
                    except Exception:
                        return _np.zeros((N, 2), dtype=float)
                except Exception:
                    return _np.zeros((N, 2), dtype=float)

                arr = _np.asarray(res)
                # (N,2) -> good
                if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] >= 2:
                    return arr[:, :2]
                # (2,N) -> transpose
                if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == N:
                    return arr.T
                # flat length 2*N -> reshape
                if arr.ndim == 1 and arr.size == 2 * N:
                    try:
                        return arr.reshape((N, 2))
                    except Exception:
                        pass
                # single pair -> tile for N
                flat = arr.ravel()
                if flat.size >= 2:
                    try:
                        u0, v0 = float(flat[0]), float(flat[1])
                        out = _np.tile(_np.array([[u0, v0]]), (N, 1))
                        return out
                    except Exception:
                        pass
                return _np.zeros((N, 2), dtype=float)
            return sampler

        # Normalize the dummy samplers immediately so callers can rely on (N,2)
        self.current_fn = _normalize_env_sampler(self.current_fn)
        self.wind_fn = _normalize_env_sampler(self.wind_fn)
        self._env_loaded = False

        
        # ------------------------------------------------------------------
        # Build a static lon/lat grid (≈150×150) over the domain for quiver
        # ------------------------------------------------------------------
        # Increase resolution to provide many more quiver sampling points
        # Note: higher values increase sampling and drawing cost; 150×150 = 22.5k samples
        nx = ny = 150
        gx = np.linspace(minx, maxx, nx)    # in lon/lat – we still have those
        gy = np.linspace(miny, maxy, ny)
        self._quiver_lon, self._quiver_lat = np.meshgrid(gx, gy)

        def _currents_grid(when: datetime):
            """Return U,V 2-D arrays (ny,nx) in m/s for viewer quiver."""
            res = self.current_fn(
                self._quiver_lon.ravel(),
                self._quiver_lat.ravel(),
                when
            )
            arr = np.asarray(res)
            expected = ny * nx
            try:
                # arr expected shape (N,2)
                if arr.ndim == 2 and arr.shape[0] == expected and arr.shape[1] >= 2:
                    flat_u = arr[:, 0]
                    flat_v = arr[:, 1]
                    return (flat_u.reshape(ny, nx), flat_v.reshape(ny, nx))
                else:
                    print(f"[Simulation][debug] current_fn returned shape={arr.shape}, expected ({expected},2); falling back to zeros")
            except Exception as e:
                print(f"[Simulation][debug] current_fn reshape error: {e}; arr.shape={getattr(arr,'shape',None)}")
            # fallback: return zeros to avoid index errors
            return (np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float))
        
        
        self.currents_grid = _currents_grid
    
        def _wind_grid(when: datetime):
            """Return wind U,V 2-D arrays (ny,nx) in m/s for viewer quiver."""
            # wind_fn returns an (N,2) array of [u, v] at each query point
            res = self.wind_fn(
                self._quiver_lon.ravel(),
                self._quiver_lat.ravel(),
                when
            )
            arr = np.asarray(res)
            expected = ny * nx
            try:
                if arr.ndim == 2 and arr.shape[0] == expected and arr.shape[1] >= 2:
                    wu = arr[:, 0]
                    wv = arr[:, 1]
                    return wu.reshape(ny, nx), wv.reshape(ny, nx)
                else:
                    print(f"[Simulation][debug] wind_fn returned shape={arr.shape}, expected ({expected},2); falling back to zeros")
            except Exception as e:
                print(f"[Simulation][debug] wind_fn reshape error: {e}; arr.shape={getattr(arr,'shape',None)}")
            return (np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float))
        self.wind_grid = _wind_grid    
   
        # Initialize placeholders for cached backgrounds
        self._bg_cache = None

        
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
        # If the viewer has registered text artists for the log, update them.
        artists = getattr(self, 'log_text_artists', None)
        if artists:
            for j, txt in enumerate(artists):
                try:
                    txt.set_text(self.log_lines[j] if j < len(self.log_lines) else "")
                    txt.set_fontsize(12)
                    txt.set_fontfamily('serif')
                except Exception:
                    # be defensive: ignore any drawing errors from GUI objects
                    pass

    def load_environmental_forcing(self, start: 'datetime | None' = None):
        """
        Load environmental forcing data (currents and winds) from NOAA sources.
        This should be called AFTER Qt GUI is fully initialized to avoid threading conflicts.
        Safe to call multiple times - only loads once.
        """
        if self._env_loaded:
            print("[Simulation] Environmental forcing already loaded, skipping")
            return

        if start is None:
            start = datetime.utcnow()
        print(f"[Simulation] Loading environmental forcing for {self.port_name} (start={start.date()})...")

        # Helper: wrap a sampler so it always returns values aligned with query points.
        def wrap_sampler_for_queries(raw_sampler):
            """Return a sampler(lons, lats, when) that guarantees output length == len(lons).

            Strategy:
            1) Try calling raw_sampler directly with query points.
            2) If the returned array length matches len(query) -> return it.
            3) Else, if raw_sampler carries metadata (native pts or valid arrays), use KDTree
               nearest-neighbor resampling from native points to query points.
            4) As a last resort, fall back to nearest-neighbor sampling by calling raw_sampler
               on the native points extracted from the sampler (if exposed) or return zeros.
            """
            import numpy as _np
            from scipy.spatial import cKDTree as _cKDTree

            # If raw_sampler already has metadata for fast resampling, extract them
            native_type = getattr(raw_sampler, '_native', None)

            # Prebuild KDTree if possible
            kdtree = None
            native_pts = None
            native_u = None
            native_v = None
            if native_type == 'roms':
                # roms exposes separate u/v pts and valid arrays
                try:
                    u_pts = getattr(raw_sampler, '_u_pts', None)
                    v_pts = getattr(raw_sampler, '_v_pts', None)
                    u_valid = getattr(raw_sampler, '_u_valid', None)
                    v_valid = getattr(raw_sampler, '_v_valid', None)
                    # merge into one representative native grid by averaging (cheap approx)
                    if u_pts is not None and v_pts is not None:
                        # stack and build tree on combined unique points
                        pts = _np.vstack((u_pts, v_pts))
                        native_pts = pts
                        kdtree = _cKDTree(pts)
                        # store u/v arrays aligned to pts by nearest mapping (approx)
                        native_u = _np.concatenate((u_valid, _np.full(len(v_valid), _np.nan)))
                        native_v = _np.concatenate((_np.full(len(u_valid), _np.nan), v_valid))
                except Exception:
                    pass
            elif native_type in ('unstructured',):
                try:
                    pts = getattr(raw_sampler, '_valid_pts', None)
                    native_u = getattr(raw_sampler, '_u_valid', None)
                    native_v = getattr(raw_sampler, '_v_valid', None)
                    if pts is not None:
                        native_pts = pts
                        kdtree = _cKDTree(pts)
                except Exception:
                    pass

            def sampler(lons, lats, when):
                lons = _np.asarray(lons, dtype=float)
                lats = _np.asarray(lats, dtype=float)
                Nq = lons.size
                try:
                    res = raw_sampler(lons, lats, when)
                    arr = _np.asarray(res)
                    if arr.ndim == 2 and arr.shape[0] == Nq and arr.shape[1] >= 2:
                        return arr
                    # many samplers return (N,2), but some return flattened shorter arrays
                except Exception:
                    arr = None

                # If we have a KDTree and native values, use it
                if kdtree is not None and native_pts is not None and native_u is not None and native_v is not None:
                    try:
                        query = _np.column_stack((lons.ravel(), lats.ravel()))
                        _, idx = kdtree.query(query)
                        uvals = native_u[idx]
                        vvals = native_v[idx]
                        print(f"[Simulation][debug] KDTree resample used (native_pts={len(native_pts)}, queries={Nq})")
                        return _np.column_stack((uvals, vvals))
                    except Exception:
                        pass

                # Last resort: try calling raw_sampler with native points (if exposed) to build a KDTree
                try:
                    # attempt to pull native lon/lat arrays from sampler attributes
                    nlon = getattr(raw_sampler, '_lon', None)
                    nlat = getattr(raw_sampler, '_lat', None)
                    if nlon is not None and nlat is not None:
                        native = _np.column_stack((_np.asarray(nlon).ravel(), _np.asarray(nlat).ravel()))
                        vals = _np.asarray(raw_sampler(nlon.ravel(), nlat.ravel(), when))
                        if vals.ndim == 2 and vals.shape[0] == native.shape[0]:
                            tree = _cKDTree(native)
                            _, idx = tree.query(_np.column_stack((lons.ravel(), lats.ravel())))
                            uvals = vals[idx, 0]
                            vvals = vals[idx, 1]
                            return _np.column_stack((uvals, vvals))
                except Exception:
                    pass

                # Fallback: return zeros
                return _np.zeros((Nq, 2), dtype=float)

            return sampler

        # local normalizer (same logic as used in __init__) to ensure any exposed
        # sampler returns an (N,2) array where N = len(lons)
        def _normalize_env_sampler(raw_sampler):
            def sampler(lons, lats, when):
                import numpy as _np
                N = int(_np.atleast_1d(lons).size)
                try:
                    res = raw_sampler(lons, lats, when)
                except TypeError:
                    try:
                        res = raw_sampler(getattr(self, 'state', None), getattr(self, 't', None))
                    except Exception:
                        return _np.zeros((N, 2), dtype=float)
                except Exception:
                    return _np.zeros((N, 2), dtype=float)

                arr = _np.asarray(res)
                if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] >= 2:
                    return arr[:, :2]
                if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == N:
                    return arr.T
                if arr.ndim == 1 and arr.size == 2 * N:
                    try:
                        return arr.reshape((N, 2))
                    except Exception:
                        pass
                flat = arr.ravel()
                if flat.size >= 2:
                    try:
                        u0, v0 = float(flat[0]), float(flat[1])
                        out = _np.tile(_np.array([[u0, v0]]), (N, 1))
                        return out
                    except Exception:
                        pass
                return _np.zeros((N, 2), dtype=float)
            return sampler


        # Try multiple times to acquire current data before falling back
        current_loaded = False
        # exponential backoff schedule in seconds
        backoff = [1, 2, 4, 8]
        for attempt in range(len(backoff)):
            try:
                from .ofs_loader import get_current_fn
                # Provide ENC land polygons (LNDARE) if available so the OFS loader
                # can avoid building KDTree entries that fall on land and thus
                # prevent inland nearest-neighbour picks.
                land_gdf = None
                try:
                    land_gdf = getattr(self, 'enc_data', {}).get('LNDARE')
                except Exception:
                    land_gdf = None

                raw_current = get_current_fn(self.port_name, start=start, land_gdf=land_gdf)
                # wrap with robust query wrapper then normalize output shape
                self.current_fn = _normalize_env_sampler(wrap_sampler_for_queries(raw_current))
                msg = f"[Simulation] ✓ Ocean currents loaded (attempt {attempt+1})"
                print(msg)
                # Diagnostic: sample the current_fn at our quiver points (if available) and print shapes
                try:
                    if hasattr(self, '_quiver_lon'):
                        res = self.current_fn(self._quiver_lon.ravel(), self._quiver_lat.ravel(), start)
                        arr = np.asarray(res)
                        print(f"[Simulation][debug] sampled current_fn -> arr.shape={getattr(arr,'shape',None)}")
                except Exception as _e:
                    print(f"[Simulation][debug] sampling current_fn failed: {_e}")
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
                current_loaded = True
                break
            except Exception as e:
                msg = f"[Simulation] ✗ Failed to load currents (attempt {attempt+1}): {e}"
                print(msg)
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
                # backoff before next attempt
                try:
                    import time
                    time.sleep(backoff[attempt])
                except Exception:
                    pass
        if not current_loaded:
            # If OFS discovery/opening failed repeatedly, fall back to a
            # simple tidal-proxy sampler so the simulation still experiences
            # spatially-uniform, time-varying currents (helpful for UI tests).
            try:
                from emergent.ship_abm.ofs_loader import make_tidal_proxy_current
                proxy = make_tidal_proxy_current(axis_deg=320.0, A_M2=0.30, A_S2=0.10)
                # wrap+normalize so caller always gets an (N,2) array
                self.current_fn = _normalize_env_sampler(wrap_sampler_for_queries(proxy))
                msg = f"[Simulation] ✓ Falling back to tidal-proxy currents after {len(backoff)} attempts"
                print(msg)
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
                current_loaded = True
            except Exception:
                msg = f"[Simulation]   Falling back to zero currents after {len(backoff)} attempts"
                print(msg)
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass

        # Try multiple times to acquire wind data before falling back
        wind_loaded = False
        for attempt in range(len(backoff)):
            try:
                from .ofs_loader import get_wind_fn
                raw_wind = get_wind_fn(self.port_name, start=start)
                # wrap with robust query wrapper then normalize output shape
                self.wind_fn = _normalize_env_sampler(wrap_sampler_for_queries(raw_wind))
                msg = f"[Simulation] ✓ Winds loaded (from OFS loader) (attempt {attempt+1})"
                print(msg)
                # Diagnostic: sample the wind_fn at our quiver points (if available) and print shapes
                try:
                    if hasattr(self, '_quiver_lon'):
                        resw = self.wind_fn(self._quiver_lon.ravel(), self._quiver_lat.ravel(), start)
                        arrw = np.asarray(resw)
                        print(f"[Simulation][debug] sampled wind_fn -> arr.shape={getattr(arrw,'shape',None)}")
                except Exception as _e:
                    print(f"[Simulation][debug] sampling wind_fn failed: {_e}")
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
                wind_loaded = True
                break
            except Exception as e:
                msg = f"[Simulation] ✗ Failed to load winds (attempt {attempt+1}): {e}"
                print(msg)
                try:
                    self.log_lines.insert(0, msg)
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
                # attempt atmospheric fallback for this attempt
                try:
                    from .atmospheric import wind_sampler
                    cfg = SIMULATION_BOUNDS[self.port_name]
                    bbox = (cfg["minx"], cfg["maxx"], cfg["miny"], cfg["maxy"])
                    self.wind_fn = wind_sampler(bbox, start)
                    msg2 = f"[Simulation] ✓ Winds loaded (from atmospheric.wind_sampler) (attempt {attempt+1})"
                    print(msg2)
                    try:
                        self.log_lines.insert(0, msg2)
                        self.log_lines = self.log_lines[: self.max_log_lines]
                    except Exception:
                        pass
                    wind_loaded = True
                    break
                except Exception as e2:
                    msg2 = f"[Simulation] ✗ Atmospheric fallback failed (attempt {attempt+1}): {e2}"
                    print(msg2)
                    try:
                        self.log_lines.insert(0, msg2)
                        self.log_lines = self.log_lines[: self.max_log_lines]
                    except Exception:
                        pass
                try:
                    import time
                    time.sleep(backoff[attempt])
                except Exception:
                    pass
        if not wind_loaded:
            msg = f"[Simulation]   Falling back to zero winds after {len(backoff)} attempts"
            print(msg)
            try:
                self.log_lines.insert(0, msg)
                self.log_lines = self.log_lines[: self.max_log_lines]
            except Exception:
                pass

        # Consider environment loaded if at least one of currents or winds is available
        if current_loaded or wind_loaded:
            self._env_loaded = True
            print("[Simulation] ✓ Environmental forcing ready!")
        else:
            self._env_loaded = False
            print("[Simulation] ✗ Environmental forcing unavailable; using zero fields")

    def get_ais_heatmap(self,
                        date_range: tuple[date,date],
                        grid_size: tuple[int,int]=(100,100)
                        ) -> tuple[np.ndarray, tuple[float,float,float,float]]:
        """
        Call ais.compute_ais_heatmap (which works in lon/lat), then
        reproject its extent into UTM so the viewer can place it.
        """
        if not getattr(self, "use_ais", True):
            return None, (0, 0, 0, 0)      # graceful no-op
        # 1) call the NOAA‐downloader / histogrammer in ais.py
        #    bbox in lon/lat:
        lonlat_bbox = (self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])
        heat, extent_ll = compute_ais_heatmap(
            bbox=lonlat_bbox,
            start_date=date_range[0],
            end_date=date_range[1],
            grid_size=grid_size,
            year=None
        )

        # 2) convert the lon/lat extent → UTM extent
        min_lon, max_lon, min_lat, max_lat = extent_ll
        from shapely.geometry import box
        import geopandas as gpd

        ll_poly = box(min_lon, min_lat, max_lon, max_lat)
        utm_poly = gpd.GeoSeries([ll_poly], crs="EPSG:4326").to_crs(self.crs_utm)
        x0, y0, x1, y1 = utm_poly.total_bounds

        # 3) return the heatmap and UTM extent
        #    Note: ais returns shape = (ny, nx), which viewer will transpose if needed
        return heat, (x0, x1, y0, y1)

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

        # handle nested zips: if any zip was inside, extract those too
        for root, dirs, files in os.walk(extract_to):
            for fname in files:
                if fname.lower().endswith('.zip'):
                    try:
                        nested = zipfile.ZipFile(os.path.join(root, fname))
                        nested.extractall(root)
                        if verbose:
                            print(f"Extracted nested zip: {fname} in {root}")
                    except Exception:
                        if verbose:
                            print(f"Failed to extract nested zip: {fname} (skipping)")

        # debug: print small tree when verbose
        if verbose:
            for root, dirs, files in os.walk(extract_to):
                rel = os.path.relpath(root, extract_to)
                print(f"  {rel}/: {len(files)} files, {len(dirs)} dirs")
                sample = files[:10]
                for f in sample:
                    print(f"    - {f}")

        # locate folder with .000 files and sanity-check them
        for root_dir, _, files in os.walk(extract_to):
            s57 = [f for f in files if f.lower().endswith('.000')]
            if s57:
                # sanity-check first file
                test_path = os.path.join(root_dir, s57[0])
                try:
                    size = os.path.getsize(test_path)
                    if size < 200:
                        if verbose:
                            print(f"Warning: extracted {s57[0]} is very small ({size} bytes)")
                    # simple header sniff: S-57 often contains 'ISO' or 'S-57' bytes; check first 256 bytes
                    with open(test_path, 'rb') as fh:
                        head = fh.read(256)
                    if b'ISO' not in head and b'S-57' not in head and b'000' not in head[:20]:
                        if verbose:
                            print(f"Note: {s57[0]} header doesn't look like a typical S-57 file; proceeding but fiona may fail to open it")
                except Exception as e:
                    if verbose:
                        print(f"Failed to inspect {test_path}: {e}")
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
        # Use only 'medium' for faster loading, or 'best' for highest detail
        # Using both creates too many charts and slows down GUI initialization
        resolutions = ['medium']  # was: ['best', 'medium']
        print(f"[ENC] Using resolutions: {resolutions}")  # DEBUG
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
        
        # Use persistent cache directory instead of temp
        cache_root = Path.home() / '.emergent_cache' / 'enc'
        cache_root.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[ENC] Cache directory: {cache_root}")
    
        for cell in cells:
            cell_id = cell.get('cell_id', 'unknown')
            cell_cache_dir = cache_root / cell_id
            
            # Check if already cached
            if cell_cache_dir.exists() and list(cell_cache_dir.glob("*.000")):
                if verbose:
                    print(f"[ENC] Using cached: {cell_id}")
                cell_dir = cell_cache_dir
            else:
                if verbose:
                    print(f"Fetching ENC cell: {cell['url']}")
                cell_cache_dir.mkdir(parents=True, exist_ok=True)
                cell_dir = self.fetch_and_extract_enc(
                    cell['url'],
                    extract_to=str(cell_cache_dir),
                    verbose=verbose
                )
            # only iterate the true S-57 base (.000) files; skip .001 updates
            enc_files = list(Path(cell_dir).glob("*.000"))
            
            # track whether key layers were present in this cell
            cell_had_lnd = False
            cell_had_coal = False
            for enc_file in enc_files:
                if verbose:
                    print(f"  Opening ENC file: {enc_file.name}")
                try:
                    layers = fiona.listlayers(str(enc_file))
                except Exception as e:
                    if verbose:
                        print(f"    SKIP: cannot list layers for {enc_file.name}: {e}")
                    continue
                for layer in layers:
                    if verbose:
                        print(f"    reading layer: {layer}")
                    if layer == 'LNDARE':
                        cell_had_lnd = True
                    if layer == 'COALNE':
                        cell_had_coal = True
                    try:
                        with fiona.open(str(enc_file), layer=layer) as src:
                            gdf = gpd.GeoDataFrame.from_features(src)
                            # skip empty frames and ensure CRS then filter and reproject
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
                    except Exception as e:
                        if verbose:
                            print(f"    Skipped {layer}: {e}")
                        continue

            # report per-cell summary so we can diagnose missing coverage (donut holes)
            if verbose:
                print(f"[ENC] cell {cell_id}: LNDARE={cell_had_lnd}, COALNE={cell_had_coal}")
            # emit progress for GUI: percent of cells processed
            try:
                total = len(cells) if len(cells) > 0 else 1
                processed = getattr(self, '_enc_processed', 0) + 1
                self._enc_processed = processed
                self._enc_progress = int(100.0 * processed / total)
                # also push a short log line for visibility
                try:
                    self.log_lines.insert(0, f"[ENC] processed {cell_id} ({self._enc_progress}%)")
                    self.log_lines = self.log_lines[: self.max_log_lines]
                except Exception:
                    pass
            except Exception:
                pass
    
        keep_layers = {
            'LNDARE',     # land areas
            'DEPARE',     # shoal/depth areas
            'DEPVAL',     # contour lines
            'COALNE',     # coastline outlines
            'BRIDGE',     # bridge polygons
            'BOY',        # buoy/marker points
            'BOYINB',     # 
            'DRVAL2',     # depth‐value labels
            'OBSTRN'      # obstructions
        }
        
        all_layers = set(self.enc_data.keys())
        for layer in (all_layers - keep_layers):
            del self.enc_data[layer]
    
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

        # If COALNE absent or empty, attempt to synthesize coastline from LNDARE polygons
        # If COALNE absent or empty, attempt to synthesize coastline from LNDARE polygons
        if ("COALNE" not in self.enc_data) or (self.waterway is None) or self.waterway.empty:
            try:
                lnd = self.enc_data.get('LNDARE')
                if lnd is not None and not lnd.empty:
                    # merge land polygons and take their exterior boundaries as coastline lines
                    merged = unary_union(list(lnd.geometry))
                    boundaries = merged.boundary
                    # boundaries can be LineString or MultiLineString
                    geoms = [boundaries] if hasattr(boundaries, 'geom_type') else list(boundaries)
                    coal_gdf = gpd.GeoDataFrame(geometry=geoms, crs=self.crs_utm)
                    self.enc_data['COALNE'] = coal_gdf
                    self.waterway = coal_gdf
                    print("[ENC] COALNE synthesized from LNDARE (merged land boundaries)")
            except Exception as e:
                print(f"[ENC] Failed to synthesize COALNE from LNDARE: {e}")

    def synthesize_coastline(self):
        """Ensure that a coastline layer (COALNE) exists in self.enc_data.
        If COALNE is missing or empty but LNDARE (land polygons) exist, synthesize
        the coastline from land polygon boundaries. Returns True if synthesized.
        """
        try:
            coal = self.enc_data.get('COALNE')
            if coal is not None and not getattr(coal, 'empty', False):
                return False
            lnd = self.enc_data.get('LNDARE')
            if lnd is None or getattr(lnd, 'empty', True):
                return False
            merged = unary_union(list(lnd.geometry))
            boundaries = merged.boundary
            geoms = [boundaries] if hasattr(boundaries, 'geom_type') else list(boundaries)
            coal_gdf = gpd.GeoDataFrame(geometry=geoms, crs=self.crs_utm)
            self.enc_data['COALNE'] = coal_gdf
            self.waterway = coal_gdf
            print("[ENC] COALNE synthesized from LNDARE (synthesize_coastline call)")
            return True
        except Exception as e:
            print(f"[ENC] synthesize_coastline failed: {e}")
            return False

        # ── NEW: re-project every GeoDataFrame to the sim’s UTM CRS ─────────
        for k, gdf in self.enc_data.items():
            if not gdf.empty and gdf.crs != self.crs_utm:
                self.enc_data[k] = gdf.to_crs(self.crs_utm)

    # def spawn_agents_at_entrance(self):
    #     """
    #     Programmatic spawn: place each of n agents evenly along the left (min-x) boundary,
    #     set their initial surge speed, and point goals at the opposite side.

    #     Returns:
    #         state0 (4×n ndarray): initial [u, v, p, r] for each ship
    #         pos0   (2×n ndarray): initial [x, y] for each ship
    #         psi0   (n,) ndarray: initial heading for each ship
    #         goals_arr (2×n ndarray): goal [x, y] for each ship
    #     """
    #     # Domain limits in UTM (meters)
    #     # x0, x1 = self.ax.get_xlim()
    #     # y0, y1 = self.ax.get_ylim()
    #     n = self.n

    #     # Evenly space along left edge for spawns
    #     ys = np.linspace(y0, y1, n)
    #     xs = np.full(n, x0)
    #     pos0 = np.vstack((xs, ys))

    #     # Goals along right edge at same y
    #     gx = np.full(n, x1)
    #     gy = ys.copy()
    #     goals_arr = np.vstack((gx, gy))

    #     # Initial heading toward each goal
    #     psi0 = np.arctan2(gy - ys, gx - xs)

    #     # Initial state: surge = spawn_speed, other components zero
    #     state0 = np.zeros((4, n), dtype=float)
    #     state0[0, :] = getattr(self, 'spawn_speed', PROPULSION['desired_speed'])

    #     # store into self for later updating in run()
    #     self.state = state0
    #     self.pos   = pos0
    #     self.psi   = psi0
    #     self.goals = goals_arr

    #     # position each ship‐patch at its spawn point
    #     # (self.base is your Nx2 polygon template)
    #     for i, patch in enumerate(self.patches):
    #         # shift the base polygon by the agent’s (x,y)
    #         patch.set_xy(self.base + pos0[:, i])

    #     return state0, pos0, psi0, goals_arr

    def spawn(self):
        """
        Interactive spawn and routing: collect waypoints first, then initialize states.

        - Uses `route()` to let user click all desired waypoints (including start and goal).
        - Sets initial position to first click, heading toward second click, and goal to last click.
        """
        # 1) Make sure way-points exist. In test-mode ("zigzag") we fabricate
        #    a trivial 2-point leg so the rest of spawn() can proceed unaltered.
        if not hasattr(self, 'waypoints') or len(self.waypoints) != self.n:
            if getattr(self, "test_mode", None) == "zigzag":
                # straight east, 100 m — arbitrary but harmless
                self.waypoints = [[(0.0, 0.0), (100.0, 0.0)]
                                  for _ in range(self.n)]
            elif getattr(self, "test_mode", None) == "turning_circle":
                # Make a full 360-degree circle with waypoints
                radius = getattr(self, "tc_radius", 500.0)  # meters
                num_pts = 36
                angles = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
                circle = list(zip(x, y))
                self.waypoints = [circle for _ in range(self.n)]
            else:
                raise RuntimeError(
                    "Cannot spawn: self.waypoints not set. "
                    "Call route() (in viewer) before spawn()."
                )

        # Debug instrumentation: if waypoints already exist at spawn time,
        # log a short stack trace at DEBUG level to help locate the caller
        # setting them. Only emit this when self.verbose is True (batch
        # runs should remain quiet).
        try:
            import traceback
            if getattr(self, 'verbose', False) and hasattr(self, 'waypoints') and len(getattr(self, 'waypoints', [])) > 0:
                log.debug("[Simulation.spawn] Spawn called with pre-existing waypoints (n_agents=%s, waypoints_len=%s)", self.n, len(self.waypoints))
                stack = traceback.format_stack(limit=8)
                log.debug("[Simulation.spawn] recent stack (most recent call last):")
                for line in stack[:-1]:
                    # skip the current frame's own line
                    log.debug(line.strip())
        except Exception:
            # Best-effort: never raise during spawn for logging
            pass



        # 2) Derive state arrays from waypoints
        n = self.n
        # Initialize arrays
        state0 = np.zeros((4, n), dtype=float)
        pos0 = np.zeros((2, n), dtype=float)
        psi0 = np.zeros(n, dtype=float)
        goals_arr = np.zeros((2, n), dtype=float)
        
        # 3) (no-op) Viewer will draw start/goal if desired
        
        # Set initial surge speed
        spawn_speed = getattr(self, 'spawn_speed', PROPULSION['desired_speed'])
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

        # 4) Instantiate ship and assign to simulation state
        self.state  = state0.copy()
        self.pos    = pos0.copy()
        self.psi    = psi0.copy()
        self.goals  = goals_arr.copy()
        self.ship = ship(state0,pos0,psi0,goals_arr)
        # propagate verbosity preference to ship model so its debug prints respect the sim flag
        try:
            setattr(self.ship, 'verbose', bool(self.verbose))
        except Exception:
            pass
        self.ship.wpts = self.waypoints
        self.ship.short_route = self.waypoints
        self.history = { i: [self.pos[:,i].copy()] for i in range(self.n) }

        # ── TEST-MODE: lock the “straight ahead” heading after spawn ──
        if getattr(self, "test_mode", None) == "zigzag":
            self.zz_base_psi = float(self.psi[0])

        # ─── Clear any AIS heatmap & reset to ENC‐only background ──────────────
        # Switch back to sim mode so that run() sees only the ENC chart.
        self.mode = 'sim'
        # ──────────────────────────────────────────────────────────────────────

        if getattr(self, "test_mode", None) == "turncircle":
            rud_deg = getattr(self, "tc_rudder_deg", 20.0)
            self.ship.test_mode = "turncircle"
            self.ship.test_rudder_deg = rud_deg
            self.ship.constant_rudder_cmd = np.deg2rad(rud_deg)  # constant rudder in rad

        return state0, pos0, psi0, goals_arr

    def route(self, *args, **kwargs):
        """
        Dummy route(): no GUI here. Viewer must set self.waypoints
        to a list of waypoint‐lists before calling spawn().
        Here we only compute final_goals and user_routes.
        """
        if not hasattr(self, 'waypoints') or len(self.waypoints) != self.n:
            raise RuntimeError(
                "Route not set: viewer must populate self.waypoints before calling route()"
            )

        # Build final_goals from last waypoint of each agent
        self.final_goals = np.zeros((2, self.n), dtype=float)
        for idx, wp in enumerate(self.waypoints):
            self.final_goals[:, idx] = wp[-1]

        # Copy into user_routes for backwards compatibility
        self.user_routes = list(self.waypoints)
        

    def load_route_heatmap(
        self,
        start_date,
        end_date,
        grid_size=(100, 100)
    ):
        """
        Downloads daily AIS ZIPs from NOAA (for the given date range),
        filters to this port’s bounding box, builds a cumulative 2D heatmap,
        and draws it as the “route” background.

        Parameters
        ----------
        start_date : datetime.date
            First date (inclusive) to request (e.g., date(2024,5,1)).
        end_date : datetime.date
            Last date (inclusive) to request (e.g., date(2024,5,31)).
        grid_size : tuple of ints, optional
            (nx, ny) bins for the 2D histogram. Defaults to (800, 600).
        """
        if not self.use_ais:
            return                         # skip when AIS disabled
        # 1) Look up this port’s lon/lat bounding box from SIMULATION_BOUNDS
        bb = SIMULATION_BOUNDS[self.port_name]
        minx_ll, maxx_ll = bb["minx"], bb["maxx"]
        miny_ll, maxy_ll = bb["miny"], bb["maxy"]
        bbox_ll = (minx_ll, miny_ll, maxx_ll, maxy_ll)
        
        # 2) Parse strings into datetime.date and generate the AIS heatmap
        import pandas as _pd
        if isinstance(start_date, str):
            start_date = _pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = _pd.to_datetime(end_date).date()
            
        # 3) Delegate to our NOAA‐AIS downloader
        # ─── Logging: start of AIS heatmap generation ─────────────────────────
        log.info(f"[AIS] Computing heatmap for {self.port_name}: "
                 f"{start_date.isoformat()} → {end_date.isoformat()}, "
                 f"bbox={bbox_ll}, grid={grid_size}")
        try:
            heatmap, extent = compute_ais_heatmap(
                bbox       = bbox_ll,
                start_date = start_date,
                end_date   = end_date,
                grid_size  = grid_size,
                year       = start_date.year
            )
        except Exception as e:
            log.error(f"[AIS] compute_ais_heatmap failed: {e}")
            return  # abort early if AIS generation fails

        # 4) Store results for later use (imshow, etc.)
        self.ais_heatmap = heatmap
        self.ais_extent  = extent

        # 5) Overlay AIS heatmap on top of current ENC background, if in 'route' mode
        if self.mode == 'route':
            # Parse dates if provided as strings
            # (allow either datetime.date or "YYYY-MM-DD" strings)
            import pandas as _pd
            if isinstance(start_date, str):
                start_date = _pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = _pd.to_datetime(end_date).date()

            # Recompute heatmap if needed (in case bbox or dates changed)
            # (heatmap & extent already populated above)
            # Now overlay it via imshow, without clearing the existing ENC chart
            minlon, maxlon, minlat, maxlat = self.ais_extent
            # Convert lon/lat extent to UTM coordinates

            ll_box = box(minlon, minlat, maxlon, maxlat)
            utm_box = (
                gpd.GeoDataFrame(geometry=[ll_box], crs="EPSG:4326")
                .to_crs(self.crs_utm)
            )
            utm_minx, utm_miny, utm_maxx, utm_maxy = utm_box.total_bounds


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
        Return a list of collision events for any overlapping ship hulls.
        Each event is a dict containing indices, contact area, relative speed,
        and computed impact energies per vessel.
        """
        events = []
        # Test every pair of ships using freshly‑built hulls
        for i in range(self.n):
            poly_i = self._current_hull_poly(i)
            for j in range(i + 1, self.n):
                poly_j = self._current_hull_poly(j)
                if poly_i.intersects(poly_j):
                    inter = poly_i.intersection(poly_j)
                    if inter.area > self.collision_tol_area:
                        # compute relative speed (world frame)
                        psi_i, psi_j = self.psi[i], self.psi[j]
                        u_i, v_i = self.state[0, i], self.state[1, i]
                        u_j, v_j = self.state[0, j], self.state[1, j]
                        vel_i = np.array([u_i * np.cos(psi_i) - v_i * np.sin(psi_i),
                                          u_i * np.sin(psi_i) + v_i * np.cos(psi_i)])
                        vel_j = np.array([u_j * np.cos(psi_j) - v_j * np.sin(psi_j),
                                          u_j * np.sin(psi_j) + v_j * np.cos(psi_j)])
                        rel_v = vel_j - vel_i
                        rel_speed = float(np.linalg.norm(rel_v))

                        # Use masses from ship model; if vectorized, extract per-index
                        try:
                            m_i = float(self.ship.m)
                            m_j = float(self.ship.m)
                        except Exception:
                            m_i = getattr(self.ship, 'm', 1.0)
                            m_j = getattr(self.ship, 'm', 1.0)

                        # Compute per-ship kinetic energies at impact in Joules
                        # KE = 0.5 * m * v^2 (use magnitude of body-relative velocities)
                        # Use half the relative speed apportioned by mass ratio for rough estimate
                        ke_rel = 0.5 * (m_i * (rel_speed ** 2) * (m_j / (m_i + m_j)))
                        ke_i = 0.5 * m_i * (rel_speed ** 2) * (m_j / (m_i + m_j))
                        ke_j = 0.5 * m_j * (rel_speed ** 2) * (m_i / (m_i + m_j))

                        ev = {
                            't': float(t),
                            'i': int(i),
                            'j': int(j),
                            'contact_area': float(inter.area),
                            'rel_speed_m_s': rel_speed,
                            'ke_i_J': ke_i,
                            'ke_j_J': ke_j,
                        }
                        log.error(f"Collision detected between Ship{i} and Ship{j} at t={t:.2f}s, rel_speed={rel_speed:.2f} m/s")
                        events.append(ev)
        return events

    def _check_allision(self, t):
        """
        Return a list of allision events (ship index and intersecting land geometry)
        if any ship hull polygon intersects land.
        """
        events = []
        # If no waterway geometry is available, return empty
        land_geoms = list(getattr(self, 'waterway', gpd.GeoDataFrame()).geometry)
        if not land_geoms:
            return events

        ship_polys = [ShapelyPolygon(patch.get_xy()) for patch in self.patches]
        for i, ship_poly in enumerate(ship_polys):
            for land in land_geoms:
                if ship_poly.intersects(land):
                    inter = ship_poly.intersection(land)
                    ev = {'t': float(t), 'i': int(i), 'contact_area': float(inter.area)}
                    log.error(f"Allision detected: Ship{i} with shore at t={t:.2f}s")
                    events.append(ev)
        # record to persistent list
        for e in events:
            self.allision_events.append(e)
        return events

    def _update_overlay(self, t):
        """
        Update wind‐vane arrow, header text, and log lines—all on self.ax.
        """
        # 1) Compute raw wind vector for vessel 0 (or dict‐style)
        wind_raw = playful_wind(self.state, t)
        if isinstance(wind_raw, dict):
            wx = float(wind_raw['speed'] * np.cos(wind_raw['dir']))
            wy = float(wind_raw['speed'] * np.sin(wind_raw['dir']))
        else:
            wx = float(wind_raw[0, 0])
            wy = float(wind_raw[1, 0])
    
        # 2) Compute wind‐arrow tip in axes‐fraction space
        theta = np.arctan2(wy, wx)                   # heading of wind
        dx = 0.05 * np.cos(theta)
        dy = 0.05 * np.sin(theta)
    
        # tx, ty = self._wing_tail
        # tip_x = tx + dx
        # tip_y = ty + dy
    
        # # clamp to [0,1] so arrow never leaves the axesframe
        # tip_x = max(0.0, min(1.0, tip_x))
        # tip_y = max(0.0, min(1.0, tip_y))
    
        # update arrow positions
        new_posB = (self._wing_tail[0] + dx, self._wing_tail[1] + dy)

    def run(self):
        """
        Advance the simulation through all time steps (headless).
        This version no longer does any plotting or GUI work.
        """
        total_steps = self.steps #int(self.T / self.dt)
        self.stop = False


        for step in range(total_steps):
            t = step * self.dt
            self.t = t

            # 1) update goals only if agent navigation is enabled
            if self.test_mode not in ("zigzag", "turning_circle"):
                self._update_goals()

            if getattr(self, 'verbose', False):
                log.debug("[SIM] Time step: %s, t=%.1f s", step, t)
            if self.stop:
                if getattr(self, 'verbose', False):
                    log.info("[SIM] Halted at t=%.1f s.", t)
                break
                break

            # 2) pack body‐fixed velocities into nu = [u; v; r]
            nu = np.vstack([self.state[0],    # surge speed u
                            self.state[1],    # sway speed v
                            self.state[3]])   # yaw‐rate r
            # 3) compute controls using the current velocities
            hd, sp, rud = self._compute_controls_and_update(nu, t)

            self.hd_cmds = hd  # <- cache for history tracking

            # Integrate dynamics and record new state
            self._step_dynamics(hd, sp, rud)

            # Check for collisions or allisions
            collision_events = self._check_collision(t)
            grounded = self._check_allision(t)
            if collision_events or grounded:
                # If there are individual collision events, stop the involved vessels and record events
                if collision_events:
                    for ev in collision_events:
                        i = ev['i']
                        j = ev['j']
                        # stop the two vessels immediately
                        try:
                            self.ship.cut_power(i)
                        except Exception:
                            pass
                        try:
                            self.ship.cut_power(j)
                        except Exception:
                            pass
                        # add to persistent collisions list
                        self.collision_events.append(ev)
                        # also log to console
                        # record collision but only log if verbose requested
                        if getattr(self, 'verbose', False):
                            log.warning("[SIM] Collision: Ships %s&%s at t=%.1f s rel_speed=%.2f m/s KE_i=%.1f J KE_j=%.1f J",
                                        i, j, t, ev.get('rel_speed_m_s', 0.0), ev.get('ke_i_J', 0.0), ev.get('ke_j_J', 0.0))

                # If grounded (allision) or any collision happened we freeze simulation state
                # Apply a conservative hard-stop across all vessels
                try:
                    self.state[[0,1,3], :] = 0.0            # u, v, r → 0
                except Exception:
                    pass
                try:
                    self.ship.current_speed[:] = 0.0
                    self.ship.commanded_rpm[:] = 0.0
                    self.ship.desired_speed[:] = 0.0
                except Exception:
                    pass

                if getattr(self, 'verbose', False):
                    log.info("[SIM] %s at t=%.1f s — vessels frozen.", 'Collision' if collision_events else 'Allision', t)
                break
            
            # -- record for post-run analysis
            self.t_history.append(t)
            # psi[0] = heading of agent 0 in radians
            # store normalized heading in [-pi, pi]
            psi0 = ((self.psi[0] + np.pi) % (2*np.pi)) - np.pi
            self.psi_history.append(psi0)
            # hd_cmds[0] = commanded heading for agent 0
            self.hd_cmd_history.append(self.hd_cmds[0])
            # rudder recorded from last _compute_rudder output (rad)
            try:
                # ensure rudder_history exists
                if not hasattr(self, 'rudder_history'):
                    self.rudder_history = []
                # store agent-0 rudder
                self.rudder_history.append(float(rud[0]))
            except Exception:
                # fallback: append nan if something went wrong
                try:
                    self.rudder_history.append(float('nan'))
                except Exception:
                    pass

        if getattr(self, 'verbose', False):
            log.info("[SIM] Completed at t=%.1f s", self.t)

    def _startup_diagnostics(self, total_steps):
        if getattr(self, 'verbose', False):
            log.info("[SIM] Starting: %s agents, dt=%s, steps=%s", self.n, self.dt, total_steps)
            log.info("[SIM] Initial pos: %s", self.pos.T.round(2))
        
    def _update_goals(self):
        """
        Pops reached way-points and recomputes LOS for each agent.
        Short-circuit when we’re in a unit-test manoeuvre (zig-zag, etc.).
        """
        if getattr(self, "test_mode", None) == "zigzag":      # ← NEW
            return                                            # skip routing
        #tol = getattr(self, 'wp_tol', self.L * 2.)
        tol = getattr(self, 'wp_tol', self.L * 2.)

        for i in range(self.n):
            wpts = self.waypoints[i]
            if not wpts:
                # No waypoints to follow, skip this agent
                continue

            # If close to the current waypoint and there are still others, advance
            if len(wpts) > 1 and np.linalg.norm(self.pos[:, i] - wpts[0]) < tol:
                wpts.pop(0)

            # Aim not just at the immediate waypoint but at a short lookahead target
            # to smooth the course over multiple legs. Use up to the next 3 waypoints.
            lookahead = 3
            n_w = min(len(wpts), lookahead)
            if n_w >= 2:
                # weighted average favoring nearer waypoints
                weights = np.linspace(1.0, 0.5, n_w)
                weights = weights / weights.sum()
                pts = np.stack([wpts[k] for k in range(n_w)], axis=0)
                target = np.dot(weights, pts)
                self.goals[:, i] = target
            else:
                # Always aim at the first (current) waypoint
                self.goals[:, i] = wpts[0]

    def _compute_controls_and_update(self, nu, t):
        # ──────────────────────────────────────────────────────────────
        # TEST-MODE OVERRIDE   (simple ±zig-zag without ENC / COLREGS)
        # ──────────────────────────────────────────────────────────────
        if getattr(self, "test_mode", None) == "zigzag":
            if t >= self.zz_next_sw:
                self.zz_sign   *= -1
                self.zz_next_sw += self.zz_hold
            hd_cmds = np.full(self.n, self.zz_base_psi + self.zz_sign * self.zz_delta)
            sp_cmds = np.full(self.n, self.zz_sp_cmd)
            roles   = ["neutral"] * self.n
            rud_cmds = self._compute_rudder(hd_cmds, roles)
            return hd_cmds, sp_cmds, rud_cmds

        # ─── Sample wind & current at each ship’s lon/lat ───────────────
        now = datetime.now(timezone.utc)
        lon, lat = self._utm_to_ll.transform(self.pos[0], self.pos[1])
        # wind_fn/current_fn expect (lon, lat)
        wind_vec    = self.wind_fn(lon, lat, now).T   # shape (2, n)
        current_vec = self.current_fn(lon, lat, now).T
    
        # Now pass the raw environmental drift vector (wind + current).
        # compute_desired expects the environmental *current* used to compute
        # a correction for steering INTO the drift. The caller should provide
        # the vector which represents how the ship should compensate (i.e.
        # steer into the drift), so pass the *negated* environmental push
        # (wind+current) here. Previously we passed the raw push which had
        # the wrong sign and produced headings ~180° off in some scenarios.
        combined_drift_vec = -(current_vec + wind_vec)
        
        # 1) COLREGS override
        col_hd, col_sp, _, roles = self.ship.colregs(
            self.dt, self.pos, nu, self.psi, self.ship.commanded_rpm
        )

        # ─── 3) Compute desired track WITH drift built-in ───────────────────────
        goal_hd, goal_sp = self.ship.compute_desired(
            self.goals,
            self.pos[0], self.pos[1],
            self.state[0], self.state[1], self.state[3], self.psi,
            current_vec = combined_drift_vec
        )
        
        # 4) Fuse COLREGS + PID, then compute rudder
        hd, sp = self._fuse_and_pid(goal_hd, goal_sp, col_hd, col_sp, roles)
        rud    = self._compute_rudder(hd, roles)
        return hd, sp, rud
        
        # # ─── Step 1: ask for the raw “no-drift” track
        # raw_goal_hd, goal_sp = self.ship.compute_desired(
        #     self.goals,
        #     self.pos[0], self.pos[1],
        #     self.state[0], self.state[1], self.state[3], self.psi,
        #     # zero drift so that compute_desired just points at the green line
        #     current_vec = np.zeros_like(current_vec)
        # )
        
        # # ─── Step 2: apply *our* compensation
        # goal_hd = raw_goal_hd.copy()
        # for i in range(self.n):
        #     # convert to compass bearing [0–360)
        #     track_deg = (np.degrees(raw_goal_hd[i]) + 360) % 360
        
        #     # --- swap & (optionally) negate so we feed [east, north] into compensate_heading ---
        #     # unpack raw [north, east]
        #     w_north, w_east = wind_vec[:, i]
        #     c_north, c_east = current_vec[:, i]
    
        #     # steer-into drift = NEGATE each axis-swapped component
        #     wind_for_comp    = np.array([-w_east, -w_north], dtype=float)
        #     current_for_comp = np.array([-c_east, -c_north], dtype=float)
        
        #     comp_deg = self.ship.compensate_heading(
        #         track_bearing_deg = track_deg,
        #         wind_vec          = wind_for_comp,
        #         current_vec       = current_for_comp,
        #         surge_speed       = float(self.state[0, i])
        #     )
        
        #     # feed back in radians
        #     goal_hd[i] = np.radians(comp_deg)
        
        #     # debug print so you can see exactly what’s happening:
        #     #print(f"[DRIFT] A{i}: track {track_deg:.1f}°, drift=({drift[0]:.2f},{drift[1]:.2f}), comp→{comp_deg:.1f}°")

        # # 1) COLREGS override
        # col_hd, col_sp, _, roles = self.ship.colregs(
        #     self.dt, self.pos, nu, self.psi, self.ship.commanded_rpm
        # )

        # # 2) “Point-at-route” heading with zero drift
        # raw_goal_hd, goal_sp = self.ship.compute_desired(
        #     self.goals,
        #     self.pos[0], self.pos[1],
        #     self.state[0], self.state[1], self.state[3], self.psi,
        #     current_vec = np.zeros_like(current_vec)
        # )

        # # 3) Manual dead-reckoning: sum wind + current and compute a lateral offset
        # goal_hd = raw_goal_hd.copy()
        # drift   = wind_vec + current_vec    # world-frame environmental forcing
        # for i in range(self.n):
        #     h = raw_goal_hd[i]
        #     # port-side unit vector in world frame
        #     perp      = np.array([-np.sin(h), np.cos(h)])
        #     lateral   = float(drift[:, i].dot(perp))
        #     surge     = float(self.state[0, i]) + 1e-6
        #     hd_offset = np.arctan2(lateral, surge)
        #     goal_hd[i] = h + hd_offset

        
        # # now fuse with COLREGS and PID as before
        # hd, sp = self._fuse_and_pid(goal_hd, goal_sp, col_hd, col_sp, roles)
        # rud = self._compute_rudder(hd, roles)
        # return hd, sp, rud
    
        # # 4) fuse nav+COLREGS, then PID → rudder
        # hd, sp = self._fuse_and_pid(goal_hd, goal_sp, col_hd, col_sp, roles)
        # rud = self._compute_rudder(hd, roles)
        # return hd, sp, rud



    def _fuse_and_pid(self, goal_hd, goal_sp, col_hd, col_sp, roles):
        #roles = np.array(self.ship.colregs(self.pos, np.vstack([self.state[0],self.state[1],self.state[3]]), self.psi, self.ship.commanded_rpm)[3])
        roles_arr = np.asarray(roles)           # ensure element-wise compare
        is_give = (np.asarray(roles) == 'give_way')     # element-wise mask
        hd = np.where(is_give, col_hd, goal_hd)
        sp = np.where(is_give, col_sp, goal_sp)
        #raw_rud = self.ship.pid_control(self.psi, hd, self.dt)
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
        # optional tracing
        try:
            from emergent.ship_abm.config import PID_TRACE
        except Exception:
            PID_TRACE = {'enabled': False, 'path': None}
    
        # 1) Heading error in [-π, π]
        # canonical form: err = hd - psi  (use helper to ensure consistent wrapping)
        try:
            from emergent.ship_abm.angle_utils import heading_diff_rad
            err = heading_diff_rad(psi_ref, self.psi)
        except Exception:
            # fallback (previous behavior used psi_ref - self.psi but we want hd - psi)
            err = ((psi_ref - self.psi + np.pi) % (2 * np.pi)) - np.pi
        # normalize to 1-D arrays of length self.n to avoid shape mismatches
        err = np.asarray(err).ravel()
        if err.size != self.n:
            # if scalar or unexpected shape, broadcast/resize to match self.n
            err = np.resize(err, (self.n,))
    
        # 2) Feed-forward: desired turn rate
        Kf = self.tuning['Kf']                # feed-forward gain
        r_rate_max = np.radians(self.tuning['r_rate_max_deg'])
        r_des = np.clip(Kf * err, -r_rate_max, r_rate_max)
        # Conservative feed-forward guard: if heading error is very large,
        # downscale feed-forward so the controller doesn't command an
        # aggressive turn-rate based solely on a large error (prevents
        # large r_des that the ship cannot follow and leads to saturation).
        try:
            ff_err_limit_deg = float(self.tuning.get('ff_err_limit_deg', 10.0))
        except Exception:
            ff_err_limit_deg = 10.0
        ff_limit_rad = np.deg2rad(ff_err_limit_deg)
        # scale down r_des where |err| exceeds the threshold
        try:
            large_err_mask = np.abs(err) > ff_limit_rad
            if np.any(large_err_mask):
                # apply conservative scale (retain a small portion of feed-forward)
                r_des = np.where(large_err_mask, r_des * 0.25, r_des)
        except Exception:
            pass
    
        # 3) Measured turn-rate
        # Compute angle difference with wrap-around protection so crossing the
        # -pi/pi boundary doesn't produce a huge false yaw-rate spike.
        # Also guard the very first call if prev_psi is not initialized.
        if not hasattr(self, 'prev_psi') or self.prev_psi is None:
            # no previous heading recorded -> assume zero turn-rate on first step
            r_meas = np.zeros_like(self.psi)
            # initialize prev_psi so subsequent steps are valid
            self.prev_psi = self.psi.copy()
        else:
            # minimal signed angle difference in [-pi, pi]
            delta_psi = ((self.psi - self.prev_psi + np.pi) % (2 * np.pi)) - np.pi
            r_meas = delta_psi / dt
        r_meas = np.asarray(r_meas).ravel()
        if r_meas.size != self.n:
            r_meas = np.resize(r_meas, (self.n,))

        # 3a) Derivative low-pass filtering (first-order) to reduce noisy D-action
        # r_filtered[k] = r_filtered[k-1] + alpha * (r_meas - r_filtered[k-1])
        deriv_tau = self.tuning.get('deriv_tau', 0.0)
        if not hasattr(self, '_r_filtered'):
            self._r_filtered = r_meas.copy()
        if deriv_tau and deriv_tau > 0.0:
            alpha = dt / (deriv_tau + dt)
            self._r_filtered = self._r_filtered + alpha * (r_meas - self._r_filtered)
        else:
            self._r_filtered = r_meas.copy()

        # 4) PID on turn-rate error (use filtered measured r)
        derr = r_des - self._r_filtered
        derr = np.asarray(derr).ravel()
        if derr.size != self.n:
            derr = np.resize(derr, (self.n,))

        # Tentative integrator update (anti-windup via conditional / back-calculation)
        # Compute a candidate integral and test for saturation after combining terms.
        Kp = self.tuning['Kp']
        Ki = self.tuning['Ki']
        Kd = self.tuning['Kd']

        # Gain-schedule: reduce proportional gain when heading error is large
        try:
            kp_reduce_deg = float(self.tuning.get('kp_reduce_deg', 15.0))
        except Exception:
            kp_reduce_deg = 15.0
        kp_reduce_rad = np.deg2rad(kp_reduce_deg)
        try:
            large_err_mask = np.abs(err) > kp_reduce_rad
            if np.any(large_err_mask):
                # conservative reduction: halve Kp where error is large
                if np.isscalar(Kp):
                    Kp_eff = np.where(large_err_mask, Kp * 0.5, Kp) if hasattr(large_err_mask, '__iter__') else (Kp * 0.5 if large_err_mask else Kp)
                else:
                    Kp_eff = np.where(large_err_mask, np.asarray(Kp) * 0.5, np.asarray(Kp))
            else:
                Kp_eff = Kp
        except Exception:
            Kp_eff = Kp

        # provisional integral (do not commit yet)
        I_proposed = self.integral_error + err * dt
        I_proposed = np.asarray(I_proposed).ravel()
        if I_proposed.size != self.n:
            I_proposed = np.resize(I_proposed, (self.n,))
        I_max = np.radians(self.tuning['I_max_deg'])
        I_proposed = np.clip(I_proposed, -I_max, I_max)

        # provisional rudder command if we used the proposed integral
        # use scheduled proportional gain (Kp_eff) to avoid overly large P-action
        rud_prov = (Kp_eff * err) + Ki * I_proposed + Kd * derr
        rud_prov = np.asarray(rud_prov).ravel()
        if rud_prov.size != self.n:
            rud_prov = np.resize(rud_prov, (self.n,))

        # Micro-trim: if provisional command is very small, zero it to avoid chatter
        trim_rad = np.radians(self.tuning.get('trim_band_deg', np.degrees(self.tuning.get('trim_band_deg', 1.0))))
        if np.isscalar(rud_prov):
            if abs(rud_prov) < trim_rad:
                rud_prov = 0.0
        else:
            small_mask = np.abs(rud_prov) < trim_rad
            rud_prov[small_mask] = 0.0

        # Now decide whether to accept the integrator update: if rud_prov would saturate
        # aggressively (sign change at saturation), avoid adding to integral (simple anti-windup)
        sat_mask = np.abs(rud_prov) >= self.ship.max_rudder
        sat_mask = np.asarray(sat_mask).ravel()
        if sat_mask.size != self.n:
            sat_mask = np.resize(sat_mask, (self.n,))
        # commit integral only where not saturating
        commit_mask = ~sat_mask
        if np.isscalar(self.integral_error):
            if commit_mask:
                self.integral_error = I_proposed
        else:
            self.integral_error[commit_mask] = I_proposed[commit_mask]

        # final rudder command (use scheduled Kp_eff)
        rud_cmd = (Kp_eff * err) + Ki * self.integral_error + Kd * derr

        # preserve pre-override / pre-saturation command for logging
        # Apply a light low-pass filter to the provisional rudder command to
        # reduce rapid sign-flips (chatter) that can be amplified by high Kp
        try:
            raw_cmd_arr = np.asarray(rud_cmd).ravel()
            prev_raw = getattr(self, 'prev_raw_cmd', np.zeros_like(raw_cmd_arr))
            alpha = 0.35  # smoothing factor: 0<alpha<=1 (smaller -> more smoothing)
            filtered = alpha * raw_cmd_arr + (1.0 - alpha) * prev_raw
            if filtered.size != self.n:
                filtered = np.resize(filtered, (self.n,))
            raw_cmd = filtered.copy()
            # store for next step
            self.prev_raw_cmd = filtered.copy()
        except Exception:
            # fallback to original value if filtering fails for any reason
            raw_cmd = rud_cmd.copy() if hasattr(rud_cmd, 'copy') else np.array(rud_cmd)
    
        # 5) (disabled) Prediction-based early release
        # predicted_err = err + r * self.tuning['lead_time']
        # release_rad   = np.radians(self.tuning['release_band_deg'])
        # early_release = np.abs(predicted_err) < release_rad
        # rud_cmd = np.where(early_release, 0.0, rud_cmd)

        # 6) (disabled) Micro-trim for chatter avoidance
        # trim_rad = np.radians(self.tuning['trim_band_deg'])
        # rud_cmd[np.abs(rud_cmd) < trim_rad] = 0.0
    
        # 6) COLREGS override for give-way: optionally hard opposite rudder beyond 30° error
        err_abs = np.abs(err)
        ov_mask = (np.array(roles) == 'give_way') & (err_abs >= np.radians(30))
        allow_override = COLLISION_AVOIDANCE.get('allow_hard_give_way_override', True)
        if allow_override:
            rud_cmd = np.where(ov_mask,
                               np.sign(err) * self.ship.max_rudder,
                               rud_cmd)
        else:
            # log when we would have overridden (useful for diagnostics)
            if np.any(ov_mask):
                log.info(f"[COLREGS] override suppressed for agents: {list(np.where(ov_mask)[0])}")
    
        # 7) Saturate to max rudder and rate-limit the change
        rud_sat = np.clip(rud_cmd, -self.ship.max_rudder, self.ship.max_rudder)
        max_delta = self.ship.max_rudder_rate * dt
        rud = np.clip(rud_sat,
                      self.prev_rudder - max_delta,
                      self.prev_rudder + max_delta)

        # Integrator back-calculation (small corrective term) when saturation occurred.
        # This gently pulls the integral term toward a value consistent with the
        # saturated actuator to reduce windup over time.
        try:
            backcalc_beta = float(self.tuning.get('backcalc_beta', 0.08))
            # raw_cmd is the filtered pre-saturation command; rud_sat is saturated.
            raw_arr = np.asarray(raw_cmd).ravel()
            sat_arr = np.asarray(rud_sat).ravel()
            if raw_arr.size != self.n:
                raw_arr = np.resize(raw_arr, (self.n,))
            if sat_arr.size != self.n:
                sat_arr = np.resize(sat_arr, (self.n,))
            diff = raw_arr - sat_arr
            # Update integral with a small back-calculation step (units: rad*s)
            # Only apply where saturation magnitude is significant
            sat_apply = np.abs(diff) > 1e-6
            if np.isscalar(self.integral_error):
                if sat_apply:
                    self.integral_error -= backcalc_beta * diff * dt
            else:
                self.integral_error = self.integral_error - (backcalc_beta * diff * dt * sat_apply)
        except Exception:
            # be conservative: don't crash the sim on any error here
            pass

        # ---------- Rudder inversion auto-detect & corrective flip ----------
        # Compare previously applied rudder (self.prev_rudder) with measured yaw-rate
        # r_meas: if prev_rudder has significant magnitude but r_meas has opposite
        # sign for several consecutive steps, assume the plant interprets rudder
        # with an inverted sign and flip the commanded rudder for that agent.
        try:
            prev_r = np.asarray(self.prev_rudder).ravel()
            prev_r = np.resize(prev_r, (self.n,))
            # detection mask: prev rudder sizeable and measured yaw sizeable and opposite signs
            # Do not attempt flip while locked out for an agent
            unlocked = self._rudder_rev_lock_until <= self.t
            detect_mask = (np.abs(prev_r) > np.radians(1.0)) & (np.abs(r_meas) > self._rudder_rev_threshold) & ((np.sign(prev_r) * np.sign(r_meas)) < 0) & unlocked
            # update per-agent counters conservatively (decrement where not detected)
            self._rudder_rev_counter = np.where(detect_mask, self._rudder_rev_counter + 1, np.maximum(self._rudder_rev_counter - 1, 0))
            newly = (self._rudder_rev_counter >= self._rudder_rev_confirm) & (~self._rudder_inverted)
            if np.any(newly):
                # apply flip and perform safe reset actions to avoid immediate oscillation
                for i in np.where(newly)[0]:
                    self._rudder_inverted[i] = True
                    # set a lockout to prevent further flips for a short window
                    lock_seconds = 5.0
                    self._rudder_rev_lock_until[i] = self.t + lock_seconds
                    # reset integrator and prev_rudder to avoid sudden discontinuities
                    try:
                        self.integral_error[i] = 0.0
                    except Exception:
                        pass
                    try:
                        self.prev_rudder[i] = 0.0
                    except Exception:
                        pass
                    # reset counter so we only trigger once until re-detected after lock
                    self._rudder_rev_counter[i] = 0
                    log.warning(f"[RUDDER-AUTO-CORRECT] Agent {i}: detected inverted plant, flipping commanded rudder; lock for {lock_seconds}s")
            # Apply flip to commanded rudder & raw_cmd for any inverted agents so dynamics see corrected sign
            if np.any(self._rudder_inverted):
                if np.isscalar(rud):
                    if bool(self._rudder_inverted):
                        rud = -rud
                        raw_cmd = -raw_cmd
                else:
                    rud = np.where(self._rudder_inverted, -rud, rud)
                    raw_cmd = np.where(self._rudder_inverted, -raw_cmd, raw_cmd)
        except Exception:
            # conservative: if detection fails for any reason, don't crash the sim
            pass

        # optional trace: write per-agent PID internals AFTER saturation/rate-limit
        if PID_TRACE.get('enabled', False):
            import csv, os
            path = PID_TRACE.get('path')
            if path is None:
                path = os.path.abspath('pid_trace_simulation.csv')
            # open file on append, write header if missing
            # add psi (heading), hd_cmd (desired heading), measured turn-rate, and position
            header = ['t','agent','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg',
                      'psi_deg','hd_cmd_deg','r_meas_deg','x_m','y_m']
            write_header = not os.path.exists(path)
            try:
                with open(path, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    if write_header:
                        writer.writerow(header)
                    for idx in range(self.n):
                        # reflect scheduled Kp in trace P-term
                        # Kp_eff may be array-like or scalar
                        try:
                            if np.isscalar(Kp_eff):
                                p_term = float(Kp_eff) * err[idx]
                            else:
                                p_term = float(np.asarray(Kp_eff).ravel()[idx]) * err[idx]
                        except Exception:
                            p_term = Kp * err[idx]
                        i_term = Ki * self.integral_error[idx]
                        d_term = Kd * derr[idx]
                        # For clarity in the trace, present P/I/D after any runtime flip so the sign
                        # shown lines up with the commanded rudder the plant saw (raw_cmd). This
                        # avoids confusing apparent sign-mismatches in post-mortem analysis.
                        if self._rudder_inverted[idx]:
                            p_out = -p_term
                            i_out = -i_term
                            d_out = -d_term
                        else:
                            p_out = p_term
                            i_out = i_term
                            d_out = d_term
                        raw_deg = float(np.degrees(raw_cmd[idx]))
                        r_des_deg = float(np.degrees(r_des[idx])) if hasattr(r_des, '__len__') or np.isscalar(r_des) else float(np.degrees(r_des))
                        # For clarity in traces, normalize (wrap) logged headings to [-180, +180)
                        try:
                            psi_deg_wrapped = ((np.degrees(self.psi[idx]) + 180.0) % 360.0) - 180.0
                        except Exception:
                            psi_deg_wrapped = float(np.degrees(self.psi[idx]))
                        try:
                            # psi_ref may be scalar or array-like
                            raw_ref = psi_ref[idx] if hasattr(psi_ref, '__len__') else psi_ref
                            hd_deg_wrapped = ((np.degrees(raw_ref) + 180.0) % 360.0) - 180.0
                        except Exception:
                            hd_deg_wrapped = float(np.degrees(psi_ref) if hasattr(psi_ref, '__len__') else np.degrees(psi_ref))

                        writer.writerow([
                            float(self.t), int(idx), float(np.degrees(err[idx])), r_des_deg, float(np.degrees(derr[idx])),
                            float(np.degrees(p_out)), float(np.degrees(i_out)), float(np.degrees(d_out)),
                            raw_deg, float(np.degrees(rud[idx])),
                            float(psi_deg_wrapped),
                            float(hd_deg_wrapped),
                            float(np.degrees(r_meas[idx])),
                            float(self.pos[0, idx]) if (hasattr(self, 'pos') and self.pos is not None) else float('nan'),
                            float(self.pos[1, idx]) if (hasattr(self, 'pos') and self.pos is not None) else float('nan')
                        ])
            except Exception:
                pass
    
        # 8) Save state for next cycle
        self.prev_psi = self.psi.copy()
        self.prev_rudder = rud
        self.ship.prev_rudder = rud
        self.ship.smoothed_rudder = rud
   
        return rud

    def _step_dynamics(self, hd, sp, rud):
        # throttle, drag, forces, integrate self.state & self.pos
        thrust = self.ship.thrust(self.ship.speed_to_rpm(sp))
        drag = self.ship.drag(self.state[0])

        # ----------------------------------------------------------------
        # A) Environmental forcing – REAL wind & currents!

        lon, lat = self._utm_to_ll.transform(self.pos[0], self.pos[1])
        # DEBUG: inspect lon/lat shapes just before sampling
        try:
            if getattr(self, 'verbose', False):
                log.debug(f"[SIM-DBG] pos.shape={getattr(self.pos,'shape',None)} lon.shape={getattr(lon,'shape',None)} lat.shape={getattr(lat,'shape',None)}")
        except Exception:
            pass
        wind_vec = self.wind_fn(lon, lat, datetime.now(timezone.utc)).T

        # 2) Sample NOAA field at (lon, lat, now)
        current_vec = self.current_fn(
            lon, lat,
            datetime.now(timezone.utc)   # explicit UTC
        ).T              # returns (N,2) – transpose → (2,N)
        try:
            if getattr(self, 'verbose', False):
                log.debug(f"[SIM-DBG] wind_vec.shape(before)={getattr(wind_vec,'shape',None)} current_vec.shape(before)={getattr(current_vec,'shape',None)}")
        except Exception:
            pass

        # Defensive normalization: ensure wind_vec/current_vec are shape (2, n)
        def _norm_env(ev, name):
            ev = np.atleast_2d(ev)
            # If transposed relative to expectation, try swapping axes
            if ev.shape[0] != 2 and ev.shape[1] == 2:
                ev = ev.T
            # If sampler returned a single column but we have n agents, tile it
            if ev.shape[1] == 1 and self.n > 1:
                ev = np.tile(ev, (1, self.n))
            # If sampler returned more columns than agents, truncate
            if ev.shape[1] > self.n:
                ev = ev[:, :self.n]
            # If sampler produced fewer columns than agents, pad by repeating last column
            if ev.shape[1] < self.n:
                last = ev[:, -1:]
                need = self.n - ev.shape[1]
                ev = np.hstack([ev, np.tile(last, (1, need))])
            return ev

        wind_vec = _norm_env(wind_vec, 'wind')
        current_vec = _norm_env(current_vec, 'current')
        try:
            if getattr(self, 'verbose', False):
                log.debug(f"[SIM-DBG] wind_vec.shape(after)={getattr(wind_vec,'shape',None)} current_vec.shape(after)={getattr(current_vec,'shape',None)}")
        except Exception:
            pass

        wind, current = self.ship.environmental_forces(
             wind_vec, current_vec,
             self.state[0], self.state[1], self.psi
        )
        
        #tide = tide_fn(self.t)
        u_dot,v_dot,p_dot,r_dot = self.ship.dynamics(
            self.state, thrust, drag, wind, current, rud
        )
        # update state & history arrays
        # integrate body-fixed accelerations
        self.state[0] += u_dot * self.dt    # surge acceleration → surge speed
        self.state[1] += v_dot * self.dt    # sway acceleration → sway speed
        self.state[3] += r_dot * self.dt    # yaw rate
        self.psi     += self.state[3] * self.dt

        # In _step_dynamics, after computing wind, current, tide:
        ws = np.hypot(wind_vec[1],wind_vec[0])
        cs = np.hypot(current_vec[1],current_vec[0])
        td = tide_fn(self.t)
        msg = f"Wind={ws} m/s  Curr={cs} m/s  Tide={td} m"
        self.log_lines.insert(0, msg)
        if len(self.log_lines) > self.max_log_lines:
            self.log_lines.pop()

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
            
def heading_error(actual_deg: np.ndarray, commanded_deg: np.ndarray) -> np.ndarray:
    """
    Compute signed error in degrees, wrapped to [–180, +180).
    actual_deg, commanded_deg: 1D arrays of the same length.
    """
    # Use shared helper for consistent degrees wrapping
    try:
        from emergent.ship_abm.angle_utils import heading_diff_deg
        # canonical error = commanded - actual, wrapped to [-180, 180)
        return heading_diff_deg(commanded_deg, actual_deg)
    except Exception:
        # fallback: compute wrapped difference manually
        diff = commanded_deg - actual_deg
        return (diff + 180) % 360 - 180

def peak_overshoot(error_deg: np.ndarray) -> float:
    """Largest absolute heading error (deg)."""
    return np.max(np.abs(error_deg))

def settling_time(t: np.ndarray, error_deg: np.ndarray, tol: float = 2.0) -> float:
    """
    t: time array (s)
    error_deg: heading_error output
    tol: degrees tolerance band
    Returns the time (s) from t[0] until settled; or np.nan if never settles.
    """
    within = np.abs(error_deg) <= tol
    # find the last index where it goes outside tol
    outside_idxs = np.where(~within)[0]
    if outside_idxs.size == 0:
        return 0.0  # already within band
    last_out = outside_idxs[-1]
    if last_out == len(t) - 1:
        return np.nan  # never settles
    # settled time is the next timestamp
    return t[last_out + 1] - t[0]

def steady_state_error(t: np.ndarray,
                       error_deg: np.ndarray,
                       window: float = 5.0) -> float:
    """
    Average signed error over the final 'window' seconds.
    """
    cutoff = t[-1] - window
    mask = t >= cutoff
    if not mask.any():
        return np.nan
    return np.mean(error_deg[mask])

def oscillation_period(t: np.ndarray, error_deg: np.ndarray) -> float:
    """
    Find times when error crosses zero (sign flips), compute
    differences, and return the average period (s).  
    If fewer than 2 crossings, returns np.nan.
    """
    signs = np.sign(error_deg)
    # indices where sign changes
    zero_cross = np.where(np.diff(signs) != 0)[0]
    if zero_cross.size < 2:
        return np.nan
    # pick the times of crossing
    cross_times = t[zero_cross]
    # periods between consecutive crossings
    periods = np.diff(cross_times)
    # full oscillation is twice the half-period
    return 2 * np.mean(periods)

def compute_zigzag_metrics(t: np.ndarray,
                           actual_rad: np.ndarray,
                           commanded_rad: np.ndarray,
                           tol: float = 2.0,
                           ss_window: float = 5.0) -> dict:
    """
    Compute key zig-zag metrics from time, actual heading, and commanded heading.
    Metrics:
      - peak_overshoot_deg: max overshoot beyond command per segment
      - settling_time_s: time to stay within ±tol band
      - steady_state_error_deg: mean error in final window
      - oscillation_period_s: average full oscillation period
    """
    # convert rad to deg
    act_deg = np.degrees(actual_rad)
    cmd_deg = np.degrees(commanded_rad)
    # wrapped error
    err = heading_error(act_deg, cmd_deg)
    # segment boundaries where command changes
    change_idxs = np.where(np.diff(cmd_deg) != 0)[0] + 1
    starts = np.concatenate(([0], change_idxs))
    ends = np.concatenate((change_idxs, [len(cmd_deg)]))
    # compute overshoot beyond commanded amplitude for each segment
    overshoots = []
    for s, e in zip(starts, ends):
        c = cmd_deg[s]
        seg = act_deg[s:e]
        if c >= 0:
            os_val = np.max(seg - c)
        else:
            os_val = np.max(c - seg)
        overshoots.append(os_val)
    peak_overshoot_deg = float(max(0.0, np.max(overshoots)))
    # other metrics
    segment_settles = []
    for s, e in zip(starts, ends):
        # relative time and error for this segment
        seg_t   = t[s:e] - t[s]
        seg_err = err[s:e]
        # compute settling time within the segment
        seg_st  = settling_time(seg_t, seg_err, tol)
        segment_settles.append(seg_st)
    # pick the worst-case finite settling across all legs
    finite = [st for st in segment_settles if not np.isnan(st)]
    settling = float(max(finite)) if finite else np.nan           
    steady = steady_state_error(t, err, ss_window)
    period = oscillation_period(t, err)
    return {
        "peak_overshoot_deg":    peak_overshoot_deg,
        "settling_time_s":       settling,
        "steady_state_error_deg": steady,
        "oscillation_period_s":  period,
    }

def compute_turning_advance(self):
    """
    Advance = distance traveled in original heading direction when yaw == 90°
    Returns distance in meters (float)
    """
    if self.n != 1:
        raise ValueError("Only implemented for a single ship (n=1)")
    
    # Convert heading history to degrees
    psi = np.unwrap(np.array(self.psi_history))  # radians, unwrapped
    psi0 = psi[0]
    psi_rel = np.rad2deg(psi - psi0)

    # Find first index where heading >= 90 deg
    idx_90 = np.argmax(psi_rel >= 90.0)
    if psi_rel[idx_90] < 90.0:
        return np.nan  # never hit 90 deg

    x0, y0 = self.pos[:, 0]  # spawn point
    x90, y90 = self.history[0][idx_90]
    advance = (x90 - x0)  # assuming heading was along +x
    return advance

def compute_turning_transfer(self):
    """
    Transfer = lateral deviation from initial track at 90° heading
    """
    if self.n != 1:
        raise ValueError("Only implemented for a single ship (n=1)")

    psi = np.unwrap(np.array(self.psi_history))
    psi0 = psi[0]
    psi_rel = np.rad2deg(psi - psi0)
    idx_90 = np.argmax(psi_rel >= 90.0)
    if psi_rel[idx_90] < 90.0:
        return np.nan

    x0, y0 = self.pos[:, 0]
    x90, y90 = self.history[0][idx_90]
    transfer = (y90 - y0)  # assuming initial heading along +x
    return transfer

def compute_tactical_diameter(self):
    """
    Tactical Diameter = distance from original track when yaw == 180°
    """
    if self.n != 1:
        raise ValueError("Only implemented for a single ship (n=1)")

    psi = np.unwrap(np.array(self.psi_history))
    psi0 = psi[0]
    psi_rel = np.rad2deg(psi - psi0)
    idx_180 = np.argmax(psi_rel >= 180.0)
    if psi_rel[idx_180] < 180.0:
        return np.nan

    x0, y0 = self.pos[:, 0]
    x180, y180 = self.history[0][idx_180]
    tactical = np.sqrt((x180 - x0)**2 + (y180 - y0)**2)
    return tactical

def compute_final_diameter(self):
    """
    Final Diameter = max distance between any two points on the turning circle after 360°
    """
    if self.n != 1:
        raise ValueError("Only implemented for a single ship (n=1)")

    psi = np.unwrap(np.array(self.psi_history))
    psi0 = psi[0]
    psi_rel = np.rad2deg(psi - psi0)
    idx_360 = np.argmax(psi_rel >= 360.0)
    if psi_rel[idx_360] < 360.0:
        return np.nan

    positions = np.array(self.history[0][idx_360:])
    if positions.shape[0] < 2:
        return np.nan

    from scipy.spatial.distance import pdist
    diameter = np.max(pdist(positions))  # full chord length
    return diameter

def compute_average_turn_rate(self):
    """
    Compute average turn rate (deg/sec) from 0 to 180 degrees.
    """
    if self.n != 1:
        raise ValueError("Only implemented for a single ship (n=1)")

    psi = np.unwrap(np.array(self.psi_history))
    psi0 = psi[0]
    psi_rel = np.rad2deg(psi - psi0)
    idx_180 = np.argmax(psi_rel >= 180.0)
    if psi_rel[idx_180] < 180.0:
        return np.nan

    t180 = self.t_history[idx_180]
    return 180.0 / t180

def compute_turning_circle_metrics(sim):
    """
    Post-run analyzer for turning circle test.

    Returns
    -------
    dict with:
        - advance_m
        - transfer_m
        - tactical_diameter_m
        - final_diameter_m
        - avg_turn_rate_deg_per_s
    """
    return {
        'advance_m': sim.compute_turning_advance(),
        'transfer_m': sim.compute_turning_transfer(),
        'tactical_diameter_m': sim.compute_tactical_diameter(),
        'final_diameter_m': sim.compute_final_diameter(),
        'avg_turn_rate_deg_per_s': sim.compute_average_turn_rate()
    }
