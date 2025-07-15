# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
from matplotlib.widgets import Button            
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch
import os
import xml.etree.ElementTree as ET
import requests, zipfile, io
import geopandas as gpd
from shapely.geometry import box, LineString, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import pandas as pd
import fiona
import urllib3
import warnings
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
import tempfile
import sys
from emergent.ship_abm.ship_model import ship
from emergent.ship_abm import enctiler
from emergent.ship_abm.config import CONTROLLER_GAINS, \
    ADVANCED_CONTROLLER, \
        PROPULSION, \
            SIMULATION_BOUNDS, \
                xml_url
from emergent.ship_abm.ais import compute_ais_heatmap


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

def tide_fn(t):
    return 1.5 * np.sin(2 * np.pi / 12.42 * (t / 3600))


class simulation:
    def __init__(
        self,
        port_name,
        dt=0.1,
        T=100,
        n_agents=None,
        coast_simplify_tol = 50.0,
        light_bg = True,
        verbose=False
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
        self.port_name = port_name
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

        # Compute UTM CRS from domain center
        midx = (minx + maxx) / 2
        utm_zone = int((midx + 180) // 6) + 1
        utm_epsg = 32600 + utm_zone  # Northern hemisphere
        self.crs_utm = f"EPSG:{utm_epsg}"
        
        # enable follow‐ship zoom (meters from ship center)
        self.dynamic_zoom = False
        self.zoom = 5000    # e.g. ±2 km view radius

        # Initialize matplotlib figure in true UTM (meters)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=300)
        # ensure Matplotlib really “locks in” 4×4 inches at 300 dpi
        self.fig.set_size_inches(10.0, 6.0)
        self.fig.set_dpi(300)
        
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        
        for tl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            tl.set_fontfamily('serif')

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
        
        # # Save spawn parameters for programmatic entrance
        # self.spawn_speed = spawn_speed
        # self.spawn_waypoints = spawn_waypoints
        
        # # Obstacle avoidance parameters
        # domain_width = maxx - minx
        # self._coast_paths = None
        # self.collision_tol_area = 1.0       
        # domain_width = maxx - minx
        # self.obs_threshold = obs_threshold or (0.1 * domain_width)
        # self.obs_gain = obs_gain or 1e5
        # # PRM sampling: how many random free‐space nodes to add per agent
        # self.prm_samples = params.get('prm_samples', 100)


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
 
        # 1) Load ENC charts

        self.load_enc_features(xml_url, verbose=verbose)
        
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
    
        self.tile_dict = enctiler.pre_tile(coast_series, tol=10.0)   # once
        self.tcache    = enctiler.TileCache(self.ax, self.tile_dict)
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
            self.spawn()

        #--- Setup steering controller state & tuning ---
        # prev_psi for turn-rate calculation
        self.prev_psi = self.psi.copy()
        # integral of heading error for I-term
        self.integral_error = np.zeros_like(self.psi)
        
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
                linewidth = 0.3,
                antialiased = True,
                zorder=5
            )
            self.ax.add_patch(poly)
            self.patches.append(poly)

            line, = self.ax.plot([], [], '--', linewidth=1)
            self.traj.append(line)

            txt = self.ax.text(self.pos[0, i], self.pos[1, i], '', fontsize=8, zorder=9)
            self.texts.append(txt)

            self.history.append([self.pos[:, i].copy()])
        
        # ── “Kill Power” control ────────────────────
        bax = inset_axes(
            self.ax,
            width="10%",   # 15% of the axes width
            height="8%",  # 10% of the axes height
            loc='lower right',  # anchored in the lower‐right
            borderpad=0.5      # small padding
        )
        self._btn_kill = Button(bax, 
                                'Kill Power',
                                color='#cc4444', 
                                hovercolor='#ff6666')
        
        self._btn_kill.label.set_fontsize(8)

        
        self._btn_kill.on_clicked(self._cut_random_power)

        # ── WIND‐VANE ARROW ───────────────────────────
        self._wing_tail = (0.05, 0.95)  # 5% in from left, 95% up from bottom
        self.wind_arrow = FancyArrowPatch(
            posA=self._wing_tail,
            posB=self._wing_tail,       # initialize “collapsed” (no length)
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=2,
            color='purple',
            transform=self.ax.transAxes,
            zorder=5,
            animated=True
        )
        self.ax.add_patch(self.wind_arrow)

        # ── HEADER & LOG TEXT on the map axes ──
        # “Header” in top‐center (above wind‐vane)
        self.header_text = self.ax.text(
            0.5, 0.98,             # x=50%, y=98% in axes coords
            "",                    # updated each frame
            transform=self.ax.transAxes,
            fontsize=10,
            fontfamily='serif',
            ha='center', va='top',
            color='black',
            animated=True
        )
        
        # “Log” lines in bottom‐left corner
        self.max_log_lines = 5
        self.log_lines = []               # a Python list of strings
        self.log_text_artists = []
        for i in range(self.max_log_lines):
            txt = self.ax.text(
                0.02, 0.05 + i * 0.04,   # stagger from 5% up by 4% per line
                "",
                transform=self.ax.transAxes,
                fontsize=8,
                fontfamily='serif',
                color='darkred',
                va='bottom',
                animated=True
            )
            self.log_text_artists.append(txt)

        
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
    
        keep_layers = {'LNDARE', 'DEPARE', 'DEPVAL'}#, 'BUOYES', 'BOYAID', 'LITFLT', 'BOYWUN', 'BOYSPP', 'BOYMKR'}
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
        state0[0, :] = getattr(self, 'spawn_speed', PROPULSION['desired_speed'])

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
        self.mode = 'route'
        self.route()

        # 2) Derive state arrays from waypoints
        n = self.n
        # Initialize arrays
        state0 = np.zeros((4, n), dtype=float)
        pos0 = np.zeros((2, n), dtype=float)
        psi0 = np.zeros(n, dtype=float)
        goals_arr = np.zeros((2, n), dtype=float)

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

        # 3) Plot start/goal markers
        for i in range(n):
            sx, sy = pos0[0, i], pos0[1, i]
            gx, gy = goals_arr[0, i], goals_arr[1, i]
            self.ax.scatter(sx, sy, marker='o', s=50, color='green', zorder=2)
            self.ax.scatter(gx, gy, marker='8', s=50, color='red',   zorder=2)

        # 4) Instantiate ship and assign to simulation state
        self.state  = state0.copy()
        self.pos    = pos0.copy()
        self.psi    = psi0.copy()
        self.goals  = goals_arr.copy()
        self.ship = ship(state0,pos0,psi0,goals_arr)
        self.ship.wpts = self.waypoints
        self.ship.short_route = self.waypoints

        # ─── Clear any AIS heatmap & reset to ENC‐only background ──────────────
        # Switch back to sim mode so that run() sees only the ENC chart.
        self.mode = 'sim'
        # Clear the axes (removes the heatmap overlay)
        self.ax.clear()
        # Restore aspect and limits
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.minx, self.maxx)
        self.ax.set_ylim(self.miny, self.maxy)
        # Draw just the ENC‐based nautical chart (no heatmap)
        self._draw_background()
        # Cache that ENC background for use in run()
        self.fig.canvas.draw()
        self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # ──────────────────────────────────────────────────────────────────────

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
            self.load_route_heatmap('2024-06-01', '2024-06-07')
        except Exception as e:
            log.error(f"[ROUTE] Failed to draw full background: {e}")
            
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


        # avoid method shadowing
        self.user_routes = list(self.waypoints)
        print("[ROUTE] Interactive routing complete.")

    def load_route_heatmap(
        self,
        start_date,
        end_date,
        grid_size=(5000, 5000)
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

            # Overlay the heatmap (semi-transparent) on top of ENC
            hm_artist = self.ax.imshow(
                self.ais_heatmap,
                origin='lower',
                extent=(utm_minx, utm_maxx, utm_miny, utm_maxy),
                cmap='hot',
                alpha=0.5,
                zorder=1  # ensure it’s above the water/land but below ship patches
            )
            # keep a reference so it doesn't get garbage-collected
            self._last_hm_artist = hm_artist
            self.fig.canvas.draw_idle()

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
        self.wind_arrow.set_positions(self._wing_tail, new_posB)
        
        #self.wind_arrow.set_positions((tx, ty), (tip_x, tip_y))
    
        # 3) Update header text: time only (we’re logging wind/current elsewhere)
        header_str = f"t = {t:.1f} s"
        self.header_text.set_text(header_str)
    
        # 4) Update log lines (already stored as self.log_lines)
        #   We assume self.log_lines[0] is most recent; up to max_log_lines
        for idx, txt_artist in enumerate(self.log_text_artists):
            if idx < len(self.log_lines):
                txt_artist.set_text(self.log_lines[idx])
            else:
                txt_artist.set_text("")


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
                self._bg_cache       = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self._blit_enabled   = True
            else:
                # Restore both map & panel backgrounds from their caches
                self.fig.canvas.restore_region(self._bg_cache)

            # ── Draw map‐side dynamic elements ───────────────────────────────
            self._draw_waypoints()
            self._draw_ships(step)
            self._draw_annotations(hd, sp, rud)

            # ── Draw control‐panel dynamic elements ──────────────────────────
            self._update_overlay(t)

            
            # ── Draw panel‐side dynamic artists (all in serif, size 12) ─────────
            # Make sure header_text is serif, size 12
            self.header_text.set_fontsize(12)
            self.header_text.set_fontfamily('serif')

            # Ensure each log line is serif, size 12
            for txt in self.log_text_artists:
                txt.set_fontsize(6)
                txt.set_fontfamily('serif')

            dynamic_panel = [self.wind_arrow, self.header_text] + self.log_text_artists
            for art in dynamic_panel:
                self.ax.draw_artist(art)

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
        wind_raw = playful_wind(self.state, self.t)

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

        tail_x, tail_y = self._wing_tail
        tip_x, tip_y = tail_x + dx, tail_y + dy

        # Clamp to [0..1] so the arrow never wanders off-screen
        tip_x = max(0.0, min(1.0, tip_x))
        tip_y = max(0.0, min(1.0, tip_y))

       # Finally update the FancyArrowPatch (both coords in axes-fraction)
        self.wind_arrow.set_positions((tail_x, tail_y),
                                      (tip_x, tip_y))

        wind_vec    = playful_wind(self.state, self.t)
        current_vec = north_south_current(self.state, self.t)
        wind, current = self.ship.environmental_forces(
            wind_vec, current_vec,
            self.state[0], self.state[1], self.psi
        )
        
        tide = tide_fn(self.t)
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
        ws = np.hypot(wx, wy)
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