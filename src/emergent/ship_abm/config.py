# -*- coding: utf-8 -*-

"""
ship_abm/config.py

This module centralizes all configuration parameters for the Ship Agent-Based Model (ABM)
within the Emergent software. By keeping physical properties, control gains, and other
constants in one place, you ensure consistency across simulation, visualization, and any
associated utilities (e.g., dashboard or testing modules).

Contents:
---------
1. SHIP_PHYSICS:
   - Physical dimensions and mass/inertia properties that define each vessel’s hydrodynamics.
   - If you ever need to simulate a different ship type (e.g. shorter/longer, heavier/lighter),
     adjust these values here.

2. CONTROLLER_GAINS:
   - PID gains for heading and speed controllers.
   - You can tune these values to affect how quickly or smoothly vessels respond to steering
     and throttle commands.
     
3. ADVANCED_CONTROLLER:
   - Additional tuning parameters that go beyond simple P/I/D:
       • Feed‐forward gain (`Kf_gain`)
       • Maximum commanded turn‐rate (`r_rate_max_deg`)
       • Anti‐windup limit on the I‐term (`I_max_deg`)
       • Prediction horizon for dead‐band (`lead_time`)
       • Dead‐band half‐angle (`trim_band_deg`)
       • Early‐release band for predictive dead‐band (`release_band_deg`)
   - If you later add hysteresis, gain‐scheduling, or micro-trim logic, expand this section.

4. COLLISION_AVOIDANCE:
   - Safe and clear distances used in COLREGS-based avoidance routines.
   - If you want vessels to start evading each other sooner or later, modify these here.

5. DEFAULT_SPAWN:
   - Default speed at which new vessels enter the simulation.
   - Number of waypoints to generate per vessel when calling the spawn helper.

6. SIMULATION_BOUNDS (EPSG:4326)

   - A nested dictionary providing default latitude/longitude extents for various ports.
     End users can select a port name to retrieve its bounding‐box coordinates without
     manually entering min/max values.

    Usage example:
        from emergent.ship_abm.config import SIMULATION_BOUNDS

        baltimore_bounds = SIMULATION_BOUNDS["Baltimore"]
        # baltimore_bounds == {
        #     "minx": -76.60, "maxx": -76.30,
        #     "miny":  39.19, "maxy":  39.50
        # }

Usage:
------
Import this file anywhere you need consistent ship parameters:

    from emergent.ship_abm.config import SHIP_PHYSICS, CONTROLLER_GAINS, COLLISION_AVOIDANCE

If you later decide to read these values from a YAML file or database, you can update this
module to load from external sources instead of hard-coding them.

"""
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# 1) PHYSICAL PROPERTIES OF THE SHIP (all units in SI: meters, kilograms, seconds)
# ───────────────────────────────────────────────────────────────────────────────
SHIP_PHYSICS = {
    # vessel geometry (m)
    'length': 400.0,            # Length overall of the ship (m)
    'beam': 60.0,               # Beam (width) of the ship (m)
    'draft': 10.0,              # Vertical draft of the ship (m)

    # Rigid-body properties
    'm': 2.5e8,                   # Mass of the vessel (kg)
    'Ixx': 5802102368.428337,                 # Moment of inertia about the x‐axis (roll) (kg·m²)
    'Izz': 8446609085.138884,                 # Moment of inertia about the z‐axis (yaw) (kg·m²)
    'xG': 0.0,                  # Longitudinal offset of center of gravity from geometric center (m)
    'zG': -3.666481,                 # Vertical offset of center of gravity below waterline (m)

    # Hydrodynamic derivatives
    'Xu': -7536261.980847,               # Surge force derivative w.r.t. surge velocity (kg/s)
    'Xv': 0.0,                  # Surge force derivative w.r.t. sway velocity (kg/s)
    'Xp': 0.0,                  # Surge force derivative w.r.t. roll rate (kg·m)
    'Xr': 0.0,                  # Surge force derivative w.r.t. yaw rate (kg·m)

    'Yu': 0.0,                  # Sway force derivative w.r.t. surge velocity (kg/s)
    'Yv': -805224.848207,             # Sway force derivative w.r.t. sway velocity (kg/s)
    'Yp': 0.0,                  # Sway force derivative w.r.t. roll rate (kg·m)
    'Yr': -66333786.294689,                  # Sway force derivative w.r.t. yaw rate (kg·m)

    'Ku': 0.0,                  # Roll moment derivative w.r.t. surge velocity (kg·m²/s)
    'Kv': 0.0,                  # Roll moment derivative w.r.t. sway velocity (kg·m²/s)
    'Kp': 0.0,                  # Roll moment derivative w.r.t. roll rate (kg·m²)
    'Kr': 0.0,                  # Roll moment derivative w.r.t. yaw rate (kg·m²)

    'Nu': 0.0,                  # Yaw moment derivative w.r.t. surge velocity (kg·m²/s)
    'Nv': -8238657164.318151,                # Yaw moment derivative w.r.t. sway velocity (kg·m²/s)
    'Np': 0.0,                  # Yaw moment derivative w.r.t. roll rate (kg·m²)
    'Nr': -16943042.526334,                  # Yaw moment derivative w.r.t. yaw rate (kg·m²)

    # Rudder control effectiveness: reduce slightly to avoid excessive yaw from small rudder
    # (temporarily reduced by ~40% from original to get gentler steering; reversible)
    'Ydelta': 30096654.456664,   # lateral force per radian of rudder (reduced)
    'Kdelta': 0.0,     # N·m of roll moment per radian (ignored)
    # Increased by 1.5x from 88,857,939.477034 to improve rudder authority for candidate tuning
    # For Option B testing we temporarily reduce Ndelta to 70% of the tuned value
    # to make the plant less aggressive and observe controller behaviour.
    'Ndelta': 133286909.215551 * 0.7,   # yaw torque per radian of rudder (reduced for test)

    # Damping
    # Increase linear yaw-related damping modestly to reduce yaw acceleration peaks
    'linear_damping': 1.4e5,      # Linear hull damping coefficient (N per m/s)
    'quad_damping': 1.2e4,        # Quadratic hull damping coefficient (N per (m/s)²)

    # Rudder limits (temporarily reduced to avoid spinouts)
    # NOTE: lowered from 35° to 20° and rate from 0.35 rad/s (~20°/s) to
    # 0.17 rad/s (~10°/s). These conservative defaults are easy to revert.
    # Reduce rudder limits to model more conservative steering and avoid
    # frequent hard-over during gusty forcing. 12° max and ~0.087 rad/s
    # (~5°/s) rate limit are conservative defaults for development.
    'max_rudder': np.radians(12),         # Maximum rudder deflection (rad)
    'max_rudder_rate': 0.087,  # Maximum rudder rate change (rad/s) (~5°/s)
    
    # Drag Stuff
    'drag_coeff': 0.0012
}
# Optional aerodynamic tuning (per-ship or global defaults)
# A_air_ref: reference projected area (m^2) exposed to crosswind. If not set,
#   the ship_model computes a default from length*beam*0.7.
# Cd_air: aerodynamic drag coefficient (dimensionless)
# wind_proj_baseline/scale: projection factor = baseline + scale * |sin(beta)|
#   where beta is wind incidence angle relative to ship heading.
SHIP_AERO_DEFAULTS = {
    'A_air_ref': None,        # None -> use length*beam*0.7
    'Cd_air': 1.0,
    'wind_proj_baseline': 0.2,
    'wind_proj_scale': 0.8,
    # global multiplier applied to computed wind force (use <1.0 to reduce wind effect)
    # conservative default: slightly reduce wind forcing to avoid excessive crabbing
    # Set to 1.0 to model realistic wind forcing by default; lower values used
    # earlier were a conservative handicap and may make vessels under-react.
    'wind_force_scale': 1.0,
}

# ───────────────────────────────────────────────────────────────────────────────
# 2) PID CONTROLLER GAINS (for heading and/or speed control)
# ───────────────────────────────────────────────────────────────────────────────
# These gains determine how aggressively each vessel adjusts heading (rudder) and speed (thrust).
CONTROLLER_GAINS = {
    # Proportional gain for heading controller
    "Kp": 0.5,
    # Integral gain for heading controller (set to 0 if no steady-state offset correction is needed)
    "Ki": 0.05,
    # Derivative gain for heading controller (damping term)
    # reduced to avoid excessive D-term reacting to transient measured r
    "Kd": 0.12,
}

# ───────────────────────────────────────────────────────────────────────────────
# 3) ADVANCED CONTROLLER / TUNING PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
# These extend the basic P/I/D loop with:
#  • Feed‐forward (Kf_gain)
#  • Turn‐rate limiting (r_rate_max_deg)
#  • Anti‐windup bound on the integral term (I_max_deg)
#  • Predictive dead‐band (lead_time, release_band_deg)
#  • Micro‐trim dead‐zone (trim_band_deg)
ADVANCED_CONTROLLER = {
    # Feed‐forward gain: multiplies heading error to compute a desired turn‐rate.
    # Typical values: 0.0 (no feed‐forward) up to ~1.0 (aggressive).
    "Kf_gain": 0.002,  

    # Maximum commanded turn rate (°/s).  After feed‐forward, we clamp
    # desired r_des to ±r_rate_max_deg before converting to radians.
    "r_rate_max_deg": 3.0,  

    # Anti‐windup limit on the I‐term (degrees).  The integral of error is
    # clipped to ±I_max_deg before using it.  Convert to radians in code.
    "I_max_deg": 10.0,  

    # Prediction horizon for dead‐band (seconds).  If |err_pred| < trim_band, we
    # set rudder=0 early to avoid chatter or overshoot.  Typical values: tens to
    # thousands of seconds depending on your dynamics.
    # Reduced lead_time to avoid over-aggressive predictive deadzone that forced
    # rudder to zero on short transients. 5s is a conservative choice to start.
    "lead_time": 5.0,

    # Dead‐zone half‐angle (degrees).  Any commanded rudder smaller than this
    # in magnitude is forced to zero to prevent constant micro‐twitching.
    # Increase trim_band to reduce twitching under strong, gusty wind forcing.
    "trim_band_deg": 0.5,  

    # Early‐release band (degrees).  Once |predicted_error| < release_band_deg,
    # rudder is released (forced to zero), even if commanded > trim_band_deg.
    # This widens the “dead‐zone” as heading error shrinks.  Typical: 3–10°.
    # Keep release band moderate to allow release when errors are small.
    "release_band_deg": 3.0,
    # Derivative low-pass time constant (seconds). Small values follow measured r closely;
    # larger values smooth noisy measurements before they reach the D-term. 0.5s is conservative.
    "deriv_tau": 1.0,
    # Dead-reckoning tuning: how aggressively to correct for current drift
    # sensitivity: fraction of U (through-water speed) at which full correction is applied
    "dead_reck_sensitivity": 0.25,
    # maximum heading correction (degrees) applied by dead-reckoning
    "dead_reck_max_corr_deg": 30.0,
    # When True, the simulation-level controller is the canonical source of
    # rudder commands. Per-ship `pid_control()` will return the last
    # simulation-commanded rudder unless explicitly asked to run locally.
    "use_simulation_controller": True,
    # Back-calculation coefficient used for anti-windup: when saturation occurs,
    # this fraction of the (raw - applied) command is subtracted from the
    # integrator each second (i.e. integral -= backcalc_beta * diff * dt).
    # Increase to more aggressively unwind I during saturation. Default was 0.08.
    "backcalc_beta": 0.16,
    # Soft pre-saturation cap (degrees): clamp PID raw command magnitude before
    # applying actuator limits. This prevents extremely large internal commands
    # from causing aggressive integrator back-calculation and numeric instability.
    # Set to None or a very large value to disable.
    "raw_cap_deg": 90.0,
}

# Enable verbose PID internals printing when True (useful for headless debugging)
# Default to False for batch/automated runs; tests can override to True when needed.
PID_DEBUG = False

# Simulation-level PID tracing: when True, the simulation will emit a CSV of
# per-step PID internals for offline QC/tuning. Path may be None to auto-generate.
PID_TRACE = {
    'enabled': False,
    'path': 'scripts/pid_trace_gui_rosario.csv'
}

# ───────────────────────────────────────────────────────────────────────────────
# 4) COLLISION AVOIDANCE / COLREGS PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
COLLISION_AVOIDANCE = {
    # ---------------------------------------------------------------------------
    # safe_dist (meters)
    # ---------------------------------------------------------------------------
    # When another ship is within this distance, the avoidance routine engages.
    # In practice, as soon as the Euclidean distance between two vessel centers
    # drops below safe_dist, we compute a repulsive‐force vector (or execute COLREGS
    # logic) to steer away.  Set this larger than any “collision_threshold” so that
    # avoidance begins well before an actual near‐miss.
    #
    # Typical value: 2000.0  (m)
    "safe_dist": 2000.0,

    # ---------------------------------------------------------------------------
    # clear_dist (meters)
    # ---------------------------------------------------------------------------
    # Used (optionally) to check “future‐position clearance” in path planning.
    # If you were to predict where both vessels will be in, say, dt_pred seconds,
    # and ensure their predicted positions remain farther apart than clear_dist,
    # you know each has room to maneuver.  In many simple implementations, clear_dist
    # is left “unused” or equal to safe_dist; but if you build a look‐ahead step that
    # checks next‐timestep collisions, you’d compare predicted positions against this.
    #
    # (Currently unused in core avoidance, but reserved for future path‐prediction logic.)
    #
    # Typical value: 500.0  (m)
    "clear_dist": 2300.0,
    # ---------------------------------------------------------------------------
    # unlock_ang (radians)
    # ---------------------------------------------------------------------------
    # After avoidance has been triggered, you might lock the rudder or keep the vessel
    # in an “evasive” heading until the bearing to the target (or threat) exceeds this
    # angle.  In other words:
    #   • While avoiding, if |bearing_to_other_ship| > unlock_ang, revert back to
    #     normal steering (e.g., resume following waypoints or neutral heading).
    #
    # This prevents the controller from twitching in and out of avoidance at very small
    # angular separations.  Expressed here in radians, but you can convert from degrees
    # with: np.radians(15).
    #
    # Typical value: np.radians(15)  →  ~0.2618 rad
    "unlock_ang": np.radians(25.0),
    
    # NEW: seconds to stay “immune” after a lock clears
    "post_avoid_time": 45.0,

    # ── predictive CPA drop logic ─────────────────────────────
    "t_cpa_max"   : 300.0,   # s  – don’t consider encounters >5 min away
    "d_cpa_safe"  : 1000.0, # m  – if predicted dCPA exceeds this, unlock
    "drop_angle"  : np.radians(15.0),  # if heading changed ≥15° from initial
    "headon_turn_deg": 30,
    "cross_turn_deg": 20,
    # If False, do not force hard-over rudder when give-way logic requests it.
    "allow_hard_give_way_override": True,
}
    
# ───────────────────────────────────────────────────────────────────────────────
# 4) PROPULSION SETTINGS 
# ───────────────────────────────────────────────────────────────────────────────
PROPULSION = {
    # Initial commanded speed (m/s) when a vessel enters the simulation
    "initial_speed": 5.0,
    # Default cruise/desired speed (m/s) per vessel
    "desired_speed": 5.0,
    # Maximum allowed speed (m/s); defaults to 1.2 × desired_speed
    "max_speed": 12,
    # Density of water (kg/m³) used in thrust/dynamic‐pressure calculations
    "rho": 1025.0,
    # Thrust coefficient (unitless) mapping RPM to thrust
    "K_T": 0.5,
    # Maximum RPM for propellers
    "max_rpm": 90.0,
    # Maximum allowable rate of change in RPM (rev/min per second)
    "max_rpm_rate": 5.0,
    # Number of propellers per vessel (integer)
    "n_prop": 1,
}

# ───────────────────────────────────────────────────────────────────────────────
# 5) SIMULATION DOMAIN BOUNDS 
# ───────────────────────────────────────────────────────────────────────────────
SIMULATION_BOUNDS = {
    "Galveston": {
        # Galveston, TX (City) bounds in EPSG:4326
        # Longitude: West → East, Latitude: South → North
        "minx": -95.5,
        "maxx": -94.5,
        "miny":  29.,
        "maxy":  30.,
    },
    "Baltimore": {
        # Baltimore, MD (City) bounds in EPSG:4326
        "minx": -76.60,
        "maxx": -76.30,
        "miny":  39.19,
        "maxy":  39.50,
    },
    "Los Angeles / Long Beach": {
        # Port of Los Angeles & Port of Long Beach (San Pedro Bay) in EPSG:4326
        "minx": -118.290046296,  # West boundary
        "maxx": -118.069953704,  # East boundary
        "miny":   33.699861117,  # South boundary
        "maxy":   33.790046302,  # North boundary
    },
    "Oakland / San Francisco Bay": {
        # Oakland/San Francisco Bay (Region) bounds in EPSG:4326
        "minx": -122.550000,
        "maxx": -121.680000,
        "miny":   37.360000,
        "maxy":   38.300000,
    },
    "Seattle": {
        # Seattle, WA (City) bounds in EPSG:4326
        "minx": -122.459696,
        "maxx": -122.224433,
        "miny":   47.491911,
        "maxy":   47.734061,
    },
    "Rosario Strait": {
        # Rosario Strait, WA (between Anacortes and San Juan Islands) in EPSG:4326
        "minx": -122.80,
        "maxx": -122.60,
        "miny":   48.50,
        "maxy":   48.75,
    },
    "New Orleans": {
        # New Orleans downriver to the Mississippi River mouth (EPSG:4326)
        # West (upriver New Orleans) → East (river mouth),
        # South (delta/mouth) → North (city limits)
        "minx": -89.50,
        "maxx": -89.00,
        "miny":  28.75,
        "maxy":  29.25,
    },
    "New York": {
        # New York Harbor bounds in EPSG:4326
        # West (Staten Island/Narrows) → East (Queens/LI shore),
        # South (Lower Bay)           → North (Upper Bay)
        "minx": -74.27,
        "maxx": -73.86,
        "miny":  40.49,
        "maxy":  40.75,
    },
}

# ───────────────────────────────────────────────────────────────────────────────
# 7)  NOAA Operational Forecast System (OFS) mapping
# ───────────────────────────────────────────────────────────────────────────────
# Used by emergent.ship_abm.ofs_loader to decide which hydrodynamic model to
# download for each port in SIMULATION_BOUNDS.  Keys must *exactly* match the
# port names you use elsewhere (e.g. spawn dialog, CLI flag, etc.).
#
# If you add a new port to SIMULATION_BOUNDS, drop the model-ID here and the
# loader will pick it up automatically.  Model IDs are the same folder names
# NOAA uses in the S3 bucket, 100 % lower-case.
#
#   https://registry.opendata.aws/noaa-ofs-pds/
#
OFS_MODEL_MAP = {
    "Galveston":                    "ngofs2",   # Northern Gulf of Mexico
    "Baltimore":                    "cbofs",    # Chesapeake Bay
    "Los Angeles / Long Beach":     "wcofs",    # West-Coast nest (4 km ROMS)
    "Oakland / San Francisco Bay":  "sfbofs",   # SF Bay FVCOM
    "Seattle":                      "sscofs",   # Salish Sea / Puget
    "Rosario Strait":               "sscofs",   # Salish Sea (same model)
    "New Orleans":                  "ngofs2",   # Northern Gulf (covers Delta)
    "New York":                     "nyofs",    # NY/NJ Harbor
}


# ───────────────────────────────────────────────────────────────────────────────
# 6) OTHER CONSTANTS / FUTURE EXTENSIONS
# ───────────────────────────────────────────────────────────────────────────────
# If there are additional, model-specific constants (e.g., environmental forcing amplitudes,
# time step defaults, or numerical solver tolerances), you can add them here. Example:
#
# ENVIRONMENT = {
#     "max_wind_speed": 15.0,   # (m/s)
#     "tide_amplitude": 0.5,    # (m)
# }
#
# TIME_STEP_DEFAULT = 0.1       # (s) if you want to override dt in simulation constructor
#
# these dictionaries allow you to modify any key aspect of the ship ABM in one place.

# NOAA ENC catalog
xml_url = 'https://charts.noaa.gov/ENCs/ENCProdCat_19115.xml'