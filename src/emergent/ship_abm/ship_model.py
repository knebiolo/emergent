import numpy as np
from emergent.ship_abm.config import SHIP_PHYSICS, \
    CONTROLLER_GAINS, \
        ADVANCED_CONTROLLER, \
            COLLISION_AVOIDANCE, \
                PROPULSION
from emergent.ship_abm.config import PID_DEBUG

class ship:
    """
    Encapsulates vessel geometry, hydrodynamics, and basic control loops.
    Pulls all numeric constants from ship_abm.config.

    Responsibilities:
      • Initialize mass, inertia, added‐mass, drag, and thrust‐map based on SHIP_PHYSICS.
      • Provide speed_to_rpm() and rpm_to_thrust() conversions.
      • Implement a minimal PID‐based heading controller (using CONTROLLER_GAINS).
      • Apply advanced terms (feed‐forward, anti‐windup, dead‐band) from ADVANCED_CONTROLLER.
      • Expose a .step(state, commanded_rpm, goal_heading, wind, current, dt) interface:
        - Compute new surge/u, sway/v, roll/p, yaw/r, heading/ψ, and positions.
        - Return an updated state vector plus any robot‐side outputs (e.g. new commanded RPM).
    """
    
    def __init__(self,state0,pos0,psi0,goals_arr):
        # infer fleet size
        n = state0.shape[-1] if hasattr(state0, 'ndim') and state0.ndim > 1 else 1
        self.n = n
        
        # ── cool-down timer after avoidance ──
        self.post_avoid_timer = np.zeros(n)  # [s]
        
        # assign initial state
        self.state = state0
        self.pos   = pos0
        self.psi   = psi0
        self.goals = goals_arr
        
        # vessel geometry 
        self.length = SHIP_PHYSICS["length"]
        self.beam = SHIP_PHYSICS["beam"]
        self.draft = SHIP_PHYSICS["draft"]
        
        # block coefficient & approximate wetted area
        self.C_B = 0.80 # typical value for container vessels
        self.A = ((self.length * self.beam)
                  + 2 * (self.beam * self.draft)
                  + 2 * (self.length * self.draft)) * self.C_B
        
        # propeller diameter heuristic
        self.D = self.beam / 4.0
        
        # mass and inertia
        self.m = SHIP_PHYSICS["m"]
        self.Ixx = SHIP_PHYSICS["Ixx"]
        self.Izz = SHIP_PHYSICS["Izz"]
        self.xG = SHIP_PHYSICS["xG"]
        self.zG = SHIP_PHYSICS["zG"]
        
        # rigid-body mass
        H = np.zeros((4, 4))
        H[0,0] = self.m
        H[1,1], H[1,2], H[1,3] = self.m, -self.m * self.zG,  self.m * self.xG
        H[2,1], H[2,2]        = -self.m * self.zG, self.Ixx
        H[3,1], H[3,3]        =  self.m * self.xG, self.Izz
        self.H_inv = np.linalg.inv(H)
        
        # hydrodynamic derivatives
        self.Xu = SHIP_PHYSICS['Xu']
        self.Xv = SHIP_PHYSICS['Xv']
        self.Xp = SHIP_PHYSICS['Xp']
        self.Xr = SHIP_PHYSICS['Xr']
        self.Yu = SHIP_PHYSICS['Yu']
        self.Yv = SHIP_PHYSICS['Yv']
        self.Yp = SHIP_PHYSICS['Yp']
        self.Yr = SHIP_PHYSICS['Yr']
        self.Ku = SHIP_PHYSICS['Ku']
        self.Kv = SHIP_PHYSICS['Kv']
        self.Kp = SHIP_PHYSICS['Kp']
        self.Kr = SHIP_PHYSICS['Kr']
        self.Nu = SHIP_PHYSICS['Nu']
        self.Nv = SHIP_PHYSICS['Nv']
        self.Np = SHIP_PHYSICS['Np']
        self.Nr = SHIP_PHYSICS['Nr'] 
        
        # rudder effectiveness and limits
        self.Ydelta = SHIP_PHYSICS['Ydelta']
        self.Kdelta = SHIP_PHYSICS['Kdelta']  
        self.Ndelta = SHIP_PHYSICS['Ndelta']
        # SHIP_PHYSICS may contain values already converted to radians in some configs,
        # so be defensive: if the stored value is plausibly in degrees (> 2π) convert,
        # otherwise assume it's already in radians and use as-is.
        raw_max_rudder = float(SHIP_PHYSICS.get("max_rudder", 0.0))
        if abs(raw_max_rudder) > 2 * np.pi:
            self.max_rudder = np.deg2rad(raw_max_rudder)
        else:
            self.max_rudder = raw_max_rudder

        raw_max_rudder_rate = float(SHIP_PHYSICS.get("max_rudder_rate", 0.0))
        if abs(raw_max_rudder_rate) > 2 * np.pi:
            self.max_rudder_rate = np.deg2rad(raw_max_rudder_rate)
        else:
            self.max_rudder_rate = raw_max_rudder_rate
        self.smoothed_rudder = np.zeros(n)
        self.prev_rudder = np.zeros(n)
        self.rudder_tau = 1.0
        
        # damping and drag coefficients 
        self.linear_damping = SHIP_PHYSICS['linear_damping']
        self.quad_damping = SHIP_PHYSICS['quad_damping']
        self.drag_coeff = SHIP_PHYSICS['drag_coeff']
        
        # basic PID gains 
        self.Kp = CONTROLLER_GAINS["Kp"]
        self.Ki = CONTROLLER_GAINS["Ki"]
        self.Kd = CONTROLLER_GAINS["Kd"]
        self.integral = np.zeros(n)
        self.prev_error = np.zeros(n)

        # advanced tuning parameters
        self.Kf      = ADVANCED_CONTROLLER["Kf_gain"]
        self.r_max   = np.radians(ADVANCED_CONTROLLER["r_rate_max_deg"])
        self.I_max   = np.radians(ADVANCED_CONTROLLER["I_max_deg"])
        self.lead_time    = ADVANCED_CONTROLLER["lead_time"]
        self.trim_band_deg    = np.radians(ADVANCED_CONTROLLER["trim_band_deg"])
        self.release_band_deg = np.radians(ADVANCED_CONTROLLER["release_band_deg"])
        self._deadzone_off_state = np.zeros(self.n, dtype=bool)
        # dead-reckoning tuning (from config)
        self.dead_reck_sensitivity = ADVANCED_CONTROLLER.get('dead_reck_sensitivity', 0.25)
        self.dead_reck_max_corr_deg = ADVANCED_CONTROLLER.get('dead_reck_max_corr_deg', 30.0)
    
        # speed & propulsion settings
        init_sp = PROPULSION['initial_speed']
        # initialize current_speed as array of length n for vectorized control
        try:
            # multi-ship case: fill an array of length n
            self.current_speed = np.full(self.n, init_sp, dtype=float)
        except Exception:
            # fallback to scalar for single-ship case
            self.current_speed = float(init_sp)

        # deceleration limit: max acceleration (m/s²), fallback to max_speed if not configured
        accel_limit = PROPULSION.get('max_accel', PROPULSION['max_speed'])
        try:
            self.max_accel = np.full(self.n, accel_limit, dtype=float)
        except Exception:
            self.max_accel = float(accel_limit)
            
        self.desired_speed = np.full(n, PROPULSION['desired_speed'])
        self.cruise_speed  = self.desired_speed.copy()
        self.max_speed     = np.full(n, PROPULSION['max_speed'])
        # maximum reverse speed (positive magnitude) – default to 30% of max_speed
        try:
            self.max_reverse_speed = 0.3 * np.array(self.max_speed)
        except Exception:
            self.max_reverse_speed = 0.3 * PROPULSION['max_speed']
        self.rho = PROPULSION['rho']
        self.K_T = PROPULSION['K_T']
        self.max_rpm = PROPULSION['max_rpm']
        self.max_rpm_rate = PROPULSION['max_rpm_rate']
        rpm0 = self.speed_to_rpm(init_sp)
        self.commanded_rpm = np.full(n, rpm0, dtype=float)
        self.prev_rpm = np.copy(self.commanded_rpm)
        self.n_prop = 2

        # aero constants
        self.rho_air   = 1.225           # kg·m⁻³
        self.A_air     = self.length * self.beam * .7  # proj. area above WL
        self.Cd_air    = 1.0            # bluff-body
        self.Cd_water  = self.drag_coeff
        
        # ── per‑vessel avoidance state ───────────────────────────────────────
        # crossing_lock holds the index of the vessel we are yielding to.
        # -1 means no active lock.  When a give-way maneuver triggers,
        # this lock stores the target index and the commanded heading/speed.
        self.lock_init_psi  = np.full(n, np.nan)  # heading at lock-on
        self.crossing_lock   = np.full(n, -1, dtype=int)
        self.crossing_heading = np.zeros(n)
        self.crossing_speed   = np.zeros(n)

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
            # NEW ── make the vessel “dead in the water”
            self.current_speed[idx] = 0.
        
        # log for debugging
        #log.warning(f"Power cut to vessel {idx}")        # ASCII → avoids CP1252 gripe

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

        #return rpm
        return np.clip(rpm, 0.0, self.max_rpm)

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

        def _force(rel_vec, rho, Cd, A_override, psi_local, baseA):
            # rel_vec: (2,) vector in Earth frame
            mag   = np.hypot(rel_vec[0], rel_vec[1])
            mag   = mag if mag >= 1e-6 else 1e-6          # avoid /0 (scalar)
            dir_x = rel_vec[0] / mag
            dir_y = rel_vec[1] / mag
            # compute relative wind angle in Earth frame, then incidence vs ship heading
            rel_angle = np.arctan2(rel_vec[1], rel_vec[0])  # global angle
            beta = (rel_angle - psi_local + np.pi) % (2*np.pi) - np.pi
            # projection factor: baseline + scale*|sin(beta)| → beam-on maximized
            try:
                from emergent.ship_abm.config import SHIP_AERO_DEFAULTS
                baseline = SHIP_AERO_DEFAULTS.get('wind_proj_baseline', 0.2)
                scale = SHIP_AERO_DEFAULTS.get('wind_proj_scale', 0.8)
            except Exception:
                baseline, scale = 0.2, 0.8
            proj = baseline + scale * abs(np.sin(beta))
            A_proj = (A_override if (A_override is not None) else baseA) * proj
            F     = 0.5 * rho * Cd * A_proj * mag**2              # scalar
            Fx_e, Fy_e = F*dir_x, F*dir_y                        # Earth frame
            # rotate to body frame (surge = +Xfwd, sway = +Yport)
            X_b =  Fx_e*np.cos(psi_local) + Fy_e*np.sin(psi_local)
            Y_b = -Fx_e*np.sin(psi_local) + Fy_e*np.cos(psi_local)
            return np.vstack([X_b, Y_b])

        rel_wind    = wind_vec    - np.vstack([u_e, v_e])
        rel_current = current_vec - np.vstack([u_e, v_e])

        # allow per-ship aero overrides from config; default to instance attributes
        try:
            from emergent.ship_abm.config import SHIP_AERO_DEFAULTS
            A_ref = SHIP_AERO_DEFAULTS.get('A_air_ref', None) or None
            Cd_air = SHIP_AERO_DEFAULTS.get('Cd_air', self.Cd_air)
        except Exception:
            A_ref = None
            Cd_air = self.Cd_air

        # compute per-ship wind force: vectorized over ships
        nships = rel_wind.shape[1]
        wind_force_cols = []
        current_force_cols = []
        for i in range(nships):
            wf = _force(rel_wind[:, i], self.rho_air, Cd_air, A_ref, psi[i], self.A_air)
            cf = _force(rel_current[:, i], self.rho, self.Cd_water, None, psi[i], self.A)
            wind_force_cols.append(wf)
            current_force_cols.append(cf)

        wind_force = np.hstack(wind_force_cols)
        current_force = np.hstack(current_force_cols)
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

    def compute_desired(self, goals, x, y, u, v, r, psi, current_vec=None):
        """
        Compute desired *water-relative* heading & speed so that the
        ship’s **course-over-ground** still aims at the goal even when
        currents (and, second-order, wind) push the hull.

        Parameters
        ----------
        goals : (2,n) goal positions in *earth* frame
        x, y  : current positions (earth frame)
        u, v  : surge/sway in *body* frame  (needed only for callers)
        r, psi: yaw-rate / heading
        current_vec : (2,n) environmental current in earth frame
                      If None → fall back to “no drift” behaviour.
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

        # ------------------------------------------------------------------
        # DEAD-RECKONING : compute a conservative correction for drift
        # ------------------------------------------------------------------
        if current_vec is not None:
            # desired ground-track unit vector (safe divide)
            mag_attr = np.linalg.norm(attract, axis=0, keepdims=True)
            cog_dir = np.divide(attract, mag_attr, out=np.zeros_like(attract), where=mag_attr>0)

            # base heading (without drift correction)
            hd_base = np.arctan2(cog_dir[1], cog_dir[0])

            # pick the speed controller's demanded through-ground speed
            sog_cmd = np.array(self.desired_speed, copy=True)

            # compute perpendicular component of current+wind relative to track
            perp = np.vstack([-cog_dir[1, :], cog_dir[0, :]])
            V_perp = (current_vec * perp).sum(axis=0)

            # Predict lateral offset over lead_time (more stable than instant asin correction)
            lead_t = getattr(self, 'lead_time', 5.0)
            lateral_offset = V_perp * lead_t

            # forward distance over which we expect to close toward waypoint
            U = np.maximum(sog_cmd, 1e-3)
            forward_dist = U * lead_t

            # correction angle (radians) from lateral offset using atan2(lateral, forward)
            corr = np.arctan2(lateral_offset, forward_dist + 1e-9)

            # clamp maximum correction to avoid wild headings when currents are uncertain
            max_corr_deg = getattr(self, 'dead_reck_max_corr_deg', 30.0)
            max_corr_rad = np.deg2rad(max_corr_deg)
            corr = np.clip(corr, -max_corr_rad, max_corr_rad)

            # compute a corrected heading per-agent
            hd_corrected = hd_base - corr

            # Blend between no-correction and corrected heading based on current strength
            current_mag = np.linalg.norm(current_vec, axis=0)
            sensitivity = getattr(self, 'dead_reck_sensitivity', 0.25)
            beta = np.clip(current_mag / (sensitivity * U), 0.0, 1.0)

            # final heading: smooth blend (handles vector shapes)
            hd = (1.0 - beta) * hd_base + beta * hd_corrected
        else:
            # original behaviour: ignore drift
            hd = np.arctan2(attract[1], attract[0])

        # desired speed stays whatever you’ve set in self.desired_speed
        sp = np.array(self.desired_speed, copy=True)
        return hd, sp

    # # === Updated PID Controller ===
    def pid_control(self, psi, hd, dt):
        """
        P-I-D with gain scheduling, anti-windup, dead-band, and rate-limit.
        """
        # If the simulation-level controller is canonical, don't run the
        # per-ship PID (it would duplicate integrator state). Instead return
        # the previously commanded rudder (set by simulation) as a fallback.
        try:
            from emergent.ship_abm.config import ADVANCED_CONTROLLER
            if ADVANCED_CONTROLLER.get('use_simulation_controller', False):
                # return the most recently stored rudder (vectorized)
                return self.prev_rudder
        except Exception:
            pass

        # 1) heading error normalized to [-π, π] (flipped sign so positive error → starboard turn)
        err = (hd - psi + np.pi) % (2*np.pi) - np.pi
        #err = (psi - hd + np.pi) % (2*np.pi) - np.pi

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

        # 3) predictive error: look ahead lead_time seconds
        #    using derivative as a rough slope
        # --- yaw_rate prediction: init prev_psi on first call ---
        if not hasattr(self, 'prev_psi'):
            # first invocation: no motion yet, so yaw_rate = 0
            self.prev_psi = psi.copy()
        yaw_rate = (psi - self.prev_psi) / dt
        # store for next timestep
        self.prev_psi = psi.copy()     
        #err_pred = err + yaw_rate * self.lead_time

        err_pred = (hd - (psi + yaw_rate * self.lead_time) + np.pi) % (2 * np.pi) - np.pi


        # 4) compute raw command *on the predicted error*,
        #    so P (and optionally D) act early
        raw = Kp_eff * err_pred + self.Ki * self.integral + self.Kd * derr

        # Optional debug: print internals for diagnosing control transients
        if PID_DEBUG:
            try:
                # print a compact per-agent summary (handles n=1 or vectorized)
                for idx in range(self.n):
                    print(f"[PID] idx={idx} err={np.degrees(err[idx]):+.3f}deg "
                          f"err_pred={np.degrees(err_pred[idx]):+.3f}deg "
                          f"P={np.degrees(Kp_eff[idx]*err_pred[idx]) if hasattr(Kp_eff, '__iter__') else np.degrees(Kp_eff*err_pred[idx]):+.3f}deg "
                          f"I={np.degrees(self.integral[idx]*self.Ki):+.3f}deg "
                          f"D={np.degrees(self.Kd*derr[idx]):+.3f}deg "
                          f"raw_deg={np.degrees(raw[idx]):+.3f}deg")
            except Exception:
                # be permissive: if shapes don't match, print scalar-friendly line
                print(f"[PID] err={np.degrees(err):+.3f} err_pred={np.degrees(err_pred):+.3f} raw_deg={np.degrees(raw):+.3f}")

        # release_band_deg and trim_band_deg were converted to radians in __init__
        release_rad = self.release_band_deg
        trim_rad    = self.trim_band_deg
        # previous off-state array
        state_off = self._deadzone_off_state
        # determine new off-state via hysteresis:
        #   stay off when err_pred < release_rad,
        #   re-engage when err_pred > trim_rad + 1°,
        #   otherwise keep previous state
        staying_off = np.abs(err_pred) < release_rad
        coming_on   = np.abs(err_pred) > (trim_rad + np.deg2rad(1.0))
        new_off = np.where(staying_off, True,
                    np.where(coming_on, False, state_off)).astype(bool)
        # zero out the rudder where off-state is True
        raw[new_off] = 0
        # save for next invocation
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

    def colregs(self, dt, positions, nu, psi, current_rpm):
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
        DROP_ANG   = COLLISION_AVOIDANCE["drop_angle"]          # radians
        CPA_SAFE   = COLLISION_AVOIDANCE["d_cpa_safe"]          # metres
        T_CPA_MAX  = COLLISION_AVOIDANCE["t_cpa_max"]           # seconds

        safe_dist  = COLLISION_AVOIDANCE['safe_dist']                                        
        clear_dist = COLLISION_AVOIDANCE['clear_dist']
        unlock_ang = COLLISION_AVOIDANCE['unlock_ang'] 

        n         = self.n
        head      = np.zeros(n)
        # start from current cruise speeds, to be overridden per scenario
        speed_des = self.desired_speed.copy()
        role      = ['neutral'] * n

        for i in range(n):
            # ────────────────────────────────
            # 0) cool-down: stay neutral
            # ────────────────────────────────
            if self.post_avoid_timer[i] > 0.0:
                self.post_avoid_timer[i] = max(
                    0.0,
                    self.post_avoid_timer[i] - dt)

                head[i]      = psi[i]
                speed_des[i] = self.desired_speed[i]
                role[i]      = "neutral"
                continue
            
            pd    = positions[:, i]
            psi_i = psi[i]
            u_i   = nu[0, i]
            v_i   = nu[1, i]
            # earth-frame velocity of vessel i
            vel_i = np.array([
                u_i * np.cos(psi_i) - v_i * np.sin(psi_i),
                u_i * np.sin(psi_i) + v_i * np.cos(psi_i)
            ])
            # --- Hysteresis: exit previous give-way when clear or bearing opens ---
            # --- NEW: stay in give-way as long as the crossing_lock is active ---
            if hasattr(self, '_last_role') \
                and self._last_role[i] == 'give_way' \
                and self.crossing_lock[i] >= 0:
                head[i]      = self.crossing_heading[i]
                speed_des[i] = self.crossing_speed[i]
                role[i] = 'give_way'
                # role[i]      = 'neutral'
                # # also launch cool-down when we drop out here
                # self.post_avoid_timer[i] = COLLISION_AVOIDANCE["post_avoid_time"]
                
           
            hd    = psi_i           # default heading
            rl    = 'neutral'       # default role
            give_way_found = False  # flag if any give_way rule fires
            contact = False         # flag if any vessel is within safe_dist
            
            # ── Maintain any active crossing lock ───────────────────────
            tgt = self.crossing_lock[i]
            if tgt >= 0 and tgt < n:
                # geometry --------------------------------------------------
                delta = positions[:, tgt] - pd
                dist  = np.linalg.norm(delta)

                # world-frame velocities of *each* ship
                c_i, s_i = np.cos(psi[i]), np.sin(psi[i])
                vel_i_w  = np.array([ nu[0, i]*c_i - nu[1, i]*s_i,
                                      nu[0, i]*s_i + nu[1, i]*c_i ])

                c_j, s_j = np.cos(psi[tgt]), np.sin(psi[tgt])
                vel_j_w  = np.array([ nu[0, tgt]*c_j - nu[1, tgt]*s_j,
                                      nu[0, tgt]*s_j + nu[1, tgt]*c_j ])

                rel_w    = vel_j_w - vel_i_w                # correct Δv              
                t_cpa  = -np.dot(delta, rel_w) / (np.dot(rel_w, rel_w) + 1e-9)
                t_cpa  = np.clip(t_cpa, 0.0, T_CPA_MAX)
                d_cpa  = np.linalg.norm(delta + rel_w * t_cpa)
                
                # heading change since lock-on
                hdg_dev = np.abs(((psi_i - self.lock_init_psi[i] + np.pi) %
                                   (2*np.pi)) - np.pi)

                # ── NEW: world-frame range-rate  (+ → opening) ───────────
                range_rate = np.dot(delta, rel_w) / (dist + 1e-9)

                # stay locked **only** while all three are true:
                #   • projected dCPA still inside safety zone *and*
                #   • ships are still closing (range_rate < 0)  *and*
                #   • helm hasn’t swung beyond DROP_ANG
                still_danger = (d_cpa < CPA_SAFE) and (range_rate < 0.0) \
                               and (hdg_dev < DROP_ANG)
                print(f"[{i}] dist={dist:6.0f} dCPA={d_cpa:6.0f} rr={range_rate: .2f} "
                          f"Δψ={np.degrees(hdg_dev):4.1f} danger={still_danger}")
    
                if still_danger:                    
                    hd = self.crossing_heading[i]
                    speed_des[i] = self.crossing_speed[i]
                    rl = 'give_way'
                    role[i] = rl
                    head[i] = hd
                    continue
                else:
                    # lock cleared
                    self.crossing_lock[i] = -1
                    self.lock_init_psi[i] = np.nan
                    self.post_avoid_timer[i] = COLLISION_AVOIDANCE["post_avoid_time"]

                    # flush controller memory so she doesn’t “drag her feet”
                    self.integral[i]        = 0.0
                    self.prev_error[i]      = 0.0
                    self.prev_rudder[i]     = 0.0
                    self.smoothed_rudder[i] = 0.0

                    # hand control back to the route follower
                    head[i]      = psi_i
                    speed_des[i] = self.desired_speed[i]
                    role[i]      = 'neutral'
                    continue

            # --- Emergency braking / astern thrust ---
            # compute stopping (braking) distance: d = v²/(2·a_max), with a safety margin
            v     = self.current_speed[i]
            a_max = self.max_accel[i]    # should be a positive deceleration limit
            d_brk = v*v / (2*a_max) * 1.2

            # examine every other vessel
            for j in range(n):
                if i == j:
                    continue
                delta = positions[:, j] - pd
                dist  = np.linalg.norm(delta)
                if dist > safe_dist:
                    continue
                contact = True

                # relative velocity of j in Earth frame
                u_j = nu[0, j]; v_j = nu[1, j]
                vel_j = np.array([
                    u_j * np.cos(psi[j]) - v_j * np.sin(psi[j]),
                    u_j * np.sin(psi[j]) + v_j * np.cos(psi[j])
                ])
                rel_vel = vel_j - vel_i
                t_cpa   = - (delta @ rel_vel).sum(-1) / (np.linalg.norm(rel_vel, axis=-1)**2 + 1e-9)
                t_cpa   = np.clip(t_cpa, 0.0, T_CPA_MAX)               # future only
                d_cpa   = np.linalg.norm(delta + rel_vel * t_cpa[..., None], axis=-1)

                # compute relative bearing
                bearing = ((np.arctan2(delta[1], delta[0])
                           - psi_i + np.pi) % (2*np.pi) - np.pi)

                # Rule 14: head-on (mutual give-way)
                TURN_14 = np.radians(COLLISION_AVOIDANCE['headon_turn_deg'])  # 30 deg

                if abs(bearing) < np.radians(10):
                    if dist < d_brk:
                        rl = 'give_way'
                        hd = (psi_i - TURN_14) % (2*np.pi)      # >> starboard turn
                        speed_des[i] = -self.max_reverse_speed[i]
                    else:
                        rl = 'give_way'
                        hd = (psi_i - TURN_14) % (2*np.pi)
                        speed_des[i] = 5.0
                    self.crossing_lock[i]    = j
                    self.lock_init_psi[i]    = psi_i     # cache heading at lock-on
                    self.crossing_heading[i] = hd
                    self.crossing_speed[i]   = speed_des[i]
                    give_way_found = True
                    break

                # Rule 15: crossing from starboard (other vessel on your right)
                TURN_15 = np.radians(COLLISION_AVOIDANCE['cross_turn_deg'])  # 20 deg

                if 0 < bearing < np.pi/2:
                    if dist < d_brk:
                        rl = 'give_way'
                        hd = (psi_i - TURN_15) % (2*np.pi)
                        speed_des[i] = -self.max_reverse_speed[i]
                        self.crossing_lock[i] = j
                        self.crossing_heading[i] = hd
                        self.crossing_speed[i] = speed_des[i]
                        give_way_found = True
                        break
                    else:
                        rl = 'give_way'
                        hd = (psi_i - TURN_15) % (2*np.pi)
                        sf = max(0.3, min(1.0, dist/safe_dist))
                        speed_des[i] = self.desired_speed[i] * sf
                        self.crossing_lock[i] = j
                        self.crossing_heading[i] = hd
                        self.crossing_speed[i] = speed_des[i]                        
                        give_way_found = True
                        break

                # Rule 13: overtaking (other vessel approaching from astern)
                if abs(bearing) > 3*np.pi/4:   # bearing within ±45° of dead-astern

                    rl = 'give_way'
                    # now turn starboard by 10°
                    hd = (psi_i - np.pi/18) % (2*np.pi)
                    speed_des[i] = min(self.desired_speed[i] * 1.2, self.max_speed[i])
                    self.crossing_lock[i] = j
                    self.crossing_heading[i] = hd
                    self.crossing_speed[i] = speed_des[i]                    
                    give_way_found = True
                    self.lock_init_psi[i] = psi_i

                    break

            # after checking all contacts:
            if not give_way_found and contact:
                # someone got close but no give-way rule → stand-on
                rl = 'stand_on'

            # neutral or stand-on → resume default cruise speed
            if rl in ['neutral', 'stand_on']:
                speed_des[i] = self.desired_speed[i]
                
            head[i] = hd
            role[i] = rl

        # single conversion: desired speeds → RPM
        rpm = self.speed_to_rpm(speed_des)
        
        # DEBUG
        #print(f"[COLREGS QC] roles = {role}")
        self._last_role = role
        return head, speed_des, rpm, role

    def compensate_heading(self,
                           track_bearing_deg: float,
                           wind_vec: np.ndarray,      # shape (2,)
                           current_vec: np.ndarray,   # shape (2,)
                           surge_speed: float,
                           max_correction_deg: float = 45.0
                          ) -> float:
        """
        Given desired ground‐track bearing (deg) and environmental drift vectors (wind+current),
        return the heading (deg) that, when combined with surge_speed through water, will result
        in net motion along track_bearing_deg.
        """
        # 1) total drift
        drift = wind_vec + current_vec  # shape = (2,)

        # 2) unit vector along desired track
        theta = np.radians(track_bearing_deg)
        u_track = np.array([np.cos(theta), np.sin(theta)])

        # 3) cross‐track component of drift (signed)
        #    = drift ⋅ (unit vector rotated +90°)
        perp = np.array([-u_track[1], u_track[0]])
        V_perp = drift.dot(perp)

        # 4) magnitude of surge speed (through water)
        U = max(abs(surge_speed), 1e-3)

        # 5) correction angle = arcsin(V_perp / U) in radians
        correction = np.arcsin(np.clip(V_perp / U, -1.0, 1.0))
        correction_deg = np.degrees(correction)

        # 6) clamp to avoid crazy angles
        correction_deg = np.clip(correction_deg, -max_correction_deg, max_correction_deg)

        # 7) compensated heading
        return track_bearing_deg - correction_deg

    def reset_behavior(self):
        """
        Reset vessel behavior when its role changes.
        """
        self.current_speed = self.desired_speed
        self.current_heading = self.default_heading
        #log.debug(f"[Vessel {self.ID}] Behavior reset to default values.")

    def step(self, state, rpms, goals, wind, current, dt):
        # state: [x;y;u;v;p;r;phi;psi]
        x,y,u,v,p,r,phi,psi = state
        #self.current_speed = u
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