import numpy as np
from emergent.ship_abm.config import SHIP_PHYSICS, \
    CONTROLLER_GAINS, \
        ADVANCED_CONTROLLER, \
            COLLISION_AVOIDANCE, \
                PROPULSION
from emergent.ship_abm.config import PID_DEBUG

# Helper: normalize a heading to the nearest angular equivalent relative to a
# reference heading (psi). Uses the shared angle utilities when available and
# falls back to a safe modular arithmetic implementation.
try:
    from emergent.ship_abm.angle_utils import heading_diff_rad
    def _nearest_heading(hd, psi_ref):
        # heading_diff_rad(hd, psi_ref) returns (hd - psi_ref) wrapped to [-pi,pi]
        return psi_ref + heading_diff_rad(hd, psi_ref)
except Exception:
    def _nearest_heading(hd, psi_ref):
        return psi_ref + ((hd - psi_ref + np.pi) % (2 * np.pi) - np.pi)

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
        # rudder geometric/physical defaults (can be overridden in SHIP_PHYSICS)
        # area (m^2) of the rudder blade; default small fraction of hull frontal
        self.rudder_area = float(SHIP_PHYSICS.get('rudder_area', 0.02 * self.length * self.draft))
        # lift slope (per rad) for small angles; thin-foil ~ 2*pi ~6.28, use slightly lower
        self.rudder_lift_slope = float(SHIP_PHYSICS.get('rudder_lift_slope', 5.7))
        # stall angle (rad) after which lift degrades
        self.rudder_stall_angle = float(np.deg2rad(SHIP_PHYSICS.get('rudder_stall_deg', 20.0)))
        # lever arm from center of lateral force to yaw center (approx distance aft of CG)
        self.rudder_lever_arm = float(SHIP_PHYSICS.get('rudder_lever_arm', 0.45 * self.length))
        # rudder viscous loss / efficiency factor [0..1]
        self.rudder_efficiency = float(SHIP_PHYSICS.get('rudder_efficiency', 0.95))
        
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
        # derivative low-pass time constant (seconds)
        self.deriv_tau = ADVANCED_CONTROLLER.get('deriv_tau', 1.0)
        # filtered derivative state (initialized per-ship)
        try:
            self.deriv_filtered = np.zeros(self.n)
        except Exception:
            self.deriv_filtered = 0.0
        self._deadzone_off_state = np.zeros(self.n, dtype=bool)
        # dead-reckoning tuning (from config)
        self.dead_reck_sensitivity = ADVANCED_CONTROLLER.get('dead_reck_sensitivity', 0.25)
        self.dead_reck_max_corr_deg = ADVANCED_CONTROLLER.get('dead_reck_max_corr_deg', 30.0)
    # Note: dead-reck damping is derived from existing parameters instead
    # of introducing a separate config key. We will map
    # `dead_reck_sensitivity` → damping when computing corrections.
    
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
        # linger timer: after a crossing lock clears, keep role='give_way'
        # for a brief period to avoid chatter (seconds)
        try:
            from emergent.ship_abm.config import COLLISION_AVOIDANCE
            self.crossing_linger_default = float(COLLISION_AVOIDANCE.get('crossing_linger', 6.0))
        except Exception:
            self.crossing_linger_default = 6.0
        self.crossing_linger_timer = np.zeros(n, dtype=float)
        # UI-visible persistent flag: mark vessels that recently had to give way.
        # This flag is intended to be shown in the UI and not automatically
        # cleared by COLREGS logic; it must be cleared explicitly by the UI
        # or higher-level code when the operator acknowledges it.
        try:
            self.flagged_give_way = np.zeros(n, dtype=bool)
        except Exception:
            self.flagged_give_way = np.array([False] * n)
        # Instrumentation state: previous goal heading for HD jump detection,
        # rudder saturation timers and reporting flags
        try:
            self._instr_prev_hd = None
            self._sat_timer = np.zeros(self.n)
            self._sat_reported = np.zeros(self.n, dtype=bool)
        except Exception:
            self._instr_prev_hd = None
            self._sat_timer = 0.0
            self._sat_reported = False
        # transient integrator-flush indicator per-agent (set when integrator is cleared,
        # cleared at the end of the colregs pass so it only signals for one tick)
        try:
            self._last_integrator_flushed = np.zeros(self.n, dtype=bool)
        except Exception:
            self._last_integrator_flushed = np.array([False] * self.n)
        # safety counter: if a give-way vessel issues near-zero rudder for many
        # consecutive ticks, escalate by nudging prev_rudder to a small avoidance
        # value to ensure motion (defensive fallback)
        try:
            self._giveway_no_rudder_counts = np.zeros(self.n, dtype=int)
        except Exception:
            self._giveway_no_rudder_counts = np.zeros(self.n, dtype=int)

    def cut_power(self, idx: int):
        """
        Instantly sets RPM and desired speed to zero for vessel *idx*.
        """
        # Support both scalar and vector storage just in case.
        # Before zeroing, write a best-effort log entry so we can detect
        # where commanded_rpm gets zeroed during headless runs.
        try:
            import os, datetime
            # prefer workspace-relative logs/ but fall back to package-relative
            workspace_logs = os.path.abspath(os.path.join(os.getcwd(), 'logs'))
            os.makedirs(workspace_logs, exist_ok=True)
            evfile = os.path.join(workspace_logs, 'cut_power_events.log')
            ts = datetime.datetime.utcnow().isoformat() + 'Z'
            prev = None
            try:
                # read prev value safely
                prev = float(self.commanded_rpm[idx]) if np.ndim(self.commanded_rpm) != 0 else float(self.commanded_rpm)
            except Exception:
                prev = None
            with open(evfile, 'a') as fh:
                fh.write(f"{ts} cut_power called idx={idx} prev_cmd={prev}\n")
        except Exception:
            # best-effort logging; do not allow logging failure to raise
            pass

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
            # make the vessel “dead in the water”
            self.current_speed[idx] = 0.0

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
        # ensure psi is array-like with length == nships for indexing
        psi_arr = np.atleast_1d(psi)
        if psi_arr.size != nships:
            # broadcast or repeat scalar heading to match nships
            try:
                psi_val = float(psi_arr.flat[0])
            except Exception:
                psi_val = 0.0
            psi_arr = np.full(nships, psi_val)

        for i in range(nships):
            psi_local = float(psi_arr[i])
            wf = _force(rel_wind[:, i], self.rho_air, Cd_air, A_ref, psi_local, self.A_air)
            cf = _force(rel_current[:, i], self.rho, self.Cd_water, None, psi_local, self.A)
            wind_force_cols.append(wf)
            current_force_cols.append(cf)

        wind_force = np.hstack(wind_force_cols)
        current_force = np.hstack(current_force_cols)
        # apply a global wind-force scaling factor (configurable)
        try:
            from emergent.ship_abm.config import SHIP_AERO_DEFAULTS
            wscale = float(SHIP_AERO_DEFAULTS.get('wind_force_scale', 1.0))
        except Exception:
            wscale = 1.0
        if abs(wscale - 1.0) > 1e-12:
            wind_force = wind_force * wscale
        return wind_force, current_force      # shape (2, n) each
    
    def coriolis_matrix(self, u, v):
        c2 = -self.m * v; c3 = self.m * u; c1 = np.zeros(self.n)
        return np.vstack([c1, c2, c3])

    def quadratic_damping(self, v, r):
        return np.vstack([np.zeros(self.n), 500*np.abs(v)*v, 1e6*np.abs(r)*r])

    def compute_rudder_torque(self, rudder_angle, lengths, u):
        """Physics-based rudder torque model (vectorized).

        Parameters
        ----------
        rudder_angle : array_like (n,) in radians (positive = starboard)
        lengths      : unused here (kept for legacy compatibility)
        u            : array_like (n,) forward speed through water (m/s)

        Returns
        -------
        torque : array_like (n,) yaw moment applied by rudder (N·m)

        Model:
          - dynamic pressure q = 0.5 * rho * u^2
          - lift (side) F_y = q * A_rudder * C_L(alpha)
          - C_L approx = a0 * alpha for |alpha| < alpha_stall, then scaled down
          - moment = F_y * lever_arm * efficiency
        """
        # ensure arrays
        rud = np.atleast_1d(rudder_angle)
        u = np.atleast_1d(u)

        # side force depends on square of forward speed; for negative u, use small positive floor
        U_eff = np.maximum(np.abs(u), 1e-3)
        q = 0.5 * self.rho * U_eff**2

        # lift coefficient with simple stall model
        a0 = self.rudder_lift_slope
        alpha = np.clip(rud, -np.pi/2, np.pi/2)

        # linear region
        Cl_lin = a0 * alpha

        # stall scaling: reduce lift gradually beyond stall angle (30° transition)
        stall_window = np.deg2rad(30.0)
        s = np.minimum(1.0, np.maximum(0.0, 1.0 - (np.abs(alpha) - self.rudder_stall_angle) / (stall_window)))
        s = np.where(np.abs(alpha) <= self.rudder_stall_angle, 1.0, s)
        Cl = Cl_lin * s

        # side force (N)
        F_y = q * self.rudder_area * Cl * self.rudder_efficiency

        # yaw moment = F_y * lever_arm (N·m)
        moment = F_y * self.rudder_lever_arm
        return moment

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
        # Debug: log the raw environmental vector and heading on the first
        # few calls so we can confirm sign/axis semantics when running the GUI.
        try:
            _dbg_cnt = getattr(self, '_dr_input_dbg', 0)
            if _dbg_cnt < 6:
                # current_vec may be None or a (2,n) array; pick index-0 for readability
                if current_vec is None:
                    cur0 = (0.0, 0.0)
                else:
                    try:
                        cur0 = (float(current_vec[0, 0]), float(current_vec[1, 0]))
                    except Exception:
                        # fallback if shapes differ
                        cur_arr = np.atleast_2d(current_vec)
                        cur0 = (float(cur_arr[0, 0]), float(cur_arr[1, 0]))
                # psi may be scalar or array-like
                try:
                    psi0 = float(np.atleast_1d(psi)[0])
                except Exception:
                    psi0 = 0.0
                print(f"DR-IN call={_dbg_cnt} raw_current=(E{cur0[0]:.3f},N{cur0[1]:.3f}) psi_deg={np.degrees(psi0):.6f}")
                self._dr_input_dbg = _dbg_cnt + 1
        except Exception:
            pass
        # Debug: show attract/positions/goals for first few calls
        try:
            cnt = getattr(self, '_attract_dbg_calls', 0)
            if cnt < 6:
                a = attract[:, 0] if attract.shape[1] >= 1 else attract
                pos0 = positions[:, 0]
                g0 = goals_arr[:, 0]
                print(f"AT-DBG call={cnt} attract={a.tolist()} pos={pos0.tolist()} goal={g0.tolist()}")
                self._attract_dbg_calls = cnt + 1
        except Exception:
            pass

        # ------------------------------------------------------------------
        # DEAD-RECKONING : compute a conservative correction for drift
        # ------------------------------------------------------------------
        # Optional experimental bypass: allow tests to disable dead-reck
        # corrections by setting `ship.disable_dead_reck = True` on the
        # instance. This is useful for isolating whether dead-reck logic is
        # responsible for transient large heading corrections.
        if getattr(self, 'disable_dead_reck', False):
            # Return raw attractor heading (no correction) and desired speed
            hd = np.arctan2(attract[1], attract[0])
            sp = np.array(self.desired_speed, copy=True)
            return hd, sp

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

            # Apply conservative damping to avoid over-correction. Instead of
            # introducing a new config parameter, derive a damping factor from
            # the existing dead_reck_sensitivity: higher sensitivity means we
            # should trust the correction more (less damping). Map
            # sensitivity ∈ [0.0, 1.0] → damping ∈ [0.7, 1.0].
            sens = getattr(self, 'dead_reck_sensitivity', 0.25)
            try:
                sens = float(sens)
            except Exception:
                sens = 0.25
            # clamp sens
            sens = np.clip(sens, 0.0, 1.0)
            # linear mapping (tunable): sens=0 → damping=0.85 (moderate damping)
            # sens=1 → damping=1.0 (no damping). This gives more correction
            # authority for typical sensitivity values while still preventing
            # full over-correction when sens is very small.
            damping = 0.85 + 0.15 * sens
            corr = corr * damping

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

            # Normalize angular quantities to the nearest-equivalent relative to
            # the current heading (psi) before performing a linear blend. This
            # prevents linear interpolation across the ±π branch cut which can
            # produce far-equivalent headings (e.g., +30° → -180° → +150° jump).
            try:
                # _nearest_heading handles vectorized inputs when angle_utils
                # is available; fallback is defined at module top.
                hd_base_norm = _nearest_heading(hd_base, psi)
                hd_corr_norm = _nearest_heading(hd_corrected, psi)
                hd_base = hd_base_norm
                hd_corrected = hd_corr_norm
            except Exception:
                # if any issue, continue with the raw values (conservative)
                pass

            # Debugging: print intermediate dead-reckoning values for the first
            # few calls to help identify why goal_hd flips early in the run.
            try:
                cnt = getattr(self, '_dr_dbg_calls', 0)
                if cnt < 6:
                    # Convert to degrees for readability
                    def _d(a):
                        a = np.atleast_1d(a)
                        return np.degrees(a).tolist()
                    # extract per-agent diagnostics for clearer logs
                    try:
                        idx = 0
                        # current vector for this agent
                        cur = current_vec[:, idx]
                        vperp = V_perp[idx]
                        lat_off = lateral_offset[idx]
                        U_val = U[idx]
                        fwd = forward_dist[idx]
                        cur_mag = current_mag[idx]
                    except Exception:
                        # fall back to scalars
                        cur = current_vec if current_vec is not None else np.array([0.0, 0.0])
                        vperp = V_perp if 'V_perp' in locals() else 0.0
                        lat_off = lateral_offset if 'lateral_offset' in locals() else 0.0
                        U_val = U if 'U' in locals() else 0.0
                        fwd = forward_dist if 'forward_dist' in locals() else 0.0
                        cur_mag = current_mag if 'current_mag' in locals() else 0.0
                    # Use simulation logger when available, else print
                    msg = (
                        f"DR-DBG call={cnt} hd_base_deg={_d(hd_base)[0]:.6f} "
                        f"hd_corr_deg={_d(hd_corrected)[0]:.6f} corr_deg={np.degrees(corr)[0]:.6f} "
                        f"beta={beta[0]:.6f} psi_deg={np.degrees(psi)[0]:.6f} "
                        f"cur=(E{cur[0]:.3f},N{cur[1]:.3f}) V_perp={vperp:.3f} "
                        f"lat_off={lat_off:.3f} U={U_val:.3f} fwd={fwd:.3f} cur_mag={cur_mag:.3f}"
                    )
                    # Emit both to the simulation logger (if present) and to
                    # stdout so short headless runs capture the diagnostics.
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        try:
                            sim_log.debug(msg)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Always print to stdout for visibility in quick runs
                    try:
                        print(msg)
                    except Exception:
                        pass
                    self._dr_dbg_calls = cnt + 1
            except Exception:
                pass

            # final heading: smooth blend (handles vector shapes)
            hd = (1.0 - beta) * hd_base + beta * hd_corrected
            # Instrumentation: detect huge heading jumps (>90°) relative to
            # the last produced goal heading and emit a structured record.
            try:
                if getattr(self, '_instr_prev_hd', None) is None:
                    self._instr_prev_hd = hd.copy()
                try:
                    from emergent.ship_abm.angle_utils import heading_diff_rad
                    hd_diff = np.abs(heading_diff_rad(hd, self._instr_prev_hd))
                except Exception:
                    hd_diff = np.abs(((hd - self._instr_prev_hd + np.pi) % (2*np.pi)) - np.pi)
                if np.any(hd_diff > np.deg2rad(90.0)):
                    # collect richer context (index-0 agent best-effort)
                    try:
                        idx0 = 0
                        pos0 = positions[:, idx0].tolist() if 'positions' in locals() else [None, None]
                        goal0 = goals_arr[:, idx0].tolist() if 'goals_arr' in locals() else [None, None]
                        attract0 = attract[:, idx0].tolist() if 'attract' in locals() else [None, None]
                        cur0 = current_vec[:, idx0].tolist() if (current_vec is not None and 'current_vec' in locals()) else [0.0, 0.0]
                        vperp0 = float(V_perp[idx0]) if 'V_perp' in locals() else None
                        lat_off0 = float(lateral_offset[idx0]) if 'lateral_offset' in locals() else None
                        U0 = float(U[idx0]) if 'U' in locals() else None
                        fwd0 = float(forward_dist[idx0]) if 'forward_dist' in locals() else None
                        beta0 = float(beta[idx0]) if 'beta' in locals() else None
                    except Exception:
                        pos0 = goal0 = attract0 = cur0 = [None, None]
                        vperp0 = lat_off0 = U0 = fwd0 = beta0 = None
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        sim_log.info('[INSTRUMENT] EVENT=HD_JUMP idxs=%s new_deg=%s prev_deg=%s Δdeg=%s pos=%s goal=%s attract=%s cur=%s V_perp=%.3f lat_off=%.3f U=%.3f fwd=%.3f beta=%.3f',
                                     np.where(hd_diff > np.deg2rad(90.0))[0].tolist(),
                                     np.round(np.degrees(hd).tolist(),3),
                                     np.round(np.degrees(self._instr_prev_hd).tolist(),3),
                                     np.round(np.degrees(hd_diff).tolist(),3),
                                     pos0, goal0, attract0, cur0, vperp0, lat_off0, U0, fwd0, beta0)
                    except Exception:
                        print(f"[INSTRUMENT] EVENT=HD_JUMP idxs={np.where(hd_diff>np.deg2rad(90.0))[0].tolist()} new={np.degrees(hd).tolist()} prev={np.degrees(self._instr_prev_hd).tolist()} Δ={np.degrees(hd_diff).tolist()} pos={pos0} goal={goal0} attract={attract0} cur={cur0} V_perp={vperp0} lat_off={lat_off0} U={U0} fwd={fwd0} beta={beta0}")
                self._instr_prev_hd = hd.copy()
            except Exception:
                pass
            # --- Defensive guard: avoid accepting a new hd that is far (>120°)
            # from the previous returned heading in this ship instance unless
            # the distance to the goal has materially changed. This prevents
            # spurious ±180° flips at startup caused by tiny geometry or
            # ordering differences. We log the first few occurrences for
            # debugging.
            try:
                # ensure we have a per-instance previous heading storage
                if not hasattr(self, '_prev_goal_hd'):
                    self._prev_goal_hd = hd.copy()
                    self._prev_goal_dist = np.linalg.norm(goals_arr - positions, axis=0)

                # compute angular difference to previous goal heading
                try:
                    from emergent.ship_abm.angle_utils import heading_diff_rad
                    ang_diff = np.abs(heading_diff_rad(hd, self._prev_goal_hd))
                except Exception:
                    ang_diff = np.abs(((hd - self._prev_goal_hd + np.pi) % (2*np.pi)) - np.pi)

                # compute current distance to first goal and compare to previous
                cur_dist = np.linalg.norm(goals_arr - positions, axis=0)
                # if angular jump > 120° but distance hasn't changed much (< 5%),
                # keep previous heading and log an informational debug line.
                big_jump_mask = ang_diff > np.deg2rad(120.0)
                small_dist_change = np.abs(cur_dist - self._prev_goal_dist) < (0.05 * np.maximum(self._prev_goal_dist, 1e-3))
                if np.any(big_jump_mask & small_dist_change):
                    # Only mute the jump for those agents meeting the condition
                    hd = np.where(big_jump_mask & small_dist_change, self._prev_goal_hd, hd)
                    if getattr(self, 'verbose', False) or PID_DEBUG:
                        try:
                            from emergent.ship_abm.simulation_core import log as sim_log
                            sim_log.debug("[DR-GUARD] muted far-equivalent hd change (deg): new=%s prev=%s dist_change=%s",
                                          np.round(np.degrees(np.atleast_1d(hd)),3).tolist(),
                                          np.round(np.degrees(np.atleast_1d(self._prev_goal_hd)),3).tolist(),
                                          np.round(cur_dist - self._prev_goal_dist,3).tolist())
                        except Exception:
                            print(f"[DR-GUARD] muted far-equivalent hd change new={np.degrees(hd)[0]:.3f} prev={np.degrees(self._prev_goal_hd)[0]:.3f} distΔ={cur_dist[0]-self._prev_goal_dist[0]:.3f}")

                # update stored values for next call
                self._prev_goal_hd = hd.copy()
                self._prev_goal_dist = cur_dist
            except Exception:
                # if anything goes wrong with guard bookkeeping, fall back silently
                pass
            # Optional debug printing to trace dead-reck signs/values
            try:
                from emergent.ship_abm.config import PID_DEBUG
                # print only when PID_DEBUG or if this instance has verbose attribute
                verbose_print = PID_DEBUG or getattr(self, 'verbose', False)
                if verbose_print:
                    for i in range(hd.shape[0] if hasattr(hd, '__iter__') else 1):
                        # handle scalar/vector shapes
                        idx = i if hasattr(hd, '__iter__') else 0
                        cur = current_vec[:, idx]
                        cd = cog_dir[:, idx]
                        vperp = V_perp[idx]
                        corr_deg = np.degrees(corr[idx]) if hasattr(corr, '__iter__') else np.degrees(corr)
                        # normalize hd_base/hd for logging to degrees in [-180,180)
                        hd_base_deg = ((np.degrees(hd_base[idx]) + 180.0) % 360.0) - 180.0
                        hd_final_deg = ((np.degrees(hd[idx]) + 180.0) % 360.0) - 180.0
                        try:
                            from emergent.ship_abm.simulation_core import log as sim_log
                            if getattr(self, 'verbose', False):
                                sim_log.debug(f"[DR] idx={idx} attract=({attract[0,idx]:.2f},{attract[1,idx]:.2f}) hd_base={hd_base_deg:.2f}° "
                                              f"cur=(E{cur[0]:.3f},N{cur[1]:.3f}) V_perp={vperp:.3f} corr={corr_deg:.2f}° hd_final={hd_final_deg:.2f}°")
                        except Exception:
                            # fallback to print only if verbose
                            if getattr(self, 'verbose', False):
                                print(f"[DR] idx={idx} attract=({attract[0,idx]:.2f},{attract[1,idx]:.2f}) hd_base={hd_base_deg:.2f}° "
                                      f"cur=(E{cur[0]:.3f},N{cur[1]:.3f}) V_perp={vperp:.3f} corr={corr_deg:.2f}° hd_final={hd_final_deg:.2f}°")
            except Exception:
                pass
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

        # Optional: when avoiding, we may want to inhibit the normal PID so
        # the avoidance heading is not fought by waypoint-following integrator.
        try:
            from emergent.ship_abm.config import COLLISION_AVOIDANCE
            disable_while_avoiding = COLLISION_AVOIDANCE.get('disable_pid_while_avoiding', False)
        except Exception:
            disable_while_avoiding = False

        if disable_while_avoiding:
            # build a mask of agents that are actively avoiding
            try:
                # crossing_lock >= 0 indicates active lock; do NOT use the UI-visible
                # `flagged_give_way` persistence flag to disable PID. The controller
                # must react to active signals (locks/timers/role) only. Using the
                # UI flag could leave the controller neutral even when avoidance
                # timers have expired or been cleared; exclude it from the mask.
                avoiding_mask = (self.crossing_lock >= 0) | (self.crossing_linger_timer > 0.0) | (self.post_avoid_timer > 0.0)
                # if any agents avoiding, zero their integrator and prev_rudder and return a mixed vector
                if np.any(avoiding_mask):
                    # ensure shapes
                    n = self.n
                    out = np.array(self.prev_rudder, copy=True)
                    # zero outputs for avoiding agents
                    try:
                        out = np.atleast_1d(out)
                        out[avoiding_mask] = 0.0
                    except Exception:
                        # scalar fallback
                        if bool(avoiding_mask):
                            out = 0.0
                    # clear integrator for avoiding agents to prevent sudden returns
                    try:
                        self.integral = np.where(avoiding_mask, 0.0, self.integral)
                    except Exception:
                        try:
                            self.integral = np.zeros_like(self.integral)
                        except Exception:
                            self.integral = 0.0
                    # update prev_rudder and return immediately
                    self.prev_rudder = out
                    return out
            except Exception:
                # if anything goes wrong with avoidance gating, fall back to normal PID
                pass

        # 1) heading error normalized to [-π, π] (positive error -> starboard turn)
        # Use shared helper to ensure consistency with simulation-level controller
        try:
            from emergent.ship_abm.angle_utils import heading_diff_rad
            err = heading_diff_rad(hd, psi)
        except Exception:
            # fallback: manual wrap if import fails
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
        # apply first-order low-pass to derivative to reduce noise sensitivity
        try:
            tau = float(getattr(self, 'deriv_tau', 1.0))
            alpha_d = dt / (tau + dt)
            # ensure shapes match for vectorized update
            self.deriv_filtered = (1.0 - alpha_d) * np.atleast_1d(self.deriv_filtered) + alpha_d * np.atleast_1d(derr)
            derr_f = np.atleast_1d(self.deriv_filtered)
        except Exception:
            derr_f = derr

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

        # Conservative feed-forward guard: if predicted error is very large,
        # reduce the effective P contribution so we don't command an overly
        # aggressive raw command that immediately saturates the actuator.
        try:
            ff_err_limit_deg = float(ADVANCED_CONTROLLER.get('ff_err_limit_deg', 10.0))
        except Exception:
            ff_err_limit_deg = 10.0
        ff_limit_rad = np.deg2rad(ff_err_limit_deg)
        try:
            large_mask = np.abs(err_pred) > ff_limit_rad
            if np.any(large_mask):
                # scale local Kp (for these agents) down when prediction is huge
                if hasattr(Kp_eff, '__iter__'):
                    Kp_eff = np.where(large_mask, Kp_eff * 0.5, Kp_eff)
                else:
                    if large_mask:
                        Kp_eff = Kp_eff * 0.5
        except Exception:
            pass


            # 4) compute raw command *on the predicted error*,
            #    so P (and optionally D) act early
            # use filtered derivative for D-term
            raw = Kp_eff * err_pred + self.Ki * self.integral + self.Kd * derr_f

        # Apply a configurable soft pre-saturation cap to the raw PID output.
        # This prevents extremely large internal commands (e.g., hundreds of degrees)
        # from causing aggressive integrator/back-calculation dynamics. The cap is
        # specified in degrees in ADVANCED_CONTROLLER['raw_cap_deg']. If not set,
        # no extra cap is applied.
        try:
            from emergent.ship_abm.config import ADVANCED_CONTROLLER
            raw_cap_deg = ADVANCED_CONTROLLER.get('raw_cap_deg', None)
            if raw_cap_deg is not None:
                cap_rad = np.deg2rad(float(raw_cap_deg))
                # raw may be scalar or array-like
                raw = np.clip(raw, -cap_rad, cap_rad)
        except Exception:
            # if config is not available for any reason, silently continue
            pass

        # Optional debug: print internals for diagnosing control transients.
        # Use the module logger at DEBUG level and only emit when PID_DEBUG
        # or self.verbose is True.
        try:
            from emergent.ship_abm.simulation_core import log as sim_log
            if PID_DEBUG or getattr(self, 'verbose', False):
                try:
                    for idx in range(self.n):
                        sim_log.debug("[PID] idx=%s err=%+.3fdeg err_pred=%+.3fdeg P=%+.3fdeg I=%+.3fdeg D=%+.3fdeg raw_deg=%+.3fdeg",
                                      idx,
                                      np.degrees(err[idx]),
                                      np.degrees(err_pred[idx]),
                                      np.degrees(Kp_eff[idx]*err_pred[idx]) if hasattr(Kp_eff, '__iter__') else np.degrees(Kp_eff*err_pred[idx]),
                                      np.degrees(self.integral[idx]*self.Ki),
                                      np.degrees(self.Kd*derr[idx]),
                                      np.degrees(raw[idx]))
                except Exception:
                    sim_log.debug("[PID] err=%s err_pred=%s raw_deg=%s", np.degrees(err), np.degrees(err_pred), np.degrees(raw))
        except Exception:
            # Last-resort fallback: use print only if verbose
            if PID_DEBUG or getattr(self, 'verbose', False):
                try:
                    for idx in range(self.n):
                        print(f"[PID] idx={idx} err={np.degrees(err[idx]):+.3f}deg "
                              f"err_pred={np.degrees(err_pred[idx]):+.3f}deg "
                              f"P={np.degrees(Kp_eff[idx]*err_pred[idx]) if hasattr(Kp_eff, '__iter__') else np.degrees(Kp_eff*err_pred[idx]):+.3f}deg "
                              f"I={np.degrees(self.integral[idx]*self.Ki):+.3f}deg "
                              f"D={np.degrees(self.Kd*derr[idx]):+.3f}deg "
                              f"raw_deg={np.degrees(raw[idx]):+.3f}deg")
                except Exception:
                    print(f"[PID] err={np.degrees(err):+.3f} err_pred={np.degrees(err_pred):+.3f} raw_deg={np.degrees(raw):+.3f}")

            # Instrumentation: report very large heading errors (>30°)
            try:
                err_mask = np.abs(err) > np.deg2rad(30.0)
                if np.any(err_mask):
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        sim_log.info('[INSTRUMENT] EVENT=ERR_LARGE idxs=%s err_deg=%s hd_deg=%s psi_deg=%s',
                                     np.where(err_mask)[0].tolist(),
                                     np.round(np.degrees(err).tolist(),3),
                                     np.round(np.degrees(hd).tolist(),3),
                                     np.round(np.degrees(psi).tolist(),3))
                    except Exception:
                        print(f"[INSTRUMENT] EVENT=ERR_LARGE idxs={np.where(err_mask)[0].tolist()} err_deg={np.round(np.degrees(err).tolist(),3)} hd_deg={np.round(np.degrees(hd).tolist(),3)} psi_deg={np.round(np.degrees(psi).tolist(),3)}")
            except Exception:
                pass

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

        # 5) anti-windup: integrate only when not saturated OR when
        # the error would drive the integrator back toward unsaturation.
        # This prevents integral from winding up when rudder is saturated.
        # Allow a small epsilon so we still integrate when very close to limit
        eps = 1e-8
        des_abs = np.abs(des)
        # integrate if desired command is not saturated, or if the sign of the
        # error would reduce the command magnitude (i.e., help recover)
        sign_err = np.sign(err)
        sign_des = np.sign(des)
        recover_mask = sign_err != sign_des
        windup_mask = (des_abs < (self.max_rudder - eps)) | recover_mask
        self.integral[windup_mask] += err[windup_mask] * dt

        # clamp integral term so Ki * integral stays within configured I_max
        try:
            I_max = float(self.I_max)
        except Exception:
            I_max = None
        if I_max is not None and self.Ki != 0.0:
            # clamp such that Ki * integral is within [-I_max, I_max]
            max_integral = I_max / (abs(self.Ki) + 1e-12)
            self.integral = np.clip(self.integral, -max_integral, max_integral)

        # Note: integrator back-calculation will be applied after we compute the
        # final applied rudder (post rate-limit & smoothing) so that the
        # integrator is unwound based on the true actuator output.

        # 6) compute I-term and construct the unclamped desired command
        # (we recompute des using the clamped integral contribution)
        I_term = self.Ki * self.integral
        # recompute raw with clamped I-term and same P/D contributions
        raw_with_I = Kp_eff * err_pred + I_term + self.Kd * derr_f
        # clamp to rudder limits
        des = np.clip(raw_with_I, -self.max_rudder, self.max_rudder)

        # 7) rate-limit: limit change per timestep
        max_delta = self.max_rudder_rate * dt
        rud_rate_limited = np.clip(
            des,
            self.prev_rudder - max_delta,
            self.prev_rudder + max_delta
        )

        # 8) first-order servo (low-pass) to emulate actuator dynamics
        alpha = dt / (self.rudder_tau + dt)
        # ensure smoothed_rudder is same shape
        self.smoothed_rudder = (1.0 - alpha) * self.smoothed_rudder + alpha * rud_rate_limited
        rud = self.smoothed_rudder

        # Instrumentation: detect sustained rudder saturation (>1s)
        try:
            # maintain a small dt estimator (updated externally by caller ideally)
            dt_est = getattr(self, '_last_dt_est', 0.1)
            sat_mask = np.abs(des) >= (self.max_rudder - 1e-8)
            if np.isscalar(self._sat_timer):
                self._sat_timer = np.zeros(self.n)
            self._sat_timer = self._sat_timer + dt_est * sat_mask.astype(float)
            over_mask = self._sat_timer > 1.0
            if np.any(over_mask & (~self._sat_reported)):
                idxs = np.where(over_mask & (~self._sat_reported))[0].tolist()
                try:
                    from emergent.ship_abm.simulation_core import log as sim_log
                    sim_log.info('[INSTRUMENT] EVENT=RUD_SAT idxs=%s duration_s=%s des_deg=%s',
                                 idxs,
                                 np.round(self._sat_timer.tolist(),2),
                                 np.round(np.degrees(des).tolist(),3))
                except Exception:
                    print(f"[INSTRUMENT] EVENT=RUD_SAT idxs={idxs} duration_s={self._sat_timer} des_deg={np.round(np.degrees(des).tolist(),3)}")
                try:
                    self._sat_reported[over_mask & (~self._sat_reported)] = True
                except Exception:
                    self._sat_reported = True
        except Exception:
            pass

        # Integrator back-calculation based on the actual applied rudder
        try:
            backcalc_beta = float(ADVANCED_CONTROLLER.get('backcalc_beta', 0.08))
        except Exception:
            backcalc_beta = 0.08
        try:
            raw_arr = np.atleast_1d(raw_with_I)
            applied_arr = np.atleast_1d(rud)
            diff = raw_arr - applied_arr
            apply_mask = np.abs(diff) > 1e-8
            if np.isscalar(self.integral):
                if apply_mask:
                    self.integral -= backcalc_beta * diff * dt
            else:
                # vectorized subtract only where meaningful
                self.integral = self.integral - (backcalc_beta * diff * dt * apply_mask)
        except Exception:
            pass
        
        # 8) update state
        self.prev_error  = err
        # update prev_error only outside the dead-band
        # self.prev_error[~mask_db] = err[~mask_db]
        self.prev_rudder = rud

        # Safety: if vessel is in active avoidance (lock/linger/post) but
        # applied rudder remains near zero for many consecutive ticks, nudge
        # prev_rudder to a small commanded avoidance to ensure motion.
        try:
            # determine active avoidance per-ship
            avoiding_mask = (self.crossing_lock >= 0) | (self.crossing_linger_timer > 0.0) | (self.post_avoid_timer > 0.0)
            # small threshold (radians) below which rudder is considered 'zero'
            zero_thresh = np.deg2rad(0.5)
            # nudge magnitude (radians)
            nudge_rad = np.deg2rad(5.0)
            # for vectorized arrays
            for i in range(self.n):
                try:
                    if avoiding_mask[i]:
                        rud_val = float(np.atleast_1d(rud)[i])
                        if abs(rud_val) < zero_thresh:
                            self._giveway_no_rudder_counts[i] += 1
                        else:
                            self._giveway_no_rudder_counts[i] = 0
                        # if persisted for >8 ticks, apply small nudge
                        if self._giveway_no_rudder_counts[i] > 8:
                            # choose sign from raw_with_I (desired before smoothing)
                            try:
                                sign = np.sign(float(np.atleast_1d(raw_with_I)[i]))
                            except Exception:
                                sign = 1.0
                            try:
                                self.prev_rudder[i] = sign * nudge_rad
                                # also reset smoothed rudder to the nudge to force motion
                                self.smoothed_rudder[i] = sign * nudge_rad
                            except Exception:
                                self.prev_rudder = sign * nudge_rad
                    else:
                        # not avoiding -> reset counter
                        self._giveway_no_rudder_counts[i] = 0
                except Exception:
                    # best-effort: if per-agent indexing fails, ignore
                    pass
        except Exception:
            pass

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
        # decrement any active crossing-linger timers so they naturally expire
        try:
            # protect shape mismatches
            self.crossing_linger_timer = np.maximum(0.0, self.crossing_linger_timer - dt)
        except Exception:
            try:
                self.crossing_linger_timer = np.zeros(n, dtype=float)
            except Exception:
                self.crossing_linger_timer = np.zeros(n)

        for i in range(n):
            # --- instrumentation: write a per-agent COLREGS row for post-mortem
            try:
                import os, csv
                logs_dir = os.path.abspath(os.path.join(os.getcwd(), 'logs'))
                os.makedirs(logs_dir, exist_ok=True)
                path = os.path.join(logs_dir, 'colregs_runtime_debug.csv')
                write_hdr = not os.path.exists(path)
                sim_time = float(getattr(self, '_sim_time', np.nan))
                with open(path, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    if write_hdr:
                        writer.writerow(['sim_time', 'agent', 'role', 'flagged_give_way', 'crossing_lock', 'crossing_heading_deg', 'crossing_speed_mps', 'crossing_linger_s', 'post_avoid_s', 'integrator_flushed'])
                    try:
                        ch_deg = float(np.degrees(self.crossing_heading[i])) if not np.isnan(self.crossing_heading[i]) else float('nan')
                    except Exception:
                        ch_deg = float('nan')
                    try:
                        cs_mps = float(self.crossing_speed[i])
                    except Exception:
                        cs_mps = float('nan')
                    try:
                        flag = int(bool(self.flagged_give_way[i]))
                    except Exception:
                        flag = 0
                    try:
                        lockv = int(self.crossing_lock[i])
                    except Exception:
                        lockv = -1
                    try:
                        linger = float(self.crossing_linger_timer[i])
                    except Exception:
                        linger = float('nan')
                    try:
                        post = float(self.post_avoid_timer[i])
                    except Exception:
                        post = float('nan')
                    # hook: any time we flush integrator on this agent we set a transient attribute
                    flushed = int(bool(getattr(self, '_last_integrator_flushed', False) and getattr(self, '_last_integrator_flushed_idx', -1) == i))
                    writer.writerow([sim_time, int(i), str('neutral'), flag, lockv, ch_deg, cs_mps, linger, post, flushed])
            except Exception:
                pass
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
            # Auto-clear the persistent UI flag when the vessel is no longer
            # at risk: no active lock, no linger/post_avoid timers, and no
            # nearby contact that is closing into a CPA inside the safety bound.
            try:
                if (getattr(self, 'flagged_give_way', None) is not None
                        and self.crossing_linger_timer[i] <= 0.0
                        and self.crossing_lock[i] < 0
                        and self.post_avoid_timer[i] <= 0.0):
                    # Determine whether any contact still represents a CPA threat
                    _safe = True
                    for j in range(self.n):
                        if i == j:
                            continue
                        delta = positions[:, j] - pd
                        dist = np.linalg.norm(delta)
                        # other vessel world-frame velocity
                        c_j, s_j = np.cos(psi[j]), np.sin(psi[j])
                        vel_j_w = np.array([ nu[0, j]*c_j - nu[1, j]*s_j,
                                              nu[0, j]*s_j + nu[1, j]*c_j ])
                        rel_w = vel_j_w - vel_i
                        t_cpa = -np.dot(delta, rel_w) / (np.dot(rel_w, rel_w) + 1e-9)
                        t_cpa = np.clip(t_cpa, 0.0, T_CPA_MAX)
                        d_cpa = np.linalg.norm(delta + rel_w * t_cpa)
                        range_rate = np.dot(delta, rel_w) / (dist + 1e-9)
                        if (d_cpa < CPA_SAFE) and (range_rate < 0.0):
                            _safe = False
                            break

                    # Use a short persistent timer to avoid eager auto-clear flicker.
                    # Configurable via COLLISION_AVOIDANCE['flag_clear_delay_s'] (seconds).
                    try:
                        delay = float(COLLISION_AVOIDANCE.get('flag_clear_delay_s', 1.0))
                    except Exception:
                        delay = 1.0

                    # initialize per-agent safe timer if missing
                    try:
                        if not hasattr(self, '_flagged_safe_timer'):
                            self._flagged_safe_timer = np.zeros(self.n, dtype=float)
                    except Exception:
                        self._flagged_safe_timer = np.zeros(n, dtype=float)

                    if _safe:
                        # accumulate safe time; only clear the UI flag once the
                        # safe interval has been exceeded to prevent brief UI flicker
                        self._flagged_safe_timer[i] = min(delay, self._flagged_safe_timer[i] + dt)
                        if self._flagged_safe_timer[i] >= delay:
                            try:
                                self.flagged_give_way[i] = False
                                print(f"[COLREGS] Auto-clear give_way for vessel {i}: no CPA threat for {delay:.1f}s; cleared.")
                            except Exception:
                                pass
                    else:
                        # reset timer when still at risk
                        try:
                            self._flagged_safe_timer[i] = 0.0
                        except Exception:
                            pass
                    
                # debug: report when flagged_give_way is still set and why
                try:
                    if getattr(self, 'flagged_give_way', None) is not None and self.flagged_give_way[i]:
                        # compute nearest contact metrics for debugging
                        nearest_j = -1
                        nearest_d = float('inf')
                        nearest_dCPA = float('nan')
                        nearest_rr = float('nan')
                        try:
                            for j in range(self.n):
                                if i == j:
                                    continue
                                delta = positions[:, j] - pd
                                dist = np.linalg.norm(delta)
                                if dist < nearest_d:
                                    nearest_d = dist
                                    nearest_j = j
                                    # world-frame velocity of other
                                    c_j, s_j = np.cos(psi[j]), np.sin(psi[j])
                                    vel_j_w = np.array([ nu[0, j]*c_j - nu[1, j]*s_j,
                                                          nu[0, j]*s_j + nu[1, j]*c_j ])
                                    rel_w = vel_j_w - vel_i
                                    t_cpa = -np.dot(delta, rel_w) / (np.dot(rel_w, rel_w) + 1e-9)
                                    t_cpa = np.clip(t_cpa, 0.0, T_CPA_MAX)
                                    d_cpa = np.linalg.norm(delta + rel_w * t_cpa)
                                    nearest_dCPA = d_cpa
                                    nearest_rr = np.dot(delta, rel_w) / (dist + 1e-9)
                        except Exception:
                            pass
                        print(f"[COLREGS] flagged_give_way persists agent={i} post_avoid={self.post_avoid_timer[i]:.1f}s linger={self.crossing_linger_timer[i]:.1f}s lock={self.crossing_lock[i]} nearest_j={nearest_j} d={nearest_d:.1f} dCPA={nearest_dCPA:5.1f} rr={nearest_rr: .3f}")
                except Exception:
                    pass
                    
            except Exception:
                pass
            # If we're in a post-lock linger period, allow early release when
            # all contacts are opening or safely distant. Otherwise preserve
            # the give-way role and stored avoidance heading/speed.
            if self.crossing_linger_timer[i] > 0.0:
                try:
                    from emergent.ship_abm.simulation_core import log as sim_log
                except Exception:
                    sim_log = None

                still_needs_linger = False
                # examine other vessels to see if any still present a danger
                for j in range(n):
                    if i == j:
                        continue
                    delta = positions[:, j] - pd
                    dist = np.linalg.norm(delta)
                    # world-frame velocity of other ship j
                    c_j, s_j = np.cos(psi[j]), np.sin(psi[j])
                    vel_j_w = np.array([ nu[0, j]*c_j - nu[1, j]*s_j,
                                          nu[0, j]*s_j + nu[1, j]*c_j ])
                    rel_w = vel_j_w - vel_i
                    # predicted time to CPA (clamped)
                    t_cpa = -np.dot(delta, rel_w) / (np.dot(rel_w, rel_w) + 1e-9)
                    t_cpa = np.clip(t_cpa, 0.0, T_CPA_MAX)
                    d_cpa = np.linalg.norm(delta + rel_w * t_cpa)
                    # world-frame range-rate (+ = opening)
                    range_rate = np.dot(delta, rel_w) / (dist + 1e-9)
                    # if any contact is still closing and dCPA inside the safety
                    # bound, we should remain in give-way for the linger period
                    if (d_cpa < CPA_SAFE) and (range_rate < 0.0):
                        still_needs_linger = True
                        break

                if not still_needs_linger:
                    # nothing important nearby — clear linger early so normal
                    # COLREGS logic can resume and the agent may manoeuvre.
                    self.crossing_linger_timer[i] = 0.0
                    if sim_log is not None and getattr(self, 'verbose', False):
                        try:
                            sim_log.debug('[COLREGS] linger_early_release agent=%s timer_cleared', i)
                        except Exception:
                            pass
                else:
                    # preserve give-way role and stored avoidance commands
                    head[i] = self.crossing_heading[i]
                    speed_des[i] = self.crossing_speed[i]
                    role[i] = 'give_way'
                    # keep acting as give-way until the next tick reevaluates
                    if sim_log is not None and getattr(self, 'verbose', False):
                        try:
                            sim_log.debug('[COLREGS] linger_active agent=%s d_cpa=%s t_cpa=%s', i, d_cpa, t_cpa)
                        except Exception:
                            pass
                    continue
            # If we already have an active crossing lock, pre-populate the
            # outputs so the rest of the simulation immediately treats the
            # agent as 'give_way' while the dedicated maintenance block
            # below still runs and may clear the lock if conditions allow.
            if self.crossing_lock[i] >= 0:
                try:
                    head[i] = self.crossing_heading[i]
                    speed_des[i] = self.crossing_speed[i]
                    role[i] = 'give_way'
                except Exception:
                    # defensive fallback: leave defaults
                    pass
           
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
                try:
                    from emergent.ship_abm.simulation_core import log as sim_log
                    if getattr(self, 'verbose', False):
                        sim_log.debug("[%s] dist=%6.0f dCPA=%6.0f rr=% .2f Δψ=%4.1f danger=%s",
                                      i, dist, d_cpa, range_rate, np.degrees(hdg_dev), still_danger)
                except Exception:
                    if getattr(self, 'verbose', False):
                        print(f"[{i}] dist={dist:6.0f} dCPA={d_cpa:6.0f} rr={range_rate: .2f} "
                              f"Δψ={np.degrees(hdg_dev):4.1f} danger={still_danger}")
    
                if still_danger:
                    hd = self.crossing_heading[i]
                    speed_des[i] = self.crossing_speed[i]
                    rl = 'give_way'
                    role[i] = rl
                    head[i] = hd
                    # instrumentation: report that we remain locked on a contact
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        msg = f'[COLREGS] lock_active agent={i} tgt={tgt} dist={dist:.1f} dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} Δψ={np.degrees(hdg_dev):4.1f} post_avoid={self.post_avoid_timer[i]:.1f} linger={self.crossing_linger_timer[i]:.1f}'
                        if getattr(self, 'verbose', False):
                            try:
                                sim_log.info(msg)
                            except Exception:
                                print(msg)
                        else:
                            # also print to stdout for quick visibility in logs
                            print(msg)
                    except Exception:
                        print(f"[COLREGS] lock_active agent={i} tgt={tgt} dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} Δψ={np.degrees(hdg_dev):4.1f}")
                    continue
                else:
                    # lock cleared: set linger/post-avoid timers to avoid immediate flip-flop.
                    # We *do not* immediately zero crossing_heading/speed here — keep them
                    # available while crossing_linger_timer elapses so the give-way state
                    # remains consistent for the configured duration.
                    # clear the lock and record it in logs for post-mortem
                    prev_tgt = tgt
                    self.crossing_lock[i] = -1
                    self.lock_init_psi[i] = np.nan
                    # Set post-avoid timer. Shorten it when relative closing
                    # speeds are low so agents don't remain forced in a
                    # post-avoid state longer than required. This reduces the
                    # time window where we suppress normal navigation when
                    # contacts are barely moving relative to each other.
                    base_post = COLLISION_AVOIDANCE["post_avoid_time"]
                    try:
                        rel_speed = np.linalg.norm(rel_w)
                    except Exception:
                        rel_speed = 0.0
                    # Assumption: for relative speeds >= 2.0 m/s use full timer;
                    # for very small rel speeds scale down to 20% of base.
                    # (These numbers are conservative defaults; adjust in
                    # config if desired.)
                    scale = np.clip(rel_speed / 2.0, 0.2, 1.0)
                    self.post_avoid_timer[i] = base_post * scale
                    # small debug print for post-mortem when verbose
                    try:
                        print(f"[COLREGS] lock_cleared agent={i} tgt={tgt} rel_speed={rel_speed:.2f} post_avoid_set={self.post_avoid_timer[i]:.2f}s (scale={scale:.2f})")
                    except Exception:
                        pass
                    # set the linger timer (seconds)
                    self.crossing_linger_timer[i] = self.crossing_linger_default

                    # debug log when lock clears for post-mortem
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        msg = (f'[COLREGS] lock_cleared agent={i} tgt={tgt} dist={dist:.1f} '
                               f'dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} '
                               f'Δψ={np.degrees(hdg_dev):4.1f} set_linger={self.crossing_linger_timer[i]:4.1f}s '
                               f'post_avoid={self.post_avoid_timer[i]:4.1f}s flush_integrator=True')
                        try:
                            sim_log.info(msg)
                        except Exception:
                            print(msg)
                    except Exception:
                        # fallback minimal print
                        try:
                            print(f"[COLREGS] lock_cleared agent={i} tgt={tgt}")
                        except Exception:
                            pass
                    # keep flagged_give_way True so the UI can persistently show the
                    # vessel that recently gave way until operator acknowledgement
                    try:
                        self.flagged_give_way[i] = True
                    except Exception:
                        pass
                    # also print a dedicated lock_cleared line for easier parsing
                    try:
                        print(f"[COLREGS] lock_cleared agent={i} prev_tgt={prev_tgt} set_linger={self.crossing_linger_timer[i]:.1f}s post_avoid={self.post_avoid_timer[i]:.1f}s")
                    except Exception:
                        try:
                            print(f"[COLREGS] lock_cleared agent={i} prev_tgt={prev_tgt}")
                        except Exception:
                            pass

                    # flush controller memory now to avoid integrator-driven late hard-over
                    try:
                        # mark transient per-agent flush indicator for instrumentation
                        try:
                            self._last_integrator_flushed[int(i)] = True
                        except Exception:
                            try:
                                # fallback to scalar attr (older runs)
                                self._last_integrator_flushed = True
                                self._last_integrator_flushed_idx = int(i)
                            except Exception:
                                pass
                        self.integral[i]        = 0.0
                    except Exception:
                        pass
                    try:
                        self.prev_error[i]      = 0.0
                    except Exception:
                        pass
                    try:
                        self.prev_rudder[i]     = 0.0
                    except Exception:
                        pass
                    try:
                        self.smoothed_rudder[i] = 0.0
                    except Exception:
                        pass

                    # preserve give-way role until linger expires (handled at top of loop)
                    role[i] = 'give_way'
                    head[i] = self.crossing_heading[i]
                    speed_des[i] = self.crossing_speed[i]
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

                # Diagnostic: print per-contact metrics when verbose to help
                # determine why give-way rules do/don't trigger.
                try:
                    if getattr(self, 'verbose', False) or PID_DEBUG:
                        print(f"[COLREGS-DBG] agent={i} contact={j} dist={dist:.1f} d_brk={d_brk:.1f} bearing_deg={np.degrees(bearing):.1f} t_cpa={t_cpa:.1f} d_cpa={d_cpa:.1f}")
                except Exception:
                    pass

                # Rule 14: head-on (mutual give-way)
                TURN_14 = np.radians(COLLISION_AVOIDANCE['headon_turn_deg'])  # 30 deg

                if abs(bearing) < np.radians(10):
                    if dist < d_brk:
                        rl = 'give_way'
                        # choose the nearest equivalent to avoid ±180° flips
                        hd_tmp = (psi_i - TURN_14) % (2*np.pi)
                        hd = _nearest_heading(hd_tmp, psi_i)
                        speed_des[i] = -self.max_reverse_speed[i]
                    else:
                        rl = 'give_way'
                        hd_tmp = (psi_i - TURN_14) % (2*np.pi)
                        hd = _nearest_heading(hd_tmp, psi_i)
                        speed_des[i] = 5.0
                    self.crossing_lock[i]    = j
                    self.lock_init_psi[i]    = psi_i     # cache heading at lock-on
                    # store nearest-equivalent heading relative to current psi
                    try:
                        self.crossing_heading[i] = _nearest_heading(hd, psi_i)
                    except Exception:
                        self.crossing_heading[i] = hd
                    self.crossing_speed[i]   = speed_des[i]
                    # mark the vessel as having given way (UI-visible)
                    try:
                        self.flagged_give_way[i] = True
                    except Exception:
                        pass
                    # ensure we always print a lock-set line for forensic logs
                    try:
                        print(f"[COLREGS] lock_set agent={i} tgt={j} dist={dist:.1f} dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} psi={np.degrees(psi_i):.1f}")
                    except Exception:
                        try:
                            print(f"[COLREGS] lock_set agent={i} tgt={j}")
                        except Exception:
                            pass
                    give_way_found = True
                    break

                # Rule 15: crossing from starboard (other vessel on your right)
                TURN_15 = np.radians(COLLISION_AVOIDANCE['cross_turn_deg'])  # 20 deg

                if 0 < bearing < np.pi/2:
                    if dist < d_brk:
                        rl = 'give_way'
                        hd_tmp = (psi_i - TURN_15) % (2*np.pi)
                        hd = _nearest_heading(hd_tmp, psi_i)
                        speed_des[i] = -self.max_reverse_speed[i]
                        self.crossing_lock[i] = j
                        try:
                            self.crossing_heading[i] = _nearest_heading(hd, psi_i)
                        except Exception:
                            self.crossing_heading[i] = hd
                        self.crossing_speed[i] = speed_des[i]
                        try:
                            self.flagged_give_way[i] = True
                        except Exception:
                            pass
                            try:
                                print(f"[COLREGS] lock_set agent={i} tgt={j} dist={dist:.1f} dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} psi={np.degrees(psi_i):.1f}")
                            except Exception:
                                try:
                                    print(f"[COLREGS] lock_set agent={i} tgt={j}")
                                except Exception:
                                    pass
                        give_way_found = True
                        break
                    else:
                        rl = 'give_way'
                        hd_tmp = (psi_i - TURN_15) % (2*np.pi)
                        hd = _nearest_heading(hd_tmp, psi_i)
                        sf = max(0.3, min(1.0, dist/safe_dist))
                        speed_des[i] = self.desired_speed[i] * sf
                        self.crossing_lock[i] = j
                        try:
                            self.crossing_heading[i] = _nearest_heading(hd, psi_i)
                        except Exception:
                            self.crossing_heading[i] = hd
                        self.crossing_speed[i] = speed_des[i]                        
                        try:
                            self.flagged_give_way[i] = True
                        except Exception:
                            pass
                        give_way_found = True
                        break

                # Rule 13: overtaking (other vessel approaching from astern)
                if abs(bearing) > 3*np.pi/4:   # bearing within ±45° of dead-astern

                    rl = 'give_way'
                    # now turn starboard by 10°
                    hd_tmp = (psi_i - np.pi/18) % (2*np.pi)
                    hd = _nearest_heading(hd_tmp, psi_i)
                    speed_des[i] = min(self.desired_speed[i] * 1.2, self.max_speed[i])
                    self.crossing_lock[i] = j
                    try:
                        self.crossing_heading[i] = _nearest_heading(hd, psi_i)
                    except Exception:
                        self.crossing_heading[i] = hd
                    self.crossing_speed[i] = speed_des[i]                    
                    try:
                        self.flagged_give_way[i] = True
                    except Exception:
                        pass
                    try:
                        print(f"[COLREGS] lock_set agent={i} tgt={j} dist={dist:.1f} dCPA={d_cpa:5.1f} tCPA={t_cpa:5.1f} rr={range_rate: .3f} psi={np.degrees(psi_i):.1f}")
                    except Exception:
                        try:
                            print(f"[COLREGS] lock_set agent={i} tgt={j}")
                        except Exception:
                            pass
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
        # clear transient per-agent integrator-flush indicators so they flag only for one tick
        try:
            if hasattr(self, '_last_integrator_flushed'):
                try:
                    # vectorized clear (reset all to False)
                    self._last_integrator_flushed[:] = False
                except Exception:
                    # fallback: scalar clear
                    self._last_integrator_flushed = False
                    self._last_integrator_flushed_idx = -1
        except Exception:
            pass
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
        # Rudder torque (legacy call passed state) — pass forward speed u explicitly
        rudder_tau = self.compute_rudder_torque(rudder, None, u)
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
        # Ensure inputs are 1D arrays of length n
        # determine n from state shape
        if hasattr(state, 'ndim') and state.ndim > 1:
            n = state.shape[1]
        else:
            n = 1

        def _ensure_arr(x, shape_len=n):
            a = np.atleast_1d(x)
            if a.size == shape_len:
                return a
            if a.size == 1:
                return np.full(shape_len, float(a.flat[0]))
            # try transpose if shape mismatched (e.g., (1,n) vs (n,))
            if a.shape[0] == 1 and a.size == shape_len:
                return a.reshape(shape_len)
            return np.reshape(a, (shape_len,)) if a.size == shape_len else np.resize(a, shape_len)

        prop_thrust = _ensure_arr(prop_thrust, n)
        drag_force = _ensure_arr(drag_force, n)
        rudder_angle = _ensure_arr(rudder_angle, n)

        # DEBUG: print shapes of the inputs to catch accidental transposes
        try:
            verbose_print = getattr(self, 'verbose', False)
            if verbose_print:
                from emergent.ship_abm.simulation_core import log as sim_log
                sim_log.debug(f"[DYN-DBG-IN] n={n} prop_thrust.shape={getattr(prop_thrust,'shape',None)} drag.shape={getattr(drag_force,'shape',None)} rud.shape={getattr(rudder_angle,'shape',None)} wind_force.shape={getattr(wind_force,'shape',None)} current_force.shape={getattr(current_force,'shape',None)} state.shape={getattr(state,'shape',None)}")
        except Exception:
            pass

        # wind_force and current_force expected shape (2, n)
        wind_force = np.atleast_2d(wind_force)
        current_force = np.atleast_2d(current_force)
        if wind_force.shape[1] != n and wind_force.shape[0] == n:
            # maybe transposed -> transpose
            wind_force = wind_force.T
        if current_force.shape[1] != n and current_force.shape[0] == n:
            current_force = current_force.T
        # if still wrong, broadcast columns
        if wind_force.shape[1] == 1 and n > 1:
            wind_force = np.tile(wind_force, (1, n))
        if current_force.shape[1] == 1 and n > 1:
            current_force = np.tile(current_force, (1, n))

        # Surge and sway forces
        X = prop_thrust + drag_force + wind_force[0] + current_force[0]
        Y =                 wind_force[1] + current_force[1]
        # Moments from rudder
        K = np.full(n, self.Kdelta) * rudder_angle
        N = np.full(n, self.Ndelta) * rudder_angle
        # Damping (linear + quadratic) applied only to u, v, r
        u, v, p, r = state
        state_uvr = state[[0,1,3], :].copy()  # u, v, r only
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
        # DEBUG: ensure shapes match before stacking
        a0 = X - lin[0] - quad[0]
        a1 = Y - lin[1] - quad[1]
        a2 = K
        a3 = N
        try:
            verbose_print = getattr(self, 'verbose', False)
            if verbose_print:
                from emergent.ship_abm.simulation_core import log as sim_log
                sim_log.debug(f"[DYN-DBG] shapes: a0={getattr(a0,'shape',None)} a1={getattr(a1,'shape',None)} a2={getattr(a2,'shape',None)} a3={getattr(a3,'shape',None)}")
        except Exception:
            pass
        Tau = np.vstack([
            a0,
            a1,
            a2,
            a3
        ])  # shape (4,n)
        # Accelerations
        acc = self.H_inv @ Tau  # (4,n)
        u_dot, v_dot, p_dot, r_dot = acc
        return u_dot, v_dot, p_dot, r_dot