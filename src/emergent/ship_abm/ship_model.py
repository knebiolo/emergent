import numpy as np
from emergent.ship_abm.config import SHIP_PHYSICS, \
    CONTROLLER_GAINS, \
        ADVANCED_CONTROLLER, \
            COLLISION_AVOIDANCE, \
                PROPULSION

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
        self.max_rudder = np.radians(SHIP_PHYSICS["max_rudder"])
        self.max_rudder_rate = np.radians(SHIP_PHYSICS["max_rudder_rate"])
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
    
        # speed & propulsion settings
        init_sp = PROPULSION['initial_speed']
        self.desired_speed = np.full(n, PROPULSION['desired_speed'])
        self.cruise_speed  = self.desired_speed.copy()
        self.max_speed     = np.full(n, PROPULSION['max_speed'])
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

    # # === Updated PID Controller ===
    def pid_control(self, psi, hd, dt):
        """
        P-I-D with gain scheduling, anti-windup, dead-band, and rate-limit.
        """
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
        
        # convert bands to radians
        release_rad = np.radians(self.release_band_deg)
        trim_rad    = np.radians(self.trim_band_deg)
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

        safe_dist  = COLLISION_AVOIDANCE['safe_dist']                                        
        clear_dist = COLLISION_AVOIDANCE['clear_dist']
        unlock_ang = COLLISION_AVOIDANCE['unlock_ang'] 

        n         = self.n
        head      = np.zeros(n)
        # start from current cruise speeds, to be overridden per scenario
        speed_des = self.desired_speed.copy()
        role      = ['neutral'] * n

        for i in range(n):
            pd    = positions[:, i]
            psi_i = psi[i]
            hd    = psi_i           # default heading
            rl    = 'neutral'       # default role
            give_way_found = False  # flag if any give_way rule fires
            contact = False         # flag if any vessel is within safe_dist

            # examine every other vessel
            for j in range(n):
                if i == j:
                    continue
                delta = positions[:, j] - pd
                dist  = np.linalg.norm(delta)
                if dist > safe_dist:
                    continue
                contact = True

                # compute relative bearing
                bearing = ((np.arctan2(delta[1], delta[0])
                           - psi_i + np.pi) % (2*np.pi) - np.pi)

                # Rule 15: crossing from starboard
                if 0 < bearing < np.pi/2:
                    rl = 'give_way'
                    hd = (psi_i - 2*np.pi/3) % (2*np.pi)
                    sf = max(0.3, min(1.0, dist/safe_dist))
                    speed_des[i] = self.desired_speed[i] * sf
                    give_way_found = True
                    break

                # Rule 14: head-on
                if abs(bearing) < np.radians(10):
                    rl = 'give_way'
                    hd = (psi_i - np.pi/3) % (2*np.pi)
                    speed_des[i] = 5.0
                    give_way_found = True
                    break

                # Rule 13: overtaking
                if -np.pi/8 < bearing < np.pi/8:
                    rl = 'give_way'
                    hd = (psi_i + np.pi/18) % (2*np.pi)
                    speed_des[i] = min(self.desired_speed[i] * 1.2, self.max_speed[i])
                    give_way_found = True
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