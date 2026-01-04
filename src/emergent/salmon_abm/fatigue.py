"""Fatigue and metabolic helper functions extracted from sockeye.py.

Performance-critical functions optimized with direct indexing and Numba JIT.
"""
import numpy as np

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


def _as_1d(a, n: int | None = None) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        if n is None:
            return arr.reshape((1,))
        return np.full((n,), arr.item(), dtype=float)
    return arr.reshape((-1,))


def _as_1d_mask(mask, n: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 0:
        return np.full((n,), bool(m), dtype=bool)
    m = m.reshape((-1,))
    if m.size != n:
        raise ValueError(f"mask size {m.size} does not match n={n}")
    return m


class fatigue():
    def __init__(self, t, dt, simulation_object):
        self.t = t
        self.dt = dt
        self.simulation = simulation_object

    def swim_speeds(self):
        water_velocities = np.column_stack((self.simulation.x_vel, self.simulation.y_vel))
        fish_velocities = np.column_stack((self.simulation.sog * np.cos(self.simulation.heading),
                                           self.simulation.sog * np.sin(self.simulation.heading)))

        swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
        self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]
        self.simulation.swim_speeds[:, -1] = np.linalg.norm(fish_velocities, axis=-1)
        return swim_speeds

    def bl_s(self, swim_speeds):
        bl_s = swim_speeds / (self.simulation.length / 1000.)
        return bl_s

    def bout_distance(self):
        dist_travelled = np.hypot(self.simulation.prev_X - self.simulation.X, self.simulation.prev_Y - self.simulation.Y)
        dist_travelled = np.asarray(dist_travelled)
        self.simulation.dist_per_bout += dist_travelled if dist_travelled.ndim == 1 else dist_travelled.reshape((-1,))
        self.simulation.bout_dur += self.dt

    def time_to_fatigue(self, swim_speeds, mask_dict, method='CastroSantos'):
        ttf = np.full_like(swim_speeds, np.nan)
        if method == 'CastroSantos':
            a_p = self.simulation.a_p
            b_p = self.simulation.b_p
            a_s = self.simulation.a_s
            b_s = self.simulation.b_s

            # Optimized: direct indexing instead of chained np.where
            # Must index a_p/b_p/a_s/b_s by mask as well since they are per-agent arrays
            mask_prolonged = mask_dict['prolonged']
            mask_sprint = mask_dict['sprint']
            ttf[mask_prolonged] = np.exp(a_p[mask_prolonged] + swim_speeds[mask_prolonged] * b_p[mask_prolonged])
            ttf[mask_sprint] = np.exp(a_s[mask_sprint] + swim_speeds[mask_sprint] * b_s[mask_sprint])
            return ttf
        elif method == 'Katapodis_Gervais':
            genus = 'Oncorhyncus'
            regression_params = {'Oncorhyncus': {'K': 3.5825, 'b': -0.2621}}
            if genus in regression_params:
                k = 6.3234
                b = regression_params[genus]['b']
            else:
                raise ValueError('Species not found')
            ttf = np.zeros(self.simulation.num_agents)
            ttf[~mask_dict['sustained']] = (swim_speeds[~mask_dict['sustained']] / k) ** (1 / b)
            return ttf
        else:
            raise ValueError('method not recognized')

    def set_swim_mode(self, mask_dict):
        mask_prolonged = mask_dict['prolonged']
        mask_sprint = mask_dict['sprint']
        
        # Optimized: direct indexing instead of chained np.where
        # Default to sustained (1)
        self.simulation.swim_mode[:] = 1
        self.simulation.swim_mode[mask_prolonged] = 2
        self.simulation.swim_mode[mask_sprint] = 3

    def recovery(self):
        # Some environments provide a `simulation.recovery()` helper that
        # returns recovery percentage (0-100). Use it when present; otherwise
        # fall back to a conservative linear recovery model so unit tests and
        # headless runs remain deterministic and do not raise AttributeError.
        if hasattr(self.simulation, 'recovery') and callable(getattr(self.simulation, 'recovery')):
            rec0 = self.simulation.recovery(self.simulation.recover_stopwatch) / 100.0
            rec1 = self.simulation.recovery(self.simulation.recover_stopwatch + self.dt) / 100.0
        else:
            # fallback: linear recovery percent scaled to seconds (0.1% per second)
            rec0 = np.clip(self.simulation.recover_stopwatch * 0.1, 0.0, 100.0) / 100.0
            rec1 = np.clip((self.simulation.recover_stopwatch + self.dt) * 0.1, 0.0, 100.0) / 100.0
        # clamp extremes
        rec0 = np.clip(np.asarray(rec0, dtype=float), 0.0, 1.0)
        rec1 = np.clip(np.asarray(rec1, dtype=float), 0.0, 1.0)
        per_rec = rec1 - rec0
        mask_station_holding = self.simulation.swim_behav == 3
        self.simulation.bout_dur[mask_station_holding] = 0.0
        self.simulation.dist_per_bout[mask_station_holding] = 0.0
        self.simulation.battery[mask_station_holding] += per_rec[mask_station_holding]
        self.simulation.recover_stopwatch[mask_station_holding] += self.dt
        return per_rec

    def calc_battery(self, per_rec, ttf, mask_dict):
        n = int(self.simulation.num_agents)
        battery = _as_1d(self.simulation.battery, n=n)
        per_rec = _as_1d(per_rec, n=n)
        ttf = _as_1d(ttf, n=n)

        mask_sustained = _as_1d_mask(mask_dict['sustained'], n=n)
        battery[mask_sustained] += per_rec[mask_sustained]

        mask_non_sustained = ~mask_sustained
        ttf0 = ttf[mask_non_sustained] * battery[mask_non_sustained]
        ttf1 = ttf0 - self.dt  # FIX: Use actual timestep, not 0.001 (0.1%/sec was too lenient)

        ratio = np.divide(
            ttf1,
            ttf0,
            out=np.zeros_like(ttf1, dtype=float),
            where=np.isfinite(ttf1) & np.isfinite(ttf0) & (ttf0 != 0),
        )
        ratio = np.clip(ratio, 0.0, 1.0)
        battery[mask_non_sustained] *= ratio

        self.simulation.battery = np.clip(battery, 0.0, 1.0)

    def set_swim_behavior(self, battery_state_dict):
        mask_low_battery = battery_state_dict['low']
        mask_mid_battery = battery_state_dict['mid']
        mask_high_battery = battery_state_dict['high']

        # Optimized: direct indexing instead of chained np.where
        self.simulation.swim_behav[mask_high_battery] = 1
        self.simulation.swim_behav[mask_mid_battery] = 2
        self.simulation.swim_behav[mask_low_battery] = 3

    def set_ideal_sog(self, mask_dict, battery_state_dict):
        mask_low_battery = battery_state_dict['low']
        mask_mid_battery = battery_state_dict['mid']
        mask_high_battery = battery_state_dict['high']

        # high battery: school_sog when full battery, otherwise scaled opt_sog
        self.simulation.ideal_sog[mask_high_battery] = np.where(
            self.simulation.battery[mask_high_battery] == 1.0,
            self.simulation.school_sog[mask_high_battery],
            np.round((self.simulation.opt_sog[mask_high_battery] * self.simulation.battery[mask_high_battery]) / 2, 2),
        )

        # set other bands
        self.simulation.ideal_sog[mask_low_battery] = 0.0
        self.simulation.ideal_sog[mask_mid_battery] = 0.1

    def ready_to_move(self):
        mask_ready_to_move = self.simulation.battery >= 0.85
        self.simulation.recover_stopwatch[mask_ready_to_move] = 0.0
        self.simulation.swim_behav[mask_ready_to_move] = 1
        self.simulation.swim_mode[mask_ready_to_move] = 1

    def PID_checks(self):
        if getattr(self.simulation, 'pid_tuning', False):
            # keep lightweight debug behavior similar to original
            pass

    def assess_fatigue(self):
        swim_speeds = self.swim_speeds()
        bl_s = self.bl_s(swim_speeds)

        mask_dict = {
            'prolonged': (self.simulation.max_s_U < bl_s) & (bl_s <= self.simulation.max_p_U),
            'sprint': bl_s > self.simulation.max_p_U,
            'sustained': bl_s <= self.simulation.max_s_U,
        }

        # DIAGNOSTIC: Print swim mode distribution at t=0
        if self.t == 0:
            n_sustained = np.sum(mask_dict['sustained'])
            n_prolonged = np.sum(mask_dict['prolonged'])
            n_sprint = np.sum(mask_dict['sprint'])
            
            # Get fish velocities and water velocities
            water_vel = np.column_stack((self.simulation.x_vel, self.simulation.y_vel))
            fish_vel = np.column_stack((self.simulation.sog * np.cos(self.simulation.heading),
                                       self.simulation.sog * np.sin(self.simulation.heading)))
            water_speed = np.linalg.norm(water_vel, axis=-1)
            fish_speed = np.linalg.norm(fish_vel, axis=-1)
            
            print(f"\nFATIGUE DEBUG t={self.t:.1f}s:")
            print(f"  Fish SOG (m/s): min={self.simulation.sog.min():.3f}, mean={self.simulation.sog.mean():.3f}, max={self.simulation.sog.max():.3f}")
            print(f"  Fish SOG (BL/s): min={self.simulation.sog.min()/(self.simulation.length[0]/1000):.3f}, mean={self.simulation.sog.mean()/(self.simulation.length.mean()/1000):.3f}")
            print(f"  Water speed (m/s): min={water_speed.min():.3f}, mean={water_speed.mean():.3f}, max={water_speed.max():.3f}")
            print(f"  Fish velocity magnitude (m/s): min={fish_speed.min():.3f}, mean={fish_speed.mean():.3f}, max={fish_speed.max():.3f}")
            print(f"  Swim speeds vs water (m/s): min={swim_speeds.min():.3f}, mean={swim_speeds.mean():.3f}, max={swim_speeds.max():.3f}")
            print(f"  Swim speeds (BL/s): min={bl_s.min():.3f}, mean={bl_s.mean():.3f}, max={bl_s.max():.3f}")
            print(f"  max_s_U threshold: {self.simulation.max_s_U[0]:.3f} BL/s")
            print(f"  Swim modes: sustained={n_sustained}, prolonged={n_prolonged}, sprint={n_sprint}")
            print(f"  Battery before update: min={self.simulation.battery.min():.3f}, mean={self.simulation.battery.mean():.3f}")

        # record bout distance
        self.bout_distance()

        # assess time to fatigue
        ttf = self.time_to_fatigue(bl_s, mask_dict)

        # set swim mode
        self.set_swim_mode(mask_dict)

        # assess recovery
        per_rec = self.recovery()

        # update battery
        self.calc_battery(per_rec, ttf, mask_dict)
        
        # DIAGNOSTIC: Print battery after update
        if self.t == 0:
            print(f"  Battery after update: min={self.simulation.battery.min():.3f}, mean={self.simulation.battery.mean():.3f}")
            print(f"  Recovery amount: min={per_rec.min():.6f}, mean={per_rec.mean():.6f}, max={per_rec.max():.6f}")

        # battery masks
        battery_dict = dict()
        battery_dict['low'] = self.simulation.battery <= 0.1
        battery_dict['mid'] = (self.simulation.battery > 0.1) & (self.simulation.battery <= 0.3)
        battery_dict['high'] = self.simulation.battery > 0.3

        # set swim behavior
        self.set_swim_behavior(battery_dict)

        # set ideal sog
        self.set_ideal_sog(mask_dict, battery_dict)

        # ready to move
        self.ready_to_move()

        # PID checks
        self.PID_checks()
