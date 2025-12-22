"""Fatigue and metabolic helper functions extracted from sockeye.py."""
import numpy as np


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
        dist_travelled = np.sqrt((self.simulation.prev_X - self.simulation.X)**2 + (self.simulation.prev_Y - self.simulation.Y)**2)
        if len(dist_travelled.shape) == 1:
            self.simulation.dist_per_bout += dist_travelled
        else:
            self.simulation.dist_per_bout += dist_travelled.flatten()
        self.simulation.bout_dur += self.dt

    def time_to_fatigue(self, swim_speeds, mask_dict, method='CastroSantos'):
        ttf = np.full_like(swim_speeds, np.nan)
        if method == 'CastroSantos':
            a_p = self.simulation.a_p
            b_p = self.simulation.b_p
            a_s = self.simulation.a_s
            b_s = self.simulation.b_s
            lengths = self.simulation.length

            ttf = np.where(mask_dict['prolonged'], np.exp(a_p + swim_speeds * b_p), ttf)
            ttf = np.where(mask_dict['sprint'], np.exp(a_s + swim_speeds * b_s), ttf)
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
        self.simulation.swim_mode = np.where(mask_prolonged, 2, self.simulation.swim_mode)
        self.simulation.swim_mode = np.where(mask_sprint, 3, self.simulation.swim_mode)
        self.simulation.swim_mode = np.where(~(mask_prolonged | mask_sprint), 1, self.simulation.swim_mode)

    def recovery(self):
        rec0 = self.simulation.recovery(self.simulation.recover_stopwatch) / 100.
        rec0[rec0 < 0.0] = 0.0
        rec1 = self.simulation.recovery(self.simulation.recover_stopwatch + self.dt) / 100.
        rec1[rec1 > 1.0] = 1.0
        rec1[rec1 < 0.] = 0.0
        per_rec = rec1 - rec0
        mask_station_holding = self.simulation.swim_behav == 3
        self.simulation.bout_dur[mask_station_holding] = 0.0
        self.simulation.dist_per_bout[mask_station_holding] = 0.0
        self.simulation.battery[mask_station_holding] += per_rec[mask_station_holding]
        self.simulation.recover_stopwatch[mask_station_holding] += self.dt
        return per_rec

    def calc_battery(self, per_rec, ttf, mask_dict):
        mask_sustained = mask_dict['sustained']
        if mask_sustained.ndim == 2:
            mask_sustained = mask_sustained.squeeze()
        if self.simulation.num_agents > 1:
            self.simulation.battery[mask_sustained] += per_rec[mask_sustained]
        else:
            self.simulation.battery[mask_sustained.flatten()] += per_rec[mask_sustained.flatten()]

        mask_non_sustained = ~mask_sustained
        if self.simulation.num_agents > 1:
            ttf0 = ttf[mask_non_sustained] * self.simulation.battery[mask_non_sustained]
        else:
            ttf0 = ttf[mask_non_sustained.flatten()] * self.simulation.battery[mask_non_sustained.flatten()]

        ttf1 = ttf0 - self.dt
        if self.simulation.num_agents > 1:
            self.simulation.battery[mask_non_sustained] *= np.nan_to_num(ttf1 / ttf0)
        else:
            self.simulation.battery[mask_non_sustained.flatten()] *= ttf1.flatten() / ttf0.flatten()

        self.simulation.battery = np.clip(self.simulation.battery, 0, 1)
