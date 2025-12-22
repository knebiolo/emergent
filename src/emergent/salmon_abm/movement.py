"""Movement helpers extracted from sockeye.py.

This module contains the `movement` class that operates on a `simulation` object.
The implementation is a near-direct extraction and imports light-weight helpers
from `emergent.salmon_abm.utils` so callers can migrate to the new module.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import distance_transform_edt

from emergent.salmon_abm.utils import geo_to_pixel, pixel_to_geo, standardize_shape, calculate_front_masks


class movement():
    def __init__(self, simulation_object):
        self.simulation = simulation_object

    def find_z(self):
        """
        Calculate the z-coordinate for an agent based on its depth and body depth.
        """
        self.simulation.z = np.where(
            self.simulation.depth < self.simulation.body_depth * 3 / 100.,
            self.simulation.depth + self.simulation.too_shallow,
            self.simulation.body_depth * 3 / 100.)

        # make sure
        self.simulation.z = np.where(self.simulation.z < 0, 0, self.simulation.z)

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
                fish_dir = np.arctan2(fish_y_vel, fish_x_vel)
                fish_mag = np.linalg.norm(np.stack((fish_x_vel, fish_y_vel)).T, axis=-1)

                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)),
                                           axis=-1)

        ideal_swim_speed = np.linalg.norm(fish_velocities - water_vel, axis=-1)

        swim_speed_cms = ideal_swim_speed * 100.

        # Data for interpolation
        length_dat = np.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
        speed_dat = np.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
        amp_dat = np.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
        wave_dat = np.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
        edge_dat = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])

        # Interpolation with extrapolation using UnivariateSpline
        A_spline = UnivariateSpline(length_dat, amp_dat, k=2, ext=0)
        V_spline = UnivariateSpline(speed_dat, wave_dat, k=1, ext=0)
        B_spline = UnivariateSpline(length_dat, edge_dat, k=1, ext=0)

        A = A_spline(length_cm)
        V = V_spline(swim_speed_cms)
        B = B_spline(length_cm)

        # Calculate thrust
        m = (np.pi * rho * B ** 2) / 4.
        W = (self.simulation.Hz * A * np.pi) / 1.414
        w = W * (1 - swim_speed_cms / V)

        # Thrust calculation
        thrust_erg_s = m * W * w * swim_speed_cms - (m * w ** 2 * swim_speed_cms) / (2. * np.cos(np.radians(theta)))
        thrust_Nm = thrust_erg_s / 10000000.
        thrust_N = thrust_Nm / (self.simulation.length / 1000.)

        # Convert thrust to vector
        thrust = np.where(mask, [thrust_N * np.cos(self.simulation.heading),
                                 thrust_N * np.sin(self.simulation.heading)], 0)

        self.simulation.thrust = thrust.T

    def frequency(self, mask, t, dt, fish_velocities=None):
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

        length_dat = np.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
        speed_dat = np.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
        amp_dat = np.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
        wave_dat = np.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
        edge_dat = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])

        A_spline = UnivariateSpline(length_dat, amp_dat, k=2, ext=0)
        V_spline = UnivariateSpline(speed_dat, wave_dat, k=1, ext=0)
        B_spline = UnivariateSpline(length_dat, edge_dat, k=1, ext=0)

        A = A_spline(lengths_cm)
        V = V_spline(swim_speeds_cms)
        B = B_spline(lengths_cm)

        if alternate:
            ideal_drag = self.ideal_drag_fun(fish_velocities=fish_velocities)
        else:
            ideal_drag = self.ideal_drag_fun()

        drags_erg_s = np.where(mask, np.linalg.norm(ideal_drag, axis=-1) * self.simulation.length / 1000 * 10000000, 0)

        min_Hz = np.interp(self.simulation.length, [450, 7.5], [690, 2.])

        Hz = np.where(self.simulation.swim_behav == 3, min_Hz,
                      np.sqrt(drags_erg_s * V ** 2 * np.cos(np.radians(theta)) /
                              (A ** 2 * B ** 2 * swim_speeds_cms * np.pi ** 3 * rho *
                               (swim_speeds_cms - V) *
                               (-0.062518880701972 * swim_speeds_cms -
                                0.125037761403944 * V * np.cos(np.radians(theta)) +
                                0.062518880701972 * V)
                               )
                              )
                      )
        Hz = np.where(self.simulation.is_stuck, 0, Hz)

        self.simulation.prev_Hz = self.simulation.Hz
        self.simulation.Hz = np.where(self.simulation.Hz > 20, 20, Hz)

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

    def wat_dens(self, temp):
        dens_temp = np.array([0.1, 1., 4., 10., 15., 20., 25., 30., 35., 40.,
                              45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
                              95., 100., 110., 120., 140., 160., 180., 200.,
                              220., 240., 260., 280., 300., 320., 340., 360.,
                              373.946])
        density = np.array([0.9998495, 0.9999017, 0.9999749, 0.9997, 0.9991026,
                            0.9982067, 0.997047, 0.9956488, 0.9940326, 0.9922152,
                            0.99021, 0.98804, 0.98569, 0.9832, 0.98055, 0.97776,
                            0.97484, 0.97179, 0.96861, 0.96531, 0.96189, 0.95835,
                            0.95095, 0.94311, 0.92613, 0.90745, 0.887, 0.86466,
                            0.84022, 0.81337, 0.78363, 0.75028, 0.71214, 0.66709,
                            0.61067, 0.52759, 0.322])
        f_density = np.interp(temp, dens_temp, density)
        return f_density

    def calc_Reynolds(self, visc, water_vel):
        length_m = self.simulation.length / 1000.
        reynolds_numbers = water_vel * length_m / visc
        return reynolds_numbers

    def calc_surface_area(self):
        a = -0.143
        b = 1.881
        surface_areas = 10 ** (a + b * np.log10(self.simulation.length))
        return surface_areas

    def drag_coeff(self, reynolds):
        reynolds_data = np.array([2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5])
        drag_data = np.array([0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
        drag_coefficients = np.interp(reynolds, reynolds_data, drag_data)
        return drag_coefficients

    def drag_fun(self, mask, t, dt, fish_velocities=None):
        tired_mask = np.where(self.simulation.swim_behav == 3, True, False)

        if fish_velocities is None:
            if t == 0:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
            else:
                fish_x_vel = (self.simulation.X - self.simulation.prev_X) / dt
                fish_y_vel = (self.simulation.Y - self.simulation.prev_Y) / dt
                fish_velocities = np.stack((fish_x_vel, fish_y_vel)).T

        water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)

        water_velocities = np.where(tired_mask[:, np.newaxis],
                                    water_velocities * 0.2,
                                    water_velocities * 1.)

        fish_speeds = np.linalg.norm(fish_velocities, axis=-1)
        fish_speeds[fish_speeds == 0.0] = 0.0001
        fish_velocities[fish_speeds == 0.0] = [0.0001, 0.0001]

        viscosity = self.kin_visc(self.simulation.water_temp)
        density = self.wat_dens(self.simulation.water_temp)

        length_m = self.simulation.length / 1000.
        reynolds_numbers = np.linalg.norm(water_velocities, axis=-1) * length_m / viscosity

        a = -0.143
        b = 1.881
        surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))

        drag_coeffs = self.drag_coeff(reynolds_numbers)

        relative_velocities = fish_velocities - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1) ** 2

        unit_relative_vector = np.nan_to_num(relative_velocities / np.linalg.norm(relative_velocities, axis=1)[:, np.newaxis])

        drags = np.where(mask[:, np.newaxis],
                         -0.5 * (density * 1000) * (surface_areas[:, np.newaxis] / 100 ** 2) \
                                       * drag_coeffs[:, np.newaxis] * relative_speeds_squared[:, np.newaxis] \
                                           * unit_relative_vector * self.simulation.wave_drag[:, np.newaxis], 0)

        max_drag_magnitude = 5.0
        drag_magnitudes = np.linalg.norm(drags, axis=1)

        excessive_drag_indices = np.where(np.logical_and(self.simulation.swim_behav == 3,
                                                         drag_magnitudes > max_drag_magnitude), True, False)
        drags[excessive_drag_indices] = (drags[excessive_drag_indices].T * (max_drag_magnitude / drag_magnitudes[excessive_drag_indices])).T

        self.simulation.drag = drags

    def ideal_drag_fun(self, fish_velocities=None):
        water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)

        if fish_velocities is None:
            fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                        self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)

        ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)

        refugia_mask = (self.simulation.swim_behav == 2) & (ideal_swim_speeds > self.simulation.max_s_U)
        holding_mask = (self.simulation.swim_behav == 3) & (ideal_swim_speeds > self.simulation.max_s_U)
        too_fast = refugia_mask + holding_mask

        fish_velocities = np.where(too_fast[:, np.newaxis],
                                   (self.simulation.max_s_U / ideal_swim_speeds[:, np.newaxis]) * fish_velocities,
                                   fish_velocities)

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
        unit_max_practical_sog = self.simulation.max_practical_sog / np.linalg.norm(self.simulation.max_practical_sog, axis=1)[:, np.newaxis]

        ideal_drags = -0.5 * (density * 1000) * (surface_areas[:, np.newaxis] / 100 ** 2) * drag_coeffs[:, np.newaxis] * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog * self.simulation.wave_drag[:, np.newaxis]

        return ideal_drags

    def swim(self, t, dt, pid_controller, mask):
        tired_mask = np.where(self.simulation.swim_behav == 3, True, False)

        if t == 0:
            fish_vel_0_x = np.where(mask, self.simulation.sog * np.cos(self.simulation.heading), 0)
            fish_vel_0_y = np.where(mask, self.simulation.sog * np.sin(self.simulation.heading), 0)
            fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
        else:
            fish_vel_0_x = (self.simulation.X - self.simulation.prev_X) / dt
            fish_vel_0_y = (self.simulation.Y - self.simulation.prev_Y) / dt
            fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T

        ideal_vel_x = np.where(mask, self.simulation.ideal_sog * np.cos(self.simulation.heading), 0)
        ideal_vel_y = np.where(mask, self.simulation.ideal_sog * np.sin(self.simulation.heading), 0)

        ideal_vel = np.stack((ideal_vel_x, ideal_vel_y)).T

        surge_ini = self.simulation.thrust + self.simulation.drag
        acc_ini = np.round(surge_ini / self.simulation.weight[:, np.newaxis], 2)

        fish_vel_1_ini = fish_vel_0 + acc_ini * dt

        error = np.where(mask[:, np.newaxis], np.round(ideal_vel - fish_vel_1_ini, 12), 0.)

        self.simulation.error = error
        self.simulation.dead = np.where(np.isnan(error[:, 0]), 1, self.simulation.dead)

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

        fish_vel_1 = np.where(~tired_mask[:, np.newaxis], fish_vel_0 + acc_ini * dt + pid_adjustment, fish_vel_0 + acc_ini * dt)

        fish_vel_1 = np.where(self.simulation.dead[:, np.newaxis] == 1, fish_vel_1 * 0, fish_vel_1)

        dxdy = np.where(mask[:, np.newaxis], fish_vel_1 * dt, np.zeros_like(fish_vel_1))
        return dxdy

    def jump(self, t, g, mask):
        self.simulation.time_of_jump = np.where(mask, t, self.simulation.time_of_jump)
        jump_angles = np.where(mask, np.random.choice([np.radians(45), np.radians(60)], size=self.simulation.ucrit.shape), 0)
        time_airborne = np.where(mask, (2 * self.simulation.ucrit * np.sin(jump_angles)) / g, 0)
        displacement = self.simulation.ucrit * time_airborne * np.cos(jump_angles)
        dx = displacement * np.cos(self.simulation.heading)
        dy = displacement * np.sin(self.simulation.heading)
        dxdy = np.stack((dx, dy)).T
        return dxdy
