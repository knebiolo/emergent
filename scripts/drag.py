# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:16:26 2023

@author: Isha Deo

Script Intent: Calculate drag force on a sockeye salmon swimming upstream 
given the fish length, water & fish velocities, and water temperature.

The drag on a fish is calculated using the drag force equation given in Mirzaei 2017:
    
    drag = -0.5 * density * surface area * drag coefficient * |fish velocity - water velocity|^2 * (fish velocity / |fish velocity|)
    
The surface area of the fish is determined using an empirical relationship developed in Oshea 2006 for salmon

The drag coefficient is determined using measured values from Brett 1963 (adapted from Webb 1975), fitted to a log function


 
"""

# this script calculates drag force on a fish based on swimming speed

# import packages
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt

# define import values - note units!!
length = 20 #cm
water_vel = np.array([0.75,1]) #meters/sec
water_temp = 20 #deg C
species = 'sockeye'
fish_vel = np.array([2,1]) #meters/sec

# identify workspaces
inputWS = r"J:\2819\005\Calcs\ABM\Data\\"
outputWS = r"J:\2819\005\Calcs\ABM\Output\\"

# read databases for kinematic viscosity and density from Engineering Toolbox
kin_visc_df = pd.read_excel(inputWS + 'Engineering Toolbox Water Values.xlsx', sheet_name='kinematic viscosity', header=0, index_col=0)
density_df = pd.read_excel(inputWS + 'Engineering Toolbox Water Values.xlsx', sheet_name='density', header=0, index_col=0)

# determine kinematic viscosity based on water temperature
f_kinvisc = scipy.interpolate.interp1d(kin_visc_df.index, kin_visc_df['Kinematic viscosity (m2/s)'])
kin_visc = f_kinvisc(water_temp)

# determine density based on water temperature
f_density = scipy.interpolate.interp1d(density_df.index, density_df['Density (g/cm3)'])
density = f_density(water_temp)

# calculate Reynold's number
def calc_Reynolds(length, kin_visc, water_vel):
    length_m = length / 100
    return water_vel * length_m / kin_visc

reynolds = calc_Reynolds(length, kin_visc, np.linalg.norm(water_vel))

# calculate surface area of fish; assuming salmon for now
def calc_surface_area(length, species):
    if species == 'sockeye': # add other salmon species if we get there
        # uses length based method for salmon - better estimates with weight
        a = -0.143
        b = 1.881
        return 10 ** (a + b*math.log10(length))
    return math.nan

surface_area = calc_surface_area(length, species)

# set up drag coefficient vs Reynolds number dataframe
drag_coeffs_df = pd.DataFrame(data = {'Reynolds Number': [2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5],
                              'Drag Coefficient': [0.23,0.19,0.15,0.14,0.12,0.12,0.11,0.10]}).set_index('Reynolds Number')

# fit drag coefficient vs Reynolds number to function
drag_fig, drag_ax = plt.subplots()
drag_ax.scatter(x = drag_coeffs_df.index, y = drag_coeffs_df['Drag Coefficient'])

dragf_test_df = pd.DataFrame(data = {'Reynolds Number': np.arange(25000, 200000, 1000)})

def fit_dragcoeffs(reynolds, a, b):
    return np.log(reynolds)*a + b

dragf_popt, dragf_pcov = scipy.optimize.curve_fit(f = fit_dragcoeffs, xdata = drag_coeffs_df.index, ydata = drag_coeffs_df['Drag Coefficient'])
dragf_test_df['Drag Coefficient'] = dragf_test_df.apply(func = fit_dragcoeffs, args = tuple(dragf_popt))

drag_ax.plot(dragf_test_df['Reynolds Number'], dragf_test_df['Drag Coefficient'])

# determine drag coefficient for calculated Reynolds number
drag_coeff = fit_dragcoeffs(reynolds, dragf_popt[0], dragf_popt[1])
drag_ax.scatter(reynolds, drag_coeff)

# calculate drag!
drag = -0.5 * (density * 1000) * (surface_area / 100**2) * drag_coeff * (np.linalg.norm(fish_vel - water_vel)**2)*(fish_vel/np.linalg.norm(fish_vel))

