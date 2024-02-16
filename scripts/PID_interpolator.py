# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:19:13 2024

@author: Kevin.Nebiolo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

# Step 1: Load your data
# Replace 'your_data.csv' with the path to your CSV file
df = pd.read_csv(r'C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent\data\pid_optimize_Nushagak.csv')

# get data arrays
length = df.loc[:, 'fish_length'].values
velocity = df.loc[:, 'avg_water_velocity'].values
P = df.loc[:, 'p'].values
I = df.loc[:, 'i'].values
D = df.loc[:, 'd'].values

# Plane model function
def plane_model(coords, a, b, c):
    length, velocity = coords
    return a * length + b * velocity + c

# Curve fitting
P_params, _ = curve_fit(plane_model, (length, velocity), P)
a, b, c = P_params

# Generating a meshgrid for the plane
len_range = np.linspace(length.min(), length.max(), 100)
vel_range = np.linspace(velocity.min(), velocity.max(), 100)
len_mesh, vel_mesh = np.meshgrid(len_range, vel_range)
P_mesh = a * len_mesh + b * vel_mesh + c

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Original data points
ax.scatter(length, velocity, P, color='r', label='Original Data')

# Fitted plane
ax.plot_surface(len_mesh, vel_mesh, P_mesh, color='b', alpha=0.5, rstride=100, cstride=100, label='Fitted Plane')

# Labels and title
ax.set_xlabel('length')
ax.set_ylabel('velocity')
ax.set_zlabel('P')

plt.show()

