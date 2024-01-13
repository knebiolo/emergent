# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
#%% Import emergent
# software directory
import sys
#sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")
sys.path.append(r"C:\Users\AYoder\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")


#%% Import dpeendencies
# import dependencies
from emergent import sockeye_SoA as sockeye
import os
# import ga optimization script

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'
model_name = 'soa_01'

# identify directories
#model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\simulations",model_name)
model_dir = os.path.join(r"C:\Users\AYoder\Desktop\simulations",model_name)
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"



#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550328.25,550410.05,6641500.76,6641550.31)                             # starting box way downstream
#bbox = (549800,550115,6641332,6641407)
bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # starting box right near the falls
pid_tuning_start = (549488.29, 6641611.84) # original
#pid_tuning_start = (549225.63, 6641906.52) # high flow speed
#pid_tuning_start = (550370.39, 6641528.46) # low flow speed

# how many agents in the simulation?
n = 1

# how many timesteps in the model?
ts = 1000

# what is the delta t
dt = 0.25

# what is the water temp?
water_temp = 20.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

#%% create a simulation object 
sim = sockeye.simulation(model_dir,
                         model_name,
                         crs,
                         basin,
                         water_temp,
                         bbox,
                         ts,
                         n,
                         use_gpu = False,
                         pid_tuning = True)

# magnitude, length of array
# minimize magnitude, maximize length

#%% Read environmental data into model
# read HECRAS model and create environment rasters
#sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model),1.0)

#%% Run the model
sim.run(model_name, n = ts, dt = dt)

#%% visualize a timestep
import matplotlib.pyplot as plt

# Example data - replace with your actual data
fish_positions_x = [sim.X]  # X-coordinates of fish positions
fish_positions_y = [sim.Y]  # Y-coordinates of fish positions
thrust_vectors_x = [sim.thrust[:,0] / 50.]  # X-components of thrust vectors
thrust_vectors_y = [sim.thrust[:,1] / 50.] # Y-components of thrust vectors
drag_vectors_x = [sim.drag[:,0] / 50.]  # X-components of drag vectors
drag_vectors_y = [sim.drag[:,1] / 50.]  # Y-components of drag vectors

# Create a plot
plt.figure(figsize=(10, 6))

# Plot thrust vectors
plt.quiver(fish_positions_x, fish_positions_y, thrust_vectors_x, thrust_vectors_y, color='r', scale=1, label='Thrust')

# Plot drag vectors
plt.quiver(fish_positions_x, fish_positions_y, drag_vectors_x, drag_vectors_y, color='b', scale=1, label='Drag')

# Add labels and legend
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Thrust and Drag Vectors of Fish')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

 