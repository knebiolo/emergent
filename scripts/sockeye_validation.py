# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
#%% Import emergent
# software directory
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")


#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os


# identify input and output model names
model_name = 'val_06'

# identify directories
model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\abm_simulations\simulations",model_name)

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550328.25,550510.05,6641500.76,6641600.31)                             # starting box way downstream
bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # kinda near the falls
#bbox = (549466.69,549520.48,6641583.35,6641625.48)                             # starting box right near the falls

# how many agents in the simulation?
n = 100

# what is the delta t
dt = 0.2

# how many timesteps in the model?
hours = 0.5
ts = 3600. * hours / dt
#ts = 2000

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
                         None,
                         ts, 
                         n, 
                         use_gpu = False,
                         pid_tuning = False)

print ('simulation object created')

#%% Run the model
sim.run(model_name, ts, dt, video = True)

#%% Createa a movie file 
#sockeye.movie_maker(model_dir, model_name, crs, dt, sim.depth_rast_transform)
 