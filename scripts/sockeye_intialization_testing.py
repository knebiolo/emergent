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
sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")


#%% Import dpeendencies
# import dependencies
from emergent import sockeye
import os

# identify directories
model_dir = r"C:\Users\Isha Deo\Documents\Projects\Nuyakuk\simulations\test_ipd_21"
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p08.hdf'
model_name = 'test_ipd_21'

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 'EPSG:32604'

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
bbox = (550328.25,550510.05,6641500.76,6641600.31)                             # starting box way downstream
#bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # starting box right near the falls

# how many agents in the simulation?
n = 50

# how many timesteps in the model?
ts = 1000
# what is the delta t
dt = 1.

# what is the water temp?
water_temp = 20.

#%% create a simulation object 
sim = sockeye.simulation(model_dir,model_name,crs)

#%% Read environmental data into model
# read HECRAS model and create environment rasters
# sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model))

# or import from directory
sim.enviro_import(os.path.join(model_dir,'vel_x.tif'),'velocity x')
sim.enviro_import(os.path.join(model_dir,'vel_y.tif'),'velocity y')
sim.enviro_import(os.path.join(model_dir,'depth.tif'),'depth')
sim.enviro_import(os.path.join(model_dir,'wsel.tif'),'wsel')
sim.enviro_import(os.path.join(model_dir,'elev.tif'),'elevation')
sim.enviro_import(os.path.join(model_dir,'vel_dir.tif'),'velocity direction')
sim.enviro_import(os.path.join(model_dir,'vel_mag.tif'),'velocity magnitude')
# sim.vel_surf()

#%% Create an array of agents
fishes = sim.create_agents(n, model_dir, bbox, water_temp) 

#%% Run the model
sim.run(model_name, fishes, ts, dt)

 