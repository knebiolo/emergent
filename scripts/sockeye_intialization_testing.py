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

#%% Import dpeendencies
# import dependencies
from emergent import sockeye
import os

# identify directories
model_dir = r"J:\2819\005\Calcs\ABM\Output"
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'
model_name = 'test_4'

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 'EPSG:32604'

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
bbox = (550328.25,550510.05,6641424.76,6641609.31)

# how many agents in the simulation?
n = 15

# how many timesteps in the model?
ts = 9000

# what is the delta t
dt = 1.

# what is the water temp?
water_temp = 20.

#%% create a simulation object 
sim = sockeye.simulation(model_dir,model_name,crs)

#%% Read environmental data into model
# read HECRAS model and create environment rasters
#sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model))

# or import from directory
sim.enviro_import(os.path.join(model_dir,'vel_x.tif'),'velocity x')
sim.enviro_import(os.path.join(model_dir,'vel_y.tif'),'velocity y')
sim.enviro_import(os.path.join(model_dir,'depth.tif'),'depth')
sim.enviro_import(os.path.join(model_dir,'wsel.tif'),'wsel')
sim.enviro_import(os.path.join(model_dir,'elev.tif'),'elevation')
sim.vel_surf()

#%% Create an array of agents
fishes = sim.create_agents(n, model_dir, bbox, water_temp) 

#%% Run the model
sim.run(model_name, fishes, ts, dt)

 