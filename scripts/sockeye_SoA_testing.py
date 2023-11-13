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
from emergent import sockeye_SoA as sockeye
import os

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'
model_name = 'soa_01'

# identify directories
model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\simulations",model_name)
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"



#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550328.25,550510.05,6641500.76,6641600.31)                             # starting box way downstream
bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # starting box right near the falls

# how many agents in the simulation?
n = 1000

# how many timesteps in the model?
ts = 100

# what is the delta t
dt = 1.

# what is the water temp?
water_temp = 20.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

#%% create a simulation object 
sim = sockeye.simulation(model_dir,model_name,crs,basin,water_temp,bbox,ts,n,use_gpu = False)

#%% Read environmental data into model
# read HECRAS model and create environment rasters
#sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model),1.0)
#sim.vel_surf()

 

#%% Run the model
sim.run(model_name, n = ts, dt = dt)

 