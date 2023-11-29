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
from emergent import sockeye_multiprocessing as sockeye
import os
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'Q:\Internal_Data\Staff_Projects\KPN\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'


# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'
model_name = 'test_1'

# identify directories
model_dir = os.path.join(r"Q:\Internal_Data\Staff_Projects\KPN\emergent_testing",model_name)
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"



#%% Set model parameters
if __name__ == '__main__':
    # identify the coordinate reference system for the model
    crs = 'EPSG:32604'

    # create a starting box - aka where are all the fish starting from?
    # W,E,S,N
    #bbox = (550328.25,550510.05,6641500.76,6641600.31)                             # starting box way downstream
    #bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # kinda near the falls
    bbox = (549466.69,549520.48,6641583.35,6641625.48)                             # starting box right near the falls

    # how many agents in the simulation?
    n = 1000

    # how many timesteps in the model?
    ts = 2

    # what is the delta t
    dt = 1.

    # what is the water temp?
    water_temp = 20.

    # what is the basin that we are simulating passage in?
    basin = "Nushagak River"

    #%% create a simulation object

    sim = sockeye.simulation(model_dir,model_name,crs)
    print ('simulation object created')

    #%% Read environmental data into model
    # read HECRAS model and create environment rasters
    #sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model),1.0)
    #sim.vel_surf()

    # or import from directory
    sim.enviro_import(os.path.join(model_dir,'vel_x.tif'),'velocity x')
    sim.enviro_import(os.path.join(model_dir,'vel_y.tif'),'velocity y')
    sim.enviro_import(os.path.join(model_dir,'depth.tif'),'depth')
    sim.enviro_import(os.path.join(model_dir,'wsel.tif'),'wsel')
    sim.enviro_import(os.path.join(model_dir,'elev.tif'),'elevation')
    sim.enviro_import(os.path.join(model_dir,'vel_dir.tif'),'velocity direction')
    sim.enviro_import(os.path.join(model_dir,'vel_mag.tif'),'velocity magnitude')

    print ('agent environment imported')


    #%% Create an array of agents
    fishes = sim.create_agents(n, model_dir, bbox, crs, basin, water_temp)
    print ('fish agents created')

    #%% Run the model
    print ('running simulation')

    sim.run(model_name, fishes, ts, dt)

