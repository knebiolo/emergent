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
model_name = 'sense_15'

# identify directories
model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\abm_simulations\sensitivity",model_name)

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
start_polygon = os.path.join(model_dir,'river_left.shp')  

# how many agents in the simulation?
n = 10

# what is the delta t
dt = 3

# how many timesteps in the model?
hours = 18
ts = 3600. * hours / dt

# what is the water temp?
water_temp = 10.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

# identify background environmental files
env_files = {'wsel':'wsel_mask_mosaic.tif',
              'depth':'depth_mask_mosaic.tif',
              'x_vel':'x_vel_mask_mosaic.tif',
              'y_vel':'y_vel_mask_mosaic.tif',
              'vel_dir':'vel_dir_mask_mosaic.tif',
              'vel_mag':'vel_mag_mask_mosaic.tif',
              'elev':'elev_mask_mosaic.tif',
              'wetted':'wetted_perimeter.tif'}

# # identify background environmental files
# env_files = {'wsel':'wsel_19900_05_mosaic.tif',
#               'depth':'depth_19900_05_mosaic.tif',
#               'x_vel':'vel_x_19900_05_mosaic.tif',
#               'y_vel':'vel_y_19900_05_mosaic.tif',
#               'vel_dir':'vel_dir_19900_05_mosaic.tif',
#               'vel_mag':'vel_mag_19900_05_mosaic.tif',
#               'elev':'elev_19900_05_mosaic.tif',
#               'wetted':'wetted_perimeter.tif'}

# identify longitudinal profile shapefile
longitudinal = os.path.join(model_dir,'longitudinal.shp')

#%% create a simulation object 
sim = sockeye.simulation(model_dir,
                         model_name,
                         crs, 
                         basin, 
                         water_temp, 
                         start_polygon,
                         env_files,
                         longitudinal,
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
 