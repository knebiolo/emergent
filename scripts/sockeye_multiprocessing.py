# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
#%% Import emergent
# software directory
import sys
#sys.path.append(r"C:\Users\EMuhlestein\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")


#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os
import shutil
import multiprocessing
from joblib import Parallel, delayed

# identify input and output model names
model_name = 'Simulation_5k'

# identify directories
model_dir = os.path.join(r"C:\Users\EMuhlestein\Documents\ABM_TEST\val_TEST",model_name)

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
bbox = (550402.28,550533.22,6641508.09,6641584.47)                             # starting box way downstream
#bbox = (549857.46,550072.26,6641405.92,6641347.85)                             # about halfway up
#bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # kinda near the falls
#bbox = (549466.69,549520.48,6641583.35,6641625.48)                             # starting box right near the falls

#How many simulations?
num_simulations=25

# how many agents in the simulation?
n = 5       #Agents need to be in 5 or 10 or by 5's

# what is the delta t
dt = 1

# how many timesteps in the model?
hours = 0.25
ts = 3600. * hours / dt

# what is the water temp?
water_temp = 40.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()
print(f"Number of available CPU cores: {num_cores}")

# Number of cores to use (e.g., use all available cores)
n_jobs = num_cores  # You can adjust this if you want to leave some cores free


# identify background environmental files
env_files = {'wsel':'wsel_mask_mosaic.tif',
             'depth':'depth_mask_mosaic.tif',
             'x_vel':'x_vel_mask_mosaic.tif',
             'y_vel':'y_vel_mask_mosaic.tif',
             'vel_dir':'vel_dir_mask_mosaic.tif',
             'vel_mag':'vel_mag_mask_mosaic.tif',
             'elev':'elev_mask_mosaic.tif',
             'wetted':'wetted_perimeter.tif'}

# identify longitudinal profile shapefile
longitudinal = os.path.join(model_dir,'longitudinal.shp')

# File extensions to copy and delete
file_extensions = [
    ".tif", ".prj", ".cpg", ".dbf", ".sbn", ".sbx", ".html"
]
#%% create a simulation object 
def run_simulation(i):
    model_name = f'val_TEST_{i}'
    simulation_dir = os.path.join(model_dir, model_name)

    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    # Copy relevant files
    for file in os.listdir(model_dir):
        if any(file.endswith(ext) for ext in file_extensions):
            src_file = os.path.join(model_dir, file)
            dst_file = os.path.join(simulation_dir, file)
            shutil.copy(src_file, dst_file)

    # Run the simulation
    sim = sockeye.simulation(simulation_dir, model_name, crs, basin, water_temp, bbox, env_files, longitudinal, None, ts, n, use_gpu=False, pid_tuning=False)
    print(f'Simulation object for {model_name} created in {simulation_dir}')
    sim.run(model_name, ts, dt, video=False)

    # Delete the copied files
    for file in os.listdir(simulation_dir):
        if any(file.endswith(ext) for ext in file_extensions):
            os.remove(os.path.join(simulation_dir, file))

if __name__ == "__main__":
    # Execute simulations in parallel using Joblib
    Parallel(n_jobs=n_jobs)(delayed(run_simulation)(i) for i in range(1, num_simulations + 1))

    print("All simulations completed.")