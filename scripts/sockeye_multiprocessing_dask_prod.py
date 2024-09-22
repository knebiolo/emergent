# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods

ADY: edits to enable multiple flow conditions processing across cpu cores using dask 
"""
#%% Import emergent
# software directory
import sys
#sys.path.append(r"C:\Users\EMuhlestein\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")
#sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
sys.path.append(r"Q:\Internal_Data\Staff_Projects\ADY\ABM\emergent")

#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os
import shutil
import multiprocessing
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import time


# identify directory with subdirectories for each flow condition
base_folder = r"D:\abm"

# base folder structure:
    # base_folder
    #       1000cfs
    #           wsel.tif
    #           depth.tif
    #           ...
    #       2000cfs
    #           wsel.tif
    #           depth.tif
    #           ...

# path to shapefiles to be used across scenarios. longitudinal, start polygon, etc
shp_folder = r"D:\shapefiles"


#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# define a starting box - aka where are all the fish starting from?
start_polygon = os.path.join(shp_folder, 'start_loc_river_left.shp' )                  

# identify longitudinal profile shapefile
longitudinal = os.path.join(shp_folder,'longitudinal.shp')

# scan the base folder for the different flows
flows = os.listdir(base_folder)

# how many simulations?
num_simulations = 3

print(f'Number of tasks to compute: {len(flows)*num_simulations}')

# how many agents in the simulation?
n = 1000       #Agents need to be in 5 or 10 or by 5's

# what is the delta t
dt = 3

# how many timesteps in the model?
hours = 6
ts = 3600. * hours / dt

# what is the water temp?
water_temp = 18.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()
print(f"Number of available CPU cores: {num_cores}")

# Number of cores to use (e.g., use all available cores)
# You can adjust this if you want to leave some cores free or use them all
#n_jobs = num_cores  
n_jobs = num_simulations * len(flows)

n_threads = (num_cores-1) // n_jobs
print(f'threads per task: {n_threads}')

# identify background environmental files
print("\nreading environmental files")
# note: software looks for image files ending in .tif, not .tiff
env_files = {'depth':'depth_mask_mosaic.tif', # generic names - flow folders should be used to differentiate
             'elev':'elev_mask_mosaic.tif',
             'vel_dir':'vel_dir_mask_mosaic.tif',
             'vel_mag':'vel_mag_mask_mosaic.tif',
             'wetted':'wetted_perimeter_rp.tif',
             'wsel':'wsel_mask_mosaic.tif',
             'x_vel':'x_vel_mask_mosaic.tif',
             'y_vel':'y_vel_mask_mosaic.tif'}      

# File extensions to copy and delete
file_extensions = [".tif", ".prj", ".cpg", ".dbf", ".sbn", ".sbx", ".html"]

#%% create a simulation object 
def run_simulation(i, flow):
    start = time.time()
    model_name = f'sim_{i}'
    model_dir = os.path.join(base_folder, flow)
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
    sim = sockeye.simulation(simulation_dir,
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
                             use_gpu=False,
                             pid_tuning=False)
    print(f'Simulation object for {model_name} created in {simulation_dir}')
    sim.run(model_name, ts, dt, video=False)

    # Delete the copied files
    for file in os.listdir(simulation_dir):
        if any(file.endswith(ext) for ext in file_extensions):
            os.remove(os.path.join(simulation_dir, file))
            
    end = time.time()
    elapsed = end - start
    print(f'Simulation object for {flow} cfs, {model_name} completed in {elapsed}.')
    
    
    
if __name__ == "__main__":
    
    # start dask client for parallel execution
    print("initalizing dask client")
    cluster = LocalCluster(n_workers = n_jobs)#, 
                           #threads_per_worker = n_threads)  # adjust threads_per_worker as needed
    client = Client(cluster)
    print(f"dask dashboard available at: {client.dashboard_link}")
    
    # assign tasks
    tasks = [delayed(run_simulation)(i, flow) for i in range(1, num_simulations + 1) for flow in flows]
    # tasks = []
    # for i in range(1, num_simulations + 1):
    #     for flow in flows:
    #         tasks.append(delayed(run_simulation)(i, flow))
    
    # if tasks > num_cores, processing may take at least twice as long
    if len(tasks) > num_cores:
        print('warning: number of tasks exceed number of cpu cores')
        
    print("computing....")
    compute(*tasks)    

    print("\nAll Simulations Completed.")


#%% Summarize - TO DO

# identify workspace
#inputWS = r'Q:\Internal_Data\Staff_Projects\ENM\Nuyakuk\Dask_TEST\Simulation'
# inputWS = model_dir
# tiffWS = os.path.join(model_dir, env_files['depth'])

# summarize
# model_summary = sockeye.summary(model_dir,tiffWS)
# image_data, tiff_extent = model_summary.load_tiff(crs)
# h5_files = model_summary.find_h5_files()
# model_summary.get_data(h5_files)
# model_summary.emergence(h5_files,model_name,crs)


