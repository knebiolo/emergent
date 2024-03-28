# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
#%% Import emergent
# software directory
import sys
sys.path.append(r"C:\Users\EMuhlestein\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")


#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os
import shutil
from joblib import Parallel, delayed
import time

# identify input and output model names
model_name = 'val_TEST'

# identify directories
model_dir = os.path.join(r"C:\Users\EMuhlestein\Documents\ABM_TEST",model_name)

#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550402.28,550533.22,6641508.09,6641584.47)                             # starting box way downstream
#bbox = (549857.46,550072.26,6641405.92,6641347.85)                             # about halfway up
#bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # kinda near the falls
bbox = (549466.69,549520.48,6641583.35,6641625.48)                             # starting box right near the falls


# how many agents in the simulation?
n = 5        #Agents need to be in 5 or 10 or by 5's

# what is the delta t
dt = 1

# how many timesteps in the model?
hours = 0.005
ts = 3600. * hours / dt

# what is the water temp?
water_temp = 20.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"


#%% create a simulation object 
def run_simulation(i, model_dir, crs, basin, water_temp, bbox, ts, n, use_gpu=False, pid_tuning=False):
    start_time = time.time()
    model_name = f'val_TEST_{i}'
    simulation_dir = os.path.join(model_dir, model_name)

    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    # Copy TIF files
    for file in os.listdir(model_dir):
        if file.endswith(".tif"):
            src_file = os.path.join(model_dir, file)
            dst_file = os.path.join(simulation_dir, file)
            shutil.copy(src_file, dst_file)

    # Run the simulation
    sim = sockeye.simulation(simulation_dir, model_name, crs, basin, water_temp, bbox, None, ts, n, use_gpu, pid_tuning)
    print(f'Simulation object for {model_name} created in {simulation_dir}')
    sim.run(model_name, ts, dt, video=False)

    # Create a movie file for this simulation
    #sockeye.movie_maker(simulation_dir, model_name, crs, dt, sim.depth_rast_transform)
    #print(f'Movie file for {model_name} created in {simulation_dir}')

    # After movie creation, delete the TIF files in the simulation directory
    for file in os.listdir(simulation_dir):
        if file.endswith(".tif"):
            os.remove(os.path.join(simulation_dir, file))

    end_time = time.time()
    execution_time = end_time - start_time
    mins, secs = divmod(execution_time, 60)
    print(f"ABM {model_name}: {int(mins)} minutes {secs:.2f} seconds.")

    return model_name, execution_time

# Assuming the necessary variables (model_dir, crs, basin, water_temp, bbox, ts, dt, n) are defined
if __name__ == "__main__":
    real_start_time = time.time()
    #Number of simulations you wish to run
    num_simulations=100

    # Execute simulations in parallel
    results = Parallel(n_jobs=-1)(delayed(run_simulation)(i, model_dir, crs, basin, water_temp, bbox, ts, n) for i in range(1, num_simulations + 1))

    real_end_time = time.time()
    real_elapsed_time = real_end_time - real_start_time
    real_elapsed_mins, real_elapsed_secs = divmod(real_elapsed_time, 60)

    # Log individual and total simulation times
    time_log_path = os.path.join(model_dir, "simulation_times.txt")
    with open(time_log_path, "w") as f:
        total_time = 0
        for model_name, execution_time in results:
            mins, secs = divmod(execution_time, 60)
            f.write(f"{model_name}: {int(mins)} minutes {secs:.2f} seconds\n")
            total_time += execution_time
        total_mins, total_secs = divmod(total_time, 60)
        f.write(f"\nTotal sum of simulation times: {int(total_mins)} minutes {total_secs:.2f} seconds.\n")
        f.write(f"Real elapsed time for all simulations: {int(real_elapsed_mins)} minutes {real_elapsed_secs:.2f} seconds.")
    print(f"Execution times written to {time_log_path}")      
