# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:16:34 2023

@author: AYoder

Runs a genetic algorithm for optimizing sockeye PID controller values. Bounded, random
values are assigned to the P, I, and D values for a population of individuals then
run with sockeye. Each generation's arrays are collected from sockeye and written
to a dictionary using the generation interation as the key. An Excel worksheet version
of this dictionary is also generated in the model folder.  
"""
import sys
import os
import shutil

#sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
sys.path.append(r"C:\Users\AYoder\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")

from emergent import sockeye_SoA as sockeye

#%% Genetic algorithm parameters

# number of individuals in the population/gene pool per generation. use >= 4 individuals
pop_size = 20
# number of generations to run the algorithm
generations = 20

# P
min_p_value = 0     # min value of P
max_p_value = 75    # max value of P
# I
min_i_value = 0     # min value of I
max_i_value = 0.1   # max value of I
# D
min_d_value = 0.1   # min value of D
max_d_value = 4     # max value of D

# Create the initial indidivdual solution
pid_solution = sockeye.solution(pop_size,
                                generations,
                                min_p_value,
                                max_p_value,
                                min_i_value,
                                max_i_value,
                                min_d_value,
                                max_d_value)

# Create a population of individuals
population = pid_solution.population_create()

#%% Sockeye model parameters

# Identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'

# Identify directories
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"

# Identify the coordinate reference system for the model
crs = 32604

# Create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550328.25,550410.05,6641500.76,6641550.31)     # starting box way downstream
#bbox = (549800,550115,6641332,6641407)
bbox = (549505.65,549589.76,6641553.32,6641564.74)      # starting box right near the falls
pid_tuning_start = (549488.29, 6641611.84) # below falls
#pid_tuning_start = (549397.33, 6641816.50) # high flow speed
#pid_tuning_start = (549400.43, 6641759.13) # d/s of chute
#pid_tuning_start = (549420.07, 6641762.23) # in chute
#pid_tuning_start = (550370.39, 6641528.46) # low flow speed

# group starting locations for production loop
start_locations = {
    'below_falls': (549488.29, 6641611.84),
    'high_flow_speed': (549397.33, 6641816.50),
    'ds_of_chute': (549400.43, 6641759.13),
    'in_chute': (549420.07, 6641762.23),
    'low_flow_speed': (550370.39, 6641528.46),
    'low_flow_speed2': (550147.74, 6641476.45),
    'low_flow_speed3': (549485.93, 6641527.83),
    'low_flow_speed4': (550565.14, 6641496.90)
    }

# How many agents in the simulation?
n = 1

# How many timesteps in the model?
ts = 250

# What is the delta t?
dt = 0.25
# limit ts in low velocity areas

# What is the water temp?
water_temp = 20.

# What is the basin that we are simulating passage in?
basin = "Nushagak River"

# What is the length of the fish (mm)?
fish_length = 550

# length stats for production loop: (min, 25%, 50%, 75%, max)
#fish_lengths = (468, 570, 598, 626, 698)
fish_lengths = (570, 598, 626, 698)

#%% Run the algorithm

# # Input for run function: population of individuals, sockeye parameters
# records = pid_solution.run(population,
#                            sockeye,
#                            model_dir,
#                            crs,
#                            basin,
#                            water_temp,
#                            pid_tuning_start,
#                            fish_length,
#                            ts,
#                            n,
#                            dt)
    
# # Export record results to excel via pandas
# sockeye.output_excel(records, model_dir, model_name)

#%% Production loop
# Run the algorithm for all fish sizes and starting locations.
# Each size/location scenario creates a new file folder. At the end of each scenario, 
# an excel file with the typical output generations is saved to the folder.

for size in fish_lengths:
    for location in start_locations:
        
        # create a new folder for the scenario
        model_name = f"soa_{location}_{size}mm_production"
        model_dir = os.path.join("C:/Users/AYoder/Desktop/simulations/", model_name)
        
        # copy required surfaces files
        sockeye_surfaces = "C:/Users/AYoder/Desktop/simulations/sockeye_surfaces"
        shutil.copytree(sockeye_surfaces, model_dir)
        
        pid_tuning_start = start_locations[location]
        fish_length = size
        
        # Input for run function: population of individuals, sockeye parameters
        records = pid_solution.run(population,
                                   sockeye,
                                   model_dir,
                                   crs,
                                   basin,
                                   water_temp,
                                   pid_tuning_start,
                                   fish_length,
                                   ts,
                                   n,
                                   dt)
            
        # Export record results to excel via pandas
        sockeye.output_excel(records, model_dir, model_name)



