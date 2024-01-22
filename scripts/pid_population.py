# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:16:34 2023

@author: AYoder

Creates a population of errors for optimizing PID controller values using a
genetic algorithm. Random values are assigned to the P, I, and D values for an 
individual then run through sockeye. Errors arrays are collected from sockeye
and written to a dictionary using the individual's number as the key. 

TO DO:
    - different fish sizes to optimize for
    - different velocity locations to optimize for    

"""
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
import numpy as np
import pandas as pd
import os
from emergent import sockeye_SoA as sockeye
import pid_solution as pid

#%% genetic algorithm parameters

# set input values
genes = 3               # three values (P,I,D)
min_gene_value = 0.1     # min value of P/I/D
max_gene_value = 50     # max value of P/I/D

# number of generations to run the algorithm
generations = 20

# dictionary will store error dataframe for each generation, with generation number as key
# stores pid values, magnitude, array length, and rank in dataframe
records = {}

## create the initial population. this gets updated after each generation
# create an individual
pid_solution = pid.solution(genes,
                            min_gene_value,
                            max_gene_value)
population = pid_solution.genes

# create a population of individuals
pop_size = 10
pid_solution.pop_size = pop_size
if pop_size > 1:
    for i in range(pop_size-1):
        # create another individual
        individual = pid.solution(genes, min_gene_value, max_gene_value)
        # add it to the population
        population = np.vstack((population, individual.genes))

#%% sockeye model parameters

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p02.hdf'
model_name = 'below_falls_750_2'

# identify directories
model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\simulations\PID_optimization",model_name)
#model_dir = os.path.join(r"C:\Users\AYoder\Desktop\simulations",model_name)
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"

# identify the coordinate reference system for the model
crs = 32604

# create a starting box - aka where are all the fish starting from?
# W,E,S,N
#bbox = (550328.25,550410.05,6641500.76,6641550.31)                             # starting box way downstream
#bbox = (549800,550115,6641332,6641407)
bbox = (549505.65,549589.76,6641553.32,6641564.74)                             # starting box right near the falls
pid_tuning_start = (549488.29, 6641611.84) # below falls
#pid_tuning_start = (549397.33, 6641816.50) # high flow speed
#pid_tuning_start = (549400.43, 6641759.13) # d/s of chute
#pid_tuning_start = (549420.07, 6641762.23) # in chute
#pid_tuning_start = (550370.39, 6641528.46) # low flow speed

# how many agents in the simulation?
n = 1

# how many timesteps in the model?
ts = 500

# what is the delta t
dt = 0.2

# what is the water temp?
water_temp = 20.

# what is the basin that we are simulating passage in?
basin = "Nushagak River"

#%% run the simulation

for generation in range(generations):
    
    # keep track of the timesteps before error (length of error array),
    # also used to calc magnitude of errors
    pop_error_array = []
    pop_avg_vel_array = []

    for i in range(len(population)):
    
        print(f'\running individual {i+1} of generation {generation+1}...')
        
        # useful to have these in pid_solution
        pid_solution.p[i] = population[i][0]
        pid_solution.i[i] = population[i][1]
        pid_solution.d[i] = population[i][2]
        
        print(f'P: {pid_solution.p[i]:0.3f}, I: {pid_solution.i[i]:0.3f}, D: {pid_solution.d[i]:0.3f}')
        
        # set up the simulation
        sim = sockeye.simulation(model_dir,
                                 'solution',
                                 crs,
                                 basin,
                                 water_temp,
                                 pid_tuning_start,
                                 ts,
                                 n,
                                 use_gpu = False,
                                 pid_tuning = True)
        
        
        # run the model and append the error array
        try:
            
            sim.run('solution',
                    pid_solution.p[i], # k_p
                    pid_solution.i[i], # k_i
                    pid_solution.d[i], # k_d
                    n = ts,
                    dt = dt)
            
        except:
            print(f'failed --> P: {pid_solution.p[i]:0.3f}, I: {pid_solution.i[i]:0.3f}, D: {pid_solution.d[i]:0.3f}\n')
            pop_error_array.append(sim.error_array)
            pid_solution.errors[i] = sim.error_array
            pid_solution.velocities[i] = np.sqrt(np.power(sim.vel_x_array,2) + np.power(sim.vel_y_array,2))
            sim.close()

            continue

    # run the fitness function -> output is a df
    error_df = pid_solution.fitness()
    # update logging dictionary
    records[generation] = error_df

    # selection -> output is list of paired parents dfs
    selected_parents = pid_solution.selection(error_df)

    # crossover -> output is list of crossover pid values
    cross_offspring = pid_solution.crossover(selected_parents)

    # mutation -> output is list of muation pid values
    mutated_offspring = pid_solution.mutation()

    # combine crossover and mutation offspring to get next generation
    population = cross_offspring + mutated_offspring
    
    print(f'completed generation {generation+1}.... ')
    

    
# export record results to excel via pandas

# Create an Excel writer object
output_excel = os.path.join(model_dir,'output.xlsx')
with pd.ExcelWriter(output_excel) as writer:
    # Iterate through the dictionary and write each DataFrame to a sheet
    for generation_name, df in records.items():
        df.to_excel(writer,
                    sheet_name = 'gen' + str(generation_name),
                    index=False)
        

# print statement to calc thrust at each ts


# after x generations... measure difference between top performers from last and second to last gen







