# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 08:56:56 2023

@author: AYoder

Script to optimize PID controller values in sockeye (SoA).

Called by pid_population script. 

"""

import numpy as np
import pandas as pd
from scipy.stats import beta
import random
import os


def output_excel(records, model_dir, model_name):
    """
    Export the records of errors and rankings to an excel file.
    
    Parameters:
    - records (dict): keys are generation iteration and values are the generation's
                      dataframe of errors and rankings.
    - model_dir: path to simulation output.
    - model_name (str): name of the model to help name the output excel file.
    
    """
    # export record results to excel via pandas
    print('\nexporting records to excel...')
    
    # Create an Excel writer object
    output_excel = os.path.join(model_dir,f'output_{model_name}.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        # iterate through the dictionary and write each dataframe to a sheet
        for generation_name, df in records.items():
            df.to_excel(writer,
                        sheet_name = 'gen' + str(generation_name),
                        index=False)
    
    print('records exported. check output excel file.')
    

class solution():
    '''
    Python class object for a solution of one individual. 
    '''
    def __init__(self,
                 pop_size,
                 generations,
                 min_p_value,
                 max_p_value,
                 min_i_value,
                 max_i_value,
                 min_d_value,
                 max_d_value):
        """
        Initializes an individual's genetic traits.
    
        """
        
        # set parameters for P,I,D
        self.num_genes = 3
        self.min_p_value = min_p_value
        self.max_p_value = max_p_value
        self.min_i_value = min_i_value
        self.max_i_value = max_i_value
        self.min_d_value = min_d_value
        self.max_d_value = max_d_value
        
        # population size, number of individuals to create
        self.pop_size = pop_size
        
        # number of generations to run the alogrithm for
        self.generations = generations
        
        ## for uniform range across p/i/d values
        # self.genes = np.random.uniform(
        #     self.min_gene_value,
        #     self.max_gene_value,
        #     size = self.num_genes)
        
        ## for non-uniform range across p/i/d values
        self.p_component = np.random.uniform(self.min_p_value, self.max_p_value, size=1)
        self.i_component = np.random.uniform(self.min_i_value, self.max_i_value, size=1)
        self.d_component = np.random.uniform(self.min_d_value, self.max_d_value, size=1)
        self.genes = np.concatenate((self.p_component, self.i_component, self.d_component), axis=None)
        
        self.cross_ratio = 0.9 # percent of offspring that are crossover vs mutation
        self.mutation_count = 0 # dummy value, will be overwritten
        self.p = {}
        self.i = {}
        self.d = {}
        self.errors = {}
        self.velocities = {}
        
        
    def fitness(self):
        """
        Rank the population using error timestep magnitude and array length.
        
        Attributes set:
        - pop_size (int): number of indidivduals in population
        - errors (dict): dictionary of indidivduals (key) and sockeye error
                         array (value)
                         
        Returns: dataframe with individual parameters and ranking, sorted by rank.
        """
        error_df = pd.DataFrame(columns=['individual',
                                         'p',
                                         'i',
                                         'd',
                                         'magnitude',
                                         'array_length',
                                         'avg_velocity',
                                         'arr_len_score',
                                         'mag_score',
                                         'rank'])

        for i in range(self.pop_size):
            # remove nan from end of error array
            filtered_array = self.errors[i][:-1]
            # calculate magnitude of errors - lower is better
            magnitude = np.sum(np.power(filtered_array,2))
                        
            row_data = {
                'individual': i,
                'p': self.p[i],
                'i': self.i[i],
                'd': self.d[i],
                'magnitude': magnitude,
                'array_length': len(filtered_array),
                'avg_velocity': np.average(self.velocities[i])}

            # append as a new row to df
            error_df = error_df.append(row_data, ignore_index =True)
        
        # Normalize the criteria
        # array length 1 (maximize): higher values are better
        # magnitude 2 (minimize): lower values are better
        error_df['arr_len_score'] = (error_df['array_length'] - error_df['array_length'].min()) \
            / (error_df['array_length'].max() - error_df['array_length'].min())
        error_df['mag_score'] = (error_df['magnitude'].max() - error_df['magnitude']) \
            / (error_df['magnitude'].max() - error_df['magnitude'].min())
        error_df.set_index('individual', inplace = True)
        
        array_len_weight = 0.8
        magnitude_weight = 1 - array_len_weight
        # Compute pairwise preference matrix
        n = len(error_df)
        preference_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    preference_matrix[i, j] = (array_len_weight * (error_df.at[i, 'arr_len_score'] > error_df.at[j, 'arr_len_score'])) + \
                        (magnitude_weight * (error_df.at[i, 'mag_score'] > error_df.at[j, 'mag_score']))        
        # Aggregate preferences
        final_scores = np.sum(preference_matrix, axis=1)
        
        # Ranking the alternatives
        error_df['rank'] = final_scores
        error_df.reset_index(drop = False, inplace = True)
        error_df.sort_values(by='rank', ascending = False, inplace = True)
            
        return error_df
    
    
    def selection(self, error_df):
        """
        Selects the highest performing indivduals to become parents, based on
        solution rank. Assigns a number of offspring to each parent pair based
        on a beta probability distribution function. Fitter parents produce more
        offspring.
        
        Parameters:
        - error_df (dataframe): a ranked dataframe of indidvidual solutions.
                                output of the self.fitness() function.
        
        Attributes set:
        - pop_size (int): number of indidivduals in population. useful for defining
                          the number of offspring to ensure population doesn't balloon.
        - cross_ratio (float): controls the ratio of crossover offspring vs mutation offspring
                          
        Returns: list of dataframes. each dataframe contained paired parents with
                 assigned number of offspring
        
        """
        # selects the top 80% of individuals to be parents
        index_80_percent = int(0.8 * len(error_df))
        parents = error_df.iloc[:index_80_percent]
        
        # create a list of dataframes -> pairs of parents by fitness
        pairs_parents = []
        for i in np.arange(0, len(parents), 2):
            pairs_parents.append(parents[i:(i + 2)])
        
        # shape parameters for the beta distribution -> have more fit parents produce more offspring
        # https://en.wikipedia.org/wiki/Beta_distribution#/media/File:Beta_distribution_pdf.svg
        a = 1
        b = 3
        
        # calculate PDF values of the beta distribution based on the length of the list
        beta_values = beta.pdf(np.linspace(0, 0.5, len(pairs_parents)), a, b)
        
        # scale values to number of offspring desired
        offspring = self.cross_ratio * self.pop_size # generate XX% of offspring as crossover
        scaled_values = offspring * beta_values / sum(beta_values)
        scaled_values = np.round(scaled_values).astype(int)
        
        # assign beta values (as offspring weight) to appropriate parent pair
        for i, df in enumerate(pairs_parents):
            df['offspring_weight'] = scaled_values[i]  # Assign array value to the column
        
        return pairs_parents
        
    
    def crossover(self, pairs_parents):
        """
        Generate new genes for offspring based on existing parent genes. Number of offspring
        per parent pair is dictated by 'offspring_weight' as set in selection function.
        
        Parameters:
        - pairs_parents (list): list of dataframes. each dataframe contained paired
                                parents with assigned number of offspring
                                
        Returns: list of lists, each list contains random p,i,d values between parent values
                                
        """
        offspring = []

        for i in pairs_parents:
            parent1 = i[:1]
            parent2 = i[1:]
            num_offspring = parent1.iloc[0]['offspring_weight'].astype(int)
            
            for j in range(num_offspring):
                p = random.uniform(parent1.iloc[0]['p'], parent2.iloc[0]['p'])
                i = random.uniform(parent1.iloc[0]['i'], parent2.iloc[0]['i'])
                d = random.uniform(parent1.iloc[0]['d'], parent2.iloc[0]['d'])
                offspring.append([p,i,d])
        
        # set a number of mutations to generate
        # this ensures the correct number of offspring are generated
        self.mutation_count = self.pop_size - len(offspring)
        
        return offspring


    def mutation(self):
        """
        Generate new genes for offspring independent of parent genes. Uses the min/max
        gene values set in the first generation population.
        
        Attributes set:
        - mutation_count (int): number of mutation individuals to create. defined by the crossover
                                function, this ensures that the offspring total are the same as the
                                previous population so it doesn't change.
        - min_gene_value: minimum for gene value. same as defined in initial population
        - max_gene_value: maximum for gene value. same as defined in initial population
        - num_genes: number of genes to create. should always be 3 for pid controller
                                
        Returns: list of lists, each list contains random p,i,d values between min/max gene values.
                 this list will be combined with the crossover offspring to produce the full
                 population of the next generation.
        
        """
        population = []

        for i in range(self.mutation_count):
            # individual = [random.uniform(self.min_gene_value, self.max_gene_value) for _ in range(self.num_genes)]
            
            self.p_component = np.random.uniform(self.min_p_value, self.max_p_value, size=1)
            self.i_component = np.random.uniform(self.min_i_value, self.max_i_value, size=1)
            self.d_component = np.random.uniform(self.min_d_value, self.max_d_value, size=1)
            individual = np.concatenate((self.p_component, self.i_component, self.d_component), axis=None)
            
            population.append(individual)
   
        return population


    def population_create(self):
        """
        Generate the population of individuals.
        
        Attributes set:
        - genes
        - pop_size
        - num_genes
        - min_gene_value
        - max_gene_value
                                
        Returns: array of population p/i/d values, one set for each individual.
        
        """
        # population = self.genes
        # if self.pop_size > 1:
        #     for i in range(self.pop_size-1):
        #         # create another individual
        #         #individual = solution(self.num_genes, self.min_gene_value, self.max_gene_value, self.pop_size, self.generations)
        #         individual = solution(self.pop_size,
        #                               self.generations,
        #                               self.min_p_value,
        #                               self.max_p_value,
        #                               self.min_i_value,
        #                               self.max_i_value,
        #                               self.min_d_value,
        #                               self.max_d_value)
        #         # add it to the population
        #         population = np.vstack((population, individual.genes))
        
        population = []

        for _ in range(self.pop_size):
        # create a new instance of the solution class for each individual
            individual = solution(self.pop_size,
                              self.generations,
                              self.min_p_value,
                              self.max_p_value,
                              self.min_i_value,
                              self.max_i_value,
                              self.min_d_value,
                              self.max_d_value)
            population.append(individual.genes)

        return population
    
    
    def run(self, population, sockeye, model_dir, crs, basin, water_temp, pid_tuning_start, fish_length, ts, n, dt):
        """
        Run the genetic algorithm.
        
        Parameters:
        - population (array): collection of solutions (population of individuals)
        - sockeye: sockeye model
        - model_dir (str): Directory where the model data will be stored.
        - crs (str): Coordinate reference system for the model.
        - basin (str): Name or identifier of the basin.
        - water_temp (float): Water temperature in degrees Celsius.
        - pid_tuning_start (tuple): A tuple of two values (x, y) defining the point where agents start.
        - ts (int, optional): Number of timesteps for the simulation. Defaults to 100.
        - n (int, optional): Number of agents in the simulation. Defaults to 100.
        - dt (float): The duration of each time step.
        
        Attributes:
        - generations
        - pop_size
        - p
        - i
        - d
        - errors
        - velocities
        
        Returns:
        - records (dict): dictionary holding each generation's errors and rankings
        
        
        """
        # Create a dictionary that will store the error dataframe for each generation,
        # We will use the generation number as key. Stores pid values, magnitude, array length,
        # and rank in each dataframe.
        records = {}
        
        for generation in range(self.generations):
            
            # keep track of the timesteps before error (length of error array),
            # also used to calc magnitude of errors
            pop_error_array = []

            #for i in range(len(self.population)):
            for i in range(self.pop_size):
            
                print(f'\nrunning individual {i+1} of generation {generation+1}...')
                
                # useful to have these in pid_solution
                self.p[i] = population[i][0]
                self.i[i] = population[i][1]
                self.d[i] = population[i][2]
                
                print(f'P: {self.p[i]:0.3f}, I: {self.i[i]:0.3f}, D: {self.d[i]:0.3f}')
                
                # set up the simulation
                sim = sockeye.simulation(model_dir,
                                         'solution',
                                         crs,
                                         basin,
                                         water_temp,
                                         pid_tuning_start,
                                         fish_length,
                                         ts,
                                         n,
                                         use_gpu = False,
                                         pid_tuning = True)
                
                
                # run the model and append the error array
                try:
                    sim.run('solution',
                            self.p[i], # k_p
                            self.i[i], # k_i
                            self.d[i], # k_d
                            n = ts,
                            dt = dt)
                    
                except:
                    print(f'failed --> P: {self.p[i]:0.3f}, I: {self.i[i]:0.3f}, D: {self.d[i]:0.3f}\n')
                    pop_error_array.append(sim.error_array)
                    self.errors[i] = sim.error_array
                    self.velocities[i] = np.sqrt(np.power(sim.vel_x_array,2) + np.power(sim.vel_y_array,2))
                    sim.close()

                    continue

            # run the fitness function -> output is a df
            error_df = self.fitness()
            # print(f'Generation {generation+1}: {error_df.head()}')
            
            # update logging dictionary
            records[generation] = error_df

            # selection -> output is list of paired parents dfs
            selected_parents = solution.selection(self, error_df)

            # crossover -> output is list of crossover pid values
            cross_offspring = solution.crossover(self, selected_parents)

            # mutation -> output is list of muation pid values
            mutated_offspring = solution.mutation(self)

            # combine crossover and mutation offspring to get next generation
            population = cross_offspring + mutated_offspring
            
            print(f'completed generation {generation+1}.... ')
            
        return records
