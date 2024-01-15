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

class solution():
    '''Python class object for a solution of one individual. 
    '''
    def __init__(self,
                 num_genes,
                 min_gene_value,
                 max_gene_value):
        """
        Initializes an individual's genetic traits.
    
        """
        
        # set parameters for P,I,D
        self.num_genes = num_genes
        self.min_gene_value = min_gene_value
        self.max_gene_value = max_gene_value
        
        ## for uniform range across p/i/d values
        # self.genes = np.random.uniform(
        #     self.min_gene_value,
        #     self.max_gene_value,
        #     size = self.num_genes)
        
        ## for non-uniform range across p/i/d values
        self.p_component = np.random.uniform(self.min_gene_value, self.max_gene_value, size=1)
        self.i_component = np.random.uniform(0, 1, size=1)
        self.d_component = np.random.uniform(0.1, 15, size=1)
        self.genes = np.concatenate((self.p_component, self.i_component, self.d_component), axis=None)
        
        self.cross_ratio = 0.9 # controls % of offspring that are crossover vs mutation
        self.pop_size = 0 # dummy value, will be overwritten
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
                                         'rank'])

        for i in range(self.pop_size):
            # remove nan from end of error array
            filtered_array = self.errors[i][:-1]
            # calculate magnitude of errors - lower is better
            magnitude = np.linalg.norm(filtered_array)
            # calculate rank - minimize magnitude & maximize steps before failure
            # lower is better
            rank = magnitude / len(filtered_array)**2
            #rank = magnitude + (1/len(filtered_array))
            
            row_data = {
                'individual': i + 1,
                'p': self.p[i],
                'i': self.i[i],
                'd': self.d[i],
                'magnitude': magnitude,
                'array_length': len(filtered_array),
                'avg_velocity': np.average(self.velocities[i]),
                'rank': rank}

            # append as a new row to df
            error_df = error_df.append(row_data, ignore_index=True)
            error_df['pd_rank'] = error_df['rank'].rank()
            error_df = error_df.sort_values('rank')
            
            # logging fitness here???
            
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
        #offspring = 1000
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
            # blend_ratio = 0.8
            
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
            
            self.p_component = np.random.uniform(self.min_gene_value, self.max_gene_value, size=1)
            self.i_component = np.random.uniform(0, 1, size=1)
            self.d_component = np.random.uniform(0, 25, size=1)
            individual = np.concatenate((self.p_component, self.i_component, self.d_component), axis=None)
            
            population.append(individual)
   
        return population


        # modify one value





















