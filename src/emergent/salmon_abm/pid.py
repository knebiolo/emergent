import os
import random

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import beta


class PID_controller:
    def __init__(self, n_agents, k_p = 0., k_i = 0., k_d = 0., tau_d = 1):
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        self.integral = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.previous_error = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.derivative_filtered = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        # Attempt to initialize PID plane parameters; fall back to safe defaults
        try:
            self.interp_PID()
        except Exception:
            # Default: P = 1.0 constant, I = 0, D = 0
            self.P_params = np.array([0.0, 0.0, 1.0])
            self.I_params = np.array([0.0, 0.0, 0.0])
            self.D_params = np.array([0.0, 0.0, 0.0])

    def update(self, error, dt, status):
        # create a mask - if this fish is fatigued, this doesn't matter
        mask = np.where(status == 3,True,False)
        
        self.integral = np.where(~mask, self.integral + error, self.integral)
        derivative = error - self.previous_error
        self.previous_error = error
    
        p_term = self.k_p[:, np.newaxis] * error
        i_term = self.k_i[:, np.newaxis] * self.integral
        d_term = self.k_d[:, np.newaxis] * derivative
        
        return np.where(~mask,p_term + i_term + d_term,0.0)
        
    def interp_PID(self):
        '''
        Parameters
        ----------
        data_ws : file directory.

        Returns
        -------
        tuple consisting of (P,I,D).
        '''
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/pid_optimize_Nushagak.csv')
        # get data
        df = pd.read_csv(data_dir)
        
        # get data arrays
        length = df.loc[:, 'fish_length'].values
        velocity = df.loc[:, 'avg_water_velocity'].values
        P = df.loc[:, 'p'].values
        I = df.loc[:, 'i'].values
        D = df.loc[:, 'd'].values
        
        # Plane model function
        def plane_model(coords, a, b, c):
            length, velocity = coords
            return a * length + b * velocity + c
        
        # fit plane for P, I, and D values
        self.P_params, _ = curve_fit(plane_model, (length, velocity), P)
        
        self.I_params, _ = curve_fit(plane_model, (length, velocity), I)
        
        self.D_params, _ = curve_fit(plane_model, (length, velocity), D)
    
    def PID_func(self, velocity, length):
        '''
        
        '''
        # Ensure parameters exist
        a_P, b_P, c_P = self.P_params
        a_I, b_I, c_I = self.I_params
        a_D, b_D, c_D = self.D_params

        # Coerce inputs to numpy arrays and broadcast to the same shape
        vel = np.asarray(velocity)
        leng = np.asarray(length)

        # If scalars, expand to 1-D arrays
        if vel.ndim == 0:
            vel = np.full(1, float(vel))
        if leng.ndim == 0:
            leng = np.full(1, float(leng))

        # Broadcast to common shape
        try:
            vel_b, leng_b = np.broadcast_arrays(vel, leng)
        except ValueError:
            # Fallback: flatten and match lengths if possible
            vel_b = vel.ravel()
            leng_b = np.broadcast_to(leng.ravel(), vel_b.shape)

        P = a_P * leng_b + b_P * vel_b + c_P
        I = a_I * leng_b + b_I * vel_b + c_I
        D = a_D * leng_b + b_D * vel_b + c_D

        # Return 1-D arrays
        return np.asarray(P).ravel(), np.asarray(I).ravel(), np.asarray(D).ravel()


class PID_optimization():
    '''
    Python class object for solving a genetic algorithm to optimize PID controller values. 
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
        self.batteries = {}
        

    def fitness(self):
        '''
        Overview

        This fitness function is designed to evaluate a population of individuals 
        based on three key criteria: error magnitude, array length, and battery life. 
        The function ranks each individual by combining these criteria into a single 
        score, with the goal of minimizing error magnitude, maximizing array length, 
        and maximizing battery life.
        
        Attributes
        
            pop_size (int): The number of individuals in the population. Each 
            individual's performance is evaluated against the set criteria.
            errors (dict): A dictionary where keys are individual identifiers and 
            values are arrays representing the error magnitude for each timestep.
            p, i, d (arrays): Parameters associated with each individual, potentially 
            relevant to the context of the evaluation (e.g., PID controller parameters).
            velocities (array): An array containing the average velocities for each 
            individual, which might be relevant for certain analyses.
            batteries (array): An array containing the battery life values for each 
            individual. Higher values indicate better performance.
        
        Returns
        
            error_df (DataFrame): A pandas DataFrame containing the following 
            columns for each individual:
                individual: The identifier for the individual.
                p, i, d: The PID controller parameters or other relevant parameters 
                for the individual.
                magnitude: The sum of squared errors, representing the error magnitude. 
                Lower values are better.
                array_length: The length of the error array, indicative of the operational 
                duration. Higher values are better.
                avg_velocity: The average velocity for the individual. Included for 
                contextual information.
                battery: The battery life of the individual. Higher values are better.
                arr_len_score: Normalized score based on array_length. Higher scores are better.
                mag_score: Normalized score based on magnitude. Higher scores are better (inverted).
                battery_score: Normalized score based on battery. Higher scores are better.
                rank: The final ranking score, calculated by combining arr_len_score, mag_score, 
                and battery_score according to their respective weights.
        
        Methodology
        
            Data Preparation: The function iterates through each individual in 
            the population, calculating the magnitude of errors and extracting 
            other relevant parameters. It then appends this information to error_df.
        
            Normalization: Each criterion (array length, magnitude, and battery) 
            is normalized to a [0, 1] scale. For array length and battery, higher 
            values result in higher scores. For magnitude, the normalization is 
            inverted so that lower values result in higher scores.
        
            Weighting and Preference Matrix: The criteria are weighted according 
            to their perceived importance to the overall fitness. A pairwise 
            preference matrix is constructed based on these weighted scores, 
            comparing each individual against every other individual.
        
            Ranking: The final rank for each individual is determined by summing 
            up their preferences in the preference matrix. The DataFrame is then 
            sorted by these ranks in descending order, with higher ranks indicating 
            better overall fitness according to the defined criteria.
        
        Customization
        
            The weights assigned to each criterion (array_len_weight, 
                                                    magnitude_weight, 
                                                    battery_weight) can be adjusted 
            to reflect their relative importance in the specific context of use. The 
            default weights are set based on a balanced assumption but should be 
            tailored to the specific requirements of the evaluation.
            Additional criteria can be incorporated into the evaluation by extending 
            the DataFrame to include new columns, normalizing these new criteria,
            and adjusting the preference matrix calculation to account for these 
            criteria.
        
        Usage
        
        To use this function, instantiate the class with the relevant data 
        (errors, parameters, velocities, and batteries) and call the fitness method. 
        The method returns a ranked DataFrame, which can be used to select the 
        top-performing individuals for further analysis or operations.
                
                
                
        '''
        error_df = pd.DataFrame(columns=['individual', 
                                         'p', 
                                         'i', 
                                         'd', 
                                         'magnitude',
                                         'array_length',
                                         'avg_velocity',
                                         'battery',
                                         'arr_len_score',
                                         'mag_score',
                                         'battery_score',
                                         'rank'])

        for i in range(self.pop_size):
            filtered_array = self.errors[i][:-1]
            magnitude = np.nansum(np.power(filtered_array, 2))

            row_data = {
                'individual': i,
                'p': self.p[i],
                'i': self.i[i],
                'd': self.d[i],
                'magnitude': magnitude,
                'array_length': len(filtered_array),
                'avg_velocity': np.nanmean(self.velocities[i]),
                'battery': self.batteries[i]  # Assuming you have battery data in self.batteries
            }

            error_df = error_df.append(row_data, ignore_index=True)

        # Normalize the criteria
        error_df['arr_len_score'] = (error_df['array_length'] - error_df['array_length'].min()) / (error_df['array_length'].max() - error_df['array_length'].min())
        error_df['mag_score'] = (error_df['magnitude'].max() - error_df['magnitude']) / (error_df['magnitude'].max() - error_df['magnitude'].min())
        error_df['battery_score'] = (error_df['battery'] - error_df['battery'].min()) / (error_df['battery'].max() - error_df['battery'].min())

        error_df.set_index('individual', inplace=True)

        # Update weights to include battery
        array_len_weight = 0.35
        magnitude_weight = 0.40
        battery_weight = 1 - array_len_weight - magnitude_weight

        n = len(error_df)
        preference_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    preference_matrix[i, j] = (array_len_weight * (error_df.at[i, 'arr_len_score'] > error_df.at[j, 'arr_len_score'])) + \
                                              (magnitude_weight * (error_df.at[i, 'mag_score'] > error_df.at[j, 'mag_score'])) + \
                                              (battery_weight * (error_df.at[i, 'battery_score'] > error_df.at[j, 'battery_score']))

        final_scores = np.sum(preference_matrix, axis=1)
        error_df['rank'] = final_scores
        error_df.reset_index(drop=False, inplace=True)
        error_df.sort_values(by='rank', ascending=False, inplace=True)

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

    def mutation(self, error_df):
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
            P = np.abs(error_df.iloc[i]['p'] + np.random.uniform(-4.0,4.0,1)[0])
            I = np.abs(error_df.iloc[i]['i'] + np.random.uniform(-0.1,0.1,1)[0])
            D = np.abs(error_df.iloc[i]['d'] + np.random.uniform(-1.0,1.0,1)[0])
            
            individual = np.concatenate((P, I, D), axis=None)
            
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
        population = []

        for _ in range(self.pop_size):
        # create a new instance of the solution class for each individual
            individual = PID_optimization(self.pop_size,
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
        - records (dict): dictionary holding each generation's errors and rankings. 
                          Generation number is used as the dictionary key. Each key's value
                          is the dataframe of PID values and ranking metrics.
        """
        records = {}
        
        for generation in range(self.generations):
            
            # keep track of the timesteps before error (length of error array),
            # also used to calc magnitude of errors
            pop_error_array = []
            
            prev_error_sum = np.zeros(1)

            #for i in range(len(self.population)):
            for i in range(self.pop_size):
            
                print(f'\nrunning individual {i+1} of generation {generation+1}, {generation+1}, {generation+1}, {generation+1}, {generation+1}...')
                
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
                            n = ts,
                            dt = dt,
                            k_p = self.p[i], # k_p
                            k_i = self.i[i], # k_i
                            k_d = self.d[i], # k_d
                            )
                    
                except:
                    print(f'failed --> P: {self.p[i]:0.3f}, I: {self.i[i]:0.3f}, D: {self.d[i]:0.3f}\n')
                    pop_error_array.append(sim.error_array)
                    self.errors[i] = sim.error_array
                    self.velocities[i] = np.sqrt(np.power(sim.vel_x_array,2) + np.power(sim.vel_y_array,2))
                    self.batteries[i] = sim.battery[-1]
                    sim.close()

                    continue

            # run the fitness function -> output is a df
            error_df = self.fitness()
            # print(f'Generation {generation+1}: {error_df.head()}')
            
            # update logging dictionary
            records[generation] = error_df

            # selection -> output is list of paired parents dfs
            selected_parents = PID_optimization.selection(self, error_df)

            # crossover -> output is list of crossover pid values
            cross_offspring = PID_optimization.crossover(self, selected_parents)

            # mutation -> output is list of muation pid values
            mutated_offspring = PID_optimization.mutation(self, error_df)
            # combine crossover and mutation offspring to get next generation
            population = cross_offspring + mutated_offspring
            
            print(f'completed generation {generation+1}.... ')
            
            if np.all(error_df.magnitude.values == 0):
                return records
                        
        return records
