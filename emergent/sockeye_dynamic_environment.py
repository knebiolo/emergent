# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:30:21 2023

@author: KNebiolo, Isha Deo

Python software for an Agent Based Model of migrating adult Sockeye salmon (spp.)  
with intent of understanding the potential ramifications of river discharge 
changes on ability of fish to succesffuly pass upstream through a riffle - 
cascade complex.  

An agent is a goal-directed, autonomous, software-object that interacts with 
other agents in simulated space.  In the case of a fish passage agent, our fish 
are motivated to move upstream to spawn, thus their goal is simply to pass the 
impediment.   Their motivation is clear, they have an overriding instinct to 
migrate upstream to their natal reach and will do so at the cost of their own 
mortality.   

Our fish agents are python class objects with initialization methods, and 
methods for movement, behaviors, and perception.  Movement is continuous in 2d 
space as our environment is a depth averaged 2d model.  Movement in 
the Z direction is handled with logic.  We will use velocity distributions and 
the agent's position within the water column to understand the forces acting on 
the body of the fish (drag).  To maintain position, the agent must generate 
enough thrust to counteract drag.  The fish generates thrust by beating its tail.  
According to Castro-Santos (2006), fish tend to migrate at a specific speed over 
ground in body lengths per second depending upon the mode of swimming it is in.  
Therefore their tail beat per minute rate is dependent on the amount of drag 
and swimming mode.   
"""
# import dependencies
#import cupy as cp
import h5py
import dask.array as da
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
# from pysal.explore import esda
# from pysal.lib import weights
# from pysal.model import spreg
from affine import Affine as AffineTransform
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform
from rasterio.features import rasterize
from shapely import Point, Polygon, box
from shapely import affinity
from shapely.wkt import loads as loads
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline, interp1d, CubicSpline, RectBivariateSpline
from scipy.optimize import curve_fit
from scipy.constants import g
from scipy.spatial import cKDTree#, cdist
from scipy.stats import beta
from scipy.ndimage import distance_transform_edt
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from datetime import datetime
import time
import warnings
import sys
import random
from collections import deque
#from sksurv.nonparametric import kaplan_meier_estimator


warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'
rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'


def get_arr(use_gpu):
    '''
    Parameters
    ----------
    use_gpu : Boolean
        AI generated function that returns a CuPy array if true, otherwise it 
        returns a Numpy array.  

    Returns
    -------
    Array
        a CuPy or Numpy array.

    '''
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except:
            print("CuPy not found. Falling back to Numpy.")
            import numpy as np
            return np
    else:
        import numpy as np
        return np

# ... other utility functions ...

def geo_to_pixel(X, Y, transform):
    """
    Convert x, y coordinates to row, column indices in the raster grid.
    This function inverts the provided affine transform to convert geographic
    coordinates to pixel coordinates.

    Parameters:
    - X: array-like of x coordinates (longitude or projected x)
    - Y: array-like of y coordinates (latitude or projected y)
    - transform: affine transform of the raster

    Returns:
    - rows: array of row indices
    - cols: array of column indices
    """
    # Invert the transform to go from geographic to pixel coordinates
    inv_transform = ~transform

    # Apply the inverted transform to each coordinate pair
    pixels = [inv_transform * (x, y) for x, y in zip(X, Y)]

    # Separate the pixel coordinates into rows and columns
    cols, rows = zip(*pixels)

    # Round the values to get pixel indices and convert to integers
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)

    return rows, cols


def pixel_to_geo(transform, rows, cols):
    """
    Convert row, column indices in the raster grid to x, y coordinates.

    Parameters:
    - transform: affine transform of the raster
    - rows: array-like or scalar of row indices
    - cols: array-like or scalar of column indices

    Returns:
    - xs: array of x coordinates
    - ys: array of y coordinates
    """
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)

    return xs, ys

def standardize_shape(arr, target_shape= (5, 5), fill_value=np.nan):
    target_shape= arr.shape
    if arr.shape != target_shape:
        # Create a new array with the target shape, filled with the fill value
        standardized_arr = np.full(target_shape, fill_value)
        # Copy data from the original array to the standardized array
        standardized_arr[:arr.shape[0], :arr.shape[1]] = arr
        return standardized_arr
    return arr

def calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0):
    num_agents = len(headings)

    # Convert headings to direction vectors (dx, dy)
    dx = np.cos(headings)[:, np.newaxis, np.newaxis]
    dy = np.sin(headings)[:, np.newaxis, np.newaxis]

    # Agent coordinates expanded to match the 5x5 grid
    agent_x_expanded = agent_x[:, np.newaxis, np.newaxis]
    agent_y_expanded = agent_y[:, np.newaxis, np.newaxis]

    # Calculate relative coordinates of each cell
    rel_x = x_coords - agent_x_expanded
    rel_y = y_coords - agent_y_expanded

    # Dot product to determine if cells are in front of the agent
    dot_product = dx * rel_x + dy * rel_y
    front_masks = (dot_product > 0).astype(int)

    # Set cells behind the agent to the user-defined value
    front_masks[dot_product <= 0] = behind_value

    return front_masks

def determine_slices_from_vectors(vectors, num_slices=4):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    normalized_angles = np.mod(angles, 2*np.pi)
    slice_width = 2*np.pi / num_slices

    slice_indices = (normalized_angles // slice_width).astype(int)
    return slice_indices

def determine_slices_from_headings(headings, num_slices=4):
    normalized_headings = np.mod(headings, 2*np.pi)
    slice_width = 2*np.pi / num_slices

    slice_indices = (normalized_headings // slice_width).astype(int)
    return slice_indices

def output_excel(records, model_dir, model_name):
    """
    Export the records of PID optimization errors and rankings to an excel file.
    
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
    
def movie_maker(directory, model_name, crs, dt, depth_rast_transform):
    # connect to model
    model_directory = os.path.join(directory,'%s.h5'%(model_name))
    hdf5 = h5py.File(model_directory, 'r')
    X_arr = hdf5['agent_data/X'][:]
    Y_arr = hdf5['agent_data/Y'][:]
    
    # calculate important things, like the number of columns which should equal the number of timesteps
    shape = X_arr.shape

    # Number of columns is the second element of the 'shape' tuple
    num_columns = shape[1]

    # get depth raster
    depth_arr = hdf5['environment/depth'][:]
    depth = rasterio.MemoryFile()
    height = depth_arr.shape[0]
    width = depth_arr.shape[1]
    
    with depth.open(
        driver ='GTiff',
        height = depth_arr.shape[0],
        width = depth_arr.shape[1],
        count =1,
        dtype ='float32',
        crs = crs,
        transform = depth_rast_transform
    ) as dataset:
        dataset.write(depth_arr, 1)

        # define metadata for movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title= model_name, artist='Matplotlib',
                        comment='emergent model run %s'%(datetime.now()))
        writer = FFMpegWriter(fps = np.round(30/dt,0), metadata=metadata)

        #initialize plot
        fig, ax = plt.subplots(figsize = (10,5))

        background = ax.imshow(dataset.read(1),
                               origin = 'upper',
                               extent = [dataset.bounds[0],
                                          dataset.bounds[2],
                                          dataset.bounds[1],
                                          dataset.bounds[3]])

        agent_pts, = plt.plot([], [], marker = 'o', ms = 1, ls = '', color = 'red')

        plt.xlabel('Easting')
        plt.ylabel('Northing')

        # Update the frames for the movie
        with writer.saving(fig, 
                           os.path.join(directory,'%s.mp4'%(model_name)), 
                           dpi = 300):

            for i in range(int(num_columns)):


                # write frame
                agent_pts.set_data(X_arr[:, i],
                                   Y_arr[:, i])
                writer.grab_frame()
                    
                print ('Time Step %s complete'%(i))

    # clean up
    writer.finish()
    
def HECRAS(model_dir, HECRAS_dir, resolution, crs, timestep_range = None):
    """
    Import environment data from a HECRAS model, process selected timesteps, 
    and store results in a new HDF5 file.
    
    Parameters:
    - model_dir (str): Directory to save the output files.
    - HECRAS_dir (str): Path to the HECRAS model in HDF format.
    - resolution (float): Desired resolution for the interpolated rasters.
    - crs (str): Coordinate reference system for the geospatial data.
    - timestep_range (list or None): List of timesteps to process. If None, only the last timestep is processed.
    - output_hdf_path (str): Path to the output HDF5 file where processed data is stored.

    """
    hdf = h5py.File(HECRAS_dir, 'r')
    pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))

    # If no timestep range is provided, default to the last timestep
    if timestep_range is None:
        timestep_range = [-1]

    with h5py.File(model_dir, 'a') as output_hdf:
        for timestep in timestep_range:
            print(f"Processing timestep: {timestep}")
            
            vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][timestep]
            vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][timestep]
            wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][timestep]

            geom = list(tuple(zip(pts[:, 0], pts[:, 1])))
            df = pd.DataFrame.from_dict({
                'index': np.arange(0, len(pts), 1),
                'geom_tup': geom,
                'vel_x': vel_x,
                'vel_y': vel_y,
                'wsel': wsel,
                'elev': elev
            })
            df['geometry'] = df.geom_tup.apply(Point)
            gdf = gpd.GeoDataFrame(df, crs=crs)

            vel_x_interp = LinearNDInterpolator(pts, gdf.vel_x)
            vel_y_interp = LinearNDInterpolator(pts, gdf.vel_y)
            wsel_interp = LinearNDInterpolator(pts, gdf.wsel)
            elev_interp = LinearNDInterpolator(pts, gdf.elev)

            xmin = np.min(pts[:, 0])
            xmax = np.max(pts[:, 0])
            ymin = np.min(pts[:, 1])
            ymax = np.max(pts[:, 1])

            xint = np.arange(xmin, xmax, resolution)
            yint = np.arange(ymax, ymin, resolution * -1.)
            xnew, ynew = np.meshgrid(xint, yint, sparse=True)

            vel_x_new = vel_x_interp(xnew, ynew)
            vel_y_new = vel_y_interp(xnew, ynew)
            wsel_new = wsel_interp(xnew, ynew)
            elev_new = elev_interp(xnew, ynew)

            depth = wsel_new - elev_new
            vel_mag = np.sqrt(np.power(vel_x_new, 2) + np.power(vel_y_new, 2))
            vel_dir = np.arctan2(vel_y_new, vel_x_new)

            # Store the results in the output HDF5 file
            timestep_key = f'timestep_{timestep}'

            output_hdf.create_dataset(f'{timestep_key}/elev', data=elev_new)
            output_hdf.create_dataset(f'{timestep_key}/wsel', data=wsel_new)
            output_hdf.create_dataset(f'{timestep_key}/depth', data=depth)
            output_hdf.create_dataset(f'{timestep_key}/vel_mag', data=vel_mag)
            output_hdf.create_dataset(f'{timestep_key}/vel_dir', data=vel_dir)
            output_hdf.create_dataset(f'{timestep_key}/vel_x', data=vel_x_new)
            output_hdf.create_dataset(f'{timestep_key}/vel_y', data=vel_y_new)

    hdf.close()
    print(f"Finished processing and saving timesteps to {HECRAS_dir}")

class PID_controller:
    def __init__(self, n_agents, k_p = 0., k_i = 0., k_d = 0., tau_d = 1):
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        self.integral = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.previous_error = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.derivative_filtered = np.zeros((np.round(n_agents,0).astype(np.int32),2))

    def update(self, error, dt, status):
        # create a mask - if this fish is fatigued, this doesn't matter
        mask = np.where(status == 3,True,False)
        
        self.integral = np.where(~mask, self.integral + error, self.integral)
        derivative = error - self.previous_error
        self.previous_error = error
    
        p_term = self.k_p[:, np.newaxis] * error
        i_term = self.k_i[:, np.newaxis] * self.integral
        d_term = self.k_d[:, np.newaxis] * derivative
        
        # # calculate unsaturated output
        # unsaturated_output = p_term + i_term + d_term

        # # Apply limits
        # max_limit = [[20.,20.]]
        # min_limit = [[-20.,-20.]]
        
        # actual_output = np.clip(unsaturated_output, min_limit, max_limit)
        
        # # Back-calculation if necessary
        # if np.any(actual_output != unsaturated_output):
        #     excess = unsaturated_output - actual_output
        #     excess_mask = np.where(excess != 0., True, False)
        #     integral_adjustment = np.where(~mask,
        #                                    excess / self.k_i[:, np.newaxis],
        #                                    [[0., 0.]])
            
        #     print ('initial integral \n %s'%(self.integral))
            
        #     self.integral = np.where(~excess_mask,
        #                              self.integral + integral_adjustment,
        #                              [[0., 0.]])
            
        #     i_term = self.k_i[:, np.newaxis] * self.integral
        #     print ('i term: \n %s'%(i_term))
        #     print ('new integral \n %s'%(self.integral))
            
        
        # # Apply low-pass filter to derivative
        # self.derivative_filtered += (dt / (self.tau_d + dt)) * (derivative - self.derivative_filtered)
    
        # # Update for next iteration
        # self.previous_measurement = error
    
        # # Calculate D term with filtered derivative
        # d_term = -self.k_d[:, np.newaxis] * self.derivative_filtered
    
        return np.where(~mask,p_term + i_term + d_term,0.0) #+ i_term + d_term
        
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
        # P plane parameters
        a_P = self.P_params[0]
        b_P = self.P_params[1]
        c_P = self.P_params[2]
        
        # I plane parameters
        a_I = self.I_params[0]
        b_I = self.I_params[1]
        c_I = self.I_params[2]
        
        # D plane parameters
        a_D = self.D_params[0]
        b_D = self.D_params[1]
        c_D = self.D_params[2]        
        
        P = a_P * length + b_P * velocity + c_P
        I = a_I * length + b_I * velocity + c_I
        D = a_D * length + b_D * velocity + c_D 
        
        return P, I, D
    
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
      
class simulation():
    '''Python class object that implements an Agent Based Model of adult upstream
    migrating sockeye salmon through a modeled riffle cascade complex.  Rather
    than traditional OOP architecture, this version of the software employs a 
    Structure of Arrays data management philosophy - in other words - we processin
    on a GPU now'''
    
    def __init__(self, 
                 model_dir, 
                 model_name, 
                 crs, 
                 basin, 
                 water_temp, 
                 start_polygon,
                 env_files,
                 longitudinal_profile,
                 fish_length = None,
                 num_timesteps = 100, 
                 num_agents = 100,
                 use_gpu = False,
                 pid_tuning = False):
        """
         Initialize the simulation environment.
         
         Parameters:
         - model_dir (str): Directory where the model data will be stored.
         - model_name (str): Name of the model.
         - crs (str): Coordinate reference system for the model.
         - basin (str): Name or identifier of the basin.
         - water_temp (float): Water temperature in degrees Celsius.
         - starting_box (tuple): A tuple of four values (xmin, xmax, ymin, ymax) defining the bounding box 
                                 where agents start.
         - env_files (dict): A dictionary of file names that make up the agent based model
                             environment.  Must inlcude keys: wsel (water surface elevation), 
                                                             depth, 
                                                             elev (elevation),
                                                             x_vel (water velocity x direction),
                                                             y_vel (water velocity y direction),
                                                             vel_dir (water direction in radians),
                                                             vel_mag (water velocity magnitude)
         - num_timesteps (int, optional): Number of timesteps for the simulation. Defaults to 100.
         - num_agents (int, optional): Number of agents in the simulation. Defaults to 100.
         - use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
         
         Attributes:
         - arr (module): Module used for array operations (either numpy or cupy).
         - model_dir, model_name, crs, basin, etc.: Various parameters and states for the simulation.
         - X, Y, prev_X, prev_Y: Arrays representing the X and Y coordinates of agents.
         - drag, thrust, Hz, etc.: Arrays representing various movement parameters of agents.
         - kcal: Array representing the kilocalorie counter for each agent.
         - hdf5 (h5py.File): HDF5 file object for storing simulation data.
         
         Notes:
         - The method initializes various agent properties and states.
         - It creates a new HDF5 file for the project and writes initial data to it.
         - Agent properties that do not change with time are written to the HDF5 file.
        """        
        self.arr = get_arr(use_gpu)
        
        # If we are tuning the PID controller, special settings used
        if pid_tuning:
            self.pid_tuning = pid_tuning
            self.vel_x_array = np.array([])
            self.vel_y_array = np.array([])
            
        else:
            self.pid_tuning = False
        
        # model directory and model name
        self.model_dir = model_dir
        self.model_name = model_name
        self.db = os.path.join(self.model_dir,'%s.h5'%(self.model_name))
                
        # coordinate reference system for the model
        self.crs = crs
        self.basin = basin
        
        # model parameters
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.water_temp = water_temp
        
        # initialize agent properties and internal states
        self.sim_sex()
        self.sim_length(fish_length)
        self.sim_weight()
        self.sim_body_depth()
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/recovery.csv')
        recover = pd.read_csv(data_dir)        
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds, recover.Recovery, extrapolate = True,)
        del recover
        self.swim_behav = self.arr.repeat(1, num_agents)               # 1 = migratory , 2 = refugia, 3 = station holding
        self.swim_mode = self.arr.repeat(1, num_agents)      # 1 = sustained, 2 = prolonged, 3 = sprint
        self.battery = self.arr.repeat(1.0, num_agents)
        self.recover_stopwatch = self.arr.repeat(0.0, num_agents)
        self.ttfr = self.arr.repeat(0.0, num_agents)
        self.time_out_of_water = self.arr.repeat(0.0, num_agents)
        self.time_of_abandon = self.arr.repeat(0.0, num_agents)
        self.time_since_abandon = self.arr.repeat(0.0, num_agents)
        self.dead = np.zeros(num_agents)
        
        # create initial positions
        gdf = gpd.read_file(start_polygon)

        # Get the geometry of the shapefile
        geometry = gdf.geometry.unary_union
        
        minx, miny, maxx, maxy = geometry.bounds
        X = []
        Y = []

        while len(X) <= self.num_agents:
            random_points = np.random.uniform([minx, miny], [maxx, maxy], size=(self.num_agents*2, 2))
            for x, y in random_points:
                pnt = Point(x, y)
                if geometry.contains(pnt):
                    X.append(x)
                    Y.append(y)

        self.X = np.array(X)[:self.num_agents]
        self.Y = np.array(Y)[:self.num_agents]
            
        self.prev_X = self.X
        self.prev_Y = self.Y
        
        # create short term memory for eddy escpement 
        max_timesteps = 600  # Maximum number of timesteps to track

        self.swim_speeds = np.full((num_agents, max_timesteps), np.nan)
        self.past_longitudes = np.full((num_agents, max_timesteps), np.nan)
        self.current_longitudes = np.zeros_like(self.X)

        self.in_eddy = np.zeros_like(self.X)
        self.time_since_eddy_escape = np.zeros_like(self.X)
        self.max_eddy_escape_seconds = 45.
        
        # Time to Fatigue values for Sockeye digitized from Bret 1964
        #TODO - we need to scale these numbers by size, way too big for tiny fish
        adult_slope_adjustment = 0.2 # 0.5 or 0.1
        adult_intercept_adjustment = 1.5 # 1.5 or 2.1
        prolonged_swim_speed_adjustment = 2.1
        self.max_s_U = 2.77      # maximum sustained swim speed in bl/s
        self.max_p_U = 4.43 + prolonged_swim_speed_adjustment  # maximum prolonged swim speed
        self.a_p = 8.643 + adult_intercept_adjustment   # prolonged intercept
        self.b_p = -2.0894 * adult_slope_adjustment  # prolonged slope
        self.a_s = 0.1746  + adult_intercept_adjustment    # sprint intercept
        self.b_s = -0.1806 * adult_slope_adjustment   # sprint slope
        
        # initialize movement parameters
        self.drag = self.arr.zeros(num_agents)           # computed theoretical drag
        self.thrust = self.arr.zeros(num_agents)         # computed theoretical thrust Lighthill 
        self.Hz = self.arr.zeros(num_agents)             # tail beats per second
        self.bout_no = self.arr.zeros(num_agents)        # bout number - new bout whenever fish recovers
        self.dist_per_bout = self.arr.zeros(num_agents)  # running counter of the distance travelled per bout
        self.bout_dur = self.arr.zeros(num_agents)       # running bout timer 
        self.time_of_jump = self.arr.zeros(num_agents)   # time since last jump - can't happen every timestep
        
        # initialize odometer
        self.kcal = self.arr.zeros(num_agents)           #kilo calorie counter
    
        # create a project database and write initial arrays to HDF
        self.hdf5 = h5py.File(self.db, 'w')
        self.initialize_hdf5()
            
        # write agent properties that do not change with time
        self.hdf5["agent_data/sex"][:] = self.sex
        self.hdf5["agent_data/length"][:] = self.length
        self.hdf5["agent_data/ucrit"][:] = self.ucrit
        self.hdf5["agent_data/weight"][:] = self.weight
        self.hdf5["agent_data/body_depth"][:] = self.body_depth
        self.hdf5["agent_data/too_shallow"][:] = self.too_shallow
        self.hdf5["agent_data/opt_wat_depth"][:] = self.sex
        
        # import environment
        self.enviro_import(os.path.join(model_dir,env_files['x_vel']),'velocity x')
        self.enviro_import(os.path.join(model_dir,env_files['y_vel']),'velocity y')
        self.enviro_import(os.path.join(model_dir,env_files['depth']),'depth')
        self.enviro_import(os.path.join(model_dir,env_files['wsel']),'wsel')
        self.enviro_import(os.path.join(model_dir,env_files['elev']),'elevation')
        self.enviro_import(os.path.join(model_dir,env_files['vel_dir']),'velocity direction')
        self.enviro_import(os.path.join(model_dir,env_files['vel_mag']),'velocity magnitude') 
        self.enviro_import(os.path.join(model_dir,env_files['wetted']),'wetted')
        self.hdf5.flush()
        
        # import longitudinal shapefile
        self.longitude = self.longitudinal_import(longitudinal_profile)

        # boundary_surface
        self.boundary_surface()

        # initialize mental maps
        self.avoid_cell_size = 10.
        self.initialize_mental_map()
        self.refugia_cell_size = 1.
        self.initialize_refugia_map()
        
        # initialize heading
        self.initial_heading()
        
        # initialize swim speed
        self.initial_swim_speed() 
        
        # error array
        self.error_array = np.array([])
        
        # initialize cumulative time
        self.cumulative_time = 0.

    def sim_sex(self):
        """
        Simulate the sex distribution of agents based on the basin.
    
        Notes:
        - The method sets the `sex` attribute of the class, which is an array representing the sex of each agent.
        - Currently, the method only has data for the "Nushagak River" basin. For this basin, the sex distribution 
          is determined based on given probabilities for male (0) and female (1).
        - If the basin is not "Nushagak River", the method does not modify the `sex` attribute.
    
        Attributes set:
        - sex (array): Array of size `num_agents` with values 0 (male) or 1 (female) representing the sex of each agent.
        """
        if self.basin == "Nushagak River":
            self.sex = self.arr.random.choice([0,1], size = self.num_agents, p = [0.503,0.497])
            
    def sim_length(self, fish_length = None):
        """
        Simulate the length distribution of agents based on the basin and their sex.
    
        Notes:
        - The method sets the `length` attribute of the class, which is an array representing the length of each agent in mm.
        - Currently, the method only has data for the "Nushagak River" basin. For this basin, the length distribution 
          is determined based on given lognormal distributions for male and female agents.
        - The method also sets other attributes that are functions of the agent's length.
    
        Attributes set:
        - length (array): Array of size `num_agents` representing the length of each agent in mm.
        - sog (array): Speed over ground for each agent, assumed to be 1 body length per second.
        - ideal_sog (array): Ideal speed over ground for each agent, set to be the same as `sog`.
        - swim_speed (array): Initial swim speed for each agent, set to be the same as `length/1000`.
        - ucrit (array): Critical swimming speed for each agent, set to be `sog * 7`.
    
        Raises:
        - ValueError: If the `sex` attribute is not recognized.
        """
        # length in mm
        if self.pid_tuning == True:
            self.length = np.repeat(fish_length,self.num_agents) # testing
        
        else:
            if self.basin == "Nushagak River":
                self.length=np.where(self.sex=='M',
                         self.arr.random.lognormal(mean = 6.426,sigma = 0.072,size = self.num_agents),
                         self.arr.random.lognormal(mean = 6.349,sigma = 0.067,size = self.num_agents))
                # if self.sex == 'M':
                #     self.length = self.arr.random.lognormal(mean = 6.426,sigma = 0.072,size = self.num_agents)
                # else:
                #     self.length = self.arr.random.lognormal(mean = 6.349,sigma = 0.067,size = self.num_agents)
        
        # we can also set these arrays that contain parameters that are a function of length
        self.length = np.where(self.length < 475.,475.,self.length)
        self.sog = self.length/1000. #* 0.8  # sog = speed over ground - assume fish maintain 1 body length per second
        self.ideal_sog = self.length/1000. # self.sog
        self.opt_sog = self.length/1000. #* 0.8
        self.school_sog = self.length/1000.
       # self.swim_speed = self.length/1000.        # set initial swim speed
        self.ucrit = self.sog * 1.6    # TODO - what is the ucrit for sockeye?
        
    def sim_weight(self):
        '''function simulates a fish weight out of the user provided basin and 
        sex of fish'''
        # using a W = a * L^b relationship given in fishbase - weight in kg
        self.weight = (0.0155 * (self.length/10.0)**3)/1000.
        
    def sim_body_depth(self):
        '''function simulates a fish body depth out of the user provided basin and 
        sex of fish'''
        # body depth is in cm
        if self.basin == "Nushagak River":
            self.body_depth=np.where(self.sex=='M',
                        self.arr.exp(-1.938 + np.log(self.length) * 1.084 + 0.0435) / 10.,
                        self.arr.exp(-1.938 + np.log(self.length) * 1.084) / 10.)
            # if self.sex == 'M':
            #     self.body_depth = self.arr.exp(-1.938 + np.log(self.length) * 1.084 + 0.0435) / 10.
            # else:
            #     self.body_depth = self.arr.exp(-1.938 + np.log(self.length) * 1.084) / 10.
                
        self.too_shallow = self.body_depth /100. / 2. # m
        self.opt_wat_depth = self.body_depth /100 * 3.0 + self.too_shallow
        
    def initialize_hdf5(self):
        '''Initialize an HDF5 database for a simulation'''
        # Create groups for organization (optional)
        agent_data = self.hdf5.create_group("agent_data")
        
        # Create datasets for agent properties that are static
        agent_data.create_dataset("sex", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("length", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("ucrit", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("weight", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("body_depth", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("too_shallow", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("opt_wat_depth", (self.num_agents,), dtype='f4')
      
        # Create datasets for agent properties that change with time
        agent_data.create_dataset("X", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("Y", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("Z", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("prev_X", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("prev_Y", (self.num_agents, self.num_timesteps), dtype='f4')            
        agent_data.create_dataset("heading", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("sog", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("ideal_sog", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("swim_speed", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("battery", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("swim_behav", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("swim_mode", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("recover_stopwatch", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("ttfr", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("time_out_of_water", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("drag", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("thrust", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("Hz", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("bout_no", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("dist_per_bout", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("bout_dur", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("time_of_jump", (self.num_agents, self.num_timesteps), dtype='f4')
        agent_data.create_dataset("kcal", (self.num_agents, self.num_timesteps), dtype='f4')
        
        # Set attributes (metadata) if needed
        self.hdf5.attrs['simulation_name'] = "%s Sockeye Movement Simulation"%(self.basin)
        self.hdf5.attrs['num_agents'] = self.num_agents 
        self.hdf5.attrs['num_timesteps'] = self.num_timesteps
        self.hdf5.attrs['basin'] = self.basin
        self.hdf5.attrs['crs'] = self.crs
        
        self.hdf5.flush()
        
    def timestep_flush(self, timestep):
        if self.pid_tuning == False:
            '''function writes to the open hdf5 file '''
            
            # write time step data to hdf
            self.hdf5['agent_data/X'][..., timestep] = self.X.astype('float32')
            self.hdf5['agent_data/Y'][..., timestep] = self.Y.astype('float32')
            self.hdf5['agent_data/Z'][..., timestep] = self.z.astype('float32')
            self.hdf5['agent_data/prev_X'][..., timestep] = self.prev_X.astype('float32')
            self.hdf5['agent_data/prev_Y'][..., timestep] = self.prev_Y.astype('float32')
            self.hdf5['agent_data/heading'][..., timestep] = self.heading.astype('float32')
            self.hdf5['agent_data/sog'][..., timestep] = self.sog.astype('float32')
            self.hdf5['agent_data/ideal_sog'][..., timestep] = self.ideal_sog.astype('float32')
            self.hdf5['agent_data/swim_speed'][..., timestep] = self.swim_speed.astype('float32')
            self.hdf5['agent_data/battery'][..., timestep] = self.battery.astype('float32')
            self.hdf5['agent_data/swim_behav'][..., timestep] = self.swim_behav.astype('float32')
            self.hdf5['agent_data/swim_mode'][..., timestep] = self.swim_mode.astype('float32')
            self.hdf5['agent_data/recover_stopwatch'][..., timestep] = self.recover_stopwatch.astype('float32')
            self.hdf5['agent_data/ttfr'][..., timestep] = self.ttfr.astype('float32')
            self.hdf5['agent_data/time_out_of_water'][..., timestep] = self.time_out_of_water.astype('float32')
            self.hdf5['agent_data/drag'][..., timestep] = np.linalg.norm(self.drag, axis = -1).astype('float32')
            self.hdf5['agent_data/thrust'][..., timestep] = np.linalg.norm(self.thrust, axis = -1).astype('float32')
            self.hdf5['agent_data/Hz'][..., timestep] = self.Hz.astype('float32')
            self.hdf5['agent_data/bout_no'][..., timestep] = self.bout_no.astype('float32')
            self.hdf5['agent_data/dist_per_bout'][..., timestep] = self.dist_per_bout.astype('float32')
            self.hdf5['agent_data/bout_dur'][..., timestep] = self.bout_dur.astype('float32')
            self.hdf5['agent_data/kcal'][..., timestep] = self.kcal.astype('float32')

            #self.hdf5['agent_data/time_of_jump'][..., timestep] = self.time_of_jump.astype('float32')
    
    
            # # Periodically flush data to ensure it's written to disk
            if timestep % 100 == 0:  # Adjust this value based on your needs
                self.hdf5.flush()            

    def enviro_import(self, data_dir, surface_type):
        """
        Imports environmental raster data and writes it to an HDF5 file.
    
        Parameters:
        - data_dir (str): Path to the raster file to be imported.
        - surface_type (str): Type of the surface data being imported. 
                              Acceptable values include 'velocity x', 'velocity y', 'depth', 
                              'wsel', 'elevation', 'velocity direction', 'velocity magnitude'.
    
        Attributes set:
        - vel_x_rast_transform, vel_y_rast_transform, depth_rast_transform, etc. (Affine): 
          Transformation matrix for the respective raster data.
        - width (int): Width of the raster data.
        - height (int): Height of the raster data.
    
        Notes:
        - The method creates a group named "environment" in the HDF5 file to organize the raster datasets.
        - The raster data is read using rasterio and written to the HDF5 file under the "environment" group.
        - The transformation matrix for each raster type is stored as an attribute of the Simulation class.
        - The method sets the width and height attributes of the Simulation class based on the raster dimensions.
    
        Raises:
        - ValueError: If the provided surface_type is not recognized.
        """
       

        
        # get raster properties
        src = rasterio.open(data_dir)
        num_bands = src.count
        width = src.width
        height = src.height
        dtype = np.float32
        transform = src.transform
        self.no_data_value = src.nodatavals[0]
        
        # Create groups for organization (optional)
        if 'environment' not in self.hdf5:
            env_data = self.hdf5.create_group("environment")
            self.width = width
            self.height = height

            # Get the dimensions of the raster
            rows, cols = src.shape
        
            # Define chunk size (you can adjust this based on your memory constraints)
            chunk_size = 1024  # Example chunk size
        
            # Set up HDF5 file and datasets
            dset_x = self.hdf5.create_dataset('x_coords', (height, width), dtype='float32')
            dset_y = self.hdf5.create_dataset('y_coords', (height, width), dtype='float32')
        
            # Process and write in chunks
            for i in range(0, rows, chunk_size):
                row_chunk = slice(i, min(i + chunk_size, rows))
                row_indices, col_indices = np.meshgrid(np.arange(row_chunk.start, row_chunk.stop), np.arange(cols), indexing='ij')
    
                # Apply the affine transformation
                x_coords, y_coords = transform * (col_indices, row_indices)
    
                # Write the chunk to the HDF5 datasets
                dset_x[row_chunk, :] = x_coords
                dset_y[row_chunk, :] = y_coords
        else:
            env_data = self.hdf5['environment']

        shape = (num_bands, height, width)
        #shape = (num_bands, width, height)
        
        if surface_type == 'wetted':
            # set transform as parameter of simulation
            self.wetted_transform = transform
                 
            # get data 
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("wetted", (height, width), dtype='f4')
            self.hdf5['environment/wetted'][:, :] = arr
            
        elif surface_type == 'velocity x':
            # set transform as parameter of simulation
            self.vel_x_rast_transform = transform
            
            # get data 
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("vel_x", (height, width), dtype='f4')
            self.hdf5['environment/vel_x'][:, :] = arr

        elif surface_type == 'velocity y':
            # set transform as parameter of simulation            
            self.vel_y_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("vel_y", (height, width), dtype='f4')
            self.hdf5['environment/vel_y'][:, :] = arr
            
        elif surface_type == 'depth':
            # set transform as parameter of simulation            
            self.depth_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)
           
            # create an hdf5 array and write to it
            env_data.create_dataset("depth", (height, width), dtype='f4')
            self.hdf5['environment/depth'][:, :] =arr
            
        elif surface_type == 'wsel':
            # set transform as parameter of simulation            
            self.wsel_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("wsel", (height, width), dtype='f4')
            self.hdf5['environment/wsel'][:, :] = src.read(1)
            
        elif surface_type == 'elevation':
            # set transform as parameter of simulation                        
            self.elev_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("elevation", (height, width), dtype='f4')#, data = src.read(1))
            self.hdf5['environment/elevation'][:, :] = arr
                
        elif surface_type == 'velocity direction':          
            # set transform as parameter of simulation                        
            self.vel_dir_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            env_data.create_dataset("vel_dir", (height, width), dtype='f4')#, data = src.read(1))
            self.hdf5['environment/vel_dir'][:, :] = src.read(1) 
                
        elif surface_type == 'velocity magnitude': 
            # set transform as parameter of simulation                        
            self.vel_mag_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)
            
            # create an hdf5 array and write to it
            env_data.create_dataset("vel_mag", (height, width), dtype='f4')#, data = src.read(1))
            self.hdf5['environment/vel_mag'][:, :] = arr
            
        self.width = width
        self.height = height
        self.hdf5.flush()
        src.close()

    def longitudinal_import(self, shapefile):
        # Load the shapefile with the longitudinal line
        line_gdf = gpd.read_file(shapefile)
        self.longitudinal = line_gdf.geometry[0]  # Assuming there's only one line feature
        
    def compute_linear_positions(self, line):
        # Assuming you have numpy arrays `x` and `y` for the coordinates of agents
        points = np.array([Point(x, y) for x, y in zip(self.X, self.Y)])
        '''Vectorized function to compute linear distance along the longitudinal line'''
        return np.array([line.project(point) for point in points])

    def boundary_surface(self):
        
        raster = self.hdf5['environment/wetted'][:]  # Adjust the path as needed

        pixel_width = self.depth_rast_transform[0]
        
        # Compute the Euclidean distance transform. This computes the distance to the nearest zero (background) for all non-zero (foreground) pixels.
        dist_to_bound = distance_transform_edt(raster != -9999) * pixel_width
        
        # Create or access 'environment' group
        if 'environment' not in self.hdf5:
            env_data = self.hdf5.create_group('environment')
        else:
            env_data = self.hdf5['environment']
        
        # Create 'distance_to' dataset and write data
        if 'distance_to' not in env_data:
            env_data.create_dataset('distance_to', (self.height, self.width), dtype='float32')
        env_data['distance_to'][:, :] = dist_to_bound  # Corrected dataset name
        
        self.hdf5.flush()  # Write changes to HDF5 file

    def initialize_mental_map(self):
        """
        Initializes the mental map for each agent.
        
        The mental map is a 3D array where each row corresponds to an agent, 
        and each agent's row is a 2D raster of shape (self.width, self.height).
        The values in the raster can represent various things, such as visited locations, 
        perceived obstacles, etc. Initially, all values are set to zero.
        
        Attributes:
            map (ndarray): A 3D array representing the mental maps of all agents.
                           Shape: (self.num_agents, self.width, self.height)
        """

        # Create groups for organization (optional)
        mem_data = self.hdf5.create_group("memory")
        avoid_height = np.round(self.height/self.avoid_cell_size,0).astype(np.int32) + 1
        avoid_width = np.round(self.width/self.avoid_cell_size,0).astype(np.int32) + 1
        # create a memory map array
        for i in np.arange(self.num_agents):
            mem_data.create_dataset('%s'%(i), (avoid_height, avoid_width), dtype = 'f4')
            self.hdf5['memory/%s'%(i)][:, :] = self.arr.zeros((avoid_height, 
                                                               avoid_width))

        # Apply the scaling
        self.mental_map_transform = Affine(self.avoid_cell_size, 
                                           self.depth_rast_transform.b, 
                                           self.depth_rast_transform.c,
                                           self.depth_rast_transform.d,
                                           -self.avoid_cell_size,
                                           self.depth_rast_transform.f)
           
        self.hdf5.flush()
        
    def initialize_refugia_map(self):
        """
        Initializes the velocity map for each agent.
        
        The velocity map is a 3D array where each row corresponds to an agent, 
        and each agent's row is a 2D raster of shape (self.width, self.height).
        The values in the raster represent veloicty refugia.
        
        Attributes:
            map (ndarray): A 3D array representing the mental maps of all agents.
                           Shape: (self.num_agents, self.width, self.height)
        """

        # Create groups for organization (optional)
        mem_data = self.hdf5.create_group("refugia")
        refugia_height = np.round(self.height/self.refugia_cell_size,0).astype(np.int32) + 1
        refugia_width = np.round(self.width/self.refugia_cell_size,0).astype(np.int32) + 1
        # create a memory map array
        for i in np.arange(self.num_agents):
            mem_data.create_dataset('%s'%(i), (refugia_height, refugia_width), dtype = 'f4')
            self.hdf5['refugia/%s'%(i)][:, :] = self.arr.zeros((refugia_height, 
                                                               refugia_width))

        # Apply the scaling
        self.refugia_map_transform = Affine(self.refugia_cell_size, 
                                           self.depth_rast_transform.b, 
                                           self.depth_rast_transform.c,
                                           self.depth_rast_transform.d,
                                           -self.refugia_cell_size,
                                           self.depth_rast_transform.f)
           
        self.hdf5.flush()
 
    def sample_environment(self, transform, raster_name):
        """
        Sample the raster values at the given x, y coordinates using an open HDF5 file.

        Parameters:
        - x_coords: array of x coordinates
        - y_coords: array of y coordinates
        - raster_name: name of the raster dataset in the HDF5 file

        Returns:
        - values: array of sampled raster values
        """
        # Get the row, col indices for the coordinates
        rows, cols = geo_to_pixel(self.X, self.Y, transform)

        # Use the already open HDF5 file object to read the specified raster dataset
        raster_dataset = self.hdf5['environment/%s'%(raster_name)][:]  # Adjust the path as needed

        # Sample the raster values using the row, col indices
        # Ensure that the indices are within the bounds of the raster data
        rows = np.clip(rows, 0, raster_dataset.shape[0] - 1)
        cols = np.clip(cols, 0, raster_dataset.shape[1] - 1)
        
        if self.num_agents > 1:
            values = raster_dataset[rows, cols]
        else:
            values = np.array([raster_dataset[rows, cols]])
        
        #self.hdf5['environment/%s'%(raster_name)] = raster_dataset
        self.hdf5.flush()
        
        return values.flatten()  
    
    def initial_swim_speed(self):
        """
        Calculates the initial swim speed required for each fish to overcome
        current water velocities and maintain their ideal speed over ground (SOG).
    
        Attributes:
        x_vel, y_vel (array): Water velocity components in m/s for each fish.
        ideal_sog (array): Ideal speed over ground in m/s for each fish.
        heading (array): Heading in radians for each fish.
        swim_speed (array): Calculated swim speed for each fish to maintain ideal SOG.
    
        Notes:
        - The function assumes that the input attributes are structured as arrays of
          values, with each index across the arrays corresponding to a different fish.
        - The function updates the swim_speed attribute for each fish based on the
          calculated swim speed necessary to maintain the ideal SOG against water currents.
        """

        self.x_vel = self.sample_environment(self.vel_x_rast_transform, 'vel_x')
        self.y_vel = self.sample_environment(self.vel_y_rast_transform, 'vel_y')
        
        # Vector components of water velocity for each fish
        water_velocities = np.sqrt(self.x_vel**2 + self.y_vel**2)
    
        # Vector components of ideal velocity for each fish
        ideal_velocities = np.stack((self.ideal_sog * np.cos(self.heading),
                                     self.ideal_sog * np.sin(self.heading)), axis=-1)
    
        # Calculate swim speed for each fish
        # Subtracting the scalar water velocity from the vector ideal velocity
        # requires broadcasting the water_velocities array to match the shape of ideal_velocities
        self.swim_speed = np.linalg.norm(ideal_velocities - np.array([self.x_vel,self.y_vel]).T, axis = -1)
                

    def initial_heading (self):
        """
        Calculate the initial heading for each agent based on the velocity direction raster.
    
        This function performs the following steps:
        - Converts the geographic coordinates of each agent to pixel coordinates.
        - Samples the environment to get the velocity direction at each agent's location.
        - Adjusts the heading based on the flow direction, ensuring it is within the range [0, 2π).
        - Calculates the maximum practical speed over ground (SOG) for each agent based on their heading and SOG.
    
        Attributes updated:
        - self.heading: The heading for each agent in radians.
        - self.max_practical_sog: The maximum practical speed over ground for each agent as a 2D vector (m/s).
        """
        # get the x, y position of the agent 
        row, col = geo_to_pixel(self.X, self.Y, self.vel_dir_rast_transform)
            
        # get the initial heading values
        values = self.sample_environment(self.vel_dir_rast_transform,'vel_dir')
        
        # set direction 
        self.heading = self.arr.where(values < 0, 
                                               (self.arr.radians(360) + values) - self.arr.radians(180), 
                                               values - self.arr.radians(180))

        # set initial max practical speed over ground as well
        self.max_practical_sog = self.arr.array([self.sog * self.arr.cos(self.heading), 
                                                          self.sog * self.arr.sin(self.heading)]) #meters/sec       

    def update_mental_map(self, current_timestep):
        """
        Update the mental map for each agent in the HDF5 dataset.
    
        This function performs the following steps:
        - Converts the geographic coordinates (X and Y) of each agent to pixel coordinates.
        - Updates the mental map for each agent in the HDF5 dataset at the corresponding pixel location with the current timestep.
    
        Parameters:
        - current_timestep: The current timestep to record in the mental map.
    
        The mental map is stored in an HDF5 dataset with shape (num_agents, width, height), where each 'slice' corresponds to an agent's mental map.
        """
        # Convert geographic coordinates to pixel coordinates for each agent
        rows, cols = geo_to_pixel(self.X, self.Y, self.mental_map_transform)
    
        # Ensure rows and cols are within the bounds of the mental map
        rows = self.arr.clip(rows, 0, self.height - 1)
        cols = self.arr.clip(cols, 0, self.width - 1)

        # get velocity and coords raster per agent
        for i in np.arange(self.num_agents):
            if self.num_agents > 1:
                try:
                    self.hdf5['memory/%s'%(i)][rows[i],cols[i]] = current_timestep
                except:
                    pass
            else:
                single_arr = np.array([self.hdf5['memory/%s'%(i)]])
                single_arr[0,rows,cols] = current_timestep
                self.hdf5['memory/%s'%(i)][:, :] = single_arr
        
        self.hdf5.flush()

    def update_refugia_map(self, current_velocity):
        """
        Update the mental map for each agent in the HDF5 dataset.
    
        This function performs the following steps:
        - Converts the geographic coordinates (X and Y) of each agent to pixel coordinates.
        - Updates the mental map for each agent in the HDF5 dataset at the corresponding pixel location with the current timestep.
    
        Parameters:
        - current_timestep: The current timestep to record in the mental map.
    
        The mental map is stored in an HDF5 dataset with shape (num_agents, width, height), where each 'slice' corresponds to an agent's mental map.
        """
        # Convert geographic coordinates to pixel coordinates for each agent
        rows, cols = geo_to_pixel(self.X, self.Y, self.refugia_map_transform)
    
        # Ensure rows and cols are within the bounds of the mental map
        rows = self.arr.clip(rows, 0, self.height - 1)
        cols = self.arr.clip(cols, 0, self.width - 1)

        # identify velocity refugia, if current velocity is less than the maximum sustained swim speed, it's a velocity refugia
        refugia = np.where(current_velocity < self.max_s_U,1,0)
        
        # get velocity and coords raster per agent
        for i in np.arange(self.num_agents):
            if self.num_agents > 1:
                try:
                    self.hdf5['refugia/%s'%(i)][rows[i],cols[i]] = refugia
                except:
                    pass
            else:
                single_arr = np.array([self.hdf5['refugia/%s'%(i)]])
                single_arr[0,rows,cols] = refugia
                self.hdf5['refugia/%s'%(i)][:, :] = single_arr
        
        self.hdf5.flush()

    def environment(self):
        """
        Updates environmental parameters for each agent and identifies neighbors within a defined buffer.
    
        This function creates a GeoDataFrame from the agents' positions and sets the coordinate reference system (CRS).
        It then samples environmental data such as depth, x-velocity, and y-velocity at each agent's position.
        The function also tracks the time each agent spends in shallow water (out of water) and identifies neighboring agents within a specified buffer.
    
        Attributes updated:
            self.depth (np.ndarray): Depth at each agent's position.
            self.x_vel (np.ndarray): X-component of velocity at each agent's position.
            self.y_vel (np.ndarray): Y-component of velocity at each agent's position.
            self.time_out_of_water (np.ndarray): Incremented time that each agent spends in water too shallow for swimming.
    
        Returns:
            agents_within_buffers_dict (dict): A dictionary where each key is an agent index and the value is a list of indices of other agents within that agent's buffer.
            closest_agent_dict (dict): A dictionary where each key is an agent index and the value is the index of the closest agent within that agent's buffer.
    
        Example:
            # Assuming `self` is an instance with appropriate attributes:
            self.environment()
            # After execution, `self.depth`, `self.x_vel`, `self.y_vel`, and `self.time_out_of_water` are updated.
            # `agents_within_buffers_dict` and `closest_agent_dict` are available for further processing.
        """
        # (The rest of your function code follows here...)        
        # create geodataframe from X and Y points
        points = [Point(x, y) for x, y in zip(self.X, self.Y)]
        
        # Now create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=points)
        
        # If you have an associated CRS (Coordinate Reference System), you can set it like this:
        gdf.set_crs(epsg=self.crs, inplace=True)
        
        # get depth, x_vel, and y_vel at every agent position
        self.depth = self.sample_environment(self.depth_rast_transform, 'depth')
        self.x_vel = self.sample_environment(self.vel_x_rast_transform, 'vel_x')
        self.y_vel = self.sample_environment(self.vel_y_rast_transform, 'vel_y')
        self.vel_mag = self.sample_environment(self.vel_mag_rast_transform, 'vel_mag')
        self.wet = self.sample_environment(self.wetted_transform, 'wetted')
        self.distance_to = self.sample_environment(self.depth_rast_transform, 'distance_to')
        self.current_longitudes = self.compute_linear_positions(self.longitudinal)
        
    
        # Avoid divide by zero by setting zero velocities to a small number
        # self.x_vel[self.x_vel == 0.0] = 0.0001
        # self.y_vel[self.y_vel == 0.0] = 0.0001
    
        # keep track of the amount of time a fish spends out of water
        self.time_out_of_water = np.where(np.logical_or(self.depth < self.too_shallow,                                            
                                          self.wet != 1.),  
                                          self.time_out_of_water + 1, 
                                          self.time_out_of_water)

        self.dead = np.where(self.time_out_of_water > 3600,
                              1,
                              self.dead)
                
        # self.dead = np.where(self.wet != 1., 
        #                      1,
        #                      self.dead)
        
        # if np.any(self.dead):
        #     print ('why did they die?')
        #     print ('wet status: %s'%(self.wet))
        #     # sys.exit()
            
            
        
        # For dead fish, zero out positions and velocity
        # self.x_vel = np.where(self.dead,np.zeros_like(self.x_vel), self.x_vel)
        # self.y_vel = np.where(self.dead,np.zeros_like(self.y_vel), self.y_vel)
        # self.X = np.where(self.dead,np.zeros_like(self.X), self.X)
        # self.Y = np.where(self.dead,np.zeros_like(self.Y), self.Y)

        clean_x = self.X.flatten()[~np.isnan(self.X.flatten())]
        clean_y = self.Y.flatten()[~np.isnan(self.Y.flatten())]
        
        positions = np.vstack([clean_x,clean_y]).T
        
        # Creating a KDTree for efficient spatial queries
        try:
           tree = cKDTree(positions)
        except ValueError:
            print ('something wrong with positions - is an agent off the map?')
            print ('XY: %s'%(positions))
            print ('wetted: %s'%(self.wet))
            print ('dead: %s' %(self.dead))
            sys.exit()
        
        # Radius for nearest neighbors search
        #TODO changed from 2 to xx
        radius = 6.
        
        # Find agents within the specified radius for each agent
        agents_within_radius = tree.query_ball_tree(tree, r=radius)
        
        # Batch query for the two nearest neighbors (including self)
        distances, indices = tree.query(positions, k=2)
        
        # Exclude self from results and handle no neighbors case
        nearest_neighbors = np.where(distances[:, 1] != np.inf, indices[:, 1], np.nan)

        # Extract the distance to the closest agent, excluding self
        nearest_neighbor_distances = np.where(distances[:, 1] != np.inf, distances[:, 1], np.nan)

        # Now `agents_within_buffers_dict` is a dictionary where each key is an agent index
        self.agents_within_buffers = agents_within_radius
        self.closest_agent = nearest_neighbors
        self.nearest_neighbor_distance = nearest_neighbor_distances
            
    def odometer(self, t, dt):
        """
        Updates the running counter of the amount of kCal consumed by each fish during a simulation timestep.
    
        Parameters:
        t (float): The current time in the simulation.
    
        Attributes:
        water_temp (array): Water temperature experienced by each fish.
        weight (array): Weight of each fish.
        wave_drag (array): Wave drag acting on each fish.
        swim_speed (array): Swimming speed of each fish.
        ucrit (array): Critical swimming speed for each fish.
        kcal (array): Cumulative kilocalories burned by each fish.
    
        Notes:
        - The function assumes that the input attributes are structured as arrays of
          values, with each index across the arrays corresponding to a different fish.
        - The function updates the kcal attribute for each fish based on their oxygen
          consumption converted to calories using metabolic equations from Brett (1964)
          and Brett and Glass (1973).
        """
    
        # Calculate active and standard metabolic rate using equations from Brett and Glass (1973)
        # O2_rate in units of mg O2/hr
        sr_o2_rate = self.arr.where(
            self.water_temp <= 5.3,
            self.arr.exp(0.0565 * np.power(self.arr.log(self.weight), 0.9141)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.1498 * self.arr.power(self.arr.log(self.weight), 0.8465)),
                self.arr.exp(0.1987 * self.arr.power(self.arr.log(self.weight), 0.8844))
            )
        )
    
        ar_o2_rate = self.arr.where(
            self.water_temp <= 5.3,
            self.arr.exp(0.4667 * self.arr.power(self.arr.log(self.weight), 0.9989)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.9513 * self.arr.power(self.arr.log(self.weight), 0.9632)),
                self.arr.exp(0.8237 * self.arr.power(self.arr.log(self.weight), 0.9947))
            )
        )
    
        # Calculate total metabolic rate
        if self.num_agents > 1:
            swim_cost = sr_o2_rate + self.wave_drag * (
                self.arr.exp(np.log(sr_o2_rate) + self.swim_speed * (
                    (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
                ) - sr_o2_rate)
            )
        else:
            swim_cost = sr_o2_rate + self.wave_drag.flatten() * (
                self.arr.exp(np.log(sr_o2_rate) + np.linalg.norm(self.swim_speed.flatten(), axis = -1) * (
                    (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
                ) - sr_o2_rate)
            )
        # swim cost is expressed in mg O2 _kg _hr.  convert to mg O2 _ kg
        hours = dt * (1./3600.)
        per_capita_swim_cost = swim_cost * hours
        mg_O2 = per_capita_swim_cost * self.weight
        # Brett (1973) used a mean oxycalorific equivalent of 3.36 cal/ mg O2 (RQ = 0.8) 
        kcal = mg_O2 * (3.36 / 1000)
        
        if np.any(kcal < 0):
            print ('why kcal negative')
        # Update kilocalories burned
        self.kcal += kcal
        
    class movement():
        
        def __init__(self, simulation_object):
            self.simulation = simulation_object
            
        def find_z(self):
            """
            Calculate the z-coordinate for an agent based on its depth and body depth.
        
            This function determines the z-coordinate (vertical position) of an agent.
            If the depth at the agent's position is less than one-third of its body depth,
            the z-coordinate is set to the sum of the depth and a predefined shallow water threshold.
            Otherwise, it is set to one-third of the agent's body depth.
        
            Attributes updated:
                self.z (array-like): The calculated z-coordinate for the agent.
        
            Note:
                The function uses `self.arr.where` for vectorized conditional operations,
                which implies that `self.depth`, `self.body_depth`, and `self.too_shallow` should be array-like
                and support broadcasting if they are not scalar values.
            """
            self.simulation.z = np.where(
                self.simulation.depth < self.simulation.body_depth * 3 / 100.,
                self.simulation.depth + self.simulation.too_shallow,
                self.simulation.body_depth * 3 / 100.)
            
            # make sure 
            self.simulation.z = np.where(self.simulation.z < 0,0,self.simulation.z)
        
        def thrust_fun(self, mask, t, dt, fish_velocities = None):
            """
            Calculates the thrust for a collection of agents based on Lighthill's elongated-body theory of fish propulsion.
            
            This method uses piecewise linear interpolation for the amplitude, wave, and trailing edge
            as functions of body length and swimming speed. It is designed to work with array operations
            and can utilize either NumPy or CuPy for calculations to support execution on both CPUs and GPUs.
            
            The method assumes a freshwater environment with a density of 1.0 kg/m^3. The thrust calculation
            uses the agents' lengths, velocities, ideal speeds over ground (SOG), headings, and tail-beat frequencies
            to compute thrust vectors for each agent.
            
            Attributes
            ----------
            length : array_like
                The lengths of the agents in meters.
            x_vel : array_like
                The x-components of the water velocity vectors for the agents in m/s.
            y_vel : array_like
                The y-components of the water velocity vectors for the agents in m/s.
            ideal_sog : array_like
                The ideal speeds over ground for the agents in m/s.
            heading : array_like
                The headings of the agents in radians.
            Hz : array_like
                The tail-beat frequencies of the agents in Hz.
            
            Returns
            -------
            thrust : ndarray
                An array of thrust vectors for each agent, where each vector is a 2D vector
                representing the thrust in the x and y directions in N/m.
            
            Notes
            -----
            The function assumes that the input arrays are of equal length, with each index
            corresponding to a different agent. The thrust calculation is vectorized to handle
            multiple agents simultaneously.
            
            The piecewise linear interpolation is used for the amplitude, wave, and trailing
            edge based on the provided data points. This approach simplifies the computation and
            is suitable for scenarios where a small amount of error is acceptable.
            
            Examples
            --------
            Assuming `agents` is an instance of the simulation class with all necessary properties set as arrays:
            
            >>> thrust = agents.thrust_fun()
            
            The `thrust` array will contain the thrust vectors for each agent after the function call.
            """

            # Constants
            rho = 1.0  # density of freshwater
            theta = 32.  # theta that produces cos(theta) = 0.85
            length_cm = self.simulation.length / 1000 * 100.
            
            # Calculate swim speed
            water_vel = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_dir = np.arctan2(fish_y_vel,fish_x_vel)
                    fish_mag = np.linalg.norm(np.stack((fish_x_vel,fish_y_vel)).T,axis = -1)

                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)),
                                               axis=-1)
                        
            ideal_swim_speed = np.linalg.norm(fish_velocities - water_vel, axis=-1)

            swim_speed_cms = ideal_swim_speed * 100.
        
            # Data for interpolation
            length_dat = np.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
            speed_dat = np.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
            amp_dat = np.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
            wave_dat = np.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
            edge_dat = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])
        
            # Interpolation with extrapolation using UnivariateSpline
            A_spline = UnivariateSpline(length_dat, amp_dat, k = 2, ext = 0)
            V_spline = UnivariateSpline(speed_dat, wave_dat, k = 1, ext = 0)
            B_spline = UnivariateSpline(length_dat, edge_dat, k = 1, ext = 0)
        
            A = A_spline(length_cm)
            V = V_spline(swim_speed_cms)
            B = B_spline(length_cm)
        
            # Calculate thrust
            m = (np.pi * rho * B**2) / 4.
            W = (self.simulation.Hz * A * np.pi) / 1.414
            w = W * (1 - swim_speed_cms / V)
        
            # Thrust calculation
            thrust_erg_s = m * W * w * swim_speed_cms - (m * w**2 * swim_speed_cms) / (2. * np.cos(np.radians(theta)))
            thrust_Nm = thrust_erg_s / 10000000.
            thrust_N = thrust_Nm / (self.simulation.length / 1000.)
        
            # Convert thrust to vector
            thrust = np.where(mask,[thrust_N * np.cos(self.simulation.heading),
                                    thrust_N * np.sin(self.simulation.heading)],0)
                
            self.simulation.thrust = thrust.T
            
        def frequency(self, mask, t, dt, fish_velocities = None):
            ''' Calculate tailbeat frequencies for a collection of agents in a vectorized manner.
            
                This method computes tailbeat frequencies based on Lighthill's elongated-body theory,
                considering each agent's length, velocity, and drag. It then adjusts these frequencies
                using a vectorized PID controller to better match the desired speed over ground.
            
                Parameters
                ----------
                mask : array_like
                    A boolean array indicating which agents to include in the calculation.
                pid_controller : VectorizedPIDController
                    An instance of the VectorizedPIDController class for adjusting Hz values.
            
                Returns
                -------
                Hzs : ndarray
                    An array of adjusted tailbeat frequencies for each agent, in Hz.
            
                Notes
                -----
                The function assumes that all input arrays are of equal length, corresponding
                to different agents. It uses vectorized operations for efficiency and is
                compatible with a structure-of-arrays approach.
            
                The PID controller adjusts frequencies based on the error between the actual
                and desired speeds, improving the model's realism and accuracy.
            
                    # ... function implementation ...'''

            # Constants
            rho = 1.0  # density of freshwater
            theta = 32.  # theta for cos(theta) = 0.85
        
            # Convert lengths from meters to centimeters
            lengths_cm = self.simulation.length / 10
        
            # Calculate swim speed in cm/s
            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            alternate = True
            
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_velocities = np.stack((fish_x_vel,fish_y_vel)).T
                
                alternate = False
            
            swim_speeds_cms = np.linalg.norm(fish_velocities - water_velocities, axis=-1) * 100 + 0.00001

            # sockeye parameters (Webb 1975, Table 20) units in CM!!! 
            length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
            speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
            amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
            wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
            edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
            
            # Interpolation with extrapolation using UnivariateSpline
            A_spline = UnivariateSpline(length_dat, amp_dat, k = 2, ext = 0)
            V_spline = UnivariateSpline(speed_dat, wave_dat, k = 1, ext = 0)
            B_spline = UnivariateSpline(length_dat, edge_dat, k = 1, ext = 0)
        
            A = A_spline(lengths_cm)
            V = V_spline(swim_speeds_cms)
            B = B_spline(lengths_cm)
            
            # get the ideal drag - aka drag if fish is moving how we want it to
            if alternate == True:
                ideal_drag = self.ideal_drag_fun(fish_velocities = fish_velocities)
            else:
                ideal_drag = self.ideal_drag_fun()
                
            # Convert drag to erg/s
            drags_erg_s = np.where(mask,np.linalg.norm(ideal_drag, axis = -1) * self.simulation.length/1000 * 10000000,0)
            
            #TODO min_Hz should be the minimum tailbeat required to match the maximum sustained swim speed 
            # self.max_s_U = 2.77 bl/s
            min_Hz = np.interp(self.simulation.length, [450, 7.5], [690, 2.])
        
            # Solve for Hz
            Hz = np.where(self.simulation.swim_behav == 3, min_Hz,
                          np.sqrt(drags_erg_s * V**2 * np.cos(np.radians(theta))/\
                                  (A**2 * B**2 * swim_speeds_cms * np.pi**3 * rho * \
                                  (swim_speeds_cms - V) * \
                                  (-0.062518880701972 * swim_speeds_cms - \
                                  0.125037761403944 * V * np.cos(np.radians(theta)) + \
                                   0.062518880701972 * V)
                                   )
                                  )
                          )
            Hz = np.where(self.simulation.is_stuck,0,Hz)
            
            self.simulation.prev_Hz = self.simulation.Hz   
            self.simulation.Hz = np.where(self.simulation.Hz > 20, 20, Hz)
             
        def kin_visc(self, temp):
            """
            Calculates the kinematic viscosity of water at a given temperature using
            interpolation from a predefined dataset.
        
            Parameters
            ----------
            temp : float
                The temperature of the water in degrees Celsius for which the kinematic
                viscosity is to be calculated.
        
            Returns
            -------
            float
                The kinematic viscosity of water at the specified temperature in m^2/s.
        
            Notes
            -----
            The function uses a dataset of kinematic viscosity values at various
            temperatures sourced from the Engineering Toolbox. It employs linear
            interpolation to estimate the kinematic viscosity at the input temperature.
        
            Examples
            --------
            >>> kin_viscosity = kin_visc(20)
            >>> print(f"The kinematic viscosity at 20°C is {kin_viscosity} m^2/s")
        
            This will output the kinematic viscosity at 20 degrees Celsius.
            """
            # Dataset for kinematic viscosity (m^2/s) at various temperatures (°C)
            kin_temp = np.array([0.01, 10., 20., 25., 30., 40., 50., 60., 70., 80.,
                                 90., 100., 110., 120., 140., 160., 180., 200.,
                                 220., 240., 260., 280., 300., 320., 340., 360.])
        
            kin_visc = np.array([0.00000179180, 0.00000130650, 0.00000100350,
                                 0.00000089270, 0.00000080070, 0.00000065790,
                                 0.00000055310, 0.00000047400, 0.00000041270,
                                 0.00000036430, 0.00000032550, 0.00000029380,
                                 0.00000026770, 0.00000024600, 0.00000021230,
                                 0.00000018780, 0.00000016950, 0.00000015560,
                                 0.00000014490, 0.00000013650, 0.00000012990,
                                 0.00000012470, 0.00000012060, 0.00000011740,
                                 0.00000011520, 0.00000011430])
        
            # Interpolate kinematic viscosity based on the temperature
            f_kinvisc = np.interp(temp, kin_temp, kin_visc)
        
            return f_kinvisc

        def wat_dens(self, temp):
            """
            Calculates the density of water at a given temperature using interpolation
            from a predefined dataset.
        
            Parameters
            ----------
            temp : float
                The temperature of the water in degrees Celsius for which the density
                is to be calculated.
        
            Returns
            -------
            float
                The density of water at the specified temperature in g/cm^3.
        
            Notes
            -----
            The function uses a dataset of water density values at various temperatures
            sourced from reliable references. It employs linear interpolation to estimate
            the water density at the input temperature.
        
            Examples
            --------
            >>> water_density = wat_dens(20)
            >>> print(f"The density of water at 20°C is {water_density} g/cm^3")
        
            This will output the density of water at 20 degrees Celsius.
            """
            # Dataset for water density (g/cm^3) at various temperatures (°C)
            dens_temp = np.array([0.1, 1., 4., 10., 15., 20., 25., 30., 35., 40.,
                                  45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
                                  95., 100., 110., 120., 140., 160., 180., 200.,
                                  220., 240., 260., 280., 300., 320., 340., 360.,
                                  373.946])
        
            density = np.array([0.9998495, 0.9999017, 0.9999749, 0.9997, 0.9991026,
                                0.9982067, 0.997047, 0.9956488, 0.9940326, 0.9922152,
                                0.99021, 0.98804, 0.98569, 0.9832, 0.98055, 0.97776,
                                0.97484, 0.97179, 0.96861, 0.96531, 0.96189, 0.95835,
                                0.95095, 0.94311, 0.92613, 0.90745, 0.887, 0.86466,
                                0.84022, 0.81337, 0.78363, 0.75028, 0.71214, 0.66709,
                                0.61067, 0.52759, 0.322])
        
            # Interpolate water density based on the temperature
            f_density = np.interp(temp, dens_temp, density)
        
            return f_density

        def calc_Reynolds(self, visc, water_vel):
            """
            Calculates the Reynolds number for each fish in an array based on their lengths,
            the kinematic viscosity of the water, and the velocity of the water.
        
            The Reynolds number is a dimensionless quantity that predicts flow patterns in
            fluid flow situations. It is the ratio of inertial forces to viscous forces and
            is used to determine whether a flow will be laminar or turbulent.
        
            This function is designed to work with libraries that support NumPy-like array
            operations, such as NumPy or CuPy, allowing for efficient computation on either
            CPUs or GPUs.
        
            Parameters
            ----------
            visc : float
                The kinematic viscosity of the water in m^2/s.
            water_vel : float
                The velocity of the water in m/s.
        
            Returns
            -------
            array_like
                An array of Reynolds numbers, one for each fish.
        
            Examples
            --------
            >>> lengths = self.arr.array([200, 250, 300])
            >>> visc = 1e-6
            >>> water_vel = 0.5
            >>> reynolds_numbers = calc_Reynolds(lengths, visc, water_vel)
            >>> print(f"The Reynolds numbers are {reynolds_numbers}")
        
            This will output the Reynolds numbers for fish of lengths 200 mm, 250 mm, and 300 mm
            in water with a velocity of 0.5 m/s and a kinematic viscosity of 1e-6 m^2/s.
            """
            # Convert length from millimeters to meters
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = water_vel * length_m / visc
        
            return reynolds_numbers
     
        def calc_surface_area(self):
            """
            Calculates the surface area of each fish in an array based on their lengths.
            
            The surface area is determined using a power-law relationship, which is a common
            empirical model in biological studies to relate the size of an organism to some
            physiological or ecological property, in this case, the surface area.
        
            This function is designed to work with libraries that support NumPy-like array
            operations, such as NumPy or CuPy, allowing for efficient computation on either
            CPUs or GPUs.
        
            Atrributes
            ----------
            length : array_like
                An array of fish lengths in millimeters.
        
            Returns
            -------
            array_like
                An array of surface areas, one for each fish.
        
            Notes
            -----
            The power-law relationship used here is given by the formula:
            SA = 10 ** (a + b * log10(length))
            where `a` and `b` are empirically derived constants.
        
            Examples
            --------
            >>> lengths = self.arr.array([200, 250, 300])
            >>> surface_areas = calc_surface_area(lengths)
            >>> print(f"The surface areas are {surface_areas}")
        
            This will output the surface areas for fish of lengths 200 mm, 250 mm, and 300 mm
            using the power-law relationship with constants a = -0.143 and b = 1.881.
            """
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
        
            # Calculate the surface area for each fish
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length))
        
            return surface_areas

        def drag_coeff(self, reynolds):
            """
            Calculates the drag coefficient for each value in an array of Reynolds numbers.
            
            The relationship between drag coefficient and Reynolds number is modeled using
            a logarithmic fit. This function is designed to work with libraries that support
            NumPy-like array operations, such as NumPy or CuPy, allowing for efficient
            computation on either CPUs or GPUs.
        
            Parameters
            ----------
            reynolds : array_like
                An array of Reynolds numbers.
            arr : module, optional
                The array library to use for calculations (default is NumPy).
        
            Returns
            -------
            array_like
                An array of drag coefficients corresponding to the input Reynolds numbers.
        
            Examples
            --------
            >>> reynolds_numbers = arr.array([2.5e4, 5.0e4, 7.4e4])
            >>> drag_coeffs = drag_coeff(reynolds_numbers)
            >>> print(f"The drag coefficients are {drag_coeffs}")
            """
            # Coefficients from the dataframe, converted to arrays for vectorized operations
            reynolds_data = np.array([2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5])
            drag_data = np.array([0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
        
            # Fit the logarithmic model to the data
            drag_coefficients = np.interp(reynolds, reynolds_data, drag_data)
        
            return drag_coefficients

        def drag_fun(self, mask, t, dt, fish_velocities = None):
            """
            Calculate the drag force on a sockeye salmon swimming upstream.
        
            This function computes the drag force experienced by a sockeye salmon as it
            swims against the current. It takes into account the fish's velocity, the
            water velocity, and the water temperature to determine the kinematic
            viscosity and density of the water. The drag force is calculated using the
            drag equation from fluid dynamics, which incorporates the Reynolds number,
            the surface area of the fish, and the drag coefficient.
        
            Attributes:
                sog (array): Speed over ground of the fish in m/s.
                heading (array): Heading of the fish in radians.
                x_vel, y_vel (array): Water velocity components in m/s.
                water_temp (array): Water temperature in degrees Celsius.
                length (array): Length of the fish in meters.
                wave_drag (array): Additional drag factor due to wave-making.
        
            Returns:
                ndarray: An array of drag force vectors for each fish, where each vector
                is a 2D vector representing the drag force in the x and y directions in N.
        
            Notes:
                - The function assumes that the input arrays are structured as arrays of
                  values, with each index across the arrays corresponding to a different
                  fish.
                - The drag force is computed in a vectorized manner, allowing for
                  efficient calculations over multiple fish simultaneously.
                - The function uses np.stack and np.newaxis to ensure proper alignment
                  and broadcasting of array operations.
                - The drag is calculated in SI units (N).
        
            Examples:
                >>> # Assuming all properties are set in the class
                >>> drags = self.drag_fun()
                >>> print(drags)
                # Output: array of drag force vectors for each fish
            """
            tired_mask = np.where(self.simulation.swim_behav == 3,True,False)

            # Calculate fish velocities
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_velocities = np.stack((fish_x_vel,fish_y_vel)).T

            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            
            
            water_velocities = np.where(tired_mask[:,np.newaxis], 
                                        water_velocities * 0.2,
                                        water_velocities * 1.)
        
            # Ensure non-zero fish velocity for calculation
            fish_speeds = np.linalg.norm(fish_velocities, axis=-1)
            fish_speeds[fish_speeds == 0.0] = 0.0001
            fish_velocities[fish_speeds == 0.0] = [0.0001, 0.0001]
        
            # Calculate kinematic viscosity and density based on water temperature
            viscosity = self.kin_visc(self.simulation.water_temp)
            density = self.wat_dens(self.simulation.water_temp)

            # Calculate Reynolds numbers
            #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
        
            # Calculate surface areas
            
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
        
            # Calculate the surface area for each fish
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))
        
            # Calculate drag coefficients
            drag_coeffs = self.drag_coeff(reynolds_numbers)
        
            # Calculate relative velocities and their norms
            relative_velocities = fish_velocities - water_velocities
            relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1)**2
        
            # Calculate unit vectors for fish velocities
            unit_relative_vector= np.nan_to_num(relative_velocities / np.linalg.norm(relative_velocities, axis=1)[:,np.newaxis])

            # Calculate drag forces
            drags = np.where(mask[:,np.newaxis],
                             -0.5 * (density * 1000) * (surface_areas[:,np.newaxis] / 100**2) \
                                           * drag_coeffs[:,np.newaxis] * relative_speeds_squared[:, np.newaxis] \
                                               * unit_relative_vector * self.simulation.wave_drag[:, np.newaxis],0)
                
            max_drag_magnitude = 5.  # Set a reasonable limit based on your system's physical reality

            # Calculate the magnitude of each drag force vector
            drag_magnitudes = np.linalg.norm(drags, axis=1)
            
            # Find where the drag exceeds the maximum and scale it down
            excessive_drag_indices = np.where(np.logical_and( self.simulation.swim_behav == 3,
                                                             drag_magnitudes > max_drag_magnitude),
                                              True,False)
            drags[excessive_drag_indices] = (drags[excessive_drag_indices].T * (max_drag_magnitude / drag_magnitudes[excessive_drag_indices])).T
                
            self.simulation.drag = drags

        def ideal_drag_fun(self, fish_velocities = None):
            """
            Calculate the ideal drag force on multiple sockeye salmon swimming upstream.
            
            This function computes the ideal drag force for each fish based on its length,
            water velocity, fish velocity, and water temperature. The drag force is computed
            using the drag equation from fluid dynamics, incorporating the Reynolds number,
            surface area, and drag coefficient.
            
            Attributes:
                x_vel, y_vel (array): Water velocity components in m/s for each fish.
                ideal_sog (array): Ideal speed over ground in m/s for each fish.
                heading (array): Heading in radians for each fish.
                water_temp (array): Water temperature in degrees Celsius for each fish.
                length (array): Length of each fish in meters.
                swim_behav (array): Swimming behavior for each fish.
                max_s_U (array): Maximum sustainable swimming speed in m/s for each fish.
                wave_drag (array): Additional drag factor due to wave-making for each fish.
            
            Returns:
                ndarray: An array of ideal drag force vectors for each fish, where each vector
                is a 2D vector representing the drag force in the x and y directions in N.
            
            Notes:
                - The function assumes that the input arrays are structured as arrays of
                  values, with each index across the arrays corresponding to a different
                  fish.
                - The drag force is computed in a vectorized manner, allowing for
                  efficient calculations over multiple fish simultaneously.
                - The function adjusts the fish velocity if it exceeds the maximum
                  sustainable speed based on the fish's behavior.
            """
            # Vector components of water velocity and speed over ground for each fish
            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            
            if fish_velocities is None:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
        
            # calculate ideal swim speed  
            ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
           
            # make sure fish isn't swimming faster than it should
            refugia_mask = (self.simulation.swim_behav == 2) & (ideal_swim_speeds > self.simulation.max_s_U)
            holding_mask = (self.simulation.swim_behav == 3) & (ideal_swim_speeds > self.simulation.max_s_U)
            too_fast = refugia_mask + holding_mask
            
            fish_velocities = np.where(too_fast[:,np.newaxis],
                                       (self.simulation.max_s_U / ideal_swim_speeds[:,np.newaxis]) * fish_velocities,
                                       fish_velocities)
        
            # Calculate the maximum practical speed over ground
            self.simulation.max_practical_sog = fish_velocities
            
            if self.simulation.num_agents > 1:
                self.simulation.max_practical_sog[np.linalg.norm(self.simulation.max_practical_sog, axis=1) == 0.0] = [0.0001, 0.0001]
            else:
                pass

            # Kinematic viscosity and density based on water temperature for each fish
            viscosity = self.kin_visc(self.simulation.water_temp)
            density = self.wat_dens(self.simulation.water_temp)
        
            # Reynolds numbers for each fish
            #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
            
            # Surface areas for each fish
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
            #surface_areas = self.calc_surface_area(self.length)
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))
        
            # Drag coefficients for each fish
            drag_coeffs = self.simulation.drag_coeff(reynolds_numbers)
        
            # Calculate ideal drag forces
            relative_velocities = self.simulation.max_practical_sog - water_velocities
            relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1)**2
            unit_max_practical_sog = self.simulation.max_practical_sog / np.linalg.norm(self.simulation.max_practical_sog, axis=1)[:, np.newaxis]
        
            # Ideal drag calculation
            ideal_drags = -0.5 * (density * 1000) * \
                (surface_areas[:,np.newaxis] / 100**2) * drag_coeffs[:,np.newaxis] \
                    * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog \
                          * self.simulation.wave_drag[:, np.newaxis]
        
            return ideal_drags
                

        def swim(self, t, dt, pid_controller, mask):
            """
            Method propels each fish agent forward by calculating its new speed over ground 
            (sog) and updating its position.
        
            Parameters:
            - dt: The time step over which to update the fish's position.
        
            The function performs the following steps for each fish agent:
            1. Calculates the initial velocity of the fish based on its speed over ground 
            (sog) and heading.
            2. Computes the surge by adding the thrust and drag forces, rounding them to 
            two decimal places.
            3. Calculates the acceleration by dividing the surge by the fish's weight and 
            rounding to two decimal places.
            4. Applies a dampening factor to the acceleration to simulate the effect of 
            water resistance.
            5. Updates the fish's velocity by adding the dampened acceleration to the 
            initial velocity.
            6. Updates the fish's speed over ground (sog) based on the new velocity.
            7. Prepares to update the fish's position in the main simulation loop 
            (not implemented here).
        
            Note: The position update is not performed within this function. The 
            'prevPosX' and 'prevPosY' attributes are set to 'self.posX' and 'self.posY' 
            to prepare for the position update, which should be handled in the main 
            simulation loop where this method is called.
            """
            tired_mask = np.where(self.simulation.swim_behav == 3,True,False)
            
            # Step 1: Calculate fish velocity in vector form for each fish
            if t == 0:
                fish_vel_0_x = np.where(mask, self.simulation.sog * np.cos(self.simulation.heading),0) 
                fish_vel_0_y = np.where(mask, self.simulation.sog * np.sin(self.simulation.heading),0)  
                fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
            else:
                fish_vel_0_x = (self.simulation.X - self.simulation.prev_X)/dt
                fish_vel_0_y = (self.simulation.Y - self.simulation.prev_Y)/dt
                fish_vel_0 = np.stack((fish_vel_0_x,fish_vel_0_y)).T
            
            if np.any(self.simulation.ideal_sog > 1.):
                print ('check')
            
            ideal_vel_x = np.where(mask, self.simulation.ideal_sog * np.cos(self.simulation.heading),0) 
            ideal_vel_y = np.where(mask, self.simulation.ideal_sog * np.sin(self.simulation.heading),0)  
            
            ideal_vel = np.stack((ideal_vel_x, ideal_vel_y)).T
            
            # Step 2: Calculate surge for each fish
            surge_ini = self.simulation.thrust + self.simulation.drag
            
            # Step 3: Calculate acceleration for each fish
            acc_ini = np.round(surge_ini / self.simulation.weight[:,np.newaxis], 2)  
            
            # Step 4: Update velocity for each fish
            fish_vel_1_ini = fish_vel_0 + acc_ini * dt
                
            # Step 5: Thrust feedback PID controller 
            error = np.where(mask[:,np.newaxis], 
                             np.round(ideal_vel - fish_vel_1_ini,12),
                             0.)
            
            #error = np.where(self.simulation.is_stuck[:,np.newaxis],np.zeros_like(error),error)
            
            self.simulation.error = error
            self.simulation.dead = np.where(np.isnan(error[:, 0]),1,self.simulation.dead)
            
                
            if self.simulation.pid_tuning == True:
                self.simulation.error_array = np.append(self.simulation.error_array, error[0])
                self.simulation.vel_x_array = np.append(self.simulation.vel_x_array, self.simulation.x_vel)
                self.simulation.vel_y_array = np.append(self.simulation.vel_y_array, self.simulation.y_vel)
                
                curr_vel = np.round(np.sqrt(np.power(self.simulation.x_vel,2) + np.power(self.simulation.y_vel,2)),2)
                
                print (f'error: {error}')
                print (f'current velocity: {curr_vel}')
                print (f'Hz: {self.Hz}')
                print (f'thrust: {np.round(self.thrust,2)}')
                print (f'drag: {np.round(self.drag,2)}')
                print (f'sog: {np.round(self.sog,4)}')

                if np.any(np.isnan(error)):
                    print('nan in error')
                    sys.exit()
                    
            else:
                k_p, k_i, k_d = pid_controller.PID_func(np.sqrt(np.power(self.simulation.x_vel,2) + np.power(self.simulation.y_vel,2)),
                                                        self.simulation.length)
                pid_controller.k_p = np.array([1.])
                pid_controller.k_i = np.array([0.])
                pid_controller.k_d = np.array([0.])
                
            # Adjust Hzs using the PID controller (vectorized)
            pid_adjustment = pid_controller.update(error, dt, None)
            self.simulation.integral = pid_controller.integral
            self.simulation.pid_adjustment = pid_adjustment
            
            # Step 6: add adjustment to original velocity computation       
            # fish_vel_1 = np.where(~tired_mask[:,np.newaxis],
            #                       fish_vel_0 + acc_ini * dt + pid_adjustment,
            #                       fish_vel_0 + acc_ini * dt)
            
            water_velocity = np.where(self.simulation.wet[:,np.newaxis] == -9999.0,
                                      np.zeros_like(np.vstack((self.simulation.x_vel,self.simulation.y_vel)).T),
                                      np.vstack((self.simulation.x_vel,self.simulation.y_vel)).T)
            
            if np.any(np.linalg.norm(water_velocity, axis=1) > 6):
                # Get a mask of which velocities exceed a magnitude of 10
                too_big = np.linalg.norm(water_velocity, axis=1) > 6
                
                # Normalize the water velocity vectors that exceed the limit
                norms = np.linalg.norm(water_velocity[too_big], axis=1, keepdims=True)
                
                # Scale those velocities to have a norm of 10 while preserving their direction
                water_velocity[too_big] = (water_velocity[too_big] / norms) * 6

            fish_vel_1 = np.where(~tired_mask[:,np.newaxis],
                                  ideal_vel,
                                  water_velocity)
            
            fish_vel_1 = np.where(self.simulation.dead[:,np.newaxis] == 1,
                                  water_velocity,
                                  fish_vel_1)
            
            # Step 7: Prepare for position update
            dxdy = np.where(mask[:,np.newaxis], fish_vel_1 * dt, np.zeros_like(fish_vel_1))
                
            # if np.any(np.linalg.norm(fish_vel_1,axis = -1) > 2* self.simulation.ideal_sog):
            #     print ('fuck - why')
            return dxdy
                            
        def jump(self, t, g, mask):
            """
            Simulates each fish jumping using a ballistic trajectory.
        
            Parameters:
            t (float): The current time in the simulation.
            g (float): The acceleration due to gravity.
        
            Attributes:
            time_of_jump (array): The time each fish jumps.
            ucrit (array): Critical swimming speed for each fish.
            sog (array): Speed over ground for each fish.
            heading (array): Heading for each fish.
            y_vel (array): Y-component of water velocity.
            x_vel (array): X-component of water velocity.
            pos_x (array): X-coordinate of the current position of each fish.
            pos_y (array): Y-coordinate of the current position of each fish.
        
            Notes:
            - The function assumes that the input attributes are structured as arrays of
              values, with each index across the arrays corresponding to a different fish.
            - The function updates the position and speed over ground (sog) for each fish
              based on their jump.
            """

            # Reset jump time for each fish
            
            self.simulation.time_of_jump = np.where(mask,t,self.simulation.time_of_jump)
        
            # Get jump angle for each fish
            jump_angles = np.where(mask,np.random.choice([np.radians(45), np.radians(60)], size=self.simulation.ucrit.shape),0)
        
            # Calculate time airborne for each fish
            time_airborne = np.where(mask,(2 * self.simulation.ucrit * np.sin(jump_angles)) / g, 0)
        
            # Calculate displacement for each fish
            displacement = self.simulation.ucrit * time_airborne * np.cos(jump_angles)
            
            # Calculate the new position for each fish
            # should we make this the water direction?  calculate unit vector, multiply by displacement, and add to current position
            
            dx = displacement * np.cos(self.simulation.heading)
            dy = displacement * np.sin(self.simulation.heading)
            
            dxdy = np.stack((dx,dy)).T
            
            if np.any(dxdy > 3):
                print ('check jump parameters')
           
            return dxdy        
        
            
    class behavior():
        
        def __init__(self, dt, simulation_object):
            self.dt = dt
            self.simulation = simulation_object
            
        def already_been_here(self, weight, t):
            """
            Calculate repulsive forces based on agents' historical locations within a specified time frame,
            simulating a tendency to avoid areas recently visited.
        
            This function retrieves the current X and Y positions of agents, converts these positions to
            row and column indices in the mental map's raster grid, and then accesses the relevant sections
            of the mental map from an HDF5 file. It computes the repulsive force exerted by each cell in the
            mental map based on the time since the agent's last visit, applying a time-dependent weighting factor.
        
            Parameters
            ----------
            weight : float
                The strength of the repulsive force.
            t : int
                The current time step in the simulation.
        
            Returns
            -------
            numpy.ndarray
                An array containing the sum of the repulsive forces in the X and Y directions for each agent.
        
            Notes
            -----
            - The method assumes that the HDF5 dataset supports numpy-style advanced indexing.
            - The mental map dataset within the HDF5 file is expected to be named in the format 'memory/{agent_idx}'.
            - The forces are normalized to unit vectors to ensure that the direction of the force is independent of the distance.
            - The method uses a buffer zone, currently set to a 4-cell radius, to limit the computation to a manageable area around each agent.
            - The time-dependent weighting factor (`multiplier`) is applied to cells visited within a specified time range, with a repulsive force applied to these cells.
            """
            # Step 1: Get the x, y position of the agents
            x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)
        
            # Step 2: Convert these positions to mental map's pixel indices
            mental_map_rows, mental_map_cols = geo_to_pixel(x, y, 
                                                            self.simulation.depth_rast_transform)
        
            # Define buffer zone around current positions
            buff = 10
            row_min = np.clip(mental_map_rows - buff, 0, None)
            row_max = np.clip(mental_map_rows + buff + 1, None,
                              self.simulation.hdf5['memory/0'].shape[0])
            col_min = np.clip(mental_map_cols - buff, 0, None)
            col_max = np.clip(mental_map_cols + buff + 1, None,
                              self.simulation.hdf5['memory/0'].shape[1])
        
            # Using list comprehension to access the relevant sections from the mental map and calculate forces
            repulsive_forces_per_agent = np.array([
                self._calculate_repulsive_force(agent_idx, rmin, rmax, cmin, cmax, weight, t)
                for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
            ])
            # if np.any(np.linalg.norm(repulsive_forces_per_agent, axis = -1) != 0.):
            #     print ('check')
            return repulsive_forces_per_agent

        def _calculate_repulsive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight, t):
            """
            A helper function to calculate the repulsive force for a single agent based on a section of the mental map.
        
            This function computes the repulsive force exerted on an agent by previously visited cells within
            a specified buffer zone around the agent's current position. It considers the time since each cell was last visited,
            applying a conditional weight based on this time to modulate the repulsive force.
        
            Parameters
            ----------
            agent_idx : int
                The index of the agent for whom the repulsive force is being calculated.
            row_min : int
                The minimum row index of the buffer zone in the mental map's raster grid.
            row_max : int
                The maximum row index of the buffer zone in the mental map's raster grid.
            col_min : int
                The minimum column index of the buffer zone in the mental map's raster grid.
            col_max : int
                The maximum column index of the buffer zone in the mental map's raster grid.
            weight : float
                The strength of the repulsive force.
            t : int
                The current time step in the simulation.
        
            Returns
            -------
            numpy.ndarray
                An array containing the repulsive force in the X and Y directions exerted on the specified agent.
        
            Notes
            -----
            - This function is designed to be called within a list comprehension in the `already_been_here` function.
            - It accesses a specific section of the mental map for the given agent, determined by the provided row and column bounds.
            - The force calculation considers the Euclidean distance from each cell to the agent's current position, normalizing the force to unit vectors.
            """
            # Access the relevant section from the mental map
            mmap_section = self.simulation.hdf5['memory/%s' % agent_idx][row_min:row_max, col_min:col_max]
        
            # Calculate time since last visit and apply conditional weight
            t_since = mmap_section - t
            multiplier = np.where((t_since > 10) & (t_since < 7200), 1 - (t_since - 5) / (7195), 0)
        
            # Relative positions and magnitudes
            delta_x = self.simulation.X[agent_idx] - np.arange(col_min, col_max)
            delta_y = self.simulation.Y[agent_idx] - np.arange(row_min, row_max)[:, np.newaxis]
            magnitudes = np.sqrt(delta_x**2 + delta_y**2)
            magnitudes[magnitudes == 0] = 0.000001  # Avoid division by zero
        
            # Unit vectors and repulsive force
            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes
            try:
                x_force = ((weight * unit_vector_x) / magnitudes) * multiplier
                y_force = ((weight * unit_vector_y) / magnitudes) * multiplier
            except ValueError:
                print (f'unit vector x: {unit_vector_x}')
                print (f'magnitudes: {magnitudes}')
                print (f'multiplier: {multiplier}')
                print (f'species status: {self.simulation.dead}')
                sys.exit()
        
            # Sum forces for this agent
            total_x_force = np.nansum(x_force)
            total_y_force = np.nansum(y_force)
        
            return np.array([total_x_force, total_y_force])    
        
        def find_nearest_refuge(self, weight):
            """
            Calculate the attractive force towards the nearest refuge cell for agents.
        
            Parameters:
            - weight: float, the weight of the attraction force.
        
            Returns:
            - np.array: Array containing the x and y components of the attraction force for the agents.
            """
            # Step 1: Get the x, y position of the agents
            x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)
        
            # Step 2: Convert these positions to mental map's pixel indices
            refugia_map_rows, refugia_map_cols = geo_to_pixel(x, y, self.simulation.refugia_map_transform)
        
            # Define buffer zone around current positions
            buff = 50
        
            row_min = np.clip(refugia_map_rows - buff, 0, None)
            row_max = np.clip(refugia_map_rows + buff + 1, None, self.simulation.hdf5['refugia/0'].shape[0])
            col_min = np.clip(refugia_map_cols - buff, 0, None)
            col_max = np.clip(refugia_map_cols + buff + 1, None, self.simulation.hdf5['refugia/0'].shape[1])
        
            if np.any(row_min < 0) or \
                np.any(row_max < 0) or \
                    np.any(col_min < 0) or \
                        np.any(col_max) < 0:
                print ('fuck')
            # Using list comprehension to access the relevant sections from the mental map and calculate forces
            attractive_forces_per_agent = np.array([
                self._calculate_attractive_force(agent_idx, rmin, rmax, cmin, cmax, weight)
                for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
            ])
            
            return attractive_forces_per_agent
            
        def _calculate_attractive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight):
            # Access the relevant section from the mental map
            refugia_section = self.simulation.hdf5['refugia/%s'% agent_idx][row_min:row_max, col_min:col_max]
        
            # Create a binary mask for the refuge cells
            refuge_mask = (refugia_section == 1)
        
            if np.any(refuge_mask):
                # Compute the distance transform
                distances = distance_transform_edt(~refuge_mask)
            
                # Find the coordinates of the nearest refuge cell
                nearest_refuge_coords = np.unravel_index(np.argmin(distances), distances.shape)
            
                # Convert pixel coordinates to geographic coordinates
                ref_xy = pixel_to_geo(self.simulation.refugia_map_transform,
                                      nearest_refuge_coords[0],
                                      nearest_refuge_coords[1])
            
                # Calculate the attraction force
                delta_x = ref_xy[0] - self.simulation.X
                delta_y = ref_xy[1] - self.simulation.Y
            
                magnitudes = np.sqrt(delta_x**2 + delta_y**2)
                magnitudes[magnitudes == 0] = 0.000001  # Avoid division by zero
            
                # Unit vectors and attractive force
                unit_vector_x = delta_x / magnitudes
                unit_vector_y = delta_y / magnitudes
                x_force = (weight * unit_vector_x)
                y_force = (weight * unit_vector_y)
            
                # Sum forces for this agent
                attract_x = np.nansum(x_force)
                attract_y = np.nansum(y_force)
            
                return np.array([attract_x, attract_y])
            
            else:
                return np.array([0,0])
        
        def vel_cue(self, weight):
            """
            Calculate the velocity cue for each agent based on the surrounding water velocity.
        
            This function determines the direction with the lowest water velocity within a specified
            buffer around each agent. The buffer size is determined by the agent's swim mode:
            if in 'refugia' mode, the buffer is 15 body lengths; otherwise, it is 5 body lengths.
            The function then computes a velocity cue that points in the direction of the lowest
            water velocity within this buffer.
        
            Parameters:
            - weight (float): A weighting factor applied to the velocity cue.
        
            Returns:
            - velocity_min (ndarray): An array of velocity cues for each agent, where each cue
              is a vector pointing in the direction of the lowest water velocity within the buffer.
              The magnitude of the cue is scaled by the given weight and the agent's body length.
        
            Notes:
            - The function assumes that the HDF5 dataset 'environment/vel_mag' is accessible and
              supports numpy-style advanced indexing.
            - The velocity cue is calculated as a unit vector in the direction of the lowest velocity,
              scaled by the weight and normalized by the square of 5 body lengths in meters.
            """
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length#.get() if isinstance(self.length, cp.ndarray) else self.length
            
            # calculate buffer size based on swim mode, if we are in refugia mode buffer is 15 body lengths else 5
            #buff = np.where(self.swim_mode == 2, 15 * length_numpy, 5 * length_numpy)
            
            # for array operations, buffers are represented as a slice (# rows and columns)
            buff = 2
            
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
            
            # find the row and column in the direction raster
            rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
            
            # Access the velocity dataset from the HDF5 file by slicing and dicing
            
            # get slices 
            xmin = cols - buff
            xmax = cols + buff + 1
            ymin = rows - buff
            ymax = rows + buff + 1
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)

            # Create slices
            slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                      for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                       ymin.flatten(), 
                                                       ymax.flatten(),
                                                       xmin.flatten(), 
                                                       xmax.flatten()
                                                       )
                      ]

            # get velocity and coords raster per agent
            vel3d = np.stack([standardize_shape(self.simulation.hdf5['environment/vel_mag'][sl[-2:]]) for sl in slices])
            x_coords = np.stack([standardize_shape(self.simulation.hdf5['x_coords'][sl[-2:]]) for sl in slices])
            y_coords = np.stack([standardize_shape(self.simulation.hdf5['y_coords'][sl[-2:]]) for sl in slices])
            
            vel3d_multiplier = calculate_front_masks(self.simulation.heading.flatten(), 
                                                     x_coords, 
                                                     y_coords, 
                                                     np.nan_to_num(self.simulation.X.flatten()), 
                                                     np.nan_to_num(self.simulation.Y.flatten()), 
                                                     behind_value = 999.9)
                
            vel3d = vel3d * vel3d_multiplier
                
            num_agents, rows, cols = vel3d.shape
            
            # Reshape the 3D array into a 2D array where each row represents an agent
            vel3d = vel3d.reshape(num_agents, rows * cols)
            
            # Find the index of the minimum value in each row (agent)
            flat_indices = np.argmin(vel3d, axis=1)
            
            # Convert flat indices to row and column indices
            min_row_indices = flat_indices // cols
            min_col_indices = flat_indices % cols
                
            # Convert the index back to geographical coordinates
            min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        min_row_indices + ymin, 
                                        min_col_indices + xmin)
            min_x = min_x
            min_y = min_y
            
            # delta_x = self.X - min_x
            # delta_y = self.Y - min_y
            delta_x = min_x - self.simulation.X
            delta_y = min_y - self.simulation.Y
            delta_x_sq = np.power(delta_x,2)
            delta_y_sq = np.power(delta_y,2)
            dist = np.sqrt(delta_x_sq + delta_y_sq)

            # Initialize an array to hold the velocity cues for each agent
            velocity_min = np.zeros((self.simulation.num_agents, 2), dtype=float)

            # attract_x = (weight * delta_x/dist) / np.power(buff,2)
            # attract_y = (weight * delta_y/dist) / np.power(buff,2)
            attract_x = weight * delta_x/dist
            attract_y = weight * delta_y/dist
            return np.array([attract_x,attract_y])

        def rheo_cue(self, weight, downstream = False):
            """
            Calculate the rheotactic heading command for each agent.
        
            This function computes a heading command based on the water velocity direction at the
            agent's current position. The heading is adjusted to face upstream by subtracting 180
            degrees from the sampled velocity direction. The resulting vector is scaled by the
            given weight and normalized by the square of twice the agent's body length in meters.
        
            Parameters:
            - weight (float): A weighting factor applied to the rheotactic cue.
        
            Returns:
            - rheotaxis (ndarray): An array of rheotactic cues for each agent, where each cue
              is a vector pointing upstream. The magnitude of the cue is scaled by the given
              weight and the agent's body length.
        
            Notes:
            - The function assumes that the method `sample_environment` is available and can
              sample the 'vel_dir' from the environment given a transformation matrix.
            - The function converts the velocity direction from degrees to radians and adjusts
              it to point upstream.
            - If `self.length` is a CuPy array, it is converted to a NumPy array for computation.
            """
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length#.get() if isinstance(self.length, cp.ndarray) else self.length
        
            if downstream == False:
                # Sample the environment to get the velocity direction and adjust to point upstream
                x_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform,
                                                           'vel_x') * -1
                y_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform,
                                                           'vel_y') * -1
            else:
                # Sample the environment to get the velocity direction and adjust to point upstream
                x_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform,
                                                           'vel_x')
                y_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform,
                                                           'vel_y')
            
            # Calculate the unit vector in the upstream direction
            v = np.column_stack([x_vel, y_vel])  
            v_hat = v / np.linalg.norm(v, axis = -1)[:,np.newaxis]
            
            # Calculate the rheotactic cue
            rheotaxis = np.zeros_like(v)
            rheotaxis = weight* v_hat

            return rheotaxis
        
        def border_cue(self, weight, t):
            """
            Calculate the border cue for each agent based on the surrounding distance 
            from the boundary.
        
            This function determines the direction with the farthest distance from the 
            boundary within a specified
            buffer around each agent. The function then computes a border cue that 
            points in the direction of the
            farthest distance from the boundary within this buffer.
        
            Parameters:
            - weight (float): A weighting factor applied to the border cue.
        
            Returns:
            - border_max (ndarray): An array of border cues for each agent, where each 
            cue is a vector pointing in the direction of the farthest distance from the 
            boundary within the buffer. The magnitude of the cue is scaled by the 
            given weight.
        
            Notes:
            - The function assumes that the HDF5 dataset 'environment/dist_to_border' 
            is accessible and supports numpy-style advanced indexing.
            """
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length  # .get() if isinstance(self.length, cp.ndarray) else self.length
            
            # For simplicity, let's use a fixed buffer size. You can adjust this as needed.
            buff = 5  # This could be dynamic based on your requirements
            #buff = 2  # This could be dynamic based on your requirements

            # get the x, y position of the agent 
            x, y = (np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y))
            
            # find the row and column in the direction raster
            rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
            
            # get slices 
            xmin = cols - buff
            xmax = cols + buff + 1
            ymin = rows - buff
            ymax = rows + buff + 1
        
            # Ensure indices are within valid range
            xmin = np.clip(xmin, 0, self.simulation.hdf5['environment/distance_to'].shape[1] - 1)
            xmax = np.clip(xmax, 0, self.simulation.hdf5['environment/distance_to'].shape[1])
            ymin = np.clip(ymin, 0, self.simulation.hdf5['environment/distance_to'].shape[0] - 1)
            ymax = np.clip(ymax, 0, self.simulation.hdf5['environment/distance_to'].shape[0])
        
            # Create slices
            slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                      for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                       ymin.flatten(), 
                                                       ymax.flatten(),
                                                       xmin.flatten(), 
                                                       xmax.flatten()
                                                       )
                      ]
            x_coords = np.stack([standardize_shape(self.simulation.hdf5['x_coords'][sl[-2:]],
                                                   target_shape=(2 * buff + 1,2 * buff + 1)) for sl in slices]) 
            y_coords = np.stack([standardize_shape(self.simulation.hdf5['y_coords'][sl[-2:]],
                                                   target_shape=(2 * buff + 1,2 * buff + 1)) for sl in slices])       

            front_multiplier = calculate_front_masks(self.simulation.heading,
                                                     x_coords,
                                                     y_coords,
                                                     self.simulation.X, 
                                                     self.simulation.Y)
            # get distance to border raster per agent
            dist3d = np.stack([standardize_shape(self.simulation.hdf5['environment/distance_to'][sl[-2:]]) for sl in slices]) * front_multiplier
            
            num_agents, rows, cols = dist3d.shape
            
            # Reshape the 3D array into a 2D array where each row represents an agent
            dist3d = dist3d.reshape(num_agents, rows * cols) 
            
            # Find the index of the maximum value in each row (agent)
            flat_indices = np.argmax(dist3d, axis=1)
            
            # Convert flat indices to row and column indices
            max_row_indices = flat_indices // cols
            max_col_indices = flat_indices % cols
            
            # Convert the index back to geographical coordinates
            max_x, max_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        max_row_indices + ymin, 
                                        max_col_indices + xmin)
            
            delta_x = np.zeros(self.simulation.X.shape)
            delta_y = np.zeros(self.simulation.Y.shape)
            
            # delta_x = self.simulation.X - max_x
            # delta_y = self.simulation.Y - max_y
            
            # rather than being repelled here, we are actually attracted to the furthest point from the border
            delta_x = max_x - self.simulation.X 
            delta_y = max_y - self.simulation.Y 
            
            dist = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            
            # check if fish is in the center of the channel?
            # Calculate the current distance to the border for each agent
            current_distances = self.simulation.sample_environment(self.simulation.depth_rast_transform, 'distance_to')
            self.simulation.current_distances = current_distances
        
            # Identify agents that are too close to the border 
            too_close = np.where(current_distances <= 1 * (self.simulation.length/1000.),1,0)# self.simulation.length / 1000.) #| \
                #(self.simulation.in_eddy == 1)
                
            # if np.any(too_close == 1):
            #     print ('boundary force needed')
                
            too_close = np.where(self.simulation.in_eddy == 1,1,too_close)
            
            # calculate repulsive force
            repulse_x = np.where(too_close, 
                                 weight * delta_x / dist,
                                 np.zeros_like(delta_x))
            repulse_y = np.where(too_close,
                                 weight * delta_y / dist,
                                 np.zeros_like(delta_y))

            return np.array([repulse_x, repulse_y])


        def shallow_cue(self, weight):
            """
            Calculate the repulsive force vectors from shallow water areas within a 
            specified buffer around each agent using a vectorized approach.
        
            This function identifies cells within a sensory buffer around each agent 
            that are shallower than a threshold depth. It then calculates the inverse 
            gravitational potential for these cells and sums up the forces to determine 
            the total repulsive force vector exerted on each agent due to shallow water.
        
            Parameters:
            - weight (float): The weighting factor to scale the repulsive force.
        
            Returns:
            - np.ndarray: A 2D array where each row corresponds to an agent and 
            contains the sum of the repulsive forces in the X and Y directions.
        
            Notes:
            - The function assumes that the depth data is accessible from an HDF5 
            file with the key 'environment/depth'.
            - The sensory buffer is set to 2 meters around each agent's position.
            - The function uses the `geo_to_pixel` method to convert geographic 
            coordinates to pixel indices.
            - The repulsive force is calculated as a vector normalized by the 
            magnitude of the distance to each shallow cell, scaled by the weight,
            and summed across all shallow cells within the buffer.
            - The function returns an array of the total repulsive force vectors 
            for each agent.
            - The vectorized approach is expected to improve performance by reducing 
            the overhead of Python loops.
            """

            buff = 2 #* self.length / 1000.  # 2 meters
        
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
        
            # find the row and column in the direction raster
            rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        
            # calculate array slice bounds for each agent
            xmin = cols - buff
            xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
            ymin = rows - buff
            ymax = rows + buff + 1  # +1 for the same reason
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)
        
            # Initialize an array to hold the repulsive forces for each agent
            repulsive_forces = np.zeros((self.simulation.num_agents,2), dtype=float)
            
            #min_depth = (self.simulation.body_depth * 1.1) / 100.# Use advanced indexing to create a boolean mask for the slices
            min_depth = self.simulation.too_shallow
            
            # Create slices
            slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                      for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),  
                                                       ymin.flatten(), 
                                                       ymax.flatten(),
                                                       xmin.flatten(),
                                                       xmax.flatten())
                      ]
            

            # get depth raster per agent
            depths = np.stack([standardize_shape(self.simulation.hdf5['environment/depth'][sl[-2:]],
                                                 target_shape=(2 * buff + 1,2 * buff + 1)) for sl in slices])        
            x_coords = np.stack([standardize_shape(self.simulation.hdf5['x_coords'][sl[-2:]],
                                                   target_shape=(2 * buff + 1,2 * buff + 1)) for sl in slices]) 
            y_coords = np.stack([standardize_shape(self.simulation.hdf5['y_coords'][sl[-2:]],
                                                   target_shape=(2 * buff + 1,2 * buff + 1)) for sl in slices])       
            
            front_multiplier = calculate_front_masks(self.simulation.heading,
                                                     x_coords,
                                                     y_coords,
                                                     self.simulation.X, 
                                                     self.simulation.Y)

            # create a multiplier
            depth_multiplier = np.where(depths < min_depth[:,np.newaxis,np.newaxis], 1, 0)

            # Calculate the difference vectors
            # delta_x = x_coords - self.simulation.X[:,np.newaxis,np.newaxis]
            # delta_y = y_coords - self.simulation.Y[:,np.newaxis,np.newaxis]
            
            delta_x =  self.simulation.X[:,np.newaxis,np.newaxis] - x_coords
            delta_y =  self.simulation.Y[:,np.newaxis,np.newaxis] - y_coords
            
            # Calculate the magnitude of each vector
            magnitudes = np.sqrt(np.power(delta_x,2) + np.power(delta_y,2))
            
            #TODO rater than norm of every delta_x, we just grab the mean delta_x, delta_y?

            # Avoid division by zero
            magnitudes = np.where(magnitudes == 0, 0.000001, magnitudes)
            
            # Normalize each vector to get the unit direction vectors
            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes
        
            # Calculate repulsive force in X and Y directions for this agent
            x_force = ((weight * unit_vector_x) / magnitudes) * depth_multiplier * front_multiplier
            y_force = ((weight * unit_vector_y) / magnitudes) * depth_multiplier * front_multiplier
        
            # Sum the forces for this agent
            if self.simulation.num_agents > 1:
                total_x_force = np.nansum(x_force, axis = (1, 2))
                total_y_force = np.nansum(y_force, axis = (1, 2))
            else:
                total_x_force = np.nansum(x_force)
                total_y_force = np.nansum(y_force)
        
            repulsive_forces =  np.array([total_x_force, total_y_force]).T
            
            return repulsive_forces

        def wave_drag_multiplier(self):
            """
            Calculate the wave drag multiplier based on the body depth of the fish 
            submerged and data from Hughes 2004.
        
            This function reads a CSV file containing digitized data from Hughes 2004 
            Figure 3, which relates the body depth of fish submerged to the wave drag 
            multiplier. It sorts this data, fits a univariate spline to it, and then 
            uses this fitted function to calculate the wave drag multiplier for the 
            current instance based on its submerged body depth.
        
            The wave drag multiplier is used to adjust the drag force experienced by 
            the fish due to waves, based on how much of the fish's body is submerged. 
            A multiplier of 1 indicates no additional drag (fully submerged), while 
            values less than 1 indicate increased drag due to the fish's body interacting 
            with the water's surface.
        
            The function updates the instance's `wave_drag` attribute with the 
            calculated wave drag multipliers.
        
            Notes:
            - The CSV file should be located at '../data/wave_drag_huges_2004_fig3.csv'.
            - The CSV file is expected to have columns 'body_depths_submerged' and 'wave_drag_multiplier'.
            - The spline fit is of degree 3 and extends with a constant value (ext=0) outside the range of the data.
            - The function assumes that the body depth of the fish (`self.body_depth`) is provided in centimeters.
            - The `self.z` attribute represents the depth at which the fish is currently swimming.
        
            Returns:
            - None: The function updates the `self.wave_drag` attribute in place.
            """

            # get data
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '../data/wave_drag_huges_2004_fig3.csv')

            hughes = pd.read_csv(data_dir)

            hughes.sort_values(by = 'body_depths_submerged',
                               ascending = True,
                               inplace = True)
            # fit function
            wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged,
                                             hughes.wave_drag_multiplier,
                                             k = 3, ext = 0)

            # how submerged are these fish - that's how many
            body_depths = self.simulation.z / (self.simulation.body_depth / 100.)

            self.simulation.wave_drag = np.where(body_depths >=3, 1, wave_drag_fun(body_depths))
           
        def wave_drag_cue(self, weight):
            """
            Calculate the direction to the optimal depth cell for each agent to minimize wave drag.
        
            This function computes the direction vectors pointing towards the depth that is closest to each agent's optimal water depth. The optimal water depth is the depth at which the agent experiences the least wave drag. The function uses a buffer around each agent to search for the optimal depth within that area.
        
            Parameters:
            - weight (float): A weighting factor applied to the direction vectors.
        
            Returns:
            - weighted_direction_vectors (ndarray): An array of weighted direction vectors for each agent. Each vector points towards the cell with the depth closest to the agent's optimal water depth.
        
            Notes:
            - The function assumes that the HDF5 dataset 'environment/depth' is accessible and contains the depth raster data.
            - The buffer size is set to 2 meters around each agent's position.
            - The function uses the `geo_to_pixel` method to convert geographic coordinates to pixel indices in the raster.
            - The direction vectors are normalized to unit vectors and then scaled by the given weight.
            - If the agent is already at the optimal depth, the direction vector is set to zero, indicating no movement is necessary.
            - The function iterates over each agent to calculate their respective direction vectors.
            """
        
            # identify buffer
            buff = 2.  # 2 meters
            
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
        
            # find the row and column in the direction raster
            rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        
            # calculate array slice bounds for each agent
            xmin = cols - buff
            xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
            ymin = rows - buff
            ymax = rows + buff + 1  # +1 for the same reason
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)
            
            # Initialize an array to hold the direction vectors for each agent
            direction_vectors = np.zeros((len(x), 2), dtype=float)
            
            # Create slices
            slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                      for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                       ymin.flatten(), 
                                                       ymax.flatten() ,
                                                       xmin.flatten(), 
                                                       xmax.flatten()
                                                       )
                      ]
            
            # get depth raster per agent
            #dep3D = np.stack([self.hdf5['environment/depth'][sl[-2:]] for sl in slices])
            dep3D = np.stack([standardize_shape(self.simulation.hdf5['environment/depth'][sl[-2:]]) for sl in slices])
            x_coords = np.stack([standardize_shape(self.simulation.hdf5['x_coords'][sl[-2:]]) for sl in slices])
            y_coords = np.stack([standardize_shape(self.simulation.hdf5['y_coords'][sl[-2:]]) for sl in slices])
            
            dep3D_multiplier = calculate_front_masks(self.simulation.heading.flatten(), 
                                                     x_coords, 
                                                     y_coords, 
                                                     self.simulation.X.flatten(), 
                                                     self.simulation.Y.flatten(), 
                                                     behind_value = 99999.9)
                
            dep3D = dep3D * dep3D_multiplier

            num_agents, rows, cols = dep3D.shape
     
            # Reshape the 3D array into a 2D array where each row represents an agent
            reshaped_dep3D = dep3D.reshape(num_agents, rows * cols)
            
            # Find the cell with the depth closest to the agent's optimal depth
            optimal_depth_diff = np.abs(reshaped_dep3D - self.simulation.opt_wat_depth[:,np.newaxis])

            # Find the index of the minimum value in each row (agent)
            flat_indices = np.argmin(optimal_depth_diff, axis=1)
            
            # Convert flat indices to row and column indices
            min_row_indices = flat_indices // cols
            min_col_indices = flat_indices % cols

            # Convert the index back to geographical coordinates
            min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        min_row_indices + ymin, 
                                        min_col_indices + xmin)
            
            # delta_x = self.X - min_x
            # delta_y = self.Y - min_y
            delta_x = min_x - self.simulation.X
            delta_y = min_y - self.simulation.Y
            delta_x_sq = np.power(delta_x,2)
            delta_y_sq = np.power(delta_y,2)
            dist = np.sqrt(delta_x_sq + delta_y_sq)

            # Initialize an array to hold the velocity cues for each agent
            velocity_min = np.zeros((self.simulation.num_agents, 2), dtype=float)

            attract_x = weight * delta_x/dist
            attract_y = weight * delta_y/dist
            
            return np.array([attract_x,attract_y])
        
        def cohesion_cue(self, weight, consider_front_only=False):
            """
            Calculate the attractive force towards the average position (cohesion) of the school for each agent.
        
            Parameters:
            - weight (float): The weighting factor to scale the attractive force.
            - consider_front_only (bool): If True, consider only the fish in front of each agent.
        
            Returns:
            - np.ndarray: An array of attractive force vectors towards the average position of the school for each agent.
        
            Notes:
            - The function assumes that `self.simulation.X` and `self.simulation.Y` are arrays 
              containing the x and y coordinates of all agents.
            """
            num_agents = self.simulation.num_agents
        
            # Flatten the list of neighbor indices and create a corresponding array of agent indices
            neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
            agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
            
            # Aggregate X and Y coordinates of all neighbors
            x_neighbors = self.simulation.X[neighbor_indices]
            y_neighbors = self.simulation.Y[neighbor_indices]
            
            # Calculate vectors from agents to their neighbors
            vectors_to_neighbors_x = x_neighbors - self.simulation.X[agent_indices]
            vectors_to_neighbors_y = y_neighbors - self.simulation.Y[agent_indices]
        
            if consider_front_only:
                # Calculate agent velocity vectors
                agent_velocities_x = self.simulation.x_vel[agent_indices]
                agent_velocities_y = self.simulation.y_vel[agent_indices]
                
                # Calculate dot products
                dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
        
                # Filter out neighbors that are behind the agent
                valid_neighbors_mask = dot_products > 0
            else:
                # Consider all neighbors
                valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)
        
            # Filter valid neighbor indices and their corresponding agent indices
            valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
            valid_agent_indices = agent_indices[valid_neighbors_mask]
        
            # Calculate Cohesion vectors
            
            # Calculate centroid for cohesion
            center_x = np.zeros(num_agents)
            center_y = np.zeros(num_agents)
            np.add.at(center_x, valid_agent_indices, x_neighbors[valid_neighbors_mask])
            np.add.at(center_y, valid_agent_indices, y_neighbors[valid_neighbors_mask])
            counts = np.bincount(valid_agent_indices, minlength=num_agents)
            center_x /= counts + (counts == 0)  # Avoid division by zero
            center_y /= counts + (counts == 0)  # Avoid division by zero
            
            # Calculate vectors to average position (centroid)
            vectors_to_center_x = center_x - self.simulation.X
            vectors_to_center_y = center_y - self.simulation.Y
        
            # Calculate distances to average position (centroid)
            distances_to_center = np.sqrt(vectors_to_center_x**2 + vectors_to_center_y**2)
        
            # Normalize vectors (add a small epsilon to distances to avoid division by zero)
            epsilon = 1e-10
            v_hat_center_x = np.divide(vectors_to_center_x, distances_to_center + epsilon, out=np.zeros_like(self.simulation.x_vel), where=distances_to_center+epsilon != 0)
            v_hat_center_y = np.divide(vectors_to_center_y, distances_to_center + epsilon, out=np.zeros_like(self.simulation.y_vel), where=distances_to_center+epsilon != 0)
        
            # Calculate attractive forces
            cohesion_array = np.zeros((num_agents, 2))
            cohesion_array[:, 0] = weight * v_hat_center_x
            cohesion_array[:, 1] = weight * v_hat_center_y
            
    
            return np.nan_to_num(cohesion_array)
        
        def alignment_cue(self, weight, consider_front_only=False):
            """
            Calculate the attractive force towards the average heading (alignment) of the school for each agent.
        
            Parameters:
            - weight (float): The weighting factor to scale the attractive force.
            - consider_front_only (bool): If True, consider only the fish in front of each agent.
        
            Returns:
            - np.ndarray: An array of attractive force vectors towards the average heading of the school for each agent.
        
            Notes:
            - The function assumes that `self.simulation.heading` is an array 
              containing the heading angles of all agents.
            - The function assumes that `self.simulation.x_vel` and `self.simulation.y_vel` are arrays 
              containing the x and y components of velocity for all agents.
            """
            num_agents = self.simulation.num_agents
        
            # Flatten the list of neighbor indices and create a corresponding array of agent indices
            neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
            agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
            
            # Aggregate headings of all neighbors
            headings_neighbors = self.simulation.heading[neighbor_indices]
            
            # Calculate vectors from agents to their neighbors
            vectors_to_neighbors_x = self.simulation.X[neighbor_indices] - self.simulation.X[agent_indices]
            vectors_to_neighbors_y = self.simulation.Y[neighbor_indices] - self.simulation.Y[agent_indices]
        
            if consider_front_only:
                # Calculate agent velocity vectors
                agent_velocities_x = self.simulation.x_vel[agent_indices]
                agent_velocities_y = self.simulation.y_vel[agent_indices]
                
                # Calculate dot products
                dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
        
                # Filter out neighbors that are behind the agent
                valid_neighbors_mask = dot_products > 0
            else:
                # Consider all neighbors
                valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)
        
            # Filter valid neighbor indices and their corresponding agent indices
            valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
            valid_agent_indices = agent_indices[valid_neighbors_mask]
        
            # Calculate average headings for valid neighbors
            avg_heading = np.zeros(num_agents)
            np.add.at(avg_heading, valid_agent_indices, headings_neighbors[valid_neighbors_mask])
            counts = np.bincount(valid_agent_indices, minlength=num_agents)
            avg_heading /= counts + (counts == 0)  # Avoid division by zero
            no_school = np.where(avg_heading == 0., 0., 1.)
        
            # Calculate unit vectors for average headings
            avg_heading_x = np.cos(avg_heading)
            avg_heading_y = np.sin(avg_heading)
        
            # Calculate vectors to average headings
            vectors_to_heading_x = avg_heading_x - self.simulation.x_vel
            vectors_to_heading_y = avg_heading_y - self.simulation.y_vel
        
            # Calculate distances to average headings
            distances = np.sqrt(vectors_to_heading_x**2 + vectors_to_heading_y**2)
        
            # Normalize vectors (add a small epsilon to distances to avoid division by zero)
            epsilon = 1e-10
            v_hat_align_x = np.divide(vectors_to_heading_x, distances + epsilon, out=np.zeros_like(self.simulation.x_vel), where=distances+epsilon != 0)
            v_hat_align_y = np.divide(vectors_to_heading_y, distances + epsilon, out=np.zeros_like(self.simulation.y_vel), where=distances+epsilon != 0)
        
            # Calculate attractive forces
            alignment_array = np.zeros((num_agents, 2))
            alignment_array[:, 0] = weight * v_hat_align_x * no_school
            alignment_array[:, 1] = weight * v_hat_align_y * no_school
            
            # Calculate a new ideal speed based on the mean speed of those fish around
            sogs = np.array([np.mean(self.simulation.sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])
            #sogs = np.array([np.mean(self.simulation.opt_sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])
            #sogs = np.array([np.min(self.simulation.opt_sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])

            # make sure sogs don't get too low
            sogs = np.where(sogs < 0.5 * self.simulation.length / 1000,
                            0.5 * self.simulation.length / 1000,
                            sogs)
            
            self.simulation.school_sog = sogs
        
            return np.nan_to_num(alignment_array)

        def collision_cue(self, weight):
            """
            Generates an array of repulsive force vectors for each agent to avoid collisions,
            based on the positions of their nearest neighbors. This function leverages
            vectorization for efficient computation across multiple agents.
        
            Parameters
            ----------
            weight : float
                The weighting factor that scales the magnitude of the repulsive forces.
            closest_agent_dict : dict
                A dictionary mapping each agent's index to the index of its closest neighbor.
        
            Returns
            -------
            collision_cue_array : ndarray
                An array where each element is a 2D vector representing the repulsive force
                exerted on an agent by its closest neighbor. The force is directed away from
                the neighbor and is scaled by the weight and the inverse square of the distance
                between the agents.
        
            Notes
            -----
            The function internally calls `collision_repulsion`, which computes the repulsive
            force for an individual agent. The `np.vectorize` decorator is used to apply this
            function across all agents, resulting in an array of repulsive forces. The
            `excluded` parameter in the vectorization process is set to exclude the last three
            parameters from vectorization, as they are common to all function calls.
        
            Example
            -------
            # Assuming an instance of the class is created and initialized as `agent_model`
            # and closest_agent_dict is already computed:
            repulsive_forces = agent_model.collision_cue(weight=0.5,
                                                         closest_agent_dict=closest_agents)
            """
            
            # Filter out invalid indices (where nearest_neighbors is nan)
            valid_indices = ~np.isnan(self.simulation.closest_agent)
            
            # Initialize arrays for closest X and Y positions
            closest_X = np.full_like(self.simulation.X, np.nan)
            closest_Y = np.full_like(self.simulation.Y, np.nan)
            
            # Extract the closest X and Y positions using the valid indices
            closest_X[valid_indices] = self.simulation.X[self.simulation.closest_agent[valid_indices].astype(int)]
            closest_Y[valid_indices] = self.simulation.Y[self.simulation.closest_agent[valid_indices].astype(int)]
            
            # calculate vector pointing from neighbor to self
            self_2_closest = np.column_stack((closest_X.flatten() - self.simulation.X.flatten(), 
                                              closest_Y.flatten() - self.simulation.Y.flatten()))
            closest_2_self = np.column_stack((self.simulation.X.flatten() - closest_X.flatten(), 
                                              self.simulation.Y.flatten() - closest_Y.flatten()))
            
            coll_slice = determine_slices_from_vectors(closest_2_self, num_slices = 8)
            head_slice = determine_slices_from_headings(self.simulation.heading, num_slices = 8)
            
            # Handling np.nan values
            # If either component of a vector is np.nan, you might want to treat the whole vector as invalid
            invalid_vectors = np.isnan(closest_2_self).any(axis=1)
            closest_2_self[invalid_vectors] = [np.nan, np.nan]
            closest_2_self = np.nan_to_num(closest_2_self)
            
            # Replace zeros and NaNs in distances to avoid division errors
            # This step assumes that a zero distance implies the agent is its own closest neighbor, 
            # which might result in a zero vector or a scenario you'll want to handle separately.
            safe_distances = np.where(self.simulation.nearest_neighbor_distance > 0, 
                                      self.simulation.nearest_neighbor_distance, 
                                      np.nan)
            
            safe_distances_mm = safe_distances * 1000
            
            # Calculate unit vector components
            v_hat_x = np.divide(closest_2_self[:,0], safe_distances, 
                                out=np.zeros_like(closest_2_self[:,0]), where=safe_distances!=0)
            v_hat_y = np.divide(closest_2_self[:,1], safe_distances, 
                                out=np.zeros_like(closest_2_self[:,1]), where=safe_distances!=0)
                            
            # Calculate collision cue components
            collision_cue_x = np.divide(weight * v_hat_x, safe_distances**2, 
                                        out=np.zeros_like(v_hat_x), where=safe_distances!=0) #* same_quad_multiplier
            collision_cue_y = np.divide(weight * v_hat_y, safe_distances**2, 
                                        out=np.zeros_like(v_hat_y), where=safe_distances!=0) #* same_quad_multiplier
            
            # Optional: Combine the components into a single array
            collision_cue_mm = np.column_stack((collision_cue_x, collision_cue_y))
            #collision_cue = collision_cue_mm / 1000.
            
            np.nan_to_num(collision_cue_mm, copy = False)
            
            return collision_cue_mm     
        
        def is_in_eddy(self,t):
            """
            Assess whether each agent is in an eddy based on several conditions,
            including displacement and behavioral states. This function updates the
            `in_eddy` attribute of the class, which is a Boolean array where each element
            indicates whether the corresponding agent is considered to be in an eddy.
            
            Parameters:
            - t (int): The current timestep. This parameter is not currently used in the
                       function but can be included for future extensions or conditional
                       checks based on time.
            
            Notes:
            - The function initializes all agents as not being in an eddy.
            - Displacement is calculated between the first and the last recorded positions
              in `self.past_x` and `self.past_y`, provided these positions are not NaN.
            - Agents are determined to be in an eddy if:
                1. Their displacement is less than 30 units.
                2. Their `swim_behav` attribute is equal to 1.
                3. Their `swim_mode` attribute is equal to 1.
                4. Their `current_distances` are less than or equal to 6 times the ratio
                   of their `length` attribute to 1000.
            - This function relies on vectorized operations for efficient computation and
              assumes that `self.past_x`, `self.past_y`, `self.swim_behav`, `self.swim_mode`,
              `self.current_distances`, and `self.length` are numpy arrays of appropriate
              dimensions and have been initialized correctly.
            
            Returns:
            - None: The function updates the `self.in_eddy` attribute in place and does not
                    return any value.
            
            Raises:
            - This function does not explicitly raise errors but will fail if the input arrays
              are not correctly formatted or initialized.
            
            Example of usage:
            - Assuming an instance `simulation` of a class where `is_in_eddy` is defined,
              you might call it at a timestep `t` as follows:
                simulation.is_in_eddy(t=100)
            """
            
            linear_positions = self.simulation.compute_linear_positions(self.simulation.longitudinal)
            self.current_longitudes = linear_positions
            # Shift data to the left
            self.simulation.past_longitudes[:, :-1] = self.simulation.past_longitudes[:, 1:]
            self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]

        
            # Insert new position data at the last column
            self.simulation.past_longitudes[:, -1] = linear_positions
            self.simulation.swim_speeds[:, -1] = self.simulation.sog
                      
            # Check for valid entries in both the first and last columns
            valid_entries = ~np.isnan(self.simulation.swim_speeds[:, 0]) & ~np.isnan(self.simulation.swim_speeds[:, -1])
            
            # initialize and calculate average speeds
            avg_speeds = np.full(self.simulation.swim_speeds.shape[0], np.nan)
            avg_speeds[valid_entries] = np.max(self.simulation.swim_speeds[valid_entries], axis = -1)
            
            # total displacements
            total_displacement = np.full(self.simulation.past_longitudes.shape[0], np.nan)
            total_displacement[valid_entries] = self.simulation.past_longitudes[valid_entries,-1] - self.simulation.past_longitudes[valid_entries,0]
            
            # calculate the change in longitude, length of memory, and expected displacement given avg velocity
            delta = self.simulation.past_longitudes[valid_entries,0] - self.simulation.past_longitudes[valid_entries,-1]
            dt = self.simulation.past_longitudes.shape[1]
            expected_displacement = avg_speeds * dt
            
            # calculate chnge in longitudinal position
            long_dir = self.simulation.past_longitudes[:,-2] - self.simulation.past_longitudes[:,-1]

            # Check if agents have moved less than expected, if they are moving backwards, and if they are sustained swimming mode
            if delta.shape == total_displacement.shape and t >= 1800.:
                # stuck_conditions = (expected_displacement >= 2* total_displacement) & \
                #     (self.simulation.swim_mode == 1) & (np.sign(delta) > 0) 
                    
                stuck_conditions = (expected_displacement >= 5. * np.abs(total_displacement)) \
                    & (self.simulation.swim_behav == 1)
            else:
                stuck_conditions = np.zeros_like(self.simulation.X)
            
            not_in_eddy_anymore = self.simulation.time_since_eddy_escape >= self.simulation.max_eddy_escape_seconds
            # Set a specific value (9999) for past positions and swim speeds where not in eddy anymore
            self.simulation.swim_speeds[not_in_eddy_anymore, :] = np.nan
            self.simulation.past_longitudes[not_in_eddy_anymore, :] = np.nan
            self.simulation.time_since_eddy_escape[not_in_eddy_anymore] = 0.0
                                 
            # Update in_eddy status based on conditions
            already_in_eddy = self.simulation.in_eddy == True
            self.simulation.in_eddy = np.where(np.logical_or(stuck_conditions,already_in_eddy), True, False) 
            self.simulation.in_eddy[not_in_eddy_anymore] = False
            self.simulation.time_since_eddy_escape[self.simulation.in_eddy == True] += 1
            
            if np.any(self.simulation.in_eddy):
                print ('check eddy escape calcs')
            
        def arbitrate(self,t):

            """
            Arbitrates between different behavioral cues to determine a new heading for each agent.
            This method considers the agent's behavioral mode and prioritizes different sensory inputs
            to calculate the most appropriate heading.
        
            Parameters
            ----------
            t : int
                The current time step in the simulation, used for cues that depend on historical data.
        
            Returns
            -------
            None
                This method updates the agent's heading in place.
        
            Notes
            -----
            - The method calculates various steering cues with predefined weights, such as rheotaxis,
              shallow water preference, wave drag, low-speed areas, historical avoidance, schooling behavior,
              and collision avoidance.
            - These cues are then organized into two dictionaries: `order_dict` which maintains the order
              of behavioral cues based on their importance, and `cue_dict` which holds all the steering cues.
            - The `f4ck_allocate` method is vectorized and called for each agent to allocate a limited
              amount of 'effort' to these cues, resulting in a new heading based on the agent's current
              swimming behavior.
            - The agent's heading is updated in place, reflecting the new direction based on the arbitration
              of cues.
        
            Example
            -------
            # Assuming an instance of the class is created and initialized as `agent_model`:
            agent_model.arbitrate(t=current_time_step)
            # The agent's heading is updated based on the arbitration of behavioral cues.
            """            
            # calculate behavioral cues
            if self.simulation.pid_tuning == True:
                rheotaxis = self.rheo_cue(50000)

            else:
                # calculate attractive forces
                rheotaxis = self.rheo_cue(25000)          # 25000
                alignment = self.alignment_cue(20500)     # 20500
                cohesion = self.cohesion_cue(11000)       # 11000
                low_speed = self.vel_cue(1500)             # 500 
                wave_drag = self.wave_drag_cue(0)         # 0                
                refugia = self.find_nearest_refuge(50000) # 50000
                # calculate high priority repusive forces
                border = self.border_cue(50000, t)        # 50000
                shallow = self.shallow_cue(100000)        # 100000
                avoid = self.already_been_here(25000, t)  # 25000
                collision = self.collision_cue(50000)     # 50000 
            
            # Create dictionary that has order of behavioral cues
            order_dict = {0: 'shallow',
                          1: 'border',
                          2: 'avoid',
                          3: 'collision', 
                          4: 'alignment', 
                          5: 'cohesion',
                          6: 'low_speed',
                          7: 'rheotaxis',  
                          8: 'wave_drag'}
            
            # Create dictionary that holds all steering cues
            cue_dict = {'rheotaxis': rheotaxis, 
                        'shallow': shallow, 
                        'border': border.T,
                        'wave_drag': wave_drag.T, 
                        'low_speed': low_speed.T, 
                        'avoid': avoid, 
                        'alignment': alignment,
                        'cohesion': cohesion,
                        'collision': collision,
                        'refugia': refugia}
            
            low_bat_cue_dict = {0:'shallow',
                                1:'border',
                                2:'refugia'}
            
            self.is_in_eddy(t)
            
            # Arbitrate between different behaviors
            # how many f4cks does this fish have?
            tolerance = 50000
            
            # add up vectors, but make sure it's not greater than the tolerance
            vec_sum_migratory = np.zeros_like(rheotaxis)
            vec_sum_tired = np.zeros_like(rheotaxis)
                    
            for i in order_dict.keys():
                cue = order_dict[i]
                vec = cue_dict[cue]
                if cue != 'refugia':
                    #print ('in school cue:%s'%(cue))
                    vec_sum_migratory = np.where(np.linalg.norm(vec_sum_migratory, axis = -1)[:,np.newaxis] < tolerance,
                                       vec_sum_migratory + vec,
                                       vec_sum_migratory)
                        
            for i in np.arange(0,3,1):
                cue = low_bat_cue_dict[i]
                vec = cue_dict[cue]
                vec_sum_tired = np.where(np.linalg.norm(vec_sum_tired, axis = -1)[:,np.newaxis] < tolerance,
                                   vec_sum_tired + vec,
                                   vec_sum_tired)
                        
            # now creating a heading vector for each fish - which is complicated because they are in different behavioral modes 
            head_vec = np.zeros_like(rheotaxis)
            
            # when actively migrating
            head_vec = np.where(self.simulation.swim_behav[:,np.newaxis] == 1,
                                vec_sum_migratory,
                                head_vec)
            
            # when fish is tired and looking for refugia
            head_vec = np.where(self.simulation.swim_behav[:,np.newaxis] == 2, 
                                vec_sum_tired,
                                head_vec)
            
            # when fish is tired and recovering
            head_vec = np.where(self.simulation.swim_behav[:,np.newaxis] == 3, 
                                vec_sum_tired,
                                head_vec)
            
            # for those unfortunate souls lost in eddies
            head_vec = np.where(self.simulation.in_eddy[:,np.newaxis] == 1, 
                                cue_dict['border'] + cue_dict['shallow'],
                                head_vec)
            
            if len(head_vec.shape) == 2:
                return np.arctan2(head_vec[:, 1], head_vec[:, 0])
            else:
                return np.arctan2(head_vec[:, 0, 1], head_vec[:, 0, 0])        

    class fatigue():
        '''
        A class dedicated to managing the fatigue and related physiological 
        parameters of a simulated fish population based on dynamic interactions 
        with their environment.
    
        Attributes:
            t (float): The current time in the simulation.
            dt (float): The time step increment for the simulation.
            simulation (object): An instance of another class handling specific 
            simulation details such as fish velocities, positions, and other metrics.
        '''
        
        def __init__ (self, t, dt, simulation_object):
            '''
            Initializes the fatigue class with time, timestep and a reference to 
            the simulation object.
            
            Parameters:
                t (float): The current time in the simulation.
                dt (float): The time step increment for the simulation.
                simulation_object (object): An instance of the class that handles 
                the environmental and biological simulation details.
            '''
            
            self.t = t
            self.dt = dt
            self.simulation = simulation_object
            self.fatigued_once = np.zeros(self.simulation.num_agents, dtype=bool)
            
            
        def swim_speeds(self):
            '''
            Calculates the swim speeds for each fish by considering the difference 
            between the fish's velocity and the water velocity.
            
            Returns:
                numpy.ndarray: Array of swim speeds for each fish.
            '''
            # Vector components of water velocity and speed over ground for each fish
            water_velocities = np.column_stack((self.simulation.x_vel, 
                                                self.simulation.y_vel))
            fish_velocities = np.column_stack((self.simulation.sog * np.cos(self.simulation.heading),
                                               self.simulation.sog * np.sin(self.simulation.heading)))
        
            # Calculate swim speeds for each fish
            swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
            
            # Shift all columns in the array one position to the left
            self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]
        
            # Insert the new swim speeds into the last column
            self.simulation.swim_speeds[:, -1] = np.linalg.norm(fish_velocities, axis = -1)
            
            return swim_speeds
        
        def bl_s(self, swim_speeds):
            '''
            Calculates the number of body lengths per second a fish swims.
            
            Parameters:
                swim_speeds (numpy.ndarray): The swim speeds of each fish as an array.
            
            Returns:
                numpy.ndarray: The number of body lengths per second for each fish.
            '''
            # calculate body lenghts per second
            bl_s = swim_speeds / (self.simulation.length/1000.)
            
            return bl_s

        def bout_distance(self):
            '''
            Updates the total distance traveled by each fish in a bout, and 
            increments the duration of the bout.
            
            Side effects:
                Modifies instance attributes related to the distance traveled 
                per bout and bout duration.
            '''
            # Calculate distances travelled and update bout odometer and duration
            dist_travelled = np.sqrt((self.simulation.prev_X - self.simulation.X)**2 + \
                                     (self.simulation.prev_Y - self.simulation.Y)**2)
                
            if len(dist_travelled.shape) == 1:
                self.simulation.dist_per_bout += dist_travelled
            else:
                self.simulation.dist_per_bout += dist_travelled.flatten()

            self.simulation.bout_dur += self.dt
            
        def time_to_fatigue(self, swim_speeds, mask_dict, method = 'CastroSantos'):
            '''
            Calculates the time to fatigue for each fish based on their current 
            swimming speeds and the selected fatigue model.
            
            Parameters:
                swim_speeds (numpy.ndarray): Array of current swim speeds for each fish.
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
                method (str): The fatigue model to apply. Default is 'CastroSantos'.
            
            Returns:
                numpy.ndarray: Array of time to fatigue for each fish.
            '''
            # Initialize time to fatigue (ttf) array
            ttf = np.full_like(swim_speeds, np.nan)
            
            if method == 'CastroSantos':
                a_p = self.simulation.a_p
                b_p = self.simulation.b_p
                a_s = self.simulation.a_s
                b_s = self.simulation.b_s
                lengths = self.simulation.length
                
                # Implement T Castro Santos (2005)
                ttf = np.where(mask_dict['prolonged'], 
                               np.exp(a_p + swim_speeds * b_p),
                               ttf)
                
                ttf = np.where(mask_dict['sprint'], 
                               np.exp(a_s + swim_speeds * b_s),
                               ttf)
                
                return ttf
                
            elif method == 'Katapodis_Gervais':
                
                genus = 'Oncorhyncus'
                                                                                            
                # Regression parameters extracted from the document, indexed by species or group
                regression_params = {
                    'Oncorhyncus':{'K':3.5825,'b':-0.2621}
                }
                
                if genus in regression_params:
                    k = 6.3234 #regression_params[genus]['K'] # at upper 95% CI  
                    b = regression_params[genus]['b']
                else:
                    raise ValueError ("Species not found. Please use a species from the provided list or check the spelling.")
                    sys.exit()
                    
                # Calculate time to fatigue using the regression equation
                ttf = np.zeros(self.num_agents)
                ttf[~mask_dict['sustained']] = \
                    (swim_speeds[~mask_dict['sustained']]/k)** (1/b)
                
                return ttf
            
            else:
                raise ValueError ('emergent does not recognize %s the method passed'%(method))
                sys.exit()   
                
        def set_swim_mode(self, mask_dict):
            '''
            Sets the swim mode for each fish based on the provided behavior masks.
            
            Parameters:
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
            '''
            # Set swimming modes based on swim speeds
            mask_prolonged = mask_dict['prolonged']
            mask_sprint = mask_dict['sprint']
            
            # set swim mode
            self.simulation.swim_mode = np.where(mask_prolonged, 
                                                 2, 
                                                 self.simulation.swim_mode)
            self.simulation.swim_mode = np.where(mask_sprint,
                                                 3, 
                                                 self.simulation.swim_mode)
            self.simulation.swim_mode = np.where(~(mask_prolonged | mask_sprint), 
                                                 1, 
                                                 self.simulation.swim_mode)
            
        def recovery(self):
            '''
            Calculates the recovery percentage for each fish at the beginning 
            and end of the time step, with a faster recovery at first that slows down 
            as the fish's battery approaches full charge.
            
            Returns:
                numpy.ndarray: Array of recovery percentages for each fish.
            '''
            recovery_duration = 30 * 60  # 45 minutes in seconds
        
            # Calculate recovery at the beginning and end of the time step
            battery_level = self.simulation.battery  # Current battery level for each fish
            rec0 = (1 - battery_level) * np.exp(-self.simulation.recover_stopwatch / recovery_duration)
            rec0[rec0 < 0.0] = 0.0
            
            rec1 = (1 - battery_level) * np.exp(-(self.simulation.recover_stopwatch + self.dt) / recovery_duration)
            rec1[rec1 > 1.0] = 1.0
            rec1[rec1 < 0.0] = 0.0
            
            per_rec = rec1 - rec0
        
            # Recovery for fish that are station holding
            mask_station_holding = self.simulation.swim_behav == 3
            self.simulation.bout_dur[mask_station_holding] = 0.0
            self.simulation.dist_per_bout[mask_station_holding] = 0.0
            self.simulation.battery[mask_station_holding] += per_rec[mask_station_holding]
            self.simulation.recover_stopwatch[mask_station_holding] += self.dt
        
            return per_rec

        def calc_battery(self, per_rec, ttf, mask_dict):
            '''
            Updates the battery levels for each fish based on their swimming mode
            and recovery.
            
            Parameters:
                dt (float): The time step of the simulation.
                per_rec (numpy.ndarray): Array of percentages representing recovery 
                for each fish.
                ttf (numpy.ndarray): Array of time to fatigue values for each fish.
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
            '''
            
            # get fish that are swimming at a sustained level
            mask_sustained = mask_dict['sustained']
            
            # Update battery levels for sustained swimming mode
            if mask_sustained.ndim == 2:
                mask_sustained = mask_sustained.squeeze()
            if self.simulation.num_agents > 1:
                self.simulation.battery[mask_sustained] += per_rec[mask_sustained]
            else:
                self.simulation.battery[mask_sustained.flatten()] += per_rec[mask_sustained.flatten()]
        
            # Update battery levels for non-sustained swimming modes
            mask_non_sustained = ~mask_sustained
            if self.simulation.num_agents > 1:
                 ttf0 = ttf[mask_non_sustained] * self.simulation.battery[mask_non_sustained]
            else:
                ttf0 = ttf[mask_non_sustained.flatten()] * self.simulation.battery[mask_non_sustained.flatten()]
    
            ttf1 = ttf0 - self.dt
            if self.simulation.num_agents > 1:
                self.simulation.battery[mask_non_sustained] *= np.nan_to_num(ttf1 / ttf0)
            else:
                self.simulation.battery[mask_non_sustained.flatten()] *= ttf1.flatten() / ttf0.flatten()            
            
            self.simulation.battery = np.clip(self.simulation.battery, 0, 1)
            
        def set_swim_behavior(self, battery_state_dict):
            '''
            Adjusts the swim behavior of the fish based on their current battery state.
            
            Parameters:
                battery_state_dict (dict): Dictionary defining categories of battery states.
            '''
            # Set swimming behavior based on battery level
            mask_low_battery = battery_state_dict['low'] 
            mask_mid_battery = battery_state_dict['mid'] 
            mask_high_battery = battery_state_dict['high'] 
        
            self.simulation.swim_behav = np.where(mask_low_battery, 
                                                  3, 
                                                  self.simulation.swim_behav)
            
            self.simulation.swim_behav = np.where(mask_mid_battery, 
                                                  2,
                                                  self.simulation.swim_behav)
            
            self.simulation.swim_behav = np.where(mask_high_battery,
                                                  1,
                                                  self.simulation.swim_behav)
            
        def set_ideal_sog(self, mask_dict, battery_state_dict):
            '''
            Calculates the optimal swim speeds for each fish based on vector flow speeds and mode.
            
            Parameters:
                flow_velocities (numpy.ndarray): The flow speeds experienced by each fish (n x 2 array).
            
            Returns:
                numpy.ndarray: Optimal swim speeds for each fish (n x 2 array).
            '''
            # Set swimming behavior based on battery level
            mask_low_battery = battery_state_dict['low'] 
            mask_mid_battery = battery_state_dict['mid'] 
            mask_high_battery = battery_state_dict['high'] 
            
            self.simulation.ideal_sog[mask_high_battery] = np.where(self.simulation.battery[mask_high_battery] == 1., 
                                                                    self.simulation.school_sog[mask_high_battery], 
                                                                    np.round((self.simulation.opt_sog[mask_high_battery] * \
                                                                             self.simulation.battery[mask_high_battery])/2, 2)
                                                                    )

            # Set ideal speed over ground based on battery level
            self.simulation.ideal_sog[mask_low_battery] = 0.0
            self.simulation.ideal_sog[mask_mid_battery] = 0.1
            
            # if np.any(mask_dict['sprint']):
            #     print ('fuck')

        def ready_to_move(self):
            '''
            Determines which fish are ready to resume movement based on their recovery state.
            
            Side effects:
                Modifies simulation attributes related to the movement readiness of the fish.
            '''
            # Fish ready to start moving again after recovery
            mask_ready_to_move = self.simulation.battery >= 0.85
            self.simulation.recover_stopwatch[mask_ready_to_move] = 0.0
            self.simulation.swim_behav[mask_ready_to_move] = 1
            self.simulation.swim_mode[mask_ready_to_move] = 1
            
        def PID_checks(self):
            '''
            Performs PID (Proportional-Integral-Derivative) checks if PID tuning
            is active in the simulation, primarily for debugging and optimization purposes.
            '''
            if self.simulation.pid_tuning == True:
                print(f'battery: {np.round(self.battery,4)}')
                print(f'swim behavior: {self.swim_behav[0]}')
                print(f'swim mode: {self.swim_mode[0]}')

                if np.any(self.simulation.swim_behav == 3):
                    print('error no longer counts, fatigued')
                    sys.exit()

        def handle_fatigue_recovery(self, battery_dict):
            '''
            Updates swim behavior of fish based on battery levels and fatigue history.
            
            Fish that have fatigued will stay in station holding (swim_behav == 3) until
            their battery reaches 10%, then switch to swim behavior 2.
            Fish will stay in swim behavior 2 until their battery reaches 85%, then switch to swim behavior 1.
            The first time fish fatigue, they are allowed to fully drain their battery.
            '''
            # Masks for fish that are recovering and have been fatigued before
            mask_station_holding = self.simulation.swim_behav == 3
            mask_recovering_low = (self.simulation.battery > 0.30) & (self.simulation.battery < 0.85)  # Battery between 10% and 85%
            mask_ready_to_swim = self.simulation.battery >= 0.85  # Battery above 85%
            mask_fatigued_once = self.fatigued_once
            
            # Fish that have been fatigued once and have a battery above 10% but less than 85% should switch to swim behavior 2
            mask_switch_to_swim_2 = mask_fatigued_once & mask_station_holding & mask_recovering_low
            self.simulation.swim_behav[mask_switch_to_swim_2] = 2  # Switch to swim behavior 2
        
            # Fish that have recovered to 85% or more and were previously in swim behavior 2 can start swimming (behavior 1)
            mask_switch_to_swim_1 = mask_fatigued_once & (self.simulation.battery >= 0.85)
            self.simulation.swim_behav[mask_switch_to_swim_1] = 1  # Switch to swim behavior 1
        
            # Fish that are fatigued and have battery less than or equal to 10% stay in swim behavior 3 (station holding)
            mask_stay_station_holding = mask_fatigued_once & (self.simulation.battery <= 0.10)
            self.simulation.swim_behav[mask_stay_station_holding] = 3  # Stay in station holding
        
            # Update the fatigued_once array: If a fish has drained its battery to 0, it marks the first fatigue
            self.fatigued_once[self.simulation.battery <= 0.0] = True

        def assess_fatigue(self):
            '''
            Comprehensive method to assess fatigue based on swim speeds, 
            calculates distances traveled, updates time to fatigue, recovery,
            and adjusts battery states accordingly.
            '''            
            # get swim speeds
            swim_speeds = self.swim_speeds()
            bl_s = self.bl_s(swim_speeds)
            
            # Calculate ttf for prolonged and sprint swimming modes
            mask_dict = dict()
            
            mask_dict['prolonged'] = np.where((self.simulation.max_s_U < bl_s) & (bl_s <= self.simulation.max_p_U),
                                              True,
                                              False)
            
            mask_dict['sprint'] = np.where(bl_s > self.simulation.max_p_U,
                                           True,
                                           False)
            
            mask_dict['sustained'] = bl_s <= self.simulation.max_s_U
            
            # calculate how far this fish has traveled this bout
            self.bout_distance()
            
            # assess time to fatigue
            ttf = self.time_to_fatigue(bl_s, mask_dict)
            
            # set swim mode
            self.set_swim_mode(mask_dict)
            
            # assess recovery
            per_rec = self.recovery()
            
            # check battery
            self.calc_battery(per_rec, ttf, mask_dict)
            
            # set battery masks
            battery_dict = dict()
            battery_dict['low'] = self.simulation.battery <= 0.1
            battery_dict['mid'] = (self.simulation.battery > 0.1) & (self.simulation.battery <= 0.3)
            battery_dict['high'] = self.simulation.battery > 0.4
            
            # Handle swim behavior based on battery and fatigue state
            self.handle_fatigue_recovery(battery_dict)
        
            # calculate ideal speed over ground
            self.set_ideal_sog(mask_dict, battery_dict)
            
            # are fatigued fish ready to move?
            self.ready_to_move()
            
            # perform PID checks if we are optimizing controller
            self.PID_checks()
        
            
    def timestep(self, t, dt, g, pid_controller, success_line_x, fallback_line_x):
        """
        Simulates a single time step for all fish in the simulation.
        
        Parameters:
        - t: Current simulation time.
        - dt: Time step duration.
        - success_line_x: The easting boundary (longitude) that determines success.
        
        The method performs the following operations for each fish:
        ...
        """
        # Create movement, behavior, and fatigue objects
        movement = self.movement(self)
        behavior = self.behavior(t, self)
        fatigue = self.fatigue(t, dt, self)
        
        # Assess mental map
        self.update_mental_map(t)
        
        # Sense the environment
        self.environment()
        
        # Update refugia map
        self.update_refugia_map(self.vel_mag)
    
        # Optimize vertical position
        movement.find_z()
        
        # Get wave drag multiplier
        behavior.wave_drag_multiplier()
        
        # Assess fatigue
        fatigue.assess_fatigue()
        
        # Calculate the ratio of ideal speed over ground to the magnitude of water velocity
        sog_to_water_vel_ratio = self.sog / np.linalg.norm(np.stack((self.x_vel, self.y_vel)).T, axis=-1)
        
        # Calculate the time since the last jump
        time_since_jump = t - self.time_of_jump
        
        # Create a boolean mask for the fish that should jump
        should_jump = (self.wet == -9999.0) | (
            (sog_to_water_vel_ratio <= 0.10) &
            (time_since_jump > 60) &
            (self.battery >= 0.25)
        )
        
        # Apply the jump or swim functions based on the condition
        dxdy_jump = movement.jump(t=t, g=g, mask=should_jump)
        movement.drag_fun(mask=~should_jump, t=t, dt=dt)
        movement.thrust_fun(mask=~should_jump, t=t, dt=dt)
        dxdy_swim = movement.swim(t, dt, pid_controller=pid_controller, mask=~should_jump)
        
        # Arbitrate among behavioral cues
        self.heading = behavior.arbitrate(t)
        
        # Store previous positions
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()
        
        # Move fish that are not successful
        success_mask = (self.X < success_line_x) | (self.X > fallback_line_x)  # Fish that have crossed the success line
        self.X = np.where(success_mask, self.X, self.X + dxdy_swim[:, 0] + dxdy_jump[:, 0])
        self.Y = np.where(success_mask, self.Y, self.Y + dxdy_swim[:, 1] + dxdy_jump[:, 1])

        # Stop fish that have crossed the success line
        self.sog = np.where(success_mask, 0, np.sqrt(np.power(self.X - self.prev_X, 2) + np.power(self.Y - self.prev_Y, 2)) / dt)
        
        if np.any(np.isnan(self.X)):
            print('fish off map - why?')
            bad_fish = np.argmax(np.linalg.norm(dxdy_swim, axis = 1))
            print('dxdy swim:', np.linalg.norm(dxdy_swim, axis = 1).max())
            print('dxdy jump:', np.linalg.norm(dxdy_jump, axis = 1).max())
            print(f'fish {bad_fish} in {self.swim_mode} swimming mode exhibiting {self.swim_behav} behavior')
            print(f'fish {bad_fish} was previously at {self.prev_X},{self.prev_Y} and landed on: {self.X},{self.Y}')
            sys.exit()
        
        # Calculate mileage
        self.odometer(t=t, dt=dt)
        
        # Log the timestep data
        self.timestep_flush(t)
        
        # Accumulate time
        self.cumulative_time = self.cumulative_time + dt
          
    def run(self, model_name, n, dt, video = False, k_p = None, k_i = None, k_d = None):
        """
        Executes the simulation model over a specified number of time steps and generates a movie of the simulation.
        
        Parameters:
        - model_name: A string representing the name of the model, used for titling the output movie.
        - agents: A list of agent objects that will be simulated.
        - n: An integer representing the number of time steps to simulate.
        - dt: The duration of each time step.
        
        The function performs the following operations:
        1. Initializes the depth raster from the HDF5 dataset.
        2. Sets up the movie writer with metadata.
        3. Initializes the plot for the simulation visualization.
        4. Iterates over the specified number of time steps, updating the agents and capturing each frame.
        5. Cleans up resources and finalizes the movie file.
        
        The simulation uses raster data for depth and agent positions to visualize the movement of agents in the environment. The output is a movie file that shows the progression of the simulation over time.
        
        Note:
        - The function assumes that the HDF5 dataset, coordinate reference system (CRS), and depth raster transformation are already set as attributes of the class instance.
        - The function prints the completion of each time step to the console.
        - The movie is saved in the directory specified by `self.model_dir`.
        
        Returns:
        None. The result of the function is the creation of a movie file visualizing the simulation.
        """        
        t0 = time.time()
        if self.pid_tuning == False:
            if video == True:
                # get depth raster
                depth_arr = self.hdf5['environment/depth'][:]
                x_vel = self.hdf5['environment/vel_x'][:]
                y_vel = self.hdf5['environment/vel_y'][:]
                center = self.hdf5['environment/distance_to'][:]
                
                # assuming raster data has been masked in ArcGIS 
                no_data_value = -9999.
                depth_masked = np.ma.masked_equal(depth_arr, no_data_value)
                depth_masked = np.ma.masked_where(depth_masked < 1e-5, depth_masked) 
                x_vel_masked = np.ma.masked_equal(x_vel, no_data_value)
                y_vel_masked = np.ma.masked_equal(y_vel, no_data_value)

                # You need to get the bounds (left, bottom, right, top) from your raster's metadata
                extent = [
                    self.depth_rast_transform[2],  # Left: x offset
                    self.depth_rast_transform[2] + self.depth_rast_transform[0] * self.width,  # Right: x offset + (pixel width * width of the image)
                    self.depth_rast_transform[5] + self.depth_rast_transform[4] * self.height,  # Bottom: y offset + (pixel height * height of the image)
                    self.depth_rast_transform[5]  # Top: y offset
                ]

            
                #depth_arr = self.hdf5['environment/wetted_perim'][:]
                height = depth_arr.shape[0]
                width = depth_arr.shape[1]
            
                # # define metadata for movie
                FFMpegWriter = manimation.writers['ffmpeg']
                metadata = dict(title= model_name, artist='Matplotlib',
                                comment='emergent model run %s'%(datetime.now()))
                writer = FFMpegWriter(fps = np.round(30/dt,0), metadata=metadata)
        
                #initialize plot
                fig, ax = plt.subplots(figsize = (9,6), dpi = 250.)
                cmap = plt.cm.gray  # Or any colormap that suits your visualization
                cmap.set_bad(color='tan')  # This sets the color for 'masked' values; adjust as needed
                # Fixed frame size in data units - this will be the size of the view window
                initial_frame_size = 100  # Adjust as needed
                min_zoom_level = 5.0  # No zoom (closest view)
                max_zoom_level = 7.0  # Maximum zoom out level
                
                ax.imshow(depth_masked,
                          origin='upper',
                          cmap=cmap,
                          extent=extent)
                #ax.set_aspect(0.5) 
                
                # Subsampling factor - adjust this to reduce the density of the quiver plot
                subsample_factor = 10
                
                # Subsampled meshgrid for the quiver plot
                x, y = np.meshgrid(
                    np.arange(extent[0], extent[1], (extent[1] - extent[0]) / width),
                    np.arange(extent[2], extent[3], (extent[3] - extent[2]) / height)
                )

                # Subsampled velocities
                subsampled_x = x[::subsample_factor, ::subsample_factor]
                subsampled_y = y[::subsample_factor, ::subsample_factor]
                subsampled_x_vel = x_vel_masked[::subsample_factor, ::subsample_factor]
                subsampled_y_vel = y_vel_masked[::subsample_factor, ::subsample_factor]
                
                # Calculate half subsample distances for offsets
                half_subsample_x = (extent[1] - extent[0]) / width / subsample_factor / 2
                half_subsample_y = (extent[3] - extent[2]) / height / subsample_factor / 2
                
                # Adjust subsampled meshgrid coordinates to center the quiver arrows
                subsampled_x_centered = subsampled_x - subsample_factor / 2. # half_subsample_x
                subsampled_y_centered = subsampled_y + subsample_factor #/ 2. # half_subsample_y
                
                ax.quiver(subsampled_x_centered,
                          subsampled_y_centered[::-1],
                          subsampled_x_vel,
                          subsampled_y_vel,
                          scale=90, 
                          color='blue',
                          width = 0.0009) 

                agent_scatter = ax.scatter([], [], s=0.5)
                # Initialize text for displaying the timestep
                timestep_text = ax.text(0.01, 0.01, '', transform=ax.transAxes, color='white', fontsize=8, ha='left', va='bottom')

                plt.xlabel('Easting')
                plt.ylabel('Northing')
                
                dpi = fig.get_dpi()
                
                # Get current size in inches and compute size in pixels
                width_inch, height_inch = fig.get_size_inches()
                width_px, height_px = width_inch * dpi, height_inch * dpi
                
                # Ensure dimensions are even
                if width_px % 2 != 0:
                    width_inch += 1 / dpi
                if height_px % 2 != 0:
                    height_inch += 1 / dpi
                
                # Update figure size
                fig.set_size_inches(width_inch, height_inch)
                
                # Update the frames for the movie
                with writer.saving(fig, 
                                    os.path.join(self.model_dir,'%s.mp4'%(model_name)),
                                    dpi = dpi):
                
                    # set up PID controller 
                    pid_controller = PID_controller(self.num_agents,
                                                    k_p, 
                                                    k_i, 
                                                    k_d)
                    
                    pid_controller.interp_PID()
                    for i in range(int(n)):
                        self.timestep(i, dt, g, pid_controller,548700,550450)
                        # we want to follow the top performing fish - calculate the top 25% by longitude
                        
                        # Step 1: Determine the threshold for the top X%
                        sorted_indices = np.argsort(self.current_longitudes)  # Get indices that would sort the array
                        
                        # Index to slice the top 25%
                        top_25_percent_index = int(len(self.current_longitudes) * 0.75)
                        threshold_top_25 = self.current_longitudes[sorted_indices[top_25_percent_index]]
                        
                        # Index to slice the top 50%
                        top_50_percent_index = int(len(self.current_longitudes) * 0.50)
                        threshold_top_50 = self.current_longitudes[sorted_indices[top_50_percent_index]]
                        
                        # Index to slice the top 75%
                        top_75_percent_index = int(len(self.current_longitudes) * 0.25)
                        threshold_top_75 = self.current_longitudes[sorted_indices[top_75_percent_index]]
                        
                        # Step 2: Create a mask for the top 75%
                        mask = (self.current_longitudes >= threshold_top_50) & (self.dead != 1)
                        
                        # Step 3: Calculate the mean x and y positions for the top 75%
                        center_x = np.median(self.X[mask])
                        center_y = np.median(self.Y[mask])
                        
                        # Step 4: Calculate the spread of agents using standard deviation
                        spread_x = np.std(self.X[mask])
                        spread_y = np.std(self.Y[mask])
                        spread = max(spread_x, spread_y)  # Use the larger spread in case the distribution is elongated
                
                        # Map the spread to a zoom level within the defined range
                        spread_normalized = (spread - np.min([spread_x, spread_y])) / (np.max([spread_x, spread_y]) - np.min([spread_x, spread_y]))
                        zoom_level = min_zoom_level + (max_zoom_level - min_zoom_level) * spread_normalized
                        
                        # Calculate dynamic frame size based on the zoom level
                        dynamic_frame_size = initial_frame_size * zoom_level
                    
                        
                        # Dynamic span for the x-axis based on zoom level or data distribution
                        x_span = dynamic_frame_size  # This can be adjusted based on your zoom logic
                        
                        # Calculate y-span to maintain a 10x5 aspect ratio
                        y_span = x_span / 2  # To maintain the 10x5 aspect ratio
    
                        # Set the axes limits while maintaining the aspect ratio
                        ax.set_xlim(center_x - x_span / 2, center_x + x_span / 2)
                        ax.set_ylim(center_y - y_span / 2, center_y + y_span / 2)            
                        
                        # Update timestep display
                        timestep_text.set_text(f'Timestep: {i}')
                        
                        # Calculate the RGB colors using vectorized operations
                        # Green (0, 1, 0) to Red (1, 0, 0) based on the battery state
                        colors = np.column_stack([1 - self.battery, 
                                                  self.battery, 
                                                  np.zeros_like(self.battery)])
                        
                        # Overriding the color for dead agents to magenta (1, 0, 1)
                        colors[self.dead == 1] = [1, 0, 1]  # Magenta
        
                        # Update the positions ('offsets') of the agents in the scatter plot
                        agent_scatter.set_offsets(np.column_stack([self.X, self.Y]))
                        # Update the colors of each agent
                        try:
                            agent_scatter.set_facecolor(colors)
                        except ValueError:
                            pass
        
                        # write frame
                        plt.tight_layout()
                        
                        dpi = fig.get_dpi()
                        
                        # Get current size in inches and compute size in pixels
                        width_inch, height_inch = fig.get_size_inches()
                        width_px, height_px = width_inch * dpi, height_inch * dpi
                        
                        # Ensure dimensions are even
                        if width_px % 2 != 0:
                            width_inch += 1 / dpi
                        if height_px % 2 != 0:
                            height_inch += 1 / dpi
                        
                        # Update figure size
                        fig.set_size_inches(width_inch, height_inch)
                        
                        writer.grab_frame()
                        plt.draw()
                        plt.pause(0.01) 
                            
                        print ('Time Step %s complete'%(i))
                
                    # clean up
                    writer.finish()
                    self.hdf5.flush()
                    self.hdf5.close()
                    #depth.close()     
                    t1 = time.time() 
        
            else:
                #TODO make PID controller a function of length and water velocity
                pid_controller = PID_controller(self.num_agents,
                                                k_p, 
                                                k_i, 
                                                k_d)
                
                pid_controller.interp_PID()
                
                # iterate over timesteps 
                for i in range(int(n)):
                    self.timestep(i, dt, g, pid_controller,548700)
                    print ('Time Step %s complete'%(i))
                    
                # close and cleanup
                self.hdf5.flush()
                self.hdf5.close()
                t1 = time.time() 
                    
        else:
            pid_controller = PID_controller(self.num_agents,
                                            k_p, 
                                            k_i, 
                                            k_d)
            for i in range(n):
                self.timestep(i, dt, g, pid_controller,548700)
                
                print ('Time Step %s %s %s %s %s %s complete'%(i,i,i,i,i,i))
                
                if i == range(n)[-1]:
                    self.hdf5.close()
                    sys.exit()
        
        print ('ABM took %s to compile'%(t1-t0))

        
    def close(self):
        self.hdf5.flush()  
        self.hdf5.close()
            

            
class summary:
    '''The power of an agent based model lies in its ability to produce emergent
    behavior of interest to managers.  novel self organized patterns that 
    only happen once are a consequence, predictable self organized patterns  
    are powerful.  Each Emergent simulation should be run no less than 30 times. 
    This summary class object is designed to iterate over a parent directory, 
    extract data from child directories, and compile statistics.  The parent 
    directory describes a single scenario (for sockeye these are discharge)
    while each child directory is an individual iteration.  
    
    The class object iterates over child directories, extracts and manipulates data,
    calculate basic descriptive statistics, manages information, and utilizes 
    Poisson kriging to produce a surface that depicts the average number of agents
    per cell per second.  High use corridors should be visible in the surface.
    These corridors are akin to the desire paths we see snaking through college
    campuses and urban parks the world over.
    
    '''
    def __init__(self, parent_directory, tif_path):
        # set the model directory path
        self.parent_directory = parent_directory
        
        # where are the background tiffs stored?
        self.tif_path = tif_path
        
        #set input WS as parent_directory for compatibility with methods
        self.inputWS = parent_directory
        
        # get h5 files associated with this model
        self.h5_files = self.find_h5_files()
        
        # create empty thigs to hold agent data
        self.ts = gpd.GeoDataFrame(columns = ['agent','timestep','X','Y','kcal','Hz','filename','geometry'])
        self.morphometrics = pd.DataFrame()
        self.success_rates = {}

    def load_tiff(self, crs):
        # Define the desired CRS
        desired_crs = CRS.from_epsg(crs)

        # Open the TIFF file with rasterio
        with rasterio.open(self.tif_path) as tiff_dataset:
            # Calculate the transformation parameters for reprojecting
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)
            
            cell_size = 2.
            # Calculate the new transform for 10x10 meter resolution
            new_transform = from_origin(transform.c, transform.f, cell_size, cell_size)
            
            # Calculate new width and height
            new_width = int((tiff_dataset.bounds.right - tiff_dataset.bounds.left) / cell_size)
            new_height = int((tiff_dataset.bounds.top - tiff_dataset.bounds.bottom) / cell_size)
            
            self.transform = new_transform
            self.width = new_width
            self.height = new_height

            # Reproject the TIFF image to the desired CRS
            image_data, _ = reproject(
                source=tiff_dataset.read(1),
                src_crs=tiff_dataset.crs,
                src_transform=tiff_dataset.transform,
                dst_crs=desired_crs,
                resampling=rasterio.enums.Resampling.bilinear)

            # Update the extent based on the reprojected data
            tiff_extent = rasterio.transform.array_bounds(height, width, transform)

        return image_data, tiff_extent
   
    # Find the .h5 files
    def find_h5_files(self):
        
        # create empty holders for all of the h5 files and child directories
        h5_files=[]
        child_dirs = []
        
        # first iterate over the parent diretory to find the children (iterations)
        for item in os.listdir(self.parent_directory):
            # create a full path object
            full_path = os.path.join(self.parent_directory, item)
            
            if full_path.endswith('.h5'):
                h5_files.append(full_path)
            
            # if full path is a directory and not a file - we found a child
            if os.path.isdir(full_path):
                child_dirs.append(full_path)
        
        # iterate over child directories and find the h5 files
        for child_dir in child_dirs:
            for filename in os.listdir(child_dir):
                if filename.endswith('.h5'):
                    h5_files.append(os.path.join(child_dir,filename))
        
        # we found our files
        return h5_files
        
    # Collect, rearrange, and manage data
    def get_data(self, h5_files):

        # Iterate through each HDF5 file in the specified directory and get data
        for filename in h5_files:

            with h5py.File(filename, 'r') as hdf:
                cell_center_x = pd.DataFrame(hdf['x_coords'][:])
                cell_center_x['row'] = np.arange(len(cell_center_x))
                cell_center_y = pd.DataFrame(hdf['y_coords'][:])
                cell_center_y['row'] = np.arange(len(cell_center_y))

                melted_center_x = pd.melt(cell_center_x, id_vars = ['row'], var_name = 'column', value_name = 'X')
                melted_center_y = pd.melt(cell_center_y, id_vars = ['row'], var_name = 'column', value_name = 'Y')
                melted_center = pd.merge(melted_center_x, melted_center_y, on = ['row','column'])
                self.melted_center = melted_center
                self.x_coords = hdf['x_coords'][:]
                self.y_coords = hdf['y_coords'][:]
                              
                if 'agent_data' in hdf:
                    # timestep data
                    X = pd.DataFrame(hdf['agent_data/X'][:])
                    X['agent'] = np.arange(X.shape[0])
                    Y = pd.DataFrame(hdf['agent_data/Y'][:])
                    Y['agent'] = np.arange(Y.shape[0])
                    Hz = pd.DataFrame(hdf['agent_data/Hz'][:])
                    Hz['agent'] = np.arange(Hz.shape[0])
                    kcal = pd.DataFrame(hdf['agent_data/kcal'][:])
                    kcal['agent'] = np.arange(kcal.shape[0])  
                    
                    # agent specific 
                    length = pd.DataFrame(hdf['agent_data/length'][:])
                    length['agent'] = np.arange(len(length))
                    length.rename(mapper = {0:'length'}, axis = 'columns', inplace = True)
                    
                    weight = pd.DataFrame(hdf['agent_data/weight'][:])
                    weight['agent'] = np.arange(len(weight))
                    weight.rename(mapper = {0:'weight'}, axis = 'columns', inplace = True)

                    body_depth = pd.DataFrame(hdf['agent_data/body_depth'][:])
                    body_depth['agent'] = np.arange(len(body_depth))
                    body_depth.rename(mapper = {0:'body_depth'}, axis = 'columns', inplace = True)

                    # melt time series data
                    melted_X = pd.melt(X, id_vars=['agent'], var_name='timestep', value_name='X')
                    melted_Y = pd.melt(Y, id_vars=['agent'], var_name='timestep', value_name='Y')
                    melted_kcal = pd.melt(kcal, id_vars=['agent'], var_name='timestep', value_name='kcal')
                    melted_Hz = pd.melt(Hz, id_vars=['agent'], var_name='timestep', value_name='Hz')
                    
                    # make one dataframe 
                    ts = pd.merge(melted_X, melted_Y, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_kcal, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_Hz, on = ['agent','timestep'])
                    ts['filename'] = filename
                    
                    print ('Data Imported ')
                    # turn ts into a geodataframe and find the fish that passed
                    geometry = [Point(xy) for xy in zip(ts['X'], ts['Y'])]
                    geo_ts = gpd.GeoDataFrame(ts, geometry=geometry)            
                    
                    # make one morphometric dataframe
                    morphometrics = pd.merge(length, weight, on = ['agent'])
                    morphometrics = pd.merge(morphometrics, body_depth, on = ['agent'])
                    
                    # add to summary data
                    self.ts = pd.concat([self.ts,geo_ts], ignore_index = True)
                    self.morphometrics = pd.concat([self.morphometrics,morphometrics], 
                                                  ignore_index = True) 
                    
                    print ('File %s imported'%(filename))
                    
                    
                    
    # Collect histograms of agent lengths
    def plot_lengths(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            pdf_filename = f"{base_name}_Lengths_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(output_folder, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        lengths = file['/agent_data/length'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            lengths_by_sex = lengths[sex_mask]
                            lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                            if lengths_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(lengths_by_sex, [75, 25])
                                    bin_width = 2 * (q75 - q25) * len(lengths_by_sex) ** (-1 / 3)

                                    if bin_width <= 0 or np.isnan(bin_width):
                                        bin_width = (max(lengths_by_sex) - min(lengths_by_sex)) / 10

                                    bins = max(1, round((max(lengths_by_sex) - min(lengths_by_sex)) / bin_width))
                                    ax.hist(lengths_by_sex, bins=bins, alpha=0.7, color='blue' if sex == 0 else 'pink')
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    continue

                                ax.set_title(f'{base_name} - {sex_label} Agent Lengths')
                                ax.set_xlabel('Length (mm)')
                                ax.set_ylabel('Frequency')
                                plt.tight_layout()
                                pdf.savefig(fig)
                                plt.close()
                            else:
                                print(f"No length values found for {sex_label}.")

    def length_statistics(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            stats_file_name = f"{base_name}_length_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(h5_file, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    lengths = file['/agent_data/length'][:]
                    sexes = file['/agent_data/sex'][:]

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        lengths_by_sex = lengths[sex_mask]
                        lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                        if lengths_by_sex.size > 1:
                            mean_length = np.mean(lengths_by_sex)
                            median_length = np.median(lengths_by_sex)
                            std_dev_length = np.std(lengths_by_sex, ddof=1)
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\n")
                            output_file.write(f"  Median Length: {median_length:.2f}\n")
                            output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\n\n")
                        elif lengths_by_sex.size == 1:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Length: {lengths_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid length values found for {sex_label}.\n\n")

    def plot_weights(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_directory = os.path.dirname(h5_file)
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            pdf_filename = f"{base_name}_Weights_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(base_directory, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        weights = file['/agent_data/weight'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            weights_by_sex = weights[sex_mask]
                            weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]

                            if weights_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(weights_by_sex, [75, 25])
                                    iqr = q75 - q25
                                    if iqr > 0:
                                        bin_width = 2 * iqr * len(weights_by_sex) ** (-1 / 3)
                                        bins = max(1, round((max(weights_by_sex) - min(weights_by_sex)) / bin_width))
                                    else:
                                        bins = 10

                                    ax.hist(weights_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                    ax.set_title(f'{base_name} - {sex_label} Agent Weights')
                                    ax.set_xlabel('Weight')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    pdf.savefig(fig)
                                    plt.close()
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    plt.close(fig)
                            else:
                                print(f"No weight values found for {sex_label} in {base_name}.")

    def weight_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_weight_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    weights = file['/agent_data/weight'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        weights_by_sex = weights[sex_mask]
                        weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]  # Filter out NaN values

                        if weights_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                            mean_weight = np.mean(weights_by_sex)
                            median_weight = np.median(weights_by_sex)
                            std_dev_weight = np.std(weights_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Weight: {mean_weight:.2f}\n")
                            output_file.write(f"  Median Weight: {median_weight:.2f}\n")
                            output_file.write(f"  Standard Deviation of Weight: {std_dev_weight:.2f}\n\n")
                        elif weights_by_sex.size == 1:
                            # Handle single value case
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Weight: {weights_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid weight values found for {sex_label}.\n\n")

    def plot_body_depths(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            pdf_filename = f"{base_name}_Body_Depths_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(output_folder, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(hdf_path, 'r') as file:
                    if 'agent_data' in file:
                        body_depths = file['/agent_data/body_depth'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            body_depths_by_sex = body_depths[sex_mask]
                            body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]

                            if body_depths_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(body_depths_by_sex, [75, 25])
                                    iqr = q75 - q25
                                    if iqr > 0:
                                        bin_width = 2 * iqr * len(body_depths_by_sex) ** (-1 / 3)
                                    else:
                                        bin_width = (max(body_depths_by_sex) - min(body_depths_by_sex)) / max(10, len(body_depths_by_sex))  # Avoid zero division

                                    bins = max(1, round((max(body_depths_by_sex) - min(body_depths_by_sex)) / bin_width))
                                    ax.hist(body_depths_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                    ax.set_title(f'{base_name} - {sex_label} Body Depths')
                                    ax.set_xlabel('Body Depth')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    pdf.savefig(fig)
                                    plt.close()
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    plt.close(fig)
                            else:
                                print(f"No body depth values found for {sex_label} in {base_name}.")

    def body_depth_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_body_depth_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    body_depths = file['/agent_data/body_depth'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        body_depths_by_sex = body_depths[sex_mask]
                        body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]  # Filter out NaN values

                        if body_depths_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                            mean_body_depth = np.mean(body_depths_by_sex)
                            median_body_depth = np.median(body_depths_by_sex)
                            std_dev_body_depth = np.std(body_depths_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Body Depth: {mean_body_depth:.2f}\n")
                            output_file.write(f"  Median Body Depth: {median_body_depth:.2f}\n")
                            output_file.write(f"  Standard Deviation of Body Depth: {std_dev_body_depth:.2f}\n\n")
                        elif body_depths_by_sex.size == 1:
                            # Handle single value case
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Body Depth: {body_depths_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid body depth values found for {sex_label}.\n\n")

    def kcal_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_kcal_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    kcals = file['/agent_data/kcal'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for i, (kcal, sex) in enumerate(zip(kcals, sexes)):
                        kcal_values = kcal[~np.isnan(kcal)]  # Remove NaN values

                        if kcal_values.size > 0:
                            mean_kcal = np.mean(kcal_values)
                            median_kcal = np.median(kcal_values)
                            std_dev_kcal = np.std(kcal_values, ddof=1)  # Use ddof=1 for sample standard deviation
                            min_kcal = np.min(kcal_values)
                            max_kcal = np.max(kcal_values)
                            sex_label = 'Male' if sex == 0 else 'Female'

                            output_file.write(f"Agent {i + 1} ({sex_label}):\n")
                            output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                            output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                            output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                            output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                            output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid kcal values found for Agent {i + 1} ({sex_label}).\n\n")
                            
                            
    def kcal_statistics_directory(self):
        # Prepare to collect cumulative statistics
        cumulative_stats = {}

        # Iterate through all HDF5 files in the directory
        for hdf_path in self.h5_files:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys():
                    kcals = hdf_file['/agent_data/kcal'][:]
                    sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        if sex_label not in cumulative_stats:
                            cumulative_stats[sex_label] = []

                        sex_mask = sexes == sex
                        kcals_by_sex = kcals[sex_mask]
                        kcals_by_sex = kcals_by_sex[~np.isnan(kcals_by_sex)]  # Remove NaN values

                        cumulative_stats[sex_label].extend(kcals_by_sex)

        # Compute and print cumulative statistics
        stats_file_path = os.path.join(self.inputWS, "kcal_statistics_directory.txt")
        with open(stats_file_path, 'w') as output_file:
            for sex_label, values in cumulative_stats.items():
                if values:
                    values = np.array(values)
                    mean_kcal = np.mean(values)
                    median_kcal = np.median(values)
                    std_dev_kcal = np.std(values, ddof=1)
                    min_kcal = np.min(values)
                    max_kcal = np.max(values)

                    output_file.write(f"Cumulative Statistics for {sex_label}:\n")
                    output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                    output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                    output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                    output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                    output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                else:
                    output_file.write(f"No valid kcal values found for {sex_label}.\n\n")

    def kcal_histograms_directory(self):
        # Dictionary to hold data for males and females
        kcal_data = {'Male': {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []},
                     'Female': {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []}}

        # Collect cumulative statistics from all HDF5 files
        for hdf_path in self.h5_files:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys():
                    kcals = hdf_file['/agent_data/kcal'][:]
                    sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'

                        sex_mask = sexes == sex
                        kcals_by_sex = kcals[sex_mask]
                        kcals_by_sex = kcals_by_sex[~np.isnan(kcals_by_sex)]  # Remove NaN values

                        if kcals_by_sex.size > 0:
                            kcal_data[sex_label]['Mean'].append(np.mean(kcals_by_sex))
                            kcal_data[sex_label]['Median'].append(np.median(kcals_by_sex))
                            kcal_data[sex_label]['Std Dev'].append(np.std(kcals_by_sex, ddof=1))
                            kcal_data[sex_label]['Min'].append(np.min(kcals_by_sex))
                            kcal_data[sex_label]['Max'].append(np.max(kcals_by_sex))

        # Create a PDF to save the cumulative histograms
        pdf_filename = os.path.join(self.inputWS, "kcal_histograms.pdf")
        with PdfPages(pdf_filename) as pdf:
            for sex, data in kcal_data.items():
                fig, ax = plt.subplots(figsize=(10, 6))

                # Colors for the different data types
                colors = {'Mean': 'blue', 'Median': 'green', 'Std Dev': 'orange', 'Min': 'red', 'Max': 'purple'}

                # Plot each data type
                for dtype, values in data.items():
                    if values:
                        ax.hist(values, bins=50, alpha=0.7, edgecolor='black', color=colors[dtype], label=dtype)

                ax.set_title(f"Kcal Distribution for {sex} Agents")
                ax.set_xlabel("Kcal")
                ax.set_ylabel("Frequency")
                ax.legend()

                # Add statistics as text on the plot
                mean_kcal = np.mean(data['Mean']) if data['Mean'] else 0
                median_kcal = np.median(data['Median']) if data['Median'] else 0
                std_dev_kcal = np.mean(data['Std Dev']) if data['Std Dev'] else 0
                min_kcal = np.min(data['Min']) if data['Min'] else 0
                max_kcal = np.max(data['Max']) if data['Max'] else 0

                stats_text = (
                    f"Mean: {mean_kcal:.2f}\n"
                    f"Median: {median_kcal:.2f}\n"
                    f"Std Dev: {std_dev_kcal:.2f}\n"
                    f"Min: {min_kcal:.2f}\n"
                    f"Max: {max_kcal:.2f}"
                )
                plt.figtext(0.15, 0.7, stats_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

                pdf.savefig(fig)
                plt.close(fig)

        # New functionality to create individual histograms for each agent
        agent_data = {'Male': {}, 'Female': {}}

        # Collect individual agent statistics from all HDF5 files
        for hdf_path in self.h5_files:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys():
                    kcals = hdf_file['/agent_data/kcal'][:]
                    sexes = hdf_file['/agent_data/sex'][:]

                    for i, (kcal, sex) in enumerate(zip(kcals, sexes)):
                        kcal_values = kcal[~np.isnan(kcal)]  # Remove NaN values
                        sex_label = 'Male' if sex == 0 else 'Female'

                        if i not in agent_data[sex_label]:
                            agent_data[sex_label][i] = {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []}

                        if kcal_values.size > 0:
                            agent_data[sex_label][i]['Mean'].append(np.mean(kcal_values))
                            agent_data[sex_label][i]['Median'].append(np.median(kcal_values))
                            agent_data[sex_label][i]['Std Dev'].append(np.std(kcal_values, ddof=1))
                            agent_data[sex_label][i]['Min'].append(np.min(kcal_values))
                            agent_data[sex_label][i]['Max'].append(np.max(kcal_values))

        # Calculate the average values for each agent
        averaged_data = {'Male': {}, 'Female': {}}
        for sex, agents in agent_data.items():
            for agent, values in agents.items():
                averaged_data[sex][agent] = {
                    'Mean': np.mean(values['Mean']) if values['Mean'] else 0,
                    'Median': np.mean(values['Median']) if values['Median'] else 0,
                    'Std Dev': np.mean(values['Std Dev']) if values['Std Dev'] else 0,
                    'Min': np.mean(values['Min']) if values['Min'] else 0,
                    'Max': np.mean(values['Max']) if values['Max'] else 0
                }

        # Find the maximum recorded kcal average across all agents for the y-axis limit
        max_kcal_value = max(
            max(kcal_data['Male']['Max']) if kcal_data['Male']['Max'] else 0,
            max(kcal_data['Female']['Max']) if kcal_data['Female']['Max'] else 0
        )

        # Create a PDF to save the individual histograms
        pdf_filename_individual = os.path.join(self.inputWS, "individual_kcal_histograms.pdf")
        with PdfPages(pdf_filename_individual) as pdf:
            for sex, agents in averaged_data.items():
                for agent, values in agents.items():
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Colors for the different data types
                    colors = ['blue', 'green', 'orange', 'red', 'purple']
                    labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
                    agent_values = [values[label] for label in labels]

                    # Plot the agent's data
                    ax.bar(labels, agent_values, color=colors, alpha=0.7, edgecolor='black')

                    ax.set_title(f"Kcal Distribution for Agent {agent} ({sex})")
                    ax.set_xlabel("Kcal Type")
                    ax.set_ylabel("Value")
                    ax.set_ylim([0, max_kcal_value])  # Set y-axis limit

                    # Add statistics as text on the plot
                    stats_text = (
                        f"Mean: {values['Mean']:.2f}\n"
                        f"Median: {values['Median']:.2f}\n"
                        f"Std Dev: {values['Std Dev']:.2f}\n"
                        f"Min: {values['Min']:.2f}\n"
                        f"Max: {values['Max']:.2f}"
                    )
                    plt.figtext(0.15, 0.7, stats_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

                    pdf.savefig(fig)
                    plt.close(fig)



    def kaplan_curve(self, shapefile_path, tiffWS):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            jpeg_filename = f"{base_name}_Kaplan_Meier_Curve.jpeg"
            jpeg_filepath = os.path.join(output_folder, jpeg_filename)

            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty:
                continue

            # Load TIFF data for coordinate reference
            with rasterio.open(tiffWS) as tif:
                tif_crs = tif.crs

            # Adjust shapefile CRS if needed
            if gdf.crs != tif_crs:
                gdf = gdf.to_crs(tif_crs)

            # Filter self.ts for the current HDF5 file
            ts_filtered = self.ts[self.ts['filename'] == h5_file]

            # Perform spatial intersection with the shapefile
            intersection = gpd.overlay(ts_filtered, gdf, how='intersection')

            # Get the list of all agent IDs in the filtered data
            all_agents = ts_filtered['agent'].unique()
            total_agents = len(all_agents)

            if intersection.empty:
                print(f"No agents intersected the rectangle - skipping {base_name}.")
                continue

            # Get the unique list of agents that are found within the rectangle
            unique_agents_in_rectangle = intersection['agent'].unique()
            num_agents_in_rectangle = len(unique_agents_in_rectangle)

            # Print the comparison
            print(f"File: {base_name}")
            print(f"Total agents in data: {total_agents}")
            print(f"Number of unique agents found within the rectangle: {num_agents_in_rectangle}")

            # Prepare the first entry times for each agent
            entry_times = {agent: intersection[intersection['agent'] == agent]['timestep'].min()
                           for agent in unique_agents_in_rectangle}

            # Convert to arrays for Kaplan-Meier analysis
            entry_times_array = np.array(list(entry_times.values()))

            # Create the survival data array (True if entered the rectangle, False if not)
            survival_data = np.array([(True, time) for time in entry_times_array], dtype=[('event', bool), ('time', int)])

            # Perform Kaplan-Meier estimation
            time, survival_prob = kaplan_meier_estimator(survival_data['event'], survival_data['time'])

            # Plot the Kaplan-Meier survival curve
            plt.figure(figsize=(10, 6))
            plt.step(time, survival_prob, where="post", label=f"Agents Entering Rectangle: {num_agents_in_rectangle}/{total_agents}")
            plt.xlabel("Time (Timesteps)")
            plt.ylabel("Proportion of Agents Remaining Outside the Rectangle")
            plt.title(f"Kaplan-Meier Curve\n{num_agents_in_rectangle} Agents Entered Rectangle out of {total_agents}")
            plt.legend()

            # Save the plot as a JPEG image
            plt.savefig(jpeg_filepath, format='jpeg')
            plt.close()

                    
                    
    def passage_success(self,finish_line):
        '''find the fish that are successful'''
        
        for filename in self.h5_files:
            dat = self.ts[self.ts.filename == filename]
            agents_in_box = dat[dat.within(finish_line)]
            
            # get the unique list of successful agents 
            succesful = agents_in_box.agent.unique()
            
            # get complete list of agents
            n_agents = dat.agent.unique()
            
            # success rate
            success = len(succesful) / n_agents 
            
            # get first detections in finish area
            passing_times = agents_in_box.groupby('agent')['timestep'].min()
            
            # get passage time stats
            pass_time_0 = passing_times.timestep.min()
            pass_time_10 = np.percentile(passing_times.timestep,10)
            pass_time_25 = np.percentile(passing_times.timestep,25)
            pass_time_50 = np.percentile(passing_times.timestep,50)
            pass_time_75 = np.percentile(passing_times.timestep,75)
            pass_time_90 = np.percentile(passing_times.timestep,90)
            pass_time_100 = passing_times.timestep.max()
            
            # make a row and add it to the success_rates dictionary
            self.success_rates[filename] = [success, 
                                            pass_time_0,
                                            pass_time_10,
                                            pass_time_25,
                                            pass_time_50,
                                            pass_time_75,
                                            pass_time_90,
                                            pass_time_100]
            
    def emergence(self,filename,scenario,crs):
        '''Method quantifies emergent spatial properties of the agent based model.
        
        Our problem is 5D, 2 spatial dimensions, 1 temporal dimension, iterations, 
        and finally scenario.  Comparison of each scenario output will typicall 
        occur in a GIS as those are the final surfaces.  
        
        Emergence first calculates the number of agents per cell per timestep.
        Then, emergence produces a 2 band raster for every iteration where the 
        first band is the average number of agents per cell per timestep and the 
        second band is the standard deviaition.  
        
        Then, Emergence statistically compares iterations, develops and index of 
        similarity, identifies a corridor threshold, and produces a final surface
        for comparison in a GIS.

        Returns
        -------
        corridor raster surface.

        '''
        # Agent coordinates and rasterio affine transform
        x_coords = self.ts.X  # X coordinates of agents
        y_coords = self.ts.Y  # Y coordinates of agents
        transform = self.transform  # affine transform from your rasterio dataset
        
        hdf5_filename = 'intermediate_results.h5'
        
        with h5py.File(os.path.join(self.parent_directory,hdf5_filename), 'w') as hdf5_file:
            for filename in self.ts.filename.unique():
                dat = self.ts[self.ts.filename == filename]
                num_timesteps = self.ts.timestep.max() + 1
                num_iterations = len(self.ts.filename.unique())
                
                # Create a dataset for each filename
                data_over_time = hdf5_file.create_dataset(
                    filename,
                    shape=(num_timesteps, self.height, self.width),
                    dtype=np.float32,
                    chunks=(1, self.height, self.width),
                    compression="gzip"
                )
                
                for timestep in range(num_timesteps):
                    t_dat = dat[dat.timestep == timestep]
                    
                    # Convert geographic coordinates to pixel indices using your function
                    rows, cols = geo_to_pixel(t_dat.X, t_dat.Y, transform)
                    
                    # Combine row and column indices to get unique cell identifiers
                    cell_indices = np.stack((cols, rows), axis=1)
                    
                    # Count unique cells
                    unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
                    valid_rows, valid_cols = unique_cells[:, 1], unique_cells[:, 0]  # Unpack the unique cell indices
                    
                    # Initialize a 2D array with zeros
                    agent_counts_grid = np.zeros((self.height, self.width), dtype=int)
                    
                    # Ensure the indices are within the grid bounds and update the agent_counts_grid
                    within_bounds = (valid_rows >= 0) & (valid_rows < self.height) & (valid_cols >= 0) & (valid_cols < self.width)
                    agent_counts_grid[valid_rows[within_bounds], valid_cols[within_bounds]] = counts[within_bounds]            
                    
                    # Insert the 2D array into the pre-allocated 3D array in HDF5
                    data_over_time[timestep, :, :] = agent_counts_grid
                    print(f'file {filename} timestep {timestep} complete')
            
            # Now aggregate results from HDF5
            all_data = []
            for filename in self.ts.filename.unique():
                data = da.from_array(hdf5_file[filename], chunks=(1, self.height, self.width))
                all_data.append(data)
            
            all_data = da.stack(all_data, axis=0)  # Stack along the new iteration axis
            
            # Calculate the average and standard deviation count per cell over all iterations and timesteps
            self.average_per_cell = da.mean(all_data, axis=(0, 1)).astype(np.float32).compute()
            self.sd_per_cell = da.std(all_data, axis=(0, 1)).astype(np.float32).compute()
        
        # Create dual band raster and write to output directory
        output_file = f'{scenario}_dual_band.tif'
        
        with rasterio.open(
            os.path.join(self.parent_directory,output_file),
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=2,  # Two bands
            dtype=self.average_per_cell.dtype,
            crs=crs,
            transform=self.transform
        ) as dst:
            dst.write(self.average_per_cell, 1)  # Write the average to the first band
            dst.write(self.sd_per_cell, 2)       # Write the standard deviation to the second band
        
        print(f'Dual band raster {output_file} created successfully.')
