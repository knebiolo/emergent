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
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.mask import mask
from shapely import Point, Polygon, box
from shapely import affinity
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline, interp1d, CubicSpline
from scipy.optimize import curve_fit
from scipy.constants import g
from scipy.spatial import cKDTree
from scipy.stats import beta
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
import sys
import random
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

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

def geo_to_pixel(X, Y, arr_type, transform):
    """
    Convert x, y coordinates to row, column indices in the raster grid.

    Parameters:
    - transform: affine transform of the raster

    Returns:
    - rows: array of row indices
    - cols: array of column indices
    """
    cols = (X - transform.c) / transform.a
    rows = (Y - transform.f) / transform.e
    
    cols = arr_type.round(cols).astype(arr_type.int32)
    rows = arr_type.round(rows).astype(arr_type.int32)

    # If using CuPy, transfer indices to CPU as NumPy arrays for HDF5 operations
    # if isinstance(cols, arr_type.ndarray):
    #     cols = cols.get()
    #     rows = rows.get()

    return rows, cols

def pixel_to_geo(arr_type, transform, rows, cols):
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
    xs = transform.c + transform.a * cols
    ys = transform.f + transform.e * rows

    # If using CuPy, ensure that the calculation is done on the GPU
    if arr_type.__name__ == 'cupy':
        xs = arr_type.asnumpy(xs)
        ys = arr_type.asnumpy(ys)

    return xs, ys

def standardize_shape(arr, target_shape=(5, 5), fill_value=np.nan):
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
    
def HECRAS (HECRAS_dir, model_dir, resolution, crs):
    """
    Import environment data from a HECRAS model and generate raster files.
    
    This method extracts data from a HECRAS model stored in HDF format and 
    interpolates the data to generate raster files for various environmental 
    parameters such as velocity, water surface elevation, and bathymetry.
    
    Parameters:
    - HECRAS_model (str): Path to the HECRAS model in HDF format.
    - resolution (float): Desired resolution for the interpolated rasters.
    
    Attributes set:
    
    Notes:
    - The method reads data from the HECRAS model, interpolates the data to the 
      desired resolution, and then writes the interpolated data to raster files.
    - The generated raster files are saved in the model directory with names corresponding 
      to the environmental parameter they represent (e.g., 'elev.tif', 'wsel.tif').
    - The method uses LinearNDInterpolator for interpolation and rasterio for raster 
      generation and manipulation.
    """
    # Initialization Part 1: Connect to HECRAS model and import environment
    hdf = h5py.File(HECRAS_dir,'r')
    
    # Extract Data from HECRAS HDF model
    print ("Extracting Model Geometry and Results")
    
    pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
    vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
    wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1]
    elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
    
    # create list of xy tuples
    geom = list(tuple(zip(pts[:,0],pts[:,1])))
    
    # create a dataframe with geom column and observations
    df = pd.DataFrame.from_dict({'index':np.arange(0,len(pts),1),
                                 'geom_tup':geom,
                                 'vel_x':vel_x,
                                 'vel_y':vel_y,
                                 'wsel':wsel,
                                 'elev':elev})
    
    # add a geometry column
    df['geometry'] = df.geom_tup.apply(Point)
    
    # convert into a geodataframe
    gdf = gpd.GeoDataFrame(df,crs = crs)
    
    print ("Create multidimensional interpolator functions for velocity, wsel, elev")
    
    vel_x_interp = LinearNDInterpolator(pts,gdf.vel_x)
    vel_y_interp = LinearNDInterpolator(pts,gdf.vel_y)
    wsel_interp = LinearNDInterpolator(pts,gdf.wsel)
    elev_interp = LinearNDInterpolator(pts,gdf.elev)
    
    # first identify extents of image
    xmin = np.min(pts[:,0])
    xmax = np.max(pts[:,0])
    ymin = np.min(pts[:,1])
    ymax = np.max(pts[:,1])
    
    # interpoate velocity, wsel, and elevation at new xy's
    ## TODO ISHA TO CHECK IF RASTER OUTPUTS LOOK DIFFERENT AT 0.5m vs 1m
    xint = np.arange(xmin,xmax,resolution)
    yint = np.arange(ymax,ymin,resolution * -1.)
    xnew, ynew = np.meshgrid(xint,yint, sparse = True)
    
    print ("Interpolate Velocity East")
    vel_x_new = vel_x_interp(xnew, ynew)
    print ("Interpolate Velocity North")
    vel_y_new = vel_y_interp(xnew, ynew)
    print ("Interpolate WSEL")
    wsel_new = wsel_interp(xnew, ynew)
    print ("Interpolate bathymetry")
    elev_new = elev_interp(xnew, ynew)
    
    # create a depth raster
    depth = wsel_new - elev_new
    
    # calculate velocity magnitude
    vel_mag = np.sqrt((np.power(vel_x_new,2)+np.power(vel_y_new,2)))
    
    # calculate velocity direction in radians
    vel_dir = np.arctan2(vel_y_new,vel_x_new)
    
    print ("Exporting Rasters")

    # create raster properties
    driver = 'GTiff'
    width = elev_new.shape[1]
    height = elev_new.shape[0]
    count = 1
    crs = crs
    #transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)
    transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution)\
        * Affine.scale(resolution,-1 * resolution)

    # write elev raster
    with rasterio.open(os.path.join(model_dir,'elev.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as elev_rast:
        elev_rast.write(elev_new,1)
    
    # write wsel raster
    with rasterio.open(os.path.join(model_dir,'wsel.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as wsel_rast:
        wsel_rast.write(wsel_new,1)
                    
    # write depth raster
    with rasterio.open(os.path.join(model_dir,'depth.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as depth_rast:
        depth_rast.write(depth,1)
        
    # write velocity dir raster
    with rasterio.open(os.path.join(model_dir,'vel_dir.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_dir_rast:
        vel_dir_rast.write(vel_dir,1)
                    
    # write velocity .mag raster
    with rasterio.open(os.path.join(model_dir,'vel_mag.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_mag_rast:
        vel_mag_rast.write(vel_mag,1)
                
    # write velocity x raster
    with rasterio.open(os.path.join(model_dir,'vel_x.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_x_rast:
        vel_x_rast.write(vel_x_new,1)
                    
    # write velocity y raster
    with rasterio.open(os.path.join(model_dir,'vel_y.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_y_rast:
        vel_y_rast.write(vel_y_new,1)


class PID_controller:
    def __init__(self, k_p, k_i, k_d, n_agents):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.integral = np.zeros(n_agents)
        self.previous_error = np.zeros(n_agents)

    def update(self, error):
        self.integral += error.flatten()
        derivative = error.flatten() - self.previous_error
        self.previous_error = error.flatten()

        p_term = self.k_p * error.flatten()
        i_term = self.k_i * self.integral
        d_term = self.k_d * derivative

        return p_term + i_term + d_term
      
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
                 starting_box,
                 fish_length,
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
        recover = pd.read_csv(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent\data\recovery.csv")        
        #recover = pd.read_csv(r"C:\Users\AYoder\OneDrive - Kleinschmidt Associates, Inc\Software\emergent\data\recovery.csv")
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds, recover.Recovery, extrapolate = True,)
        del recover
        self.swim_behav = self.arr.repeat(1, num_agents)               # 1 = migratory , 2 = refugia, 3 = station holding
        self.swim_mode = self.arr.repeat(1, num_agents)      # 1 = sustained, 2 = prolonged, 3 = sprint
        self.battery = self.arr.repeat(1.0, num_agents)
        self.recover_stopwatch = self.arr.repeat(0.0, num_agents)
        self.ttfr = self.arr.repeat(0.0, num_agents)
        self.time_out_of_water = self.arr.repeat(0.0, num_agents)
        if pid_tuning != True:
            self.X = self.arr.random.uniform(starting_box[0], starting_box[1],num_agents)
            self.Y = self.arr.random.uniform(starting_box[2], starting_box[3],num_agents)
        else:
            self.X = np.array([starting_box[0]])
            self.Y = np.array([starting_box[1]])
            
        self.prev_X = self.X
        self.prev_Y = self.Y

        # Time to Fatigue values for Sockeye digitized from Bret 1964
        self.max_s_U = 2.77  # maximum sustained swim speed
        self.max_p_U = 4.43  # maximum prolonged swim speed
        self.a_p = 8.643     # prolonged intercept
        self.b_p = -2.0894   # prolonged slope
        self.a_s = 0.1746    # sprint intercept
        self.b_s = -0.1806   # sprint slope
        
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
        self.enviro_import(os.path.join(model_dir,'vel_x.tif'),'velocity x')
        self.enviro_import(os.path.join(model_dir,'vel_y.tif'),'velocity y')
        self.enviro_import(os.path.join(model_dir,'depth.tif'),'depth')
        self.enviro_import(os.path.join(model_dir,'wsel.tif'),'wsel')
        self.enviro_import(os.path.join(model_dir,'elev.tif'),'elevation')
        self.enviro_import(os.path.join(model_dir,'vel_dir.tif'),'velocity direction')
        self.enviro_import(os.path.join(model_dir,'vel_mag.tif'),'velocity magnitude') 
        self.hdf5.flush()

        # initialize mental map
        self.initialize_mental_map()
        
        # initialize heading
        self.initial_heading()
        
        # initialize swim speed
        self.initial_swim_speed() 
        
        # error array
        self.error_array = np.array([])

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
            
    def sim_length(self, fish_length):
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
                if self.sex == 'M':
                    self.length = self.arr.random.lognormal(mean = 6.426,sigma = 0.072,size = self.num_agents)
                else:
                    self.length = self.arr.random.lognormal(mean = 6.349,sigma = 0.067,size = self.num_agents)
        
        # we can also set these arrays that contain parameters that are a function of length
        self.sog = self.length/1000.  # sog = speed over ground - assume fish maintain 1 body length per second
        self.ideal_sog = self.sog
        self.swim_speed = self.length/1000.        # set initial swim speed
        self.ucrit = self.sog * 7.    # TODO - what is the ucrit for sockeye?
        
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
            if self.sex == 'M':
                self.body_depth = self.arr.exp(-1.938 + np.log(self.length) * 1.084 + 0.0435) / 10.
            else:
                self.body_depth = self.arr.exp(-1.938 + np.log(self.length) * 1.084) / 10.
                
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

        if surface_type == 'velocity x':
            # set transform as parameter of simulation
            self.vel_x_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("vel_x", (height, width), dtype='f4', data = src.read(1))
            self.hdf5['environment/vel_x'][:, :] = src.read(1)

        elif surface_type == 'velocity y':
            # set transform as parameter of simulation            
            self.vel_y_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("vel_y", (height, width), dtype='f4')
            self.hdf5['environment/vel_y'][:, :] = src.read(1)
            
        elif surface_type == 'depth':
            # set transform as parameter of simulation            
            self.depth_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("depth", (height, width), dtype='f4')
            self.hdf5['environment/depth'][:, :] = src.read(1)
            
        elif surface_type == 'wsel':
            # set transform as parameter of simulation            
            self.wsel_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("wsel", (height, width), dtype='f4')
            self.hdf5['environment/wsel'][:, :] = src.read(1)
            
        elif surface_type == 'elevation':
            # set transform as parameter of simulation                        
            self.elev_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("elevation", (height, width), dtype='f4')
            self.hdf5['environment/elevation'][:, :] = src.read(1)  
                
        elif surface_type == 'velocity direction':          
            # set transform as parameter of simulation                        
            self.vel_dir_rast_transform = transform

            # create an hdf5 array and write to it
            env_data.create_dataset("vel_dir", (height, width), dtype='f4')
            self.hdf5['environment/vel_dir'][:, :] = src.read(1)  
                
        elif surface_type == 'velocity magnitude': 
            # set transform as parameter of simulation                        
            self.vel_mag_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("vel_mag", (height, width), dtype='f4')
            self.hdf5['environment/vel_mag'][:, :] = src.read(1)  
            
        self.width = width
        self.height = height
        self.hdf5.flush()
        src.close()

            
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
        
        # create a memory map array
        for i in np.arange(self.num_agents):
            mem_data.create_dataset('%s'%(i), (self.height, self.width), dtype = 'f4')
            self.hdf5['memory/%s'%(i)][:, :] = self.arr.zeros((self.height, self.width))
        
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
        rows, cols = geo_to_pixel(self.X, self.Y, self.arr, transform)

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
        row, col = geo_to_pixel(self.X, self.Y, self.arr, self.vel_dir_rast_transform)
            
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
        rows, cols = geo_to_pixel(self.X, self.Y, self.arr, self.vel_dir_rast_transform)
    
        # Ensure rows and cols are within the bounds of the mental map
        rows = self.arr.clip(rows, 0, self.height - 1)
        cols = self.arr.clip(cols, 0, self.width - 1)

        # get velocity and coords raster per agent
        for i in np.arange(self.num_agents):
            if self.num_agents > 1:
                self.hdf5['memory/%s'%(i)][rows[i],cols[i]] = current_timestep
            else:
                single_arr = np.array([self.hdf5['memory/%s'%(i)]])
                single_arr[0,rows,cols] = current_timestep
                self.hdf5['memory/%s'%(i)][:, :] = single_arr


    
        # # Update the mental map for all agents in the HDF5 dataset at once
        # mental_map_dataset = self.hdf5['memory/maps'][:]
            
        # # Use advanced indexing to update the mental map
        # # Note: This assumes that the HDF5 dataset supports numpy-style advanced indexing
        # mental_map_dataset[indices] = current_timestep
        
        self.hdf5.flush()

    def already_been_here(self, weight, t):
        """
        Calculates a repulsive force on agents based on their historical locations within a specified time frame.
        This method uses the agents' mental maps, stored in a 3D array within an HDF5 file, to determine areas
        they have visited previously. The force simulates a tendency to avoid areas recently visited, with a
        time-dependent weighting factor.
    
        Parameters
        ----------
        weight : float
            A scalar that determines the strength of the repulsive force.
        t : int
            The current time step in the simulation.
    
        Returns
        -------
        list
            A list containing the sum of the repulsive forces in the X and Y directions across all agents.
    
        Workflow
        --------
        1. Retrieves the current X and Y positions of the agents.
        2. Converts these positions to row and column indices in the raster grid using the `geo_to_pixel` method.
        3. Constructs an index array for advanced indexing, defining a buffer zone around the current positions.
        4. Retrieves the relevant sections of the mental map from the HDF5 file for each agent using advanced indexing.
        5. Calculates the time since each agent last visited each cell within the buffer zone.
        6. Applies a conditional weight to the cells based on the elapsed time since the last visit.
        7. Computes the relative positions of each cell from the agents' current positions using broadcasting.
        8. Calculates the Euclidean distance from each agent to each cell within the buffer zone.
        9. Determines the direction from each cell to the agent's position.
        10. Calculates the repulsive force exerted by each cell on the agent, scaled by the weight and inversely proportional to the square of the distance.
        11. Sums the forces in the X and Y directions and returns the result as a list.
    
        Usage Example
        -------------
        # Assuming an instance of the class is created and initialized as `agent_model`
        force_vector = agent_model.already_been_here(weight=0.5, t=current_time_step)
        force_x, force_y = force_vector
    
        Notes
        -----
        - The method assumes that the HDF5 dataset supports numpy-style advanced indexing and that the mental map dataset is named 'memory/maps'.
        - The forces are normalized to unit vectors to ensure that the direction of the force is independent of the distance.
        - The method returns the sum of the forces for all agents, which may need to be handled individually depending on the simulation requirements.
        - The method uses a buffer zone (`buff`) to limit the computation to a manageable area around each agent, which is currently set to a 4-cell radius.
        - The time-dependent weighting factor (`multiplier`) is applied to cells visited between 600 and 3600 time steps ago, with a repulsive force applied to these cells.
        - The method uses broadcasting to efficiently handle calculations for multiple agents simultaneously, avoiding the need for explicit loops.
        """

        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
        
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
        
        # Construct an index array for advanced indexing
        agent_indices = np.arange(self.num_agents)
        
        # create array slice bounds
        buff = 2
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
        
        # Initialize an array to hold the velocity cues for each agent
        repulsive_forces_per_agent = np.zeros((self.num_agents, 2), dtype=float)
                    
        # Create slice objects for indexing
        slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.num_agents),
                                                   ymin.flatten(), 
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten()
                                                   )
                  ]
        
        # get velocity and coords raster per agent
        mmap = np.stack([standardize_shape(self.hdf5['memory/%s'%(sl[0])][sl[-2:]]) for sl in slices])        
        x_coords = np.stack([standardize_shape(self.hdf5['x_coords'][sl[-2:]]) for sl in slices])        
        y_coords = np.stack([standardize_shape(self.hdf5['y_coords'][sl[-2:]]) for sl in slices])

        # get shape parameters of mental map array
        num_agents, map_width, map_height = mmap.shape
        
        t_since = mmap - t
        
        multiplier = np.where(np.logical_and(t_since > 600, t_since < 3600),1,0)

        # Calculate the difference vectors
        delta_x = self.X[:,np.newaxis,np.newaxis] - x_coords
        delta_y = self.Y[:,np.newaxis,np.newaxis] - y_coords
        
        # Calculate the magnitude of each vector
        magnitudes = np.sqrt(np.power(delta_x,2) + np.power(delta_x,2))

        # Avoid division by zero
        magnitudes = np.where(magnitudes == 0, 0.000001, magnitudes)
        
        # Normalize each vector to get the unit direction vectors
        unit_vector_x = delta_x / magnitudes
        unit_vector_y = delta_y / magnitudes
    
        # Calculate repulsive force in X and Y directions for this agent
        x_force = ((weight * unit_vector_x) / magnitudes) * multiplier
        y_force = ((weight * unit_vector_y) / magnitudes) * multiplier
    
        # Sum the forces for this agent
        if self.num_agents > 1:
            total_x_force = np.nansum(x_force, axis = (1,2))
            total_y_force = np.nansum(y_force, axis = (1,2))
        else:
            total_x_force = np.array([np.nansum(x_force)])
            total_y_force = np.array([np.nansum(y_force)])

    
        repulsive_forces =  np.array([total_x_force, total_y_force]).T
             
        return repulsive_forces

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
    
        # Avoid divide by zero by setting zero velocities to a small number
        self.x_vel[self.x_vel == 0.0] = 0.0001
        self.y_vel[self.y_vel == 0.0] = 0.0001
    
        # keep track of the amount of time a fish spends out of water
        self.time_out_of_water = np.where(self.depth < self.too_shallow, 
                                          self.time_out_of_water + 1, 
                                          self.time_out_of_water)
    
        positions = np.vstack([self.X.flatten(),self.Y.flatten()]).T
        # Creating a KDTree for efficient spatial queries
        tree = cKDTree(positions)
        
        # Radius for nearest neighbors search
        radius = 2
        
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
        self.z = self.arr.where(
            self.depth < self.body_depth * 3 / 100.,
            self.depth + self.too_shallow,
            self.body_depth * 3 / 100.)

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
        length_numpy = self.length#.get() if isinstance(self.length, cp.ndarray) else self.length
        
        # calculate buffer size based on swim mode, if we are in refugia mode buffer is 15 body lengths else 5
        #buff = np.where(self.swim_mode == 2, 15 * length_numpy, 5 * length_numpy)
        
        # for array operations, buffers are represented as a slice (# rows and columns)
        buff = 2
        
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
        
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
        
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
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.num_agents),
                                                   ymin.flatten(), 
                                                   ymax.flatten(),
                                                   xmin.flatten(), 
                                                   xmax.flatten()
                                                   )
                  ]

        # get velocity and coords raster per agent
        vel3d = np.stack([standardize_shape(self.hdf5['environment/vel_mag'][sl[-2:]]) for sl in slices])
        x_coords = np.stack([standardize_shape(self.hdf5['x_coords'][sl[-2:]]) for sl in slices])
        y_coords = np.stack([standardize_shape(self.hdf5['y_coords'][sl[-2:]]) for sl in slices])
        
        vel3d_multiplier = calculate_front_masks(self.heading.flatten(), 
                                                 x_coords, 
                                                 y_coords, 
                                                 self.X.flatten(), 
                                                 self.Y.flatten(), 
                                                 behind_value = 999.9)
        
        # if self.num_agents > 1:
        #     vel3d = vel3d * vel3d_multiplier
        # else:
        #     vel3d = vel3d[0] * vel3d_multiplier[0]
            
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
        min_x, min_y = pixel_to_geo(self.arr, 
                                    self.vel_mag_rast_transform, 
                                    min_row_indices + ymin, 
                                    min_col_indices + xmin)
        
        delta_x = self.X - min_x
        delta_y = self.Y - min_y
        delta_x_sq = np.power(delta_x,2)
        delta_y_sq = np.power(delta_y,2)
        dist = np.sqrt(delta_x_sq + delta_y_sq)

        # Initialize an array to hold the velocity cues for each agent
        velocity_min = np.zeros((self.num_agents, 2), dtype=float)

        attract_x = (weight * delta_x/dist) / np.power(buff,2)
        attract_y = (weight * delta_y/dist) / np.power(buff,2)
        
        return np.array([attract_x,attract_y])

    def rheo_cue(self, weight):
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
        length_numpy = self.length#.get() if isinstance(self.length, cp.ndarray) else self.length
    
        # Sample the environment to get the velocity direction and adjust to point upstream
        vel_dir = self.sample_environment(self.vel_dir_rast_transform, 'vel_dir') - np.radians(180)
        
        # Calculate the unit vector in the upstream direction
        v_hat = np.vstack([np.cos(vel_dir), np.sin(vel_dir)]).T
        
        # Calculate the rheotactic cue
        rheotaxis = (weight * v_hat)/(2**2)
        
        return rheotaxis

    def shallow_cue(self, weight):
        """
        Calculate the repulsive force vectors from shallow water areas within a specified buffer around each agent using a vectorized approach.
    
        This function identifies cells within a sensory buffer around each agent that are shallower than a threshold depth. It then calculates the inverse gravitational potential for these cells and sums up the forces to determine the total repulsive force vector exerted on each agent due to shallow water.
    
        Parameters:
        - weight (float): The weighting factor to scale the repulsive force.
    
        Returns:
        - np.ndarray: A 2D array where each row corresponds to an agent and contains the sum of the repulsive forces in the X and Y directions.
    
        Notes:
        - The function assumes that the depth data is accessible from an HDF5 file with the key 'environment/depth'.
        - The sensory buffer is set to 2 meters around each agent's position.
        - The function uses the `geo_to_pixel` method to convert geographic coordinates to pixel indices.
        - The repulsive force is calculated as a vector normalized by the magnitude of the distance to each shallow cell, scaled by the weight, and summed across all shallow cells within the buffer.
        - The function returns an array of the total repulsive force vectors for each agent.
        - The vectorized approach is expected to improve performance by reducing the overhead of Python loops.
        """

        buff = 4.  # 2 meters
    
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
    
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
    
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
        repulsive_forces = np.zeros((self.num_agents,2), dtype=float)
        
        min_depth = (self.body_depth * 1.1) / 100.# Use advanced indexing to create a boolean mask for the slices
            
        # Create slices
        slices = [(agent, slice(y0, y1), slice(x0, x1)) 
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.num_agents),  
                                                   ymin.flatten(), 
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten())
                  ]
        

        # get depth raster per agent
        depths = np.stack([standardize_shape(self.hdf5['environment/depth'][sl[-2:]], target_shape=(9, 9)) for sl in slices])        
        x_coords = np.stack([standardize_shape(self.hdf5['x_coords'][sl[-2:]], target_shape=(9, 9)) for sl in slices]) 
        y_coords = np.stack([standardize_shape(self.hdf5['y_coords'][sl[-2:]], target_shape=(9, 9)) for sl in slices])       
        
        front_multiplier = calculate_front_masks(self.heading, x_coords, y_coords, self.X, self.Y)

        # create a multiplier
        depth_multiplier = np.where(depths < min_depth[:,np.newaxis,np.newaxis], 1, 0)

        # Calculate the difference vectors
        delta_x = x_coords - self.X[:,np.newaxis,np.newaxis]
        delta_y = y_coords - self.Y[:,np.newaxis,np.newaxis]
        
        # Calculate the magnitude of each vector
        magnitudes = np.sqrt(np.power(delta_x,2) + np.power(delta_x,2))

        # Avoid division by zero
        magnitudes = np.where(magnitudes == 0, 0.000001, magnitudes)
        
        # Normalize each vector to get the unit direction vectors
        unit_vector_x = delta_x / magnitudes
        unit_vector_y = delta_y / magnitudes
    
        # Calculate repulsive force in X and Y directions for this agent
        x_force = ((weight * unit_vector_x) / magnitudes) * depth_multiplier * front_multiplier
        y_force = ((weight * unit_vector_y) / magnitudes) * depth_multiplier * front_multiplier
    
        # Sum the forces for this agent
        if self.num_agents > 1:
            total_x_force = np.nansum(x_force, axis = (1, 2))#, axis = (0))
            total_y_force = np.nansum(y_force, axis = (1, 2))
        else:
            total_x_force = np.nansum(x_force)#, axis = (1, 2))#, axis = (0))
            total_y_force = np.nansum(y_force)#, axis = (1, 2))
    
        repulsive_forces =  np.array([total_x_force, total_y_force]).T
        
        # if np.any(x_force != 0.0):
        #     print ('fuck')
        
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
        hughes = pd.read_csv(r'../data/wave_drag_huges_2004_fig3.csv')

        hughes.sort_values(by = 'body_depths_submerged',
                           ascending = True,
                           inplace = True)
        # fit function
        wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged,
                                         hughes.wave_drag_multiplier,
                                         k = 3, ext = 0)

        # how submerged are these fish - that's how many
        body_depths = self.z / (self.body_depth / 100.)

        self.wave_drag = self.arr.where(body_depths >=3, 1, wave_drag_fun(body_depths))
       
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
        x, y = (self.X, self.Y)
    
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
    
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
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.num_agents),
                                                   ymin.flatten(), 
                                                   ymax.flatten() ,
                                                   xmin.flatten(), 
                                                   xmax.flatten()
                                                   )
                  ]
        
        # get depth raster per agent
        #dep3D = np.stack([self.hdf5['environment/depth'][sl[-2:]] for sl in slices])
        dep3D = np.stack([standardize_shape(self.hdf5['environment/depth'][sl[-2:]]) for sl in slices])
        x_coords = np.stack([standardize_shape(self.hdf5['x_coords'][sl[-2:]]) for sl in slices])
        y_coords = np.stack([standardize_shape(self.hdf5['y_coords'][sl[-2:]]) for sl in slices])
        
        dep3D_multiplier = calculate_front_masks(self.heading.flatten(), 
                                                 x_coords, 
                                                 y_coords, 
                                                 self.X.flatten(), 
                                                 self.Y.flatten(), 
                                                 behind_value = 99999.9)
        
        # if self.num_agents > 1:
        #     dep3D = dep3D * dep3D_multiplier
        # else:
        #     dep3D = dep3D[0] * dep3D_multiplier[0]
            
        dep3D = dep3D * dep3D_multiplier

        num_agents, rows, cols = dep3D.shape
 
        # Reshape the 3D array into a 2D array where each row represents an agent
        reshaped_dep3D = dep3D.reshape(num_agents, rows * cols)
        
        # Find the cell with the depth closest to the agent's optimal depth
        optimal_depth_diff = np.abs(reshaped_dep3D - self.opt_wat_depth[:,np.newaxis])

        # Find the index of the minimum value in each row (agent)
        flat_indices = np.argmin(optimal_depth_diff, axis=1)
        
        # Convert flat indices to row and column indices
        min_row_indices = flat_indices // cols
        min_col_indices = flat_indices % cols

        # Convert the index back to geographical coordinates
        min_x, min_y = pixel_to_geo(self.arr, 
                                    self.vel_mag_rast_transform, 
                                    min_row_indices + ymin, 
                                    min_col_indices + xmin)
        
        delta_x = self.X - min_x
        delta_y = self.Y - min_y
        delta_x_sq = np.power(delta_x,2)
        delta_y_sq = np.power(delta_y,2)
        dist = np.sqrt(delta_x_sq + delta_y_sq)

        # Initialize an array to hold the velocity cues for each agent
        velocity_min = np.zeros((self.num_agents, 2), dtype=float)

        attract_x = (weight * delta_x/dist) / np.power(buff,2)
        attract_y = (weight * delta_y/dist) / np.power(buff,2)
        
        return np.array([attract_x,attract_y])

    def school_cue(self, weight):
        """
        Calculate the attractive force towards the centroid of the school for 
        each agent.
    
        This function applies a vectorized custom function that computes the 
        centroid of the neighboring agents within a specified buffer and determines 
        the attractive force exerted by the school on each agent towards this 
        centroid. The force is inversely proportional to the square of the distance 
        to the centroid, scaled by a weight.
    
        Parameters:
        - weight (float): The weighting factor to scale the attractive force.
        - agents_within_buffers_dict (dict): A dictionary where the key is the 
        index of an agent and the value
          is a list of indices of agents within its buffer.
    
        Returns:
        - np.ndarray: An array of attractive force vectors towards the centroid 
        of the school for each agent.
    
        Notes:
        - The function assumes that `self.X` and `self.Y` are arrays containing 
        the x and y coordinates of all agents.
        - The `school_attraction` method, which must be defined elsewhere in the 
        class, is vectorized and applied to each agent.
        - The `excluded` parameter in `np.vectorize` is used to prevent certain 
        arguments from being broadcasted, allowing
          them to be passed as-is to the vectorized function. In this case, 
          indices 4 to 7 in the argument list are excluded
          from broadcasting, which likely corresponds to the `agents_within_buffers_dict`,
          `self.X`, and `self.Y` arguments.
        - The function returns an array where each element is the calculated 
        attractive force for the corresponding agent.
        - The vectorization allows for the calculation of forces for multiple agents 
        simultaneously, improving performance over a loop-based approach.
        """       
        # Initialize arrays for centroids
        centroid_x = np.zeros(self.num_agents)
        centroid_y = np.zeros(self.num_agents)

        # Initialize arrays for school cue
        school_cue_array = np.zeros((self.num_agents, 2))

        # Flatten the list of neighbor indices and create a corresponding array of agent indices
        neighbor_indices = np.concatenate(self.agents_within_buffers).astype(np.int32)
        agent_indices = np.repeat(np.arange(self.num_agents), [len(neighbors) for neighbors in self.agents_within_buffers]).astype(np.int32)
        
        # Aggregate X and Y coordinates of all neighbors
        x_neighbors = self.X[neighbor_indices]
        y_neighbors = self.Y[neighbor_indices]
        
        # Calculate the means; use np.add.at for unbuffered in-place operation
        centroid_x = np.zeros(self.num_agents)
        centroid_y = np.zeros(self.num_agents)
        centroid_x = np.array([np.mean(self.X[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(self.num_agents)])
        centroid_y = np.array([np.mean(self.Y[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(self.num_agents)])
        
        # Calculate vectors to centroids
        vectors_to_centroid_x = self.X - centroid_x
        vectors_to_centroid_y = self.Y - centroid_x
        
        # Calculate distances to centroids
        distances = np.sqrt(vectors_to_centroid_x**2 + vectors_to_centroid_y**2)
        
        # Normalize vectors (add a small epsilon to distances to avoid division by zero)
        epsilon = 1e-10
        v_hat_x = np.divide(vectors_to_centroid_x, distances + epsilon, out=np.zeros_like(self.X), where=distances+epsilon != 0)
        v_hat_y = np.divide(vectors_to_centroid_y, distances + epsilon, out=np.zeros_like(self.Y), where=distances+epsilon != 0)
        
        # Calculate attractive forces
        school_cue_array[:, 0] = weight * v_hat_x / (distances**2 + epsilon)
        school_cue_array[:, 1] = weight * v_hat_y / (distances**2 + epsilon)
        
        #TODO - we also need to perform velocity matching so.... update ideal_sog
        # Calcaluate a new ideal_sog based on the average sogs of those fish around me
        sogs =  np.array([np.mean(self.sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(self.num_agents)])
        self.school_sog = sogs
        
        return school_cue_array
        
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
        valid_indices = ~np.isnan(self.closest_agent)
        
        # Initialize arrays for closest X and Y positions
        closest_X = np.full_like(self.X, np.nan)
        closest_Y = np.full_like(self.Y, np.nan)
        
        # Extract the closest X and Y positions using the valid indices
        closest_X[valid_indices] = self.X[self.closest_agent[valid_indices].astype(int)]
        closest_Y[valid_indices] = self.Y[self.closest_agent[valid_indices].astype(int)]
        
        # calculate vector pointing from neighbor to self
        self_2_closest = np.column_stack((closest_X.flatten() - self.X.flatten(), closest_Y.flatten() - self.Y.flatten()))
        closest_2_self = np.column_stack((self.X.flatten() - closest_X.flatten(), self.Y.flatten() - closest_Y.flatten()))
        
        coll_slice = determine_slices_from_vectors(closest_2_self, num_slices = 8)
        head_slice = determine_slices_from_headings(self.heading, num_slices = 8)
        
        same_quad_multiplier = np.where(coll_slice == head_slice,0,1)
        
        #TODO - run tests to see if we need to increase the number of slices
        # if np.any(same_quad_multiplier == 1):
        #     print ('fuck')
        
        # Handling np.nan values
        # If either component of a vector is np.nan, you might want to treat the whole vector as invalid
        invalid_vectors = np.isnan(closest_2_self).any(axis=1)
        closest_2_self[invalid_vectors] = [np.nan, np.nan]
        closest_2_self = np.nan_to_num(closest_2_self)
        
        # Replace zeros and NaNs in distances to avoid division errors
        # This step assumes that a zero distance implies the agent is its own closest neighbor, 
        # which might result in a zero vector or a scenario you'll want to handle separately.
        safe_distances = np.where(self.nearest_neighbor_distance > 0, 
                                  self.nearest_neighbor_distance, 
                                  np.nan)
        
        safe_distances_mm = safe_distances * 1000
        
        # Calculate unit vector components
        v_hat_x = np.divide(closest_2_self[:,0], safe_distances, 
                            out=np.zeros_like(closest_2_self[:,0]), where=safe_distances!=0)
        v_hat_y = np.divide(closest_2_self[:,1], safe_distances, 
                            out=np.zeros_like(closest_2_self[:,1]), where=safe_distances!=0)
                        
        # Calculate collision cue components
        collision_cue_x = np.divide(weight * v_hat_x, safe_distances**2, 
                                    out=np.zeros_like(v_hat_x), where=safe_distances!=0) * same_quad_multiplier
        collision_cue_y = np.divide(weight * v_hat_y, safe_distances**2, 
                                    out=np.zeros_like(v_hat_y), where=safe_distances!=0) * same_quad_multiplier
        
        # Optional: Combine the components into a single array
        collision_cue_mm = np.column_stack((collision_cue_x, collision_cue_y))
        collision_cue = collision_cue_mm / 1000.
        
        np.nan_to_num(collision_cue, copy = False)
        
        return collision_cue     
            
    def arbitrate(self, t):

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
        # rheotaxis = self.rheo_cue(10000)
        # shallow = self.shallow_cue(10000)
        # wave_drag = self.wave_drag_cue(5000)
        # low_speed = self.vel_cue(8000)
        # avoid = self.already_been_here(3000, t)
        # school = self.school_cue(9000)
        # collision = self.collision_cue(2500)
        rheotaxis = self.rheo_cue(50000)
        shallow = self.shallow_cue(1)
        wave_drag = self.wave_drag_cue(1)
        low_speed = self.vel_cue(1)
        avoid = self.already_been_here(1, t)
        school = self.school_cue(1)
        collision = self.collision_cue(1)
        
        # Create dictionary that has order of behavioral cues
        order_dict = {0: shallow, 
                      1: collision, 
                      2: avoid, 
                      3: school, 
                      4: rheotaxis, 
                      5: low_speed.T, 
                      6: wave_drag.T}
        
        # Create dictionary that holds all steering cues
        cue_dict = {'rheotaxis': rheotaxis, 
                    'shallow': shallow, 
                    'wave_drag': wave_drag.T, 
                    'low_speed': low_speed.T, 
                    'avoid': avoid, 
                    'school': school, 
                    'collision': collision}
        
        head_vec = np.zeros_like(rheotaxis)
        
        # Arbitrate between different behaviors
        head_vec = np.where(self.swim_behav[:,np.newaxis] == 1,
                            sum([np.where(np.linalg.norm(head_vec, axis=1, keepdims=True) >= 7500, 
                                          head_vec, 
                                          head_vec + cue) for cue in order_dict.values()]),
                            head_vec)
        
        head_vec = np.where(self.swim_behav[:,np.newaxis] == 2, 
                            cue_dict['shallow'] + cue_dict['collision'] + cue_dict['low_speed'],
                            head_vec)
        head_vec = np.where(self.swim_behav[:,np.newaxis] == 3, 
                            cue_dict['rheotaxis'], 
                            head_vec)
        
        # Calculate heading for each agent
        if len(head_vec.shape) == 2:
            self.heading = np.arctan2(head_vec[:, 1], head_vec[:, 0])
        else: 
            self.heading = np.arctan2(head_vec[:, 0, 1], head_vec[:, 0, 0])        
        
        if self.pid_tuning == True:
            if np.any(np.isnan(self.heading)):
                print ('debug check point Nans in headings')
                sys.exit()
    
    def thrust_fun(self, mask, fish_velocities = None):
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
        length_cm = self.length / 1000 * 100.
        
        # Calculate swim speed
        water_vel = self.arr.stack((self.x_vel, self.y_vel), axis=-1)
        if fish_velocities is None:
            fish_velocities = self.arr.stack((self.ideal_sog * self.arr.cos(self.heading),
                                                self.ideal_sog * self.arr.sin(self.heading)), axis=-1)
            
        ideal_swim_speed = np.linalg.norm(fish_velocities - water_vel, axis=-1)

        swim_speed_cms = ideal_swim_speed * 100.
    
        # Data for interpolation
        length_dat = self.arr.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
        speed_dat = self.arr.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
        amp_dat = self.arr.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
        wave_dat = self.arr.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
        edge_dat = self.arr.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])
    
        # Interpolation with extrapolation using UnivariateSpline
        A_spline = UnivariateSpline(length_dat, amp_dat, k = 2, ext = 0)
        V_spline = UnivariateSpline(speed_dat, wave_dat, k = 1, ext = 0)
        B_spline = UnivariateSpline(length_dat, edge_dat, k = 1, ext = 0)
    
        A = A_spline(length_cm)
        V = V_spline(swim_speed_cms)
        B = B_spline(length_cm)
    
        # Calculate thrust
        m = (self.arr.pi * rho * B**2) / 4.
        W = (self.Hz * A * self.arr.pi) / 1.414
        w = W * (1 - swim_speed_cms / V)
    
        # Thrust calculation
        thrust_erg_s = m * W * w * swim_speed_cms - (m * w**2 * swim_speed_cms) / (2. * np.cos(np.radians(theta)))
        thrust_Nm = thrust_erg_s / 10000000.
        thrust_N = thrust_Nm / (self.length / 1000.)
    
        # Convert thrust to vector
        thrust = np.where(mask,[thrust_N * np.cos(self.heading),
                                thrust_N * np.sin(self.heading)],0)
            
        self.thrust = thrust.T
        
    def frequency(self, mask, fish_velocities = None):
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
        lengths_cm = self.length / 10
    
        # Calculate swim speed in cm/s
        water_velocities = self.arr.stack((self.x_vel, self.y_vel), axis=-1)
        alternate = True
        
        if fish_velocities is None:
            fish_velocities = self.arr.stack((self.ideal_sog * self.arr.cos(self.heading),
                                                  self.ideal_sog * self.arr.sin(self.heading)), axis=-1)
            alternate = False
        
        swim_speeds_cms = self.arr.linalg.norm(fish_velocities - water_velocities, axis=-1) * 100

        # sockeye parameters (Webb 1975, Table 20) units in CM!!! 
        length_dat = self.arr.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
        speed_dat = self.arr.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
        amp_dat = self.arr.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
        wave_dat = self.arr.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
        edge_dat = self.arr.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
        
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
        drags_erg_s = np.where(mask,np.linalg.norm(ideal_drag, axis = -1) * self.length/1000 * 10000000,0)
    
        # Solve for Hz
        Hz = np.where(self.swim_behav == 3, 1.0,
                      np.sqrt(drags_erg_s * V**2 * np.cos(np.radians(theta))/\
                              (A**2 * B**2 * swim_speeds_cms * np.pi**3 * rho * \
                              (swim_speeds_cms - V) * \
                              (-0.062518880701972 * swim_speeds_cms - \
                              0.125037761403944 * V * np.cos(np.radians(theta)) + \
                               0.062518880701972 * V)
                               )
                              )
                      )

        self.Hz = Hz
         
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
        kin_temp = self.arr.array([0.01, 10., 20., 25., 30., 40., 50., 60., 70., 80.,
                             90., 100., 110., 120., 140., 160., 180., 200.,
                             220., 240., 260., 280., 300., 320., 340., 360.])
    
        kin_visc = self.arr.array([0.00000179180, 0.00000130650, 0.00000100350,
                             0.00000089270, 0.00000080070, 0.00000065790,
                             0.00000055310, 0.00000047400, 0.00000041270,
                             0.00000036430, 0.00000032550, 0.00000029380,
                             0.00000026770, 0.00000024600, 0.00000021230,
                             0.00000018780, 0.00000016950, 0.00000015560,
                             0.00000014490, 0.00000013650, 0.00000012990,
                             0.00000012470, 0.00000012060, 0.00000011740,
                             0.00000011520, 0.00000011430])
    
        # Interpolate kinematic viscosity based on the temperature
        f_kinvisc = self.arr.interp(temp, kin_temp, kin_visc)
    
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
        dens_temp = self.arr.array([0.1, 1., 4., 10., 15., 20., 25., 30., 35., 40.,
                              45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
                              95., 100., 110., 120., 140., 160., 180., 200.,
                              220., 240., 260., 280., 300., 320., 340., 360.,
                              373.946])
    
        density = self.arr.array([0.9998495, 0.9999017, 0.9999749, 0.9997, 0.9991026,
                            0.9982067, 0.997047, 0.9956488, 0.9940326, 0.9922152,
                            0.99021, 0.98804, 0.98569, 0.9832, 0.98055, 0.97776,
                            0.97484, 0.97179, 0.96861, 0.96531, 0.96189, 0.95835,
                            0.95095, 0.94311, 0.92613, 0.90745, 0.887, 0.86466,
                            0.84022, 0.81337, 0.78363, 0.75028, 0.71214, 0.66709,
                            0.61067, 0.52759, 0.322])
    
        # Interpolate water density based on the temperature
        f_density = self.arr.interp(temp, dens_temp, density)
    
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
        length_m = self.length / 1000.
    
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
        surface_areas = 10 ** (a + b * self.arr.log10(self.length))
    
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
        reynolds_data = self.arr.array([2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5])
        drag_data = self.arr.array([0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
    
        # Fit the logarithmic model to the data
        drag_coefficients = self.arr.interp(reynolds, reynolds_data, drag_data)
    
        return drag_coefficients

    def drag_fun(self, mask):
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

        # Calculate fish velocities
        fish_velocities = np.stack((self.sog * np.cos(self.heading), self.sog * np.sin(self.heading)), axis=-1)
        water_velocities = np.stack((self.x_vel, self.y_vel), axis=-1)
    
        # Ensure non-zero fish velocity for calculation
        fish_speeds = np.linalg.norm(fish_velocities, axis=-1)
        fish_speeds[fish_speeds == 0.0] = 0.0001
        fish_velocities[fish_speeds == 0.0] = [0.0001, 0.0001]
    
        # Calculate kinematic viscosity and density based on water temperature
        viscosity = self.kin_visc(self.water_temp)
        density = self.wat_dens(self.water_temp)

        # Calculate Reynolds numbers
        #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
        length_m = self.length / 1000.
    
        # Calculate the Reynolds number for each fish
        reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
    
        # Calculate surface areas
        
        # Constants for the power-law relationship
        a = -0.143
        b = 1.881
    
        # Calculate the surface area for each fish
        surface_areas = 10 ** (a + b * self.arr.log10(self.length / 1000. * 100.))
        #surface_areas = self.calc_surface_area((self.length / 1000.) * 100.)
    
        # Calculate drag coefficients
        drag_coeffs = self.drag_coeff(reynolds_numbers)
    
        # Calculate relative velocities and their norms
        relative_velocities = fish_velocities - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=1)**2
    
        # Calculate unit vectors for fish velocities
        unit_fish_velocities = fish_velocities / self.arr.linalg.norm(fish_velocities, axis=1)[:,self.arr.newaxis]
    
        # Calculate drag forces
        drags = np.where(mask[:,np.newaxis],
                         -0.5 * (density * 1000) * (surface_areas[:,np.newaxis] / 100**2) \
                                       * drag_coeffs[:,self.arr.newaxis] * relative_speeds_squared[:, np.newaxis] \
                                           * unit_fish_velocities * self.wave_drag[:, np.newaxis],0)

        # drags = np.where(mask, -0.5 * (densities * 1000) * (surface_areas / 100**2) * drag_coeffs * relative_speeds_squared \
        #     * self.arr.linalg.norm(fish_velocities, axis = 1) * self.wave_drag,0)
        
        self.drag = drags

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
        water_velocities = np.stack((self.x_vel, self.y_vel), axis=-1)
        avg_sog = (self.ideal_sog + self.school_sog)/2.
        if fish_velocities is None:
            fish_velocities = np.stack((self.ideal_sog * np.cos(self.heading),
                                        self.ideal_sog * np.sin(self.heading)), axis=-1)
    
        # calculate ideal swim speed  
        ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
       
        # make sure fish isn't swimming faster than it should
        refugia_mask = (self.swim_behav == 2) & (ideal_swim_speeds > self.max_s_U)
        holding_mask = (self.swim_behav == 3) & (ideal_swim_speeds > self.max_s_U)
        too_fast = refugia_mask + holding_mask
        
        fish_velocities = np.where(too_fast[:,np.newaxis],
                                   (self.max_s_U / ideal_swim_speeds[:,np.newaxis]) * fish_velocities,
                                   fish_velocities)
    
        # Calculate the maximum practical speed over ground
        self.max_practical_sog = fish_velocities
        
        if self.num_agents > 1:
            self.max_practical_sog[np.linalg.norm(self.max_practical_sog, axis=1) == 0.0] = [0.0001, 0.0001]
        else:
            pass

        # Kinematic viscosity and density based on water temperature for each fish
        viscosity = self.kin_visc(self.water_temp)
        density = self.wat_dens(self.water_temp)
    
        # Reynolds numbers for each fish
        #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
        length_m = self.length / 1000.
    
        # Calculate the Reynolds number for each fish
        reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
        
        # Surface areas for each fish
        # Constants for the power-law relationship
        a = -0.143
        b = 1.881
        #surface_areas = self.calc_surface_area(self.length)
        surface_areas = 10 ** (a + b * self.arr.log10(self.length / 1000. * 100.))
    
        # Drag coefficients for each fish
        drag_coeffs = self.drag_coeff(reynolds_numbers)
    
        # Calculate ideal drag forces
        relative_velocities = self.max_practical_sog - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1)**2
        unit_max_practical_sog = self.max_practical_sog / np.linalg.norm(self.max_practical_sog, axis=1)[:, np.newaxis]
    
        # Ideal drag calculation
        ideal_drags = -0.5 * (density * 1000) * \
            (surface_areas[:,np.newaxis] / 100**2) * drag_coeffs[:,np.newaxis] \
                * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog \
                      * self.wave_drag[:, np.newaxis]
    
        return ideal_drags
            
    def fatigue(self, t, dt):
        """
        Method tracks battery levels and assigns swimming modes for multiple fish.
    
        Parameters:
        t (float): The current time step.
    
        Attributes:
        x_vel, y_vel (array): Water velocity components in m/s for each fish.
        sog (array): Speed over ground in m/s for each fish.
        heading (array): Heading in radians for each fish.
        pos (array): Current position for each fish.
        prevPos (array): Previous position for each fish.
        swim_behav (array): Swimming behavior for each fish.
        swim_mode (array): Swimming mode for each fish.
        battery (array): Battery level for each fish.
        recover_stopwatch (array): Recovery stopwatch for each fish.
        bout_dur (array): Duration of current swimming bout for each fish.
        dist_per_bout (array): Distance travelled in the current bout for each fish.
        a_p, b_p, a_s, b_s (array): Parameters for calculating time to fatigue.
        max_s_U, max_p_U (array): Maximum sustainable and prolonged swimming speeds for each fish.
        length (array): Length of each fish in meters.
        wave_drag (array): Additional drag factor due to wave-making for each fish.
    
        Notes:
        - The function assumes that the input arrays are structured as arrays of
          values, with each index across the arrays corresponding to a different
          fish.
        - The function adjusts the fish's swimming mode and behavior based on its
          energy expenditure and recovery.
        """
        #dt = 1.0  # Time step duration
    
        # Vector components of water velocity and speed over ground for each fish
        water_velocities = self.arr.stack((self.x_vel, self.y_vel), axis=-1)
        fish_velocities = self.arr.stack((self.sog * self.arr.cos(self.heading),
                                    self.sog * self.arr.sin(self.heading)), axis=-1)
    
        # Calculate swim speeds for each fish
        swim_speeds = self.arr.linalg.norm(fish_velocities - water_velocities, axis=-1)
    
        # Calculate distances travelled and update bout odometer and duration
        dist_travelled = self.arr.sqrt((self.prev_X - self.X)**2 + (self.prev_Y - self.Y)**2)
        if len(dist_travelled.shape) == 1:
            self.dist_per_bout += dist_travelled
        else:
            self.dist_per_bout += dist_travelled.flatten()

        self.bout_dur += dt
    
        # Initialize time to fatigue (ttf) array
        ttf = self.arr.full_like(swim_speeds, self.arr.nan)
    
        # Calculate ttf for prolonged and sprint swimming modes
        mask_prolonged = (self.max_s_U < swim_speeds) & (swim_speeds <= self.max_p_U)
        mask_sprint = swim_speeds > self.max_p_U
        ttf[mask_prolonged] = 10. ** (self.a_p + swim_speeds[mask_prolonged] * self.b_p) * 60.
        ttf[mask_sprint] = 10. ** (self.a_s + swim_speeds[mask_sprint] * self.b_s) * 60.
    
        # Set swimming modes based on swim speeds
        self.swim_mode = self.arr.where(mask_prolonged, 2, self.swim_mode)
        self.swim_mode = self.arr.where(mask_sprint, 3, self.swim_mode)
        self.swim_mode = self.arr.where(~(mask_prolonged | mask_sprint), 1, self.swim_mode)
    
        # Calculate recovery at the beginning and end of the time step
        rec0 = self.recovery(self.recover_stopwatch) / 100.
        rec0[rec0 < 0.0] = 0.0
        rec1 = self.recovery(self.recover_stopwatch + dt) / 100.
        rec1[rec1 > 1.0] = 1.0
        rec1[rec1 < 0.] = 0.0
        per_rec = rec1 - rec0
    
        # Update battery levels for sustained swimming mode
        mask_sustained = self.swim_mode == 1
        if self.num_agents > 1:
            self.battery[mask_sustained] += per_rec[mask_sustained]
        else:
            self.battery[mask_sustained.flatten()] += per_rec[mask_sustained.flatten()]
        self.battery[self.battery > 1.0] = 1.0
    
        # Update battery levels for non-sustained swimming modes
        mask_non_sustained = ~mask_sustained
        if self.num_agents > 1:
            ttf0 = ttf[mask_non_sustained] * self.battery[mask_non_sustained]
        else:
            ttf0 = ttf[mask_non_sustained.flatten()] * self.battery[mask_non_sustained.flatten()]

        ttf1 = ttf0 - dt
        if self.num_agents > 1:
            self.battery[mask_non_sustained] *= ttf1 / ttf0
        else:
            self.battery[mask_non_sustained.flatten()] *= ttf1.flatten() / ttf0.flatten()

        self.battery[self.battery < 0.0] = 0.0
    
        # Set swimming behavior based on battery level
        mask_low_battery = self.battery <= 0.1
        mask_mid_battery = (self.battery > 0.1) & (self.battery <= 0.3)
        mask_high_battery = self.battery > 0.3
    
        self.swim_behav = self.arr.where(mask_low_battery, 3, self.swim_behav)
        self.swim_behav = self.arr.where(mask_mid_battery, 2, self.swim_behav)
        self.swim_behav = self.arr.where(mask_high_battery, 1, self.swim_behav)
    
        # Set ideal speed over ground based on battery level
        self.ideal_sog[mask_low_battery] = 0.0
        self.ideal_sog[mask_mid_battery] = 0.02
        ideal_bls = 0.0075 * self.arr.exp(4.89 * self.battery[mask_high_battery])
        self.ideal_sog[mask_high_battery] = self.arr.round(ideal_bls * (self.length[mask_high_battery] / 1000.), 2)
    
        # Check if the fish should switch to station holding based on bout duration and distance
        mask_bout_check = (self.bout_dur > 300) & (self.dist_per_bout / self.bout_dur < 0.1)
        self.swim_behav[mask_bout_check] = 3
        self.swim_mode[mask_bout_check] = 1
        self.ideal_sog[mask_bout_check] = 0.0
    
        # Recovery for fish that are station holding
        mask_station_holding = self.swim_behav == 3
        self.bout_dur[mask_station_holding] = 0.0
        self.dist_per_bout[mask_station_holding] = 0.0
        self.battery[mask_station_holding] += per_rec[mask_station_holding]
        self.recover_stopwatch[mask_station_holding] += dt
    
        # Fish ready to start moving again after recovery
        mask_ready_to_move = self.battery >= 0.85
        self.recover_stopwatch[mask_ready_to_move] = 0.0
        self.swim_behav[mask_ready_to_move] = 1
        self.swim_mode[mask_ready_to_move] = 1
        
        if self.pid_tuning == True:
            print(f'battery: {np.round(self.battery,4)}')
            print(f'swim behavior: {self.swim_behav[0]}')
            print(f'swim mode: {self.swim_mode[0]}')

            if np.any(self.swim_behav == 3):
                print('error no longer counts, fatigued')
                sys.exit()
        
        # if np.any(self.battery != 1):
        #     print ('debug battery change QC')
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
        self.swim_speed = np.linalg.norm(ideal_velocities - water_velocities[:, np.newaxis], axis=1)
            

    def swim(self, dt, pid_controller, mask):
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
        
        # Step 1: Calculate fish velocity in vector form for each fish
        fish_vel_0_x = np.where(mask, self.sog * np.cos(self.heading),0) 
        fish_vel_0_y = np.where(mask, self.sog * np.sin(self.heading),0)  
        
        fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
        
        # Step 2: Calculate surge for each fish
        surge_ini = self.thrust + self.drag
        
        # Step 3: Calculate acceleration for each fish
        acc_ini = np.round(surge_ini / self.weight[:,np.newaxis], 2)  
        
        # Step 4: Update velocity for each fish
        fish_vel_1_ini = fish_vel_0.flatten() + acc_ini.flatten() * dt  
        
        new_sog = np.linalg.norm(fish_vel_1_ini, axis = -1)


        # Step 5: Thrust feedback PID controller 
        error = np.where(mask, 
                         np.round(self.ideal_sog - new_sog,12),
                         0.)
        
        self.error = error
        
    
        # Adjust Hzs using the PID controller (vectorized)
        pid_adjustment = pid_controller.update(error)

        if self.pid_tuning == True:
            self.error_array = np.append(self.error_array, error[0])
            self.vel_x_array = np.append(self.vel_x_array, self.x_vel)
            self.vel_y_array = np.append(self.vel_y_array, self.y_vel)
            
            curr_vel = np.round(np.sqrt(np.power(self.x_vel,2) + np.power(self.y_vel,2)),2)
            
            print (f'error: {error}')
            print (f'current velocity: {curr_vel}')
            print(f'Hz: {self.Hz}')
            print(f'thrust: {np.round(self.thrust,2)}')
            print(f'drag: {np.round(self.drag,2)}')
            print(f'sog: {np.round(self.sog,4)}')

        
            if np.isnan(error):
                print('nan in error')
                sys.exit()
        
        # add adjustment to the magnitude of thrust
        thrust_mag_0 = np.linalg.norm(self.thrust, axis = -1)
        thrust_mag_1 = thrust_mag_0 + pid_adjustment
        if self.num_agents > 1:
            self.thrust = self.thrust * (thrust_mag_1[:,np.newaxis]/thrust_mag_0[:,np.newaxis])
        else:
            self.thrust = self.thrust * (thrust_mag_1/thrust_mag_0)

        # Step 6: Calculate adjusted surge for each fish
        surge_adj = self.thrust + self.drag
        
        # Step 7: Calculate acceleration for each fish
        acc_adj = np.round(surge_adj / self.weight[:,np.newaxis], 2)  
        
        # Step 8: Update velocity for each fish
        if self.num_agents > 1:
            fish_vel_1_adj = fish_vel_0 + acc_adj * dt  

        else:
            fish_vel_1_adj = fish_vel_0.flatten() + acc_adj.flatten() * dt  
        
        # Step 9: calculate tailbeat frequency at the new velocity
        self.frequency(mask, fish_vel_1_adj)
        
        # Step 10: if any adjusted frequencies are less than 0 or greater than 20, adjust
        self.Hz = np.where(mask, np.clip(self.Hz, 0, 20),self.Hz)
        
        # Step 11: calculate thrust - again, this in case
        self.thrust_fun(mask = mask, fish_velocities = fish_vel_1_adj)
        
        # Step 12: calculate final surge and acceleration
        surge_final = np.where(mask, self.thrust + self.drag, surge_ini)
        
        # Step 7: Calculate acceleration for each fish
        acc_final = np.round(surge_final / self.weight[:,np.newaxis], 2)  
        
        # Step 8: Update velocity for each fish
        fish_vel_1 = fish_vel_0.flatten() + acc_final.flatten() * dt  # X component of new velocity   
        
        # Step 6: Update sog for each fish
        self.sog = np.array([np.linalg.norm(fish_vel_1, axis = -1)])
        
        # Step 7: Prepare for position update
        # Note: Actual position update should be done in the main simulation loop
        self.prev_X = np.where(mask,self.X.copy(),self.prev_X)
        self.prev_Y = np.where(mask,self.Y.copy(),self.prev_Y)
        if self.num_agents > 1:
            self.X = np.where(mask, self.X + fish_vel_1[:,0] * dt, self.X)
            self.Y = np.where(mask, self.Y + fish_vel_1[:,1] * dt, self.Y)
        else:
            self.X = np.where(mask, self.X + fish_vel_1[0] * dt, self.X)
            self.Y = np.where(mask, self.Y + fish_vel_1[1] * dt, self.Y)
                        
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
        self.time_of_jump = np.where(mask,t,self.time_of_jump)
    
        # Get jump angle for each fish
        jump_angles = np.where(mask,self.arr.random.choice([self.arr.radians(45), self.arr.radians(60)], size=self.ucrit.shape),0)
    
        # Calculate time airborne for each fish
        time_airborne = np.where(mask,(2 * self.ucrit * self.arr.sin(jump_angles)) / g, 0)
    
        # Calculate displacement for each fish
        displacement = self.ucrit * time_airborne * self.arr.cos(jump_angles)
    
        # Set speed over ground to ucrit for each fish
        self.sog = np.where(mask, self.ucrit, self.sog)
    
        # Calculate new heading angle for each fish based solely on flow direction
        self.heading = np.where(mask,
                                self.arr.arctan2(self.y_vel.flatten(), 
                                                 self.x_vel.flatten()) - self.arr.radians(180),
                                self.heading
                                )
    
        # Calculate the new position for each fish
        if self.num_agents > 1:
            self.X += displacement * self.arr.cos(self.heading)
            self.Y += displacement * self.arr.sin(self.heading)
        else:
            self.X += displacement.flatten() * self.arr.cos(self.heading.flatten())
            self.Y += displacement.flatten() * self.arr.sin(self.heading.flatten())        

            
    def odometer(self, t):
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
            self.arr.exp(0.0565 * np.power(self.arr.log(self.weight * 1000), 0.9141)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.1498 * self.arr.power(self.arr.log(self.weight * 1000), 0.8465)),
                self.arr.exp(0.1987 * self.arr.power(self.arr.log(self.weight * 1000), 0.8844))
            )
        )
    
        ar_o2_rate = self.arr.where(
            self.water_temp <= 5.3,
            self.arr.exp(0.4667 * self.arr.power(self.arr.log(self.weight * 1000), 0.9989)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.9513 * self.arr.power(self.arr.log(self.weight * 1000), 0.9632)),
                self.arr.exp(0.8237 * self.arr.power(self.arr.log(self.weight * 1000), 0.9947))
            )
        )
    
        # Calculate total metabolic rate
        if self.num_agents > 1:
            swim_cost = sr_o2_rate + self.wave_drag * (
                self.arr.exp(np.log(sr_o2_rate) + self.swim_speed * (
                    (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
                )) - sr_o2_rate
            )
        else:
            swim_cost = sr_o2_rate + self.wave_drag.flatten() * (
                self.arr.exp(np.log(sr_o2_rate) + np.linalg.norm(self.swim_speed.flatten(), axis = -1) * (
                    (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
                )) - sr_o2_rate
            )
        # Update kilocalories burned
        self.kcal += swim_cost
            
    def timestep(self, t, dt, g, pid_controller):
        """
        Simulates a single time step for all fish in the simulation.
    
        Parameters:
        - t: Current simulation time.
        - dt: Time step duration.
    
        The method performs the following operations for each fish:
        1. Updates the mental map based on the current time.
        2. Senses the environment to gather necessary data.
        3. Optimizes vertical position within the water column.
        4. Calculates the wave drag multiplier based on environmental conditions.
        5. Assesses fatigue levels to determine energy reserves and swimming capabilities.
        6. Arbitrates among behavioral cues to decide on actions.
        7. Decides whether each fish should jump or swim based on their speed, heading, and energy levels.
        8. Calculates the energy expenditure for the time step.
        9. Logs the simulation data for the current time step.
        """
    
        # Assess mental map
        self.update_mental_map(t)
        
        # Sense the environment
        self.environment()
            
        # Optimize vertical position
        self.find_z()
        
        # Get wave drag multiplier
        self.wave_drag_multiplier()
        
        # Assess fatigue
        self.fatigue(t, dt)
        
        # Arbitrate amongst behavioral cues
        self.arbitrate(t)
        
        # Calculate the ratio of ideal speed over ground to the magnitude of water velocity
        sog_to_water_vel_ratio = self.ideal_sog / self.arr.linalg.norm([self.x_vel, self.y_vel], axis=0)
        
        # Calculate the sign of the heading and the water flow direction
        heading_sign = np.sign(self.heading)
        water_flow_direction_sign = self.arr.sign(self.arr.arctan2(self.y_vel, self.x_vel))
        
        # Calculate the time since the last jump
        time_since_jump = t - self.time_of_jump
        
        # Create a boolean mask for the fish that should jump
        should_jump = (sog_to_water_vel_ratio < 0.05) & (heading_sign != water_flow_direction_sign) & \
                      (time_since_jump > 180) & (self.battery > 0.9999) # default value battery 0.4
        
        # Apply the jump or swim functions based on the condition
        # For each fish that should jump
        self.jump(t=t, g = g, mask=should_jump)
        
        # For each fish that should swim
        self.drag_fun(mask=~should_jump)
        self.frequency(mask=~should_jump)
        self.thrust_fun(mask=~should_jump)
        self.swim(dt, pid_controller = pid_controller, mask=~should_jump)
        
        # Calculate mileage
        self.odometer(t=t)  
        
        # Log the timestep data
        self.timestep_flush(t)

            
    def run(self, model_name, k_p, k_i, k_d, n, dt):
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
        # get depth raster
        depth_arr = self.hdf5['environment/depth'][:]
        depth = rasterio.MemoryFile()
        height = depth_arr.shape[0]
        width = depth_arr.shape[1]
        
        with depth.open(
            driver ='GTiff',
            height = depth_arr.shape[0],
            width = depth_arr.shape[1],
            count =1,
            dtype ='float32',
            crs = self.crs,
            transform = self.depth_rast_transform
        ) as dataset:
            dataset.write(depth_arr, 1)

            # define metadata for movie
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title= model_name, artist='Matplotlib',
                            comment='emergent model run %s'%(datetime.now()))
            writer = FFMpegWriter(fps=30, metadata=metadata)

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
            with writer.saving(fig, os.path.join(self.model_dir,'%s.mp4'%(model_name)), 300):
                # set up PID controller 
                #TODO make PID controller a function of length and water velocity
                pid_controller = PID_controller(k_p, 
                                                k_i, 
                                                k_d, 
                                                self.num_agents)
                for i in range(n):
                    self.timestep(i, dt, g, pid_controller)

                    if self.pid_tuning == True:
                        if i == range(n)[-1]:
                            sys.exit()
                    else:
                        # write frame
                        agent_pts.set_data(self.X,
                                           self.Y)
                        writer.grab_frame()
                        
                    print ('Time Step %s complete'%(i))


        # clean up
        writer.finish()
        if self.pid_tuning == False:
            self.hdf5.flush()
        self.hdf5.close()
        depth.close()     
        t1 = time.time()     
        
        print ('ABM took %s to compile'%(t1-t0))
        
    def close(self):
        self.hdf5.close()
            
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
            magnitude = np.nansum(np.power(filtered_array,2))
                        
            row_data = {
                'individual': i,
                'p': self.p[i],
                'i': self.i[i],
                'd': self.d[i],
                'magnitude': magnitude,
                'array_length': len(filtered_array),
                'avg_velocity': np.nanmean(self.velocities[i])}

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
            selected_parents = PID_optimization.selection(self, error_df)

            # crossover -> output is list of crossover pid values
            cross_offspring = PID_optimization.crossover(self, selected_parents)

            # mutation -> output is list of muation pid values
            mutated_offspring = PID_optimization.mutation(self)

            # combine crossover and mutation offspring to get next generation
            population = cross_offspring + mutated_offspring
            
            print(f'completed generation {generation+1}.... ')
            
        return records
            
            
            
            
            
            
            
            
            
            
