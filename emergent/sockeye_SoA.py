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
import cupy as cp
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
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def get_array_module(use_gpu):
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
        except ImportError:
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
    if isinstance(cols, arr_type.ndarray):
        cols = cols.get()
        rows = rows.get()

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
                 num_timesteps = 100, 
                 num_agents = 100, 
                 use_gpu = False,):
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
         - array_module (module): Module used for array operations (either numpy or cupy).
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
        self.array_module = get_array_module(use_gpu)
        
        # model directory and model name
        self.model_dir = model_dir
        self.model_name = model_name
        self.db = os.path.join(self.model_dir,'%s.hf'%(self.model_name))
                
        # coordinate reference system for the model
        self.crs = crs
        self.basin = basin
        
        # model parameters
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        
        # initialize agent properties and internal states
        self.sim_sex()
        self.sim_length()
        self.sim_weight()
        self.sim_body_depth()
        recover = pd.read_csv("../data/recovery.csv")
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds, recover.Recovery, extrapolate = True,)
        del recover
        self.swim_behav = self.array_module.repeat(1, num_agents)               # 1 = migratory , 2 = refugia, 3 = station holding
        self.swim_mode = self.array_module.repeat('sustained', num_agents)      # 1 = sustained, 2 = prolonged, 3 = sprint
        self.battery = self.array_module.repeat(1.0, num_agents)
        self.recover_stopwatch = self.array_module.repeat(0.0, num_agents)
        self.ttfr = self.array_module.repeat(0.0, num_agents)
        self.time_out_of_water = self.array_module.repeat(0.0, num_agents)
        
        self.X = self.array_module.random.uniform(starting_box[0], starting_box[1],num_agents)
        self.Y = self.array_module.random.uniform(starting_box[2], starting_box[3],num_agents)
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
        self.drag = self.array_module.zeros(num_agents)           # computed theoretical drag
        self.thrust = self.array_module.zeros(num_agents)         # computed theoretical thrust Lighthill 
        self.Hz = self.array_module.zeros(num_agents)             # tail beats per second
        self.bout_no = self.array_module.zeros(num_agents)        # bout number - new bout whenever fish recovers
        self.dist_per_bout = self.array_module.zeros(num_agents)  # running counter of the distance travelled per bout
        self.bout_dur = self.array_module.zeros(num_agents)       # running bout timer 
        self.time_of_jump = self.array_module.zeros(num_agents)   # time since last jump - can't happen every timestep
        
        # initialize odometer
        self.kcal = self.array_module.zeros(num_agents)           #kilo calorie counter
        
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
        self.hdf5.flush()
        
   
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
            self.sex = self.array_module.random.choice([0,1], size = self.num_agents, p = [0.503,0.497])
            
    def sim_length(self):
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
        if self.basin == "Nushagak River":
            if self.sex == 'M':
                self.length = self.array_module.random.lognormal(mean = 6.426,sigma = 0.072,size = self.num_agents)
            else:
                self.length = self.array_module.random.lognormal(mean = 6.349,sigma = 0.067,size = self.num_agents)
        
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
                self.body_depth = self.array_module.exp(-1.938 + np.log(self.length) * 1.084 + 0.0435) / 10.
            else:
                self.body_depth = self.array_module.exp(-1.938 + np.log(self.length) * 1.084) / 10.
                
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
        agent_data.create_dataset("Y", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("Z", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("prev_X", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("prev_X", (self.num_timesteps, self.num_agents), dtype='f4')            
        agent_data.create_dataset("heading", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("sog", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("ideal_sog", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("swim_speed", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("battery", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("swim_behav", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("swim_mode", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("recover_stopwatch", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("ttfr", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("time_out_of_water", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("drag", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("thrust", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("Hz", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("bout_no", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("dist_per_bout", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("bout_dur", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("time_of_jump", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("kcal", (self.num_timesteps, self.num_agents), dtype='f4')
        
        # Set attributes (metadata) if needed
        self.hdf5.attrs['simulation_name'] = "%s Sockeye Movement Simulation"%(self.basin)
        self.hdf5.attrs['num_agents'] = self.num_agents 
        self.hdf5.attrs['num_timesteps'] = self.num_timesteps
        self.hdf5.attrs['basin'] = self.basin
        self.hdf5.attrs['crs'] = self.crs
        
    def timestep_flush(self, timestep):
        '''function writes to the open hdf5 file '''
        
        # write time step data to hdf
        self.hdf5['agent_data/X'][:, timestep] = self.X
        self.hdf5['agent_data/Y'][:, timestep] = self.Y
        self.hdf5['agent_data/Z'][:, timestep] = self.Z
        self.hdf5['agent_data/prev_X'][:, timestep] = self.prev_X
        self.hdf5['agent_data/prev_Y'][:, timestep] = self.prev_Y
        self.hdf5['agent_data/heading'][:, timestep] = self.heading
        self.hdf5['agent_data/sog'][:, timestep] = self.sog
        self.hdf5['agent_data/ideal_sog'][:, timestep] = self.ideal_sog
        self.hdf5['agent_data/swim_speed'][:, timestep] = self.swim_speed
        self.hdf5['agent_data/battery'][:, timestep] = self.battery
        self.hdf5['agent_data/swim_behav'][:, timestep] = self.swim_behav
        self.hdf5['agent_data/swim_mode'][:, timestep] = self.swim_mode
        self.hdf5['agent_data/recover_stopwatch'][:, timestep] = self.recover_stopwatch
        self.hdf5['agent_data/ttfr'][:, timestep] = self.ttfr
        self.hdf5['agent_data/time_out_of_water'][:, timestep] = self.time_out_of_water
        self.hdf5['agent_data/drag'][:, timestep] = self.drag
        self.hdf5['agent_data/thrust'][:, timestep] = self.thrust
        self.hdf5['agent_data/Hz'][:, timestep] = self.Hz
        self.hdf5['agent_data/bout_no'][:, timestep] = self.bout_no
        self.hdf5['agent_data/dist_per_bout'][:, timestep] = self.dist_per_bout
        self.hdf5['agent_data/bout_dur'][:, timestep] = self.bout_dur
        self.hdf5['agent_data/time_of_jump'][:, timestep] = self.time_of_jump

        # Periodically flush data to ensure it's written to disk
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
       
        # Create groups for organization (optional)
        env_data = self.hdf5.create_group("environment")
        
        # get raster properties
        src = rasterio.open(data_dir)
        num_bands = src.count
        width = src.width
        height = src.height
        dtype = np.float32
        transform = src.transform

        shape = (num_bands, height, width)
        #shape = (num_bands, width, height)

        if surface_type == 'velocity x':
            # set transform as parameter of simulation
            self.vel_x_rast_transform = transform
            
            # create an hdf5 array and write to it
            env_data.create_dataset("vel_x", (height, width), dtype='f4')
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
            env_data.create_dataset("vel_dir", (height, width), dtype='f4')
            self.hdf5['environment/vel_dir'][:, :] = src.read(1)  
            
        self.width = width
        self.height = height
        self.hdf5.flush()
        src.close()

    def HECRAS (self,HECRAS_model,resolution):
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
        hdf = h5py.File(HECRAS_model,'r')
        
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
        gdf = gpd.GeoDataFrame(df,crs = self.crs)
        
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
        crs = self.crs
        #transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)
        transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution)\
            * Affine.scale(resolution,-1 * resolution)

        # write elev raster
        with rasterio.open(os.path.join(self.model_dir,'elev.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'wsel.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'depth.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'vel_dir.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'vel_mag.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'vel_x.tif'),
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
        with rasterio.open(os.path.join(self.model_dir,'vel_y.tif'),
                           mode = 'w',
                           driver = driver,
                           width = width,
                           height = height,
                           count = count,
                           dtype = 'float64',
                           crs = crs,
                           transform = transform) as vel_y_rast:
            vel_y_rast.write(vel_y_new,1)
            
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
        self.map = self.array_module.zeros((self.num_agents, self.width, self.height))
               
        # Create groups for organization (optional)
        mem_data = self.hdf5.create_group("memory")
        
        # create a memory map array
        mem_data.create_dataset('maps', (self.height, self.width), dtype = 'f4')
        
        # write it to the hdf5
        self.hdf5['memory/maps'][:, :] = self.map
        
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
        rows, cols = self.geo_to_pixel(self.X, self.Y, transform)

        # Use the already open HDF5 file object to read the specified raster dataset
        raster_dataset = self.hdf5['environment'][raster_name]  # Adjust the path as needed
        # Sample the raster values using the row, col indices
        # Ensure that the indices are within the bounds of the raster data
        rows = np.clip(rows, 0, raster_dataset.shape[0] - 1)
        cols = np.clip(cols, 0, raster_dataset.shape[1] - 1)
        values = raster_dataset[rows, cols]

        return values  
     
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
        row, col = self.geo_to_pixel(self.x, self.y, self.vel_dir_rast_transform)
            
        # get the initial heading values
        values = self.sample_environment(self.vel_dir_rast_transform,'vel_dir')
        
        # set direction 
        self.heading = self.array_module.where(values < 0, 
                                               (self.array_module.radians(360) + values) - self.array_module.radians(180), 
                                               values - self.array_module.radians(180))

        # set initial max practical speed over ground as well
        self.max_practical_sog = self.array_module.array([self.sog * self.array_module.cos(self.heading), 
                                                          self.sog * self.array_module.sin(self.heading)]) #meters/sec       

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
        rows, cols = self.geo_to_pixel(self.X, self.Y, self.vel_dir_rast_transform)
    
        # Ensure rows and cols are within the bounds of the mental map
        rows = self.array_module.clip(rows, 0, self.height - 1)
        cols = self.array_module.clip(cols, 0, self.width - 1)
    
        # Construct an index array for advanced indexing
        agent_indices = np.arange(self.num_agents)
        indices = (agent_indices, rows, cols)
    
        # Update the mental map for all agents in the HDF5 dataset at once
        mental_map_dataset = self.hdf5['memory/maps']
            
        # Use advanced indexing to update the mental map
        # Note: This assumes that the HDF5 dataset supports numpy-style advanced indexing
        mental_map_dataset[indices] = current_timestep

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
        rows, cols = self.geo_to_pixel(x, y, self.depth_rast_transform)
        
        # Construct an index array for advanced indexing
        agent_indices = np.arange(self.num_agents)
        
        # create array slice bounds
        buff = 4
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
        
        # Create slice objects for indexing
        col_slice = slice(xmin, xmax)
        row_slice = slice(ymin, ymax)
        
        # Construct the indices tuple for advanced indexing
        indices = (agent_indices, row_slice, col_slice)
        
        # Access the mental map dataset from the HDF5 file
        mental_map_dataset = self.hdf5['memory/maps']
        
        # Use advanced indexing to retrieve the slices of the mental map
        # Note: This assumes that the HDF5 dataset supports numpy-style advanced indexing
        mmap = mental_map_dataset[indices]
        
        # get shape parameters of mental map array
        num_agents, map_width, map_height = mmap.shape
        
        t_since = mmap - t
        
        multiplier = np.where(np.logical_and(t_since > 600, t_since < 3600),1,0)

        # create an array of x and y coordinates of cells 
        repx, repy = np.meshgrid(np.arange(map_width), np.arange(map_height), indexing='ij')
        
        # Initialize an array to hold the distances for each agent
        dist_grid = np.zeros((num_agents, map_width, map_height))

        # reshape position arrays for broadcasting
        agent_x_positions = self.X[:, np.newaxis, np.newaxis]  # Reshape for broadcasting
        agent_y_positions = self.Y[:, np.newaxis, np.newaxis]  # Reshape for broadcasting

        # Compute the relative coordinates of each cell from the agents' positions
        relative_x = repx - agent_x_positions
        relative_y = repy - agent_y_positions
        
        # Calculate the Euclidean distance using broadcasting
        dist_grid = np.power(np.sqrt(relative_x**2 + relative_y**2),2)
        
        # Calculate the unit vector components
        magnitude = np.sqrt(relative_x**2 + relative_y**2)
        unit_vector_x = np.divide(relative_x, magnitude, out=np.zeros_like(relative_x), where=magnitude!=0)
        unit_vector_y = np.divide(relative_y, magnitude, out=np.zeros_like(relative_y), where=magnitude!=0)
        
        # Calculate the direction using arctan2, which handles the division and quadrant determination
        dir_grid = np.arctan2(unit_vector_y, unit_vector_x)
        
        # calculate repulsive force in X and Y directions 
        x_force = ((weight * np.cos(dir_grid))/ dist_grid) * multiplier
        y_force = ((weight * np.sin(dir_grid))/ dist_grid) * multiplier
        
        # Calculate the repulsive force in X and Y directions for each agent
        x_force_per_agent = np.nansum(x_force, axis=(1, 2))  # Sum over the map dimensions, keep the agent dimension
        y_force_per_agent = np.nansum(y_force, axis=(1, 2))  # Sum over the map dimensions, keep the agent dimension
        
        # Stack the forces into a single array with shape (num_agents, 2)
        repulsive_forces_per_agent = np.stack((x_force_per_agent, y_force_per_agent), axis=-1)
        
        return repulsive_forces_per_agent

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
    
        # Create the buffer rectangles (bounding boxes) for each agent
        gdf['buffer'] = gdf.apply(
            lambda row: box(
                row.geometry.x - 1,
                row.geometry.y - 1,
                row.geometry.x + 1,
                row.geometry.y + 1
            ),
            axis=1
        )
        
        # Prepare the spatial index on the agents' buffers
        spatial_index = gdf.sindex
        
        # Initialize an empty dictionary to store the results
        agents_within_buffers_dict = {}
        closest_agent_dict = {}
        
        for index, agent in gdf.iterrows():
            # Use the spatial index to get the possible matches
            possible_matches_index = list(spatial_index.intersection(agent['buffer'].bounds))
            possible_matches = gdf.iloc[possible_matches_index]
        
            # Further refine the possible matches by checking if they are actually within the buffer
            precise_matches = possible_matches[possible_matches.intersects(agent['buffer'])]
        
            # Exclude the current agent from the matches
            precise_matches = precise_matches.drop(index, errors='ignore')
        
            # Add the indices of the matching agents to the dictionary
            agents_within_buffers_dict[index] = list(precise_matches.index)
        
            # Calculate distances to all agents within the buffer and find the closest
            if not precise_matches.empty:
                distances = precise_matches.distance(agent['geometry'])
                closest_agent_index = distances.idxmin()
                closest_agent_dict[index] = closest_agent_index
            else:
                closest_agent_dict[index] = None  # Or some placeholder to indicate no agents are close
        
        # Now `agents_within_buffers_dict` is a dictionary where each key is an agent index
        return agents_within_buffers_dict, closest_agent_dict

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
            The function uses `self.array_module.where` for vectorized conditional operations,
            which implies that `self.depth`, `self.body_depth`, and `self.too_shallow` should be array-like
            and support broadcasting if they are not scalar values.
        """
        self.z = self.array_module.where(
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
        length_numpy = self.length.get() if isinstance(self.length, cp.ndarray) else self.length
        
        # calculate buffer size based on swim mode, if we are in refugia mode buffer is 15 body lengths else 5
        buff = np.where(self.swim_mode == 2, 15 * length_numpy, 5 * length_numpy)
        
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
        
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.array_module, self.depth_rast_transform)
        
        # Access the velocity dataset from the HDF5 file
        velocity = self.hdf5['environment/vel_mag'][:]
        
        # Initialize an array to hold the velocity cues for each agent
        velocity_min = np.zeros((self.num_agents, 2), dtype=float)
        
        for i in range(self.num_agents):
            # calculate array slice bounds
            xmin = max(0, cols[i] - buff[i])
            xmax = min(velocity.shape[1], cols[i] + buff[i] + 1)  # +1 because slicing is exclusive on the upper bound
            ymin = max(0, rows[i] - buff[i])
            ymax = min(velocity.shape[0], rows[i] + buff[i] + 1)  # +1 for the same reason
            
            # Retrieve the slice of the velocity array for this agent
            vel_slice = velocity[ymin:ymax, xmin:xmax]
            
            # Find the indices of the minimum velocity within the slice
            min_idx = np.unravel_index(np.argmin(vel_slice, axis=None), vel_slice.shape)
            
            # Convert indices to projected coordinates
            min_x, min_y = pixel_to_geo(
                self.depth_rast_transform, 
                min_idx[0] + ymin, 
                min_idx[1] + xmin
            )
            
            # Calculate the direction vector to the minimum velocity cell
            diff_x = min_x - x[i]
            diff_y = min_y - y[i]
            
            # Normalize the direction vector
            magnitude = np.sqrt(diff_x**2 + diff_y**2)
            if magnitude > 0:
                velocity_min[i] = [diff_x / magnitude, diff_y / magnitude]
            else:
                velocity_min[i] = [0, 0]  # No movement if the minimum velocity is at the agent's position
        
        # Scale the velocity cues by the weight and normalize by the square of 5 body lengths in meters
        velocity_min *= weight / ((5 * length_numpy / 1000) ** 2)
        
        return velocity_min

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
        length_numpy = self.length.get() if isinstance(self.length, cp.ndarray) else self.length
    
        # Sample the environment to get the velocity direction and adjust to point upstream
        vel_dir = self.sample_environment(self.vel_dir_rast_transform, 'vel_dir') - np.radians(180)
        
        # Calculate the unit vector in the upstream direction
        v_hat = np.array([np.cos(vel_dir), np.sin(vel_dir)])
        
        # Calculate the rheotactic cue
        rheotaxis = (weight * v_hat)/((2 * length_numpy/1000.)**2)
        
        return rheotaxis
    
    #create a function that returns a total force vector in x and y for each agent
    @np.vectorize
    def calculate_shallow_repulsive_force(agent_idx, x, y, xmin, xmax, ymin, ymax, body_depth, weight, depth_array, arr_type, depth_rast_transform):
        """
        Calculate the total repulsive force vector for a single agent based on the surrounding shallow water cells.
    
        This function computes the repulsive force exerted on an agent by shallow water areas within a specified buffer. It identifies shallow cells, computes the direction and magnitude of the repulsive force from each cell, and sums these to obtain the total force vector for the agent.
    
        Parameters:
        - agent_idx (int): Index of the agent for which to calculate the force.
        - x (float): X-coordinate of the agent's position.
        - y (float): Y-coordinate of the agent's position.
        - xmin (float): Minimum x-coordinate of the buffer area.
        - xmax (float): Maximum x-coordinate of the buffer area.
        - ymin (float): Minimum y-coordinate of the buffer area.
        - ymax (float): Maximum y-coordinate of the buffer area.
        - body_depth (float): The body depth of the agent.
        - weight (float): The weighting factor to scale the repulsive force.
        - depth_array (np.ndarray): Array containing depth data.
        - depth_rast_transform (affine.Affine): Transformation from geographic coordinates to pixel coordinates.
    
        Returns:
        - tuple: A tuple containing the total repulsive force in the X and Y directions for the agent.
    
        Notes:
        - The function assumes that the depth data is provided as a 2D NumPy array.
        - The buffer area is defined by the xmin, xmax, ymin, and ymax parameters, which should be calculated beforehand based on the agent's position and the desired buffer size.
        - The repulsive force is calculated as a vector normalized by the magnitude of the distance to each shallow cell, scaled by the weight.
        - The function returns the total repulsive force vector, which is the sum of the individual forces from all shallow cells within the buffer.
        """        
        # Calculate max depth - body depth in cm - make sure we divide by 100.
        min_depth = (body_depth * 1.1) / 100.# Use advanced indexing to create a boolean mask for the slices
        
        # calculate a force multiplier
        multiplier = np.where(depth_array < min_depth, 1, 0)
        
        # create a mask for this agent
        mask = np.zeros_like(depth_array, dtype=bool)
        mask[int(ymin[agent_idx]):int(ymax[agent_idx]), int(xmin[agent_idx]):int(xmax[agent_idx])] = True
    
        # Find the indices of cells with value 1 for this agent
        agent_multiplier = multiplier[mask]
        row_indices, col_indices = np.where(agent_multiplier == 1)
    
        # Convert indices to projected coordinates
        projected_x, projected_y = pixel_to_geo(arr_type,
            depth_rast_transform, 
            row_indices, 
            col_indices
        )
    
        # Calculate the difference vectors
        diff_x = projected_x - x[agent_idx]
        diff_y = projected_y - y[agent_idx]
        diff_vectors = np.stack((diff_x, diff_y), axis=-1)
    
        # Calculate the magnitude of each vector
        magnitudes = np.linalg.norm(diff_vectors, axis=1)
    
        # Avoid division by zero
        magnitudes[magnitudes == 0] = np.finfo(float).eps
    
        # Normalize each vector to get the unit direction vectors
        direction_vectors = diff_vectors / magnitudes[:, np.newaxis]
    
        # Calculate repulsive force in X and Y directions for this agent
        x_force = (weight * direction_vectors[:, 0]) / magnitudes
        y_force = (weight * direction_vectors[:, 1]) / magnitudes
    
        # Sum the forces for this agent
        total_x_force = np.nansum(x_force)
        total_y_force = np.nansum(y_force)
    
        return total_x_force, total_y_force

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


        buff = 2.  # 2 meters
    
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
    
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.array_module, self.depth_rast_transform)
    
        # calculate array slice bounds for each agent
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
    
        # Load the entire depth dataset into memory (if feasible)
        depth_array = self.hdf5['environment/depth'][:]
    
        # Initialize an array to hold the repulsive forces for each agent
        repulsive_forces = np.zeros((self.num_agents, 2), dtype=float)
    
        repulsive_forces = np.vectorize(self.calculate_shallow_repulsive_force, excluded = [8,9,10,11])
        
        repulsive_forces_array = repulsive_forces(np.arange(0,1,len(self.num_agents)),
                                                  self.X,
                                                  self.Y,
                                                  xmin,
                                                  xmax,
                                                  ymin,
                                                  ymax,
                                                  self.body_depth,
                                                  weight,
                                                  depth_array,
                                                  self.array_module,
                                                  self.depth_rast_transform)

        return repulsive_forces_array

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

        self.wave_drag = self.array_module.where(body_depths >=3, 1, wave_drag_fun(body_depths))
       
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
        
        # get depth raster
        depth_rast = self.hdf5['environment/depth'][:]
        
        # identify buffer
        buff = 2.  # 2 meters
        
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
    
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.array_module, self.depth_rast_transform)
    
        # calculate array slice bounds for each agent
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
        
        # Initialize an array to hold the direction vectors for each agent
        direction_vectors = np.zeros((len(x), 2), dtype=float)
        
        # Iterate over each agent to calculate direction vectors
        for i in range(len(x)):
            # Create a mask for the buffer area around the agent
            mask = np.zeros_like(depth_rast, dtype=bool)
            mask[int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])] = True
            
            # Extract the buffered depths for this agent
            buffered_depths = np.where(mask, depth_rast, np.nan)
            
            # Find the cell with the depth closest to the agent's optimal depth
            optimal_depth_diff = np.abs(buffered_depths - self.opt_wat_depth[i])
            idx_min_diff = np.nanargmin(optimal_depth_diff)
            row_idx, col_idx = np.unravel_index(idx_min_diff, buffered_depths.shape)
            
            # Convert indices to projected coordinates
            projected_x, projected_y = pixel_to_geo(
                self.depth_rast_transform, 
                row_idx, 
                col_idx
            )
            
            # Calculate the direction vector to the optimal depth cell
            diff_x = projected_x - x[i]
            diff_y = projected_y - y[i]
            
            # Normalize the direction vector
            magnitude = np.sqrt(diff_x**2 + diff_y**2)
            if magnitude > 0:
                direction_vectors[i] = [diff_x / magnitude, diff_y / magnitude]
            else:
                direction_vectors[i] = [0, 0]  # No movement if the agent is already at the optimal depth
        
        # Apply the weight to the direction vectors
        weighted_direction_vectors = direction_vectors * weight
        
        return weighted_direction_vectors

    @np.vectorize
    def school_attraction(agent_idx, x, y, weight, agents_within_buffers_dict, x_arr, y_arr):
        """
        Calculate the attractive force towards the centroid of the school for a given agent.
    
        This function computes the centroid of the school by averaging the positions of all neighboring agents within a certain buffer and the agent itself. It then calculates an attractive force that pulls the agent towards this centroid. If the agent is isolated and there are no neighbors, the function returns a zero vector.
    
        Parameters:
        - agent_idx (int): The index of the current agent.
        - x (float): The x-coordinate of the current agent.
        - y (float): The y-coordinate of the current agent.
        - weight (float): The weighting factor to scale the attractive force.
        - agents_within_buffers_dict (dict): A dictionary where the key is the current agent index and the value is a list of indices of agents within the buffer.
        - x_arr (np.ndarray): An array of x-coordinates for all agents.
        - y_arr (np.ndarray): An array of y-coordinates for all agents.
    
        Returns:
        - np.ndarray: A 2-element array representing the attractive force vector towards the school centroid.
    
        Notes:
        - The function uses a vectorized approach for efficient computation across multiple agents.
        - If the agent is alone (i.e., no other agents in the buffer), the function returns [0, 0] to indicate no attraction.
        - The function handles division by zero and other exceptions by returning a zero vector, ensuring stability in the simulation.
        """
        # Get the indices of neighboring agents
        neighbors = agents_within_buffers_dict.get(agent_idx, [])
    
        # Get the positions of neighboring agents
        xs = x_arr[neighbors]
        ys = y_arr[neighbors]
    
        # Include the current agent's position in the centroid calculation
        xs = np.append(xs, x)
        ys = np.append(ys, y)
    
        # Calculate the centroid of the school
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
    
        # Calculate the Euclidean distance from the agent to the centroid
        dist = np.sqrt((x_mean - x)**2 + (y_mean - y)**2)
    
        # Calculate the vector pointing from the agent to the centroid
        v = np.array([x_mean - x, y_mean - y])
        
        try:    
            # Normalize the vector to get the unit direction vector
            v_hat = v / (np.linalg.norm(v) + 1e-9)  # Add a small epsilon to avoid division by zero
    
            # Calculate the attractive force towards the centroid, scaled by weight and the inverse square of the distance
            school_cue = (weight * v_hat) / (dist**2)  # Add a small epsilon to avoid division by zero
        
        except (ZeroDivisionError, ValueError):
            # Return a zero vector if there's an error or no other agents
            school_cue = np.array([0, 0])
    
        return school_cue

    def school_cue(self, weight, agents_within_buffers_dict):
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
        school = np.vectorize(self.school_attraction, excluded = [4,5,6])
        
        school_cue_array = school(np.arange(0,self.num_agents + 1),
                                  self.X,
                                  self.Y,
                                  weight, 
                                  agents_within_buffers_dict,
                                  self.X,
                                  self.Y)
        
        return school_cue_array
    
    @np.vectorize
    def collision_repulsion(self,agent_idx,x,y,weight,closest_agent_dict,x_arr,y_arr):
        """
        Computes a repulsive force vector for an agent to avoid collisions, based on
        the position of the nearest neighboring agent. This function is vectorized to
        efficiently handle multiple agents' calculations simultaneously.
    
        Parameters
        ----------
        agent_idx : int
            The index of the agent for which to calculate the repulsive force.
        x : float
            The x-coordinate of the agent's current position.
        y : float
            The y-coordinate of the agent's current position.
        weight : float
            The weighting factor that scales the magnitude of the repulsive force.
        closest_agent_dict : dict
            A dictionary where keys are agent indices and values are indices of the
            closest agent to them.
        x_arr : ndarray
            An array containing the x-coordinates for all agents in the simulation.
        y_arr : ndarray
            An array containing the y-coordinates for all agents in the simulation.
    
        Returns
        -------
        collision_cue : ndarray
            A 2D vector representing the repulsive force exerted on the agent by the
            closest neighbor. The force is directed away from the neighbor and has a
            magnitude inversely proportional to the square of the distance between
            them, scaled by the given weight.
    
        Notes
        -----
        The function calculates the Euclidean distance between the current agent and
        its closest neighbor to determine the magnitude of the repulsion. A small
        epsilon value is added during the normalization step to prevent division by
        zero, ensuring stability in the force calculation.
    
        Example
        -------
        # Assuming agent positions and closest neighbors are known:
        repulsive_forces = collision_repulsion(
            agent_indices,
            agents_x_positions,
            agents_y_positions,
            repulsion_weight,
            closest_agents_dictionary,
            agents_x_positions,
            agents_y_positions
        )
        """        
        # get closest agent to the current
        closest = closest_agent_dict[agent_idx]
        
        # get position of closest agent
        x1 = x_arr[closest]
        y1 = y_arr[closest]
        
        # calculate Euclidean distance 
        dist = np.sqrt((x1 - x)**2 + (y1 - y)**2)
        
        # calculate vector pointing from neighbor to self
        v = np.array([x - x1, y - y1])
        
        # calculate unit vector
        v_hat = v / (np.linalg.norm(v) + 1e-9)  # Add a small epsilon to avoid division by zero

        # calculate collision cue
        collision_cue = (weight * v_hat) / (dist**2)
        
        return collision_cue
    
    def collision_cue(self, weight, closest_agent_dict):
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
        # Function implementation remains unchanged

        collision = np.vectorize(self.collision_repulsion, excluded = [4,5,6])
        
        collision_cue_array = collision(np.arange(0,self.num_agents + 1),
                                        self.X,
                                        self.Y,
                                        weight,
                                        closest_agent_dict,
                                        self.X,
                                        self.Y)
        
        return collision_cue_array


             
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
