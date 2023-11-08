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
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")


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
        
        # initialize agent properties and internal states
        self.sim_sex()
        self.sim_length()
        self.sim_weight()
        self.sim_body_depth()
        recover = pd.read_csv(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent\data\recovery.csv")
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds, recover.Recovery, extrapolate = True,)
        del recover
        self.swim_behav = self.arr.repeat(1, num_agents)               # 1 = migratory , 2 = refugia, 3 = station holding
        self.swim_mode = self.arr.repeat('sustained', num_agents)      # 1 = sustained, 2 = prolonged, 3 = sprint
        self.battery = self.arr.repeat(1.0, num_agents)
        self.recover_stopwatch = self.arr.repeat(0.0, num_agents)
        self.ttfr = self.arr.repeat(0.0, num_agents)
        self.time_out_of_water = self.arr.repeat(0.0, num_agents)
        
        self.X = self.arr.random.uniform(starting_box[0], starting_box[1],num_agents)
        self.Y = self.arr.random.uniform(starting_box[2], starting_box[3],num_agents)
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

        # initialize mental map
        self.initialize_mental_map()
        
        # initialize heading
        self.initial_heading()
        
        # initialize swim speed
        self.initial_swim_speed()    

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
            self.sex = self.arr.random.choice([0,1], size = self.num_agents, p = [0.503,0.497])
            
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
        agent_data.create_dataset("Y", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("Z", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("prev_X", (self.num_timesteps, self.num_agents), dtype='f4')
        agent_data.create_dataset("prev_Y", (self.num_timesteps, self.num_agents), dtype='f4')            
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
        
        self.hdf5 = h5py.File(self.db, 'w')
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
        self.hdf5.flush()

        # # Periodically flush data to ensure it's written to disk
        # if timestep % 100 == 0:  # Adjust this value based on your needs
            

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
        self.map = self.arr.zeros((self.num_agents, self.height, self.width))
               
        # Create groups for organization (optional)
        mem_data = self.hdf5.create_group("memory")
        
        # create a memory map array
        mem_data.create_dataset('maps', (self.num_agents, self.height, self.width), dtype = 'f4')
        
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
        rows, cols = geo_to_pixel(self.X, self.Y, self.arr, transform)

        # Use the already open HDF5 file object to read the specified raster dataset
        raster_dataset = self.hdf5['environment/%s'%(raster_name)][:]  # Adjust the path as needed
        # Sample the raster values using the row, col indices
        # Ensure that the indices are within the bounds of the raster data
        rows = np.clip(rows, 0, raster_dataset.shape[0] - 1)
        cols = np.clip(cols, 0, raster_dataset.shape[1] - 1)
        values = raster_dataset[rows, cols]
        #self.hdf5['environment/%s'%(raster_name)] = raster_dataset
        self.hdf5.flush()
        
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
    
        # Construct an index array for advanced indexing
        agent_indices = np.arange(self.num_agents)
        indices = (agent_indices, rows, cols)
    
        # Update the mental map for all agents in the HDF5 dataset at once
        mental_map_dataset = self.hdf5['memory/maps'][:]
            
        # Use advanced indexing to update the mental map
        # Note: This assumes that the HDF5 dataset supports numpy-style advanced indexing
        mental_map_dataset[indices] = current_timestep
        
        self.hdf5['memory/maps'][:, :] = mental_map_dataset
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
        buff = 4
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
        
        # Create slice objects for indexing
        col_slice = slice(xmin, xmax)
        row_slice = slice(ymin, ymax)
        
        # Construct the indices tuple for advanced indexing
        indices = (agent_indices, np.floor(row_slice), np.floor(col_slice))
        
        # Access the mental map dataset from the HDF5 file
        mental_map_dataset = self.hdf5['memory/maps'][:]
        
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
        self.agents_within_buffers_dict = agents_within_buffers_dict
        self.closest_agent_dict = closest_agent_dict

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
        
    def velocity_slice(self, agent_idx, x, y, columns, rows, buffer, excluded):
        #excluded = {0:arr_type,1:transform,2:velocity}
        weight = excluded[0]
        arr_type = excluded[1]
        transform = excluded[2]
        velocity = excluded[3]
        print ('analyzing agent %s'%(agent_idx))
        # calculate array slice bounds
        xmin = max(0, columns - buffer)
        xmax = min(velocity.shape[1], columns + buffer + 1)  # +1 because slicing is exclusive on the upper bound
        ymin = max(0, rows - buffer)
        ymax = min(velocity.shape[0], rows + buffer + 1)  # +1 for the same reason
        
        # Retrieve the slice of the velocity array for this agent
        vel_slice = velocity[ymin:ymax, xmin:xmax]
        
        # Find the indices of the minimum velocity within the slice
        min_idx = np.unravel_index(np.argmin(vel_slice, axis=None), vel_slice.shape)
        
        # Convert indices to projected coordinates
        min_x, min_y = pixel_to_geo(arr_type,
            transform, 
            min_idx[0] + ymin, 
            min_idx[1] + xmin
        )
        
        # Calculate the direction vector to the minimum velocity cell
        diff_x = min_x - x
        diff_y = min_y - y
        
        # Normalize the direction vector
        magnitude = np.sqrt(diff_x**2 + diff_y**2)
        if magnitude > 0:
            velocity_min = [diff_x / magnitude, diff_y / magnitude]
            # Scale the velocity cues by the weight and normalize by the square of 5 body lengths in meters
            velocity_min *= np.array(weight / ((5 * magnitude / 1000) ** 2))
        else:
            velocity_min = np.array([0, 0]) # No movement if the minimum velocity is at the agent's position
            
        return velocity_min

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
        buff = np.where(self.swim_mode == 2, 15 * length_numpy, 5 * length_numpy)
        
        # get the x, y position of the agent 
        x, y = (self.X, self.Y)
        
        # find the row and column in the direction raster
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
        
        # Access the velocity dataset from the HDF5 file
        velocity = self.hdf5['environment/vel_mag'][:]
        
        # Initialize an array to hold the velocity cues for each agent
        velocity_min = np.zeros((self.num_agents, 2), dtype=float)
        
        for i in range(self.num_agents):
            # Define the slice bounds for the current agent
            xmin = int(cols[i] - buff[i])
            xmax = int(cols[i] + buff[i] + 1)
            ymin = int(rows[i] - buff[i])
            ymax = int(rows[i] + buff[i] + 1)
            
            # Retrieve the slice of the velocity array for the current agent
            vel_slice = self.hdf5['environment/vel_mag'][:][ymin:ymax, xmin:xmax]
            
            # Find the index of the minimum velocity within the slice
            min_idx = np.unravel_index(np.argmin(vel_slice), vel_slice.shape)
            
            # Convert the index back to geographical coordinates
            min_x, min_y = pixel_to_geo(self.arr, self.vel_mag_rast_transform, min_idx[0] + ymin, min_idx[1] + xmin)
            
            velocity_min[i] = min_x, min_y
        
        diff = velocity_min - np.vstack((self.X, self.Y)).T
        squared = diff**2
        dist = np.sqrt(squared)

        repulsive = (weight * diff/dist) / dist
        return repulsive

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
        v_hat = np.array([np.cos(vel_dir), np.sin(vel_dir)])
        
        # Calculate the rheotactic cue
        rheotaxis = (weight * v_hat)/((2 * length_numpy/1000.)**2)
        
        return rheotaxis
    
    #create a function that returns a total force vector in x and y for each agent
    #@np.vectorize
    def calculate_shallow_repulsive_force(self, agent_idx, x, y, xmin, xmax, ymin, ymax, body_depth, excluded):
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
        weight = excluded[0]
        depth_array = excluded[1] 
        arr_type = excluded[2] 
        depth_rast_transform = excluded[3] # Calculate max depth - body depth in cm - make sure we divide by 100.
        
        #
        min_depth = (body_depth * 1.1) / 100.# Use advanced indexing to create a boolean mask for the slices
        
        # calculate a force multiplier
        multiplier = np.where(depth_array < min_depth, 1, 0)
        
        # create a mask for this agent
        mask = np.zeros_like(depth_array, dtype=bool)
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = True
    
        # Find the indices of cells with value 1 for this agent
        agent_multiplier = multiplier * mask
        row_indices, col_indices = np.where(agent_multiplier == 1)
        
        if len(row_indices) > 0:
    
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
        else:
            return np.array([0]), np.array([0])

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
        rows, cols = geo_to_pixel(x, y, self.arr, self.depth_rast_transform)
    
        # calculate array slice bounds for each agent
        xmin = cols - buff
        xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
        ymin = rows - buff
        ymax = rows + buff + 1  # +1 for the same reason
    
        # Load the entire depth dataset into memory (if feasible)
        depth_array = self.hdf5['environment/depth'][:]
    
        # Initialize an array to hold the repulsive forces for each agent
        repulsive_forces = np.zeros((self.num_agents, 2), dtype=float)
    
        repulsive_forces = np.vectorize(self.calculate_shallow_repulsive_force, excluded = [8])
        
        excluded = (weight, depth_array, self.arr, self.depth_rast_transform)
        
        repulsive_forces_array = repulsive_forces(np.arange(0,1,self.num_agents + 1),
                                                  self.X,
                                                  self.Y,
                                                  xmin,
                                                  xmax,
                                                  ymin,
                                                  ymax,
                                                  self.body_depth,
                                                  excluded)

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
        
        # get depth raster
        depth_rast = self.hdf5['environment/depth'][:]
        
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
        
        # Initialize an array to hold the direction vectors for each agent
        direction_vectors = np.zeros((len(x), 2), dtype=float)
        
        #TODO vectorize this - why over agents?
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
            projected_x, projected_y = pixel_to_geo(self.arr,
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
        school = np.vectorize(self.school_attraction, excluded = [4,5,6])
        
        school_cue_array = school(np.arange(0,self.num_agents + 1),
                                  self.X,
                                  self.Y,
                                  weight, 
                                  self.agents_within_buffers_dict,
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
        # Function implementation remains unchanged

        collision = np.vectorize(self.collision_repulsion, excluded = [4,5,6])
        
        collision_cue_array = collision(np.arange(0,self.num_agents + 1),
                                        self.X,
                                        self.Y,
                                        weight,
                                        self.closest_agent_dict,
                                        self.X,
                                        self.Y)
        
        return collision_cue_array

    @np.vectorize
    def f4ck_allocate(self, swim_behav, order_dict, cue_dict):
        """
        Allocates 'f4cks' (a metaphorical currency representing the agent's effort or
        attention) to generate a steering vector based on the agent's swimming behavior
        and sensory cues. This function is inspired by Reynolds' 1987 boids model, where
        agents allocate limited resources to various steering behaviors.
    
        Parameters
        ----------
        agent_idx : int
            The index of the agent for which to calculate the heading.
        swim_behav : int
            The swimming behavior of the agent, where 1 represents active migration,
            2 represents seeking refugia, and other values represent station holding.
        order_dict : dict
            A dictionary of ordered steering vectors from various cues, indexed from 0 to 6.
        cue_dict : dict
            A dictionary containing sensory cue vectors, with keys like 'shallow',
            'collision', and 'low_speed'.
    
        Returns
        -------
        heading : float
            The preferred heading for the agent in radians, calculated as the arctangent
            of the y and x components of the resulting steering vector.
    
        Notes
        -----
        - The function uses a vectorized approach to apply the allocation logic to each
          agent based on its behavior.
        - For active migration (swim_behav == 1), the function iteratively adds cue vectors
          from `order_dict` until a limit of 'f4cks' (7500) is reached, ensuring that the
          sum of the norms of the cue vectors does not exceed this limit.
        - For seeking refugia (swim_behav == 2), the heading vector is a sum of 'shallow',
          'collision', and 'low_speed' cue vectors from `cue_dict`.
        - For station holding (all other `swim_behav` values), the heading is determined
          solely by the 'rheotaxis' cue from `cue_dict`.
        - The metaphorical 'f4cks' represent a limit on the amount of effort the agent can
          expend on steering, acting as a constraint on the cumulative influence of the cues.
    
        Example
        -------
        # Assuming an instance of the class is created and initialized as `agent_model`
        # and the dictionaries `order_dict` and `cue_dict` are prepared:
        agent_heading = agent_model.f4ck_allocate(agent_idx=5,
                                                  swim_behav=1,
                                                  order_dict=steering_orders,
                                                  cue_dict=sensory_cues)
        """

        # create array to store steering vector
        head_vec = np.array([0, 0])
        
        # of fish is currently actively migrating
        if swim_behav == 1:
            f4cks = 0
            ''' this will fail if the amount of f4cks is larger than could be
            generated by rheotaxis, low speed, and wave drag cues - make sure the
            f4ck limit is less than the sum of rheotaxis, low speed, and wave drag'''
            while f4cks < 7500:
                for i in np.arange(0,7,1):
                    head_vec = head_vec + order_dict[i]
                    f4cks = f4cks + np.linalg.norm(order_dict[i])

        # else if fish is seeking refugia
        elif swim_behav == 2:
            # create a heading vector - based on input from sensory cues
            head_vec = cue_dict['shallow'] + cue_dict['collision'] + cue_dict['low_speed']

        # otherwise we are station holding
        else:
            # create a heading vector - based on input from sensory cues
            head_vec = cue_dict['rheotaxis']                    
    
        # convert into preferred heading for timestep
        heading = np.arctan2(head_vec[1],head_vec[0]) 

        return heading          
            
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
        rheotaxis = self.rheo_cue(10000)
        shallow = self.shallow_cue(15000)
        wave_drag = self.wave_drag_cue( 8000)
        low_speed = self.vel_cue( 12000)
        avoid = self.already_been_here(8000, t)
        school = self.school_cue(8000)
        collision = self.collision_cue(3000)

        # create dictionary that has order of behavioral cues and their norm
        order_dict = {0:shallow,
                      1:collision,
                      2:avoid,
                      3:school,
                      4:rheotaxis,
                      5:low_speed,
                      6:wave_drag}     
        
        # create dictionary that holds all steering cues
        cue_dict = {'rheotaxis':rheotaxis,
                    'shallow':shallow,
                    'wave_drag':wave_drag,
                    'low_speed':low_speed,
                    'avoid':avoid,
                    'school':school,
                    'collision':collision}
        
        # the agent has only so many f4cks to give - vectorize
        only_so_many = np.vectorize(self.f4ck_allocate,excluded = [1,2])
        
        # calculate heading for all agents over the next time step
        self.heading = only_so_many(self.swim_behav,order_dict,cue_dict)
        
    def thrust_fun(self):
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
        ideal_vel_vec = self.arr.stack((self.ideal_sog * self.arr.cos(self.heading),
                                            self.ideal_sog * self.arr.sin(self.heading)), axis=-1)
        ideal_swim_speed = self.arr.linalg.norm(ideal_vel_vec - water_vel, axis=-1)
        swim_speed_cms = ideal_swim_speed * 100.
    
        # Data for interpolation
        length_dat = self.arr.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
        speed_dat = self.arr.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
        amp_dat = self.arr.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
        wave_dat = self.arr.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
        edge_dat = self.arr.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])
    
        # Interpolation
        A = self.arr.interp(length_cm, length_dat, amp_dat)
        V = self.arr.interp(swim_speed_cms, speed_dat, wave_dat)
        B = self.arr.interp(length_cm, length_dat, edge_dat)
    
        # Calculate thrust
        m = (self.arr.pi * rho * B**2) / 4.
        W = (self.Hz * A * self.arr.pi) / 1.414
        w = W * (1 - swim_speed_cms / V)
    
        # Thrust calculation
        thrust_erg_s = m * W * w * swim_speed_cms - (m * w**2 * swim_speed_cms) / (2. * self.arr.cos(self.arr.radians(theta)))
        thrust_Nm = thrust_erg_s / 10000000.
        thrust_N = thrust_Nm / (self.length / 1000.)
    
        # Convert thrust to vector
        thrust = self.arr.stack((thrust_N * self.arr.cos(self.heading),
                                     thrust_N * self.arr.sin(self.heading)), axis=-1)
    
        self.thrust = thrust
        
    def frequency(self):
        """
        Calculates the tailbeat frequency for a collection of agents based on the 
        balance of propulsive forces and drag, following Lighthill's (1970) 
        elongated-body theory of fish propulsion.
    
        This method applies piecewise linear interpolation to estimate parameters 
        such as amplitude, propulsive wave velocity, and trailing edge span as 
        functions of body length and swimming speed. It is designed to work with 
        array operations, allowing for simultaneous calculations across multiple 
        agents.
    
        The method assumes a freshwater environment with a density of 1.0 kg/m^3 
        and uses the agents' lengths, velocities, and drag forces to compute the 
        tailbeat frequency for each agent.
    
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
        drag : array_like
            The drag forces experienced by the agents in N.
        swim_behav : array_like
            An array indicating the swimming behavior of each agent, where a 
            specific value (e.g., 3) indicates 'station holding'.
        
        Returns
        -------
        Hzs : ndarray
            An array of tailbeat frequencies for each agent in Hz.
        
        Notes
        -----
        The function assumes that the input arrays are of equal length, with each 
        index corresponding to a different agent. The tailbeat frequency calculation 
        is vectorized to handle multiple agents simultaneously.
    
        The piecewise linear interpolation is used for the amplitude (A), propulsive 
        wave velocity (V), and trailing edge span (B) based on the provided data points. 
        This approach simplifies the computation and is suitable for scenarios 
        where a small amount of error is acceptable.
    
        The function also accounts for different swimming behaviors, assigning a 
        default frequency for agents that are 'station holding'.
    
        Examples
        --------
        Assuming `agents` is an instance of the simulation class with all necessary 
        properties set as arrays:
        
        >>> Hzs = agents.frequency()
        
        The `Hzs` array will contain the tailbeat frequencies for each agent after 
        the function call.
        """
        # ... function implementation ...

        # Constants
        rho = 1.0  # density of freshwater
        theta = 32.  # theta for cos(theta) = 0.85
    
        # Convert lengths from meters to centimeters
        lengths_cm = self.length * 100
    
        # Calculate swim speed in cm/s
        water_velocities = self.arr.stack((self.x_vel, self.y_vel), axis=-1)
        fish_velocities = self.arr.stack((self.ideal_sog * self.arr.cos(self.heading),
                                              self.ideal_sog * self.arr.sin(self.heading)), axis=-1)
        swim_speeds_cms = self.arr.linalg.norm(fish_velocities - water_velocities, axis=-1) * 100

        # sockeye parameters (Webb 1975, Table 20) units in CM!!! 
        length_dat = self.arr.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
        speed_dat = self.arr.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
        amp_dat = self.arr.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
        wave_dat = self.arr.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
        edge_dat = self.arr.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
    
        # Interpolate A, V, B using piecewise linear functions based on provided data
        # Replace with actual piecewise linear interpolation based on your data
        A = self.arr.interp(lengths_cm, length_dat, amp_dat, self.arr)
        V = self.arr.interp(swim_speeds_cms, speed_dat, wave_dat, self.arr)
        B = self.arr.interp(lengths_cm, length_dat, edge_dat, self.arr)
    
        # Convert drag to erg/s
        drags_erg_s = self.drag * self.length * 10000000
    
        # Solve for Hz
        Hzs = self.arr.where(self.swim_behav == 3,
                             1.0,
                             self.arr.sqrt(drags_erg_s * V**2 * \
                                           self.arr.cos(self.arr.radians(theta)) \
                                               /(A**2 * B**2 * swim_speeds_cms * \
                                                 self.arr.pi**3 * rho * (swim_speeds_cms - V)\
                                                     * (-0.062518880701972 * swim_speeds_cms \
                                                        - 0.125037761403944 * V * \
                                                            self.arr.cos(self.arr.radians(theta))\
                                                                + 0.062518880701972 * V))))
    
        self.Hz = Hzs
         
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
        def fit_dragcoeffs(reynolds, a, b):
            return self.arr.log(reynolds) * a + b
    
        dragf_popt, _ = curve_fit(fit_dragcoeffs, reynolds_data, drag_data)
    
        # Calculate the drag coefficient for the input Reynolds numbers
        drag_coefficients = self.arr.abs(fit_dragcoeffs(reynolds, *dragf_popt))
    
        return drag_coefficients

    def drag_fun(self):
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
        fish_speeds = np.linalg.norm(fish_velocities, axis=1)
        fish_speeds[fish_speeds == 0.0] = 0.0001
        fish_velocities[fish_speeds == 0.0] = [0.0001, 0.0001]
    
        # Calculate kinematic viscosity and density based on water temperature
        viscosities = self.kin_visc(self.water_temp)
        densities = self.wat_dens(self.water_temp)

        # Calculate Reynolds numbers
        reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
    
        # Calculate surface areas
        surface_areas = self.calc_surface_area((self.length / 1000.) * 100.)
    
        # Calculate drag coefficients
        drag_coeffs = self.drag_coeff(reynolds_numbers)
    
        # Calculate relative velocities and their norms
        relative_velocities = fish_velocities - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=1)**2
    
        # Calculate unit vectors for fish velocities
        unit_fish_velocities = fish_velocities / self.arr.linalg.norm(fish_velocities, axis=1)[:, np.newaxis]
    
        # Calculate drag forces
        drags = -0.5 * (densities * 1000) * (surface_areas / 100**2) * drag_coeffs * relative_speeds_squared[:, self.arr.newaxis] \
            * unit_fish_velocities * self.wave_drag[:, self.arr.newaxis]
    
        self.drag = drags

    def ideal_drag_fun(self):
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
        fish_velocities = np.stack((self.ideal_sog * np.cos(self.heading),
                                    self.ideal_sog * np.sin(self.heading)), axis=-1)
    
        # Calculate ideal swim speeds and adjust based on max sustainable speed
        ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=1)
        mask = (self.swim_behav == 2) | (self.swim_behav == 3)
        mask &= (ideal_swim_speeds > self.max_s_U)
        fish_velocities[mask] *= (self.max_s_U / ideal_swim_speeds[mask])[:, np.newaxis]
    
        # Calculate the maximum practical speed over ground
        self.max_practical_sog = np.where(mask[:, np.newaxis],
                                          fish_velocities + water_velocities,
                                          fish_velocities)
        self.max_practical_sog[np.linalg.norm(self.max_practical_sog, axis=1) == 0.0] = [0.0001, 0.0001]
    
        # Kinematic viscosity and density based on water temperature for each fish
        viscosities = self.kin_visc(self.water_temp)
        densities = self.wat_dens(self.water_temp)
    
        # Reynolds numbers for each fish
        reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
    
        # Surface areas for each fish
        surface_areas = self.calc_surface_area(self.length)
    
        # Drag coefficients for each fish
        drag_coeffs = self.drag_coeff(reynolds_numbers)
    
        # Calculate ideal drag forces
        relative_velocities = self.max_practical_sog - water_velocities
        relative_speeds_squared = np.linalg.norm(relative_velocities, axis=1)**2
        unit_max_practical_sog = self.max_practical_sog / np.linalg.norm(self.max_practical_sog, axis=1)[:, np.newaxis]
    
        # Ideal drag calculation
        ideal_drags = -0.5 * (densities * 1000) * (surface_areas / 100**2) * drag_coeffs \
                      * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog \
                      * self.wave_drag[:, np.newaxis]
    
        self.ideal_drag = ideal_drags
            
    def fatigue(self, t):
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
        dt = 1.0  # Time step duration
    
        # Vector components of water velocity and speed over ground for each fish
        water_velocities = self.arr.stack((self.x_vel, self.y_vel), axis=-1)
        fish_velocities = self.arr.stack((self.sog * self.arr.cos(self.heading),
                                    self.sog * self.arr.sin(self.heading)), axis=-1)
    
        # Calculate swim speeds for each fish
        swim_speeds = self.arr.linalg.norm(fish_velocities - water_velocities, axis=1)
    
        # Calculate distances travelled and update bout odometer and duration
        dist_travelled = self.arr.sqrt((self.prev_X - self.X)**2 + (self.prev_Y - self.Y)**2)
        self.dist_per_bout += dist_travelled
        self.bout_dur += dt
    
        # Initialize time to fatigue (ttf) array
        ttf = self.arr.full_like(swim_speeds, self.arr.nan)
    
        # Calculate ttf for prolonged and sprint swimming modes
        mask_prolonged = (self.max_s_U < swim_speeds) & (swim_speeds <= self.max_p_U)
        mask_sprint = swim_speeds > self.max_p_U
        ttf[mask_prolonged] = 10. ** (self.a_p + swim_speeds[mask_prolonged] * self.b_p) * 60.
        ttf[mask_sprint] = 10. ** (self.a_s + swim_speeds[mask_sprint] * self.b_s) * 60.
    
        # Set swimming modes based on swim speeds
        self.swim_mode = self.arr.where(mask_prolonged, 'prolonged', self.swim_mode)
        self.swim_mode = self.arr.where(mask_sprint, 'sprint', self.swim_mode)
        self.swim_mode = self.arr.where(~(mask_prolonged | mask_sprint), 'sustained', self.swim_mode)
    
        # Calculate recovery at the beginning and end of the time step
        rec0 = self.recovery(self.recover_stopwatch) / 100.
        rec0[rec0 < 0.0] = 0.0
        rec1 = self.recovery(self.recover_stopwatch + dt) / 100.
        rec1[rec1 > 1.0] = 1.0
        per_rec = rec1 - rec0
    
        # Update battery levels for sustained swimming mode
        mask_sustained = self.swim_mode == 'sustained'
        self.battery[mask_sustained] += per_rec[mask_sustained]
        self.battery[self.battery > 1.0] = 1.0
    
        # Update battery levels for non-sustained swimming modes
        mask_non_sustained = ~mask_sustained
        ttf0 = ttf[mask_non_sustained] * self.battery[mask_non_sustained]
        ttf1 = ttf0 - dt
        self.battery[mask_non_sustained] *= ttf1 / ttf0
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
            
    def swim(self, dt):
        """
        Propels each fish forward based on their thrust, drag, and weight.
    
        Parameters:
        dt (float): The time step over which to advance the simulation.
    
        Attributes:
        sog (array): Speed over ground for each fish in m/s.
        heading (array): Heading for each fish in radians.
        thrust (array): Thrust force for each fish in Newtons.
        drag (array): Drag force for each fish in Newtons.
        weight (array): Weight of each fish in kilograms.
        pos (array): Current position of each fish in meters.
        prevPos (array): Previous position of each fish in meters.
    
        Notes:
        - The function assumes that the input attributes are structured as arrays of
          values, with each index across the arrays corresponding to a different fish.
        - The function updates the position and speed over ground (sog) for each fish
          based on the calculated surge and acceleration.
        """
    
        # Calculate fish velocity components for each fish
        fish_vel_0_x = self.sog * np.cos(self.heading)
        fish_vel_0_y = self.sog * np.sin(self.heading)
    
        # Calculate surge for each fish
        surge_x = np.round(self.thrust, 2) + np.round(self.drag, 2)
        surge_y = np.round(self.thrust, 2) + np.round(self.drag, 2)  # Assuming thrust and drag have y components
    
        # Calculate acceleration for each fish
        acc_x = np.round(surge_x / self.weight, 2)
        acc_y = np.round(surge_y / self.weight, 2)
    
        # Calculate the magnitude of acceleration and apply dampening for each fish
        acc_mag = np.sqrt(acc_x**2 + acc_y**2)
        damp = np.where(acc_mag > 0, (-0.067 * np.log(acc_mag) + 0.3718), 0.0000001)
        damp = np.clip(damp, a_min=0, a_max=None)  # Ensure dampening is not negative
    
        # Apply dampening to acceleration
        acc_x *= damp
        acc_y *= damp
    
        # Calculate new velocity at the end of the time step for each fish
        fish_vel_1_x = fish_vel_0_x + acc_x * dt
        fish_vel_1_y = fish_vel_0_y + acc_y * dt
    
        # Update speed over ground for each fish
        self.sog = np.round(np.sqrt(fish_vel_1_x**2 + fish_vel_1_y**2), 6)
    
        # Update positions for each fish
        self.prevPos_x = self.pos_x
        self.prevPos_y = self.pos_y
        self.pos_x += fish_vel_1_x * dt
        self.pos_y += fish_vel_1_y * dt
       
    def jump(self, t, g):
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
        self.time_of_jump = t
    
        # Get jump angle for each fish
        jump_angles = self.arr.random.choice([self.arr.radians(45), self.arr.radians(60)], size=self.ucrit.shape)
    
        # Calculate time airborne for each fish
        time_airborne = (2 * self.ucrit * self.arr.sin(jump_angles)) / g
    
        # Calculate displacement for each fish
        displacement = self.ucrit * time_airborne * self.arr.cos(jump_angles)
    
        # Set speed over ground to ucrit for each fish
        self.sog = self.ucrit
    
        # Calculate new heading angle for each fish based solely on flow direction
        self.heading = self.arr.arctan2(self.y_vel, self.x_vel) - self.arr.radians(180)
    
        # Calculate the new position for each fish
        self.pos_x += displacement * self.arr.cos(self.heading)
        self.pos_y += displacement * self.arr.sin(self.heading)
            
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
        swim_cost = sr_o2_rate + self.wave_drag * (
            self.arr.exp(np.log(sr_o2_rate) + self.swim_speed * (
                (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
            )) - sr_o2_rate
        )
    
        # Update kilocalories burned
        self.kcal += swim_cost
            
    def timestep(self, t, dt):
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
        self.fatigue(t)
        
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
                      (time_since_jump > 180) & (self.battery > 0.4)
        
        # Apply the jump or swim functions based on the condition
        # For each fish that should jump
        self.jump(t=t, mask=should_jump)
        
        # For each fish that should swim
        self.drag_fun(mask=~should_jump)
        self.frequency(mask=~should_jump)
        self.thrust_fun(mask=~should_jump)
        self.swim(dt, t=t, mask=~should_jump)
        
        # Calculate mileage
        self.odometer(t=t)  
        
        # Log the timestep data
        self.timestep_flush(t)

            
    def run(self, model_name, n, dt):
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
                for i in range(n):
                    self.timestep(i,1)

                    # write frame
                    agent_pts.set_data(self.X,
                                       self.Y)
                    writer.grab_frame()

                    print ('Time Step %s complete'%(i))


        # clean up
        writer.finish()
        self.hdf.flush()
        self.hdf.close()
        depth.close()
        t1 = time.time()     
        
        print ('ABM took %s to compile'%(t1-t0))
            
            
            
            
            
            
            
            
            
            
            
            
