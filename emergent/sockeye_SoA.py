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
import h5py
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.mask import mask
from shapely import Point, Polygon
from shapely import affinity
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline, interp1d, CubicSpline
from scipy.optimize import curve_fit
from scipy.constants import g
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from datetime import datetime
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
        
    def geo_to_pixel(self, transform):
        """
        Convert x, y coordinates to row, column indices in the raster grid.

        Parameters:
        - transform: affine transform of the raster

        Returns:
        - rows: array of row indices
        - cols: array of column indices
        """
        cols = (self.x - transform.c) / transform.a
        rows = (self.y - transform.f) / transform.e
        
        cols = self.array_module.round(cols).astype(self.array_module.int32)
        rows = self.array_module.round(rows).astype(self.array_module.int32)

        # If using CuPy, transfer indices to CPU as NumPy arrays for HDF5 operations
        if isinstance(cols, self.array_module.ndarray):
            cols = cols.get()
            rows = rows.get()

        return rows, cols
 
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
        
        return [np.nansum(x_force),np.nansum(y_force)]   




              
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
    def enviro_import(self,data_dir,surface_type):
        '''Function imports existing environmental surfaces into new simulation'''
        if surface_type == 'velocity x':
            self.vel_x_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'velocity y':
            self.vel_y_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'depth':
            self.depth_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'wsel':
            self.wsel_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'elevation':
            self.elev_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'velocity direction':
            self.vel_dir_rast = rasterio.open(data_dir, masked=True)
        elif surface_type == 'velocity magnitude':
            self.vel_mag_rast = rasterio.open(data_dir, masked=True)
                    
    def HECRAS (self,HECRAS_model,resolution):
        '''Function reads 2D HECRAS model and creates environmental surfaces 

        Parameters
        ----------
        HECRAS_model : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        '''
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
            
        self.elev_rast = rasterio.open(os.path.join(self.model_dir,'elev.tif'))

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
            
        self.wsel_rast = rasterio.open(os.path.join(self.model_dir,'wsel.tif'))
            
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
            
        self.depth_rast = rasterio.open(os.path.join(self.model_dir,'depth.tif'))

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
            
        self.vel_dir_rast = rasterio.open(os.path.join(self.model_dir,'vel_dir.tif'))
            
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
            
        self.vel_mag_rast = rasterio.open(os.path.join(self.model_dir,'vel_mag.tif'))
        
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
            
        self.vel_x_rast = rasterio.open(os.path.join(self.model_dir,'vel_x.tif'))
            
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
            
        self.vel_y_rast = rasterio.open(os.path.join(self.model_dir,'vel_y.tif'))
        
    def create_agents(self, numb_agnts, model_dir, starting_box, crs, basin, water_temp):
        '''method that creates a set of agents for simulation'''
        
        agents_list = []
        
        for i in np.arange(0,numb_agnts,1):
            # create a fish
            fishy = fish(i,model_dir, starting_box, crs, basin, water_temp)
            
            # set initial parameters
            fishy.morphological_parameters()
            fishy.initial_heading(self.vel_dir_rast)
            fishy.initial_swim_speed()
            fishy.mental_map(self.depth_rast)

            # add it to the output list
            agents_list.append(fishy)
            
            # create a dataframe for this agent 
            df = pd.DataFrame.from_dict(data = {'id':[i],
                                                'loc':[Point(fishy.pos)],
                                                'E':[np.round(fishy.pos[0],4)],
                                                'N':[np.round(fishy.pos[1],4)],
                                                'vel':[fishy.sog],
                                                'dir':[fishy.heading]},
                                        orient = 'columns')
            
            gdf = gpd.GeoDataFrame(df, geometry='loc', crs= self.crs) 
            self.agents = pd.concat([self.agents,gdf])
            
        return agents_list
    
    def ts_log(self, ts):
        '''method that writes to log at end of timestep and updates agents 
        geodataframe for next time step'''
    
        # first step writes current positions to the project database
        self.agents['ts'] = np.repeat(ts,len(self.agents))
        timestep = pd.DataFrame()
        timestep['ts'] = self.agents['ts']
        timestep['id'] = self.agents['id'].astype(str)
        timestep['E'] = np.round(self.agents['E'],4)
        timestep['N'] = np.round(self.agents['N'],4)
        timestep['loc'] = self.agents['loc'].astype(str)
        timestep['vel'] = self.agents['vel']
        timestep['dir'] = self.agents['dir']
        timestep.to_hdf(self.hdf,'TS',mode = 'a',format = 'table', append = True)
        self.hdf.flush()        
        
        # second step creates a new agents geo dataframe
        self.agents = gpd.GeoDataFrame(columns=['id', 'loc', 'vel', 'dir','E','N'],
                                       geometry='loc',
                                       crs= self.crs) 

        agent_dict = {'id':[],'loc':[],'E':[],'N':[],'vel':[],'dir':[]}
        
        for i in self.agents_list: 
            agent_dict['id'].append(i.ID)
            agent_dict['loc'].append(Point(i.pos))
            agent_dict['E'].append(np.round(i.pos[0],4))
            agent_dict['N'].append(np.round(i.pos[1],4))
            agent_dict['vel'].append(i.sog)
            agent_dict['dir'].append(i.heading)
            
        # create an agents dataframe 
        df = pd.DataFrame.from_dict(agent_dict, orient = 'columns')
        
        gdf = gpd.GeoDataFrame(df, geometry='loc', crs= self.crs) 
        self.agents = pd.concat([self.agents,gdf])
    
    def run(self, model_name, agents, n, dt):
        
        # define metadata for movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title= model_name, artist='Matplotlib',
                        comment='emergent model run %s'%(datetime.now()))
        writer = FFMpegWriter(fps=30, metadata=metadata)
        
        self.agents_list = agents
        
        # initialize 
        fig, ax = plt.subplots(figsize = (10,5))
        
        background = ax.imshow(self.depth_rast.read(1), 
                               origin = 'upper',
                               extent = [self.depth_rast.bounds[0],
                                          self.depth_rast.bounds[2],
                                          self.depth_rast.bounds[1],
                                          self.depth_rast.bounds[3]])
        
        agent_pts, = plt.plot([], [], marker = 'o', ms = 1, ls = '', color = 'red')
        
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        
        # Update the frames for the movie
        with writer.saving(fig, os.path.join(self.model_dir,'%s.mp4'%(model_name)), 300):
            for i in range(n):
                for agent in agents:
                    # check the environment 
                    agent.mental_map(depth_rast = self.depth_rast, t = i)
                    agent.environment(depth = self.depth_rast,
                                      x_vel = self.vel_x_rast,
                                      y_vel = self.vel_y_rast,
                                      agents_df = self.agents)
                    
                    # take stock of internal states
                    agent.find_z()
                    agent.wave_drag_multiplier()
                    agent.fatigue(t = i)
                    
                    # arbitrate behavioral cues
                    agent.arbitrate(vel_mag_rast = self.vel_mag_rast, 
                                    depth_rast = self.depth_rast,
                                    vel_dir_rast = self.vel_dir_rast,
                                    t = i)
                    
                    '''if the ratio to ideal speed over ground to water velocity 
                    is less than 0.05 
                    and the agent is travelling against flow
                    and it's been more than 60 seconds since its last jump
                    and there is more than 40% remaining in battery'''
                    if agent.ideal_sog / np.linalg.norm([agent.x_vel,agent.y_vel]) < 0.05 and\
                        np.sign(agent.heading) != np.sign(np.arctan2(agent.y_vel,agent.x_vel)) and\
                            i - agent.time_of_jump > 60 and agent.battery > 0.4:
                                # jump
                                agent.jump(t = i)
                    else:
                        # swim like your life depended on it
                        agent.drag_fun()
                        agent.frequency()
                        agent.thrust_fun()
                        agent.swim(dt, t = i)
                    
                    # calculate mileage
                    agent.odometer(t = i)
                
                # write frame
                agent_pts.set_data(self.agents.geometry.x.values, self.agents.geometry.y.values)
                writer.grab_frame() 
                
                # log data to hdf
                self.ts_log(i)
                
                print ('Time Step %s complete'%(i))
            for agent in agents:
                agent.mental_map_export(self.depth_rast)
        
        # clean up
        writer.finish()
        self.hdf.flush()
        self.hdf.close()
        self.wsel_rast.close()
        self.vel_dir_rast.close()
        self.vel_mag_rast.close()
        self.depth_rast.close()
        self.vel_x_rast.close()
        self.vel_y_rast.close()
        self.elev_rast.close()
        

 
    
