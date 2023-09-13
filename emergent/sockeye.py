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
    
# create a sockeye agent 
class fish():
    ''' Python class object for a sockeye agent. 
    
    Class object contains all of the sockeye's attributes, while the methods
    control how the sockeye agent senses its environment, reacts with changing 
    flow conditions, and interacts with other agents.  These methods, which are 
    in reality simple behavioral rules, will lead to complex, self organizing
    behavior.
    '''
    
    def __init__(self, ID, model_dir, starting_block, water_temp):
        '''initialization function for a sockeye agent.  this function creates
        an agent and parameterizes morphometric parameters from basin specific
        distributions
        
        Units
            length = mm
            weight = kg
            body depth = mm
            velocity = cms SOG
            
        fish are randomly positioned within the starting block, which is passed
        as a tuple (xmin, xmax, ymin, ymax) '''
            
        # initialization methods
        def sex(self, basin):
            '''function simulates a sex for a given basin'''
            
        def length(self, basin, sex):
            '''function simulates a fish length out of the user provided basin and 
            sex of fish'''
            
        def weight(self, basin, sex):
            '''function simulates a fish weight out of the user provided basin and 
            sex of fish'''
            
        def body_depth(self, basin, sex):
            '''function simulates a fish body depth out of the user provided basin and 
            sex of fish'''    
        # initialize morphometric paramters
        self.ID = ID
        self.sex = 'F'
        self.length = 700.                             # mm
        self.weight = 4.3                              # kg
        self.body_depth = 15.                          # cm
        self.too_shallow = self.body_depth /100. / 2. # m
        self.opt_wat_depth = self.body_depth /100 * 3.0 + self.too_shallow
        
        # initialize environmental states
        self.water_temp = water_temp
        self.x_vel = 0.
        self.y_vel = 0.
        self.depth = 0.
        self.z = 0.0
        
        # initialize internal states
        ## TODO KEVIN I CHANGED THIS!! ## CHANGE BACK!!!! I CAN'T MAKE IT WORK RN
        recover = pd.read_csv("../data/recovery.csv")
        
        #recover = pd.read_csv(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent\data\recovery.csv")
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds,recover.Recovery,extrapolate = True,)
        self.swim_behav = 'migratory'# swimming behavior, migratory, refugia, station holding
        self.swim_mode = 'sustained' # swimming mode, prolonged, sprint, or sustained
        self.battery = 1.            # at start of simulation battery is full
        #self.recharge = 1.           # recharge state - initial battery drain has recharge state of 1.
        self.recover_stopwatch = 0.0 # total recovery time    
        self.ttfr = 0.0              # running time to fatigue in seconds
        self.time_out_of_water = 0.0
        del recover
        
        # Time to Fatigue values for Sockeye digitized from Bret 1964
        self.max_s_U = 2.77  # maximum sustained swim speed
        self.max_p_U = 4.43  # maximum prolonged swim speed
        self.a_p = 8.643     # prolonged intercept
        self.b_p = -2.0894   # prolonged slope
        self.a_s = 0.1746    # sprint intercept
        self.b_s = -0.1806   # spring slope
        
        # initialize movement parameters
        self.sog = self.length/1000  # sog = speed over ground - assume fish maintain 1 body length per second
        self.ideal_sog = self.sog
        self.swim_speed = self.length/1000        # set initial swim speed
        self.drag = 0.               # computed theoretical drag
        self.thrust = 0.             # computed theoretical thrust Lighthill 
        self.Hz = 0.                 # tail beats per second
        self.bout_no = 0.            # bout number - new bout whenever fish recovers
        self.dist_per_bout = 0.      # running counter of the distance travelled per bout
        self.bout_dur = 0.           # running bout timer 
        self.ucrit = self.sog * 7    # TODO - what is the ucrit for sockeye?
        self.time_of_jump = 0.0   # time since last jump - can't happen every timestep
        
        # initialize the odometer
        self.kcal = 0.
        
        #position the fish within the starting block
        x = np.random.uniform(starting_block[0],starting_block[1])
        y = np.random.uniform(starting_block[2],starting_block[3])
        self.pos = (x,y)
        self.prevPos = self.pos
        
        # create agent database and write agent parameters 
        self.hdf = pd.HDFStore(os.path.join(model_dir,'%s.h5'%('agent_%s.h5'%(ID))))
        self.hdf['agent'] = pd.DataFrame.from_dict({'ID':[self.ID],
                                                    'sex':[self.sex],
                                                    'length':[self.length],
                                                    'weight':[self.weight],
                                                    'body_depth':[self.body_depth]})
        self.hdf.flush()
        self.model_dir = model_dir
        
        # create an empty map array
        self.map = None
        
    def initial_heading (self,vel_dir):
        '''function that sets the initial heading of a fish, inputs are itself
        and a velocity direction raster.  Use spatial indexing to find direction,
        then we are -180 degrees.'''
        
        # get the x, y position of the agent 
        x, y = (self.pos[0], self.pos[1])
        
        # find the row and column in the direction raster
        row, col = vel_dir.index(x, y)
        
        # flow direction
        flow_dir = vel_dir.read(1)[row, col]
        
        # set direction 
        if flow_dir < 0:
            self.heading = (np.radians(360) + flow_dir) - np.radians(180)
        else:
            self.heading = flow_dir - np.radians(180) 
        
        # set initial max practical speed over ground as well
        self.max_practical_sog = np.array([self.sog * np.cos(self.heading), 
                                           self.sog * np.sin(self.heading)]) #meters/sec
            
    def mental_map (self, depth_rast = None, t = None):
        '''function creates and updates a mental map with the time step when the
        cell was last visited by an agent'''
        
        # if there is no mental map, make one
        if self.map is None:
            
            # create meshgrid of same size and shape as depth_rast
            X = np.zeros(depth_rast.width)
            Y = np.zeros(depth_rast.height)
            X, Y = np.meshgrid(X, Y)
            Z = X * Y
                
            self.map = Z

        # if there is one, figure out where the agent is and write the timestep
        else:
            # get the x, y position of the agent 
            x, y = (self.pos[0], self.pos[1])
            
            # find the row and column in the direction raster
            row, col = depth_rast.index(x, y)

            # write timestep to cell
            if t - 60 < 0:
                self.map[row, col] = t
            elif self.map[row, col] < t - 60.:
                self.map[row, col] = t
             
    def mental_map_export(self, depth_rast):
        '''function exports mental map to model directory'''
            
        with rasterio.open(
              os.path.join(self.model_dir,'agent_%s_mental_map.tif'%(self.ID)),
              mode = 'w', 
              driver = 'GTiff',
              width = depth_rast.width,
              height = depth_rast.height,
              count = 1,
              dtype = np.float32,
              crs = depth_rast.crs,
              transform = depth_rast.transform,
        ) as new_dataset:
            new_dataset.write(self.map,1)        
    
    def already_been_here(self, depth_rast, weight, t):
        '''function that reads mental map and quantifies repulsive force emitted
        from previously visited locations.  The repulsive force does not present 
        itself until 30 seconds after the visit starts and persists for 1 hour.'''

        # get the x, y position of the agent 
        x, y = (self.pos[0], self.pos[1])
        
        # find the row and column in the direction raster
        row, col = depth_rast.index(x, y) 
        
        # create array slice bounds
        buff = 4
        xmin = col - buff
        xmax = col + buff
        ymin = row - buff
        ymax = row + buff
        
        # sclice up the mental map
        mmap = self.map[ymin:ymax+1,xmin:xmax+1]
        
        def force_multiplier(i, t):
            '''function that incorporates time since visit - we only have 
            repulsive force between 1 minute and 1 hour since visiting a cell'''
        
            if i != 0.0:
                t_since = t - i 
            else:
                t_since = 0.0
            
            # calc force
            if 60. < t_since <= 3600.:
                #return  -0.0003 * t_since + 1.0084
                return  1.

            else:
                return 0.
            
        v_force_multiplier = np.vectorize(force_multiplier,excluded = [1])
        
        multiplier = v_force_multiplier(mmap, t)

        # create an array of x and y coordinates of cells
        ys = np.arange(y + buff,y - (buff +1),-1)
        xs = np.arange(x - buff,x + (buff +1) ,1)
        
        # create a meshgrid of coordinates
        repx, repy = np.meshgrid(xs,ys)
        
        # make a function to calculate distance squared from each cell to agent
        def distance (xi, yi, x, y):
            '''function to calculate distance between mental map cell and agent
            xs = mental map x, 
            ys = mental map y
            x = fish x
            y = fisy y'''
            
            dx = xi - x
            dy = yi - y
            return np.linalg.norm([dx,dy])
        
        # vectorize 
        v_distance = np.vectorize(distance,excluded = [2,3])
        
        # calculate distance 
        dist_grid = np.power(v_distance(repx, repy, x, y),2)
        
        # make a functio to calculate direction from each cell to agent
        def direction (xi, yi, x, y):
            '''function that calculates a unit vector from mental map cell to agent'''
            
            v = np.array([x, y]) - np.array([xi, yi])
                
            vhat = v / np.linalg.norm(v)
            
            return np.arctan2(vhat[1],vhat[0])
        
        # vectorize
        v_direction = np.vectorize(direction, excluded = [2,3])
        
        # calculate direction 
        dir_grid = v_direction(repx, repy, x, y)
        
        # calculate repulsive force in X and Y directions 
        x_force = ((weight * np.cos(dir_grid))/ dist_grid) * multiplier
        y_force = ((weight * np.sin(dir_grid))/ dist_grid) * multiplier
        
        return [np.nansum(x_force),np.nansum(y_force)]                                     
   
    def buffer_poly(self):
        
        # create sensory buffer
        l = (self.length) * 2.
                
        # create wedge looking in front of fish 
        theta = np.radians(np.linspace(-120,120,100))
        arc_x = self.pos[0] + l * np.cos(theta)
        arc_y = self.pos[1] + l * np.sin(theta)
        arc_x = np.insert(arc_x,0,self.pos[0])
        arc_y = np.insert(arc_y,0,self.pos[1])
        arc = np.column_stack([arc_x, arc_y])
        arc = Polygon(arc)
        arc_rot = affinity.rotate(arc,np.degrees(self.heading), origin = (self.pos[0],self.pos[1]))    
            
        arc_gdf = gpd.GeoDataFrame(index = [0],
                                   crs = 'EPSG:3473',
                                   geometry = [arc_rot])
        return arc_rot        
    
    def environment(self, depth, x_vel, y_vel, agents_df):
        '''method finds the current depth, x velocity, y velocity, and neighbors'''
        
        # get current position 
        x, y = (self.pos[0],self.pos[1])
        row, col = depth.index(x, y)
        
        # set variables
        self.depth = depth.read(1)[row, col]
        self.x_vel = x_vel.read(1)[row, col]
        self.y_vel = y_vel.read(1)[row, col]
        
        # can't have divide by zero
        if self.x_vel == 0.0 and self.y_vel == 0.0:
            self.x_vel = 0.0001
            self.y_vel = 0.0001
        
        # keep track of the amount of time a fish spends out of water
        if self.depth < self.too_shallow:
            print ('FISH OUT OF WATER OH SHIT')
            self.time_out_of_water = self.time_out_of_water + 1
            
        buffer_poly = self.buffer_poly()
        nearbyfish_ser = agents_df.intersection(buffer_poly)
        nearbyfish_df = agents_df[~nearbyfish_ser.is_empty]
        nearbyfish_df = nearbyfish_df[nearbyfish_df.id != self.ID]
        
        self.neighbors = nearbyfish_df
            
    def find_z(self):
        '''Method resolves agent depth.  
        
        The agent wants to minimize wave drag. 
        
        If the water column depth is less than the minimum wave drag depth of 
        3 body depths, then the agent obtains its minimum depth. Else the agent
        is at the mimum wave drag depth

        Returns
        -------
        None.

        '''
        if self.depth < self.body_depth * 3 / 100.:
            self.z = self.depth + self.too_shallow
        else:
            self.z = self.body_depth * 3 / 100.  
        
    def vel_cue (self, vel_mag_rast, weight):
        '''Function that returns a lowest velocity heading command - 
        the way upstream within this narrow arc in front of me - looking for 
        lowest velocity'''       

        if self.swim_mode == 'refugia':        
            # create sensory buffer
            l = (self.length/1000.) * 15.
            
            # create wedge looking in front of fish 
            theta = np.radians(np.linspace(-120,120,100))
            
        else:
            # create sensory buffer
            l = (self.length/1000.) * 4.
            
            # create wedge looking in front of fish 
            theta = np.radians(np.linspace(-15,15,100))
            
        arc_x = self.pos[0] + l * np.cos(theta)
        arc_y = self.pos[1] + l * np.sin(theta)
        arc_x = np.insert(arc_x,0,self.pos[0])
        arc_y = np.insert(arc_y,0,self.pos[1])
        arc = np.column_stack([arc_x, arc_y])
        arc = Polygon(arc)
        arc_rot = affinity.rotate(arc,np.degrees(self.heading), origin = (self.pos[0],self.pos[1]))    
            
        arc_gdf = gpd.GeoDataFrame(index = [0],
                                   crs = vel_mag_rast.crs,
                                   geometry = [arc_rot])
        
        # perform mask
        masked = mask(vel_mag_rast,
                      arc_gdf.loc[0],
                      crop = True,
                      nodata = np.nan,
                      pad = True)    
        
        # get mask origin
        mask_x = masked[1][2]
        mask_y = masked[1][5]
        
        idx = np.where(masked[0] == np.nanmin(masked[0]))
        
        # compute position of max value
        # TODO - figure out why this fails so I'm not using an exception
        try:
            min_x = mask_x + idx[1][-1] * masked[1][0]
            min_y = mask_y + idx[2][-1] * masked[1][4]
            
            # vector of min velocity position relative to position of fish
            v = np.array([min_x,min_y]) - np.array([self.pos[0],self.pos[1]])
             
            # unit vector                               
            v_hat = v/np.linalg.norm(v)         
            
            velocity_min = (weight * v_hat)/((5 * self.length/1000.)**2)
            
            return velocity_min
        
        except:
            return [0., 0.]
        
        
    def rheo_cue (self, vel_dir_rast, weight):
        '''function rheotactic heading command.  
        
        Use spatial indexing to find current direction, heading is -180 degrees.'''
        
        # get the x, y position of the agent 
        x, y = (self.pos[0], self.pos[1])
        
        # find the row and column in the direction raster
        row, col = vel_dir_rast.index(x, y)
        
        # get velocity direction
        vel_dir = vel_dir_rast.read(1)[row, col] - np.radians(180)
        
        v_hat = np.array([np.cos(vel_dir), np.sin(vel_dir)])
        
        # calculate attractive force
        rheotaxis = (weight * v_hat)/((5 * self.length/1000.)**2)
        
        return rheotaxis
        
    def shallow_cue(self, depth_rast, weight):
        '''

        Function finds all cells that are too shallow within the sensory buffer
        and then calculates their inverse gravitational potential.  Then adds up 
        all forces to produce the sum total repulsive force.
        '''
        
        # create a sensory wedge looking in front of fish 
        theta = np.radians(np.linspace(-15,15,100))
        arc_x = self.pos[0] + (2. * self.length/1000.) * np.cos(theta)
        arc_y = self.pos[1] + (2. * self.length/1000.) * np.sin(theta)
        arc_x = np.insert(arc_x,0,self.pos[0])
        arc_y = np.insert(arc_y,0,self.pos[1])
        arc = np.column_stack([arc_x, arc_y])
        arc = Polygon(arc)
        arc_rot = affinity.rotate(arc,np.degrees(self.heading),origin = (self.pos[0],self.pos[1]))
        
        arc_gdf = gpd.GeoDataFrame(index = [0],
                                   crs = depth_rast.crs,
                                   geometry = [arc_rot])        
    
        # perform mask
        masked = mask(depth_rast,
                      arc_gdf.loc[0],
                      all_touched = True,
                      crop = True,
                      nodata = np.nan)
        
        # get mask origin
        mask_x = masked[1][2]
        mask_y = masked[1][5]
        pixel_size_x = masked[1][0]
        pixel_size_y = masked[1][4]

        # calculate max depth - body depth in cm - make sure we divide by 100. 
        min_depth = (self.body_depth * 1.1) / 100.  
               
        repArr = []
        
        def force_multiplier(i, min_depth):
            '''function that incorporates depth requirement'''
        
            if i <= min_depth:
                return 1.0 
            else:
                return 0.0
  
        v_force_multiplier = np.vectorize(force_multiplier,excluded = [1])
        
        multiplier = v_force_multiplier(masked[0], min_depth)

        # create an array of x and y coordinates of cells
        ys = np.arange(mask_y,mask_y + (pixel_size_y * masked[0].shape[1]), pixel_size_y)
        xs = np.arange(mask_x,mask_x + (pixel_size_x * masked[0].shape[2]) ,pixel_size_x)
        
        # create a meshgrid of coordinates
        repx, repy = np.meshgrid(xs,ys)
        
        # make a function to calculate distance squared from each cell to agent
        def distance (xi, yi, x, y):
            '''function to calculate distance between mental map cell and agent
            xs = mental map x, 
            ys = mental map y
            x = fish x
            y = fisy y'''
            
            dx = xi - x
            dy = yi - y
            return np.linalg.norm([dx,dy])
        
        # vectorize 
        v_distance = np.vectorize(distance,excluded = [2,3])
        
        # calculate distance 
        dist_grid = np.power(v_distance(repx, repy, self.pos[0], self.pos[1]),2)
        
        # make a functio to calculate direction from each cell to agent
        def direction (xi, yi, x, y):
            '''function that calculates a unit vector from mental map cell to agent'''
            
            v = np.array([x, y]) - np.array([xi, yi])
                
            vhat = v / np.linalg.norm(v)
            
            return np.arctan2(vhat[1],vhat[0])
        
        # vectorize
        v_direction = np.vectorize(direction, excluded = [2,3])
        
        # calculate direction 
        dir_grid = v_direction(repx, repy, self.pos[0], self.pos[1])
        
        # calculate repulsive force in X and Y directions 
        x_force = ((weight * np.cos(dir_grid))/ dist_grid) * multiplier
        y_force = ((weight * np.sin(dir_grid))/ dist_grid) * multiplier
        
        if np.linalg.norm([np.nansum(x_force),np.nansum(y_force)]) != 0.0:
            print ('fuck - lets look at these arrays to make sure we good')
        
        return [np.nansum(x_force),np.nansum(y_force)]                                     
     
    def wave_drag_multiplier(self):
        '''Function calculates the wave drag multiplier from data digitized from 
        Hughes 2004 Figure 3'''
        
        # get data
        try:
            hughes = pd.read_csv(r'../data/wave_drag_huges_2004_fig3.csv')

        except:
            hughes = pd.read_csv(r'C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent\data/wave_drag_huges_2004_fig3.csv')
                                 
        hughes.sort_values(by = 'body_depths_submerged', 
                           ascending = True,
                           inplace = True)
        # fit function 
        wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged,
                                         hughes.wave_drag_multiplier,
                                         k = 3, ext = 0) 
        
        # how submerged is this fish - that's how many
        body_depths = self.z / (self.body_depth / 100.) 
        
        if body_depths >= 3:
            self.wave_drag = 1.
        else:
            self.wave_drag = wave_drag_fun(body_depths)
        
    def wave_drag_cue(self, depth_rast, weight):
        '''Function finds the direction to the optimal depth cell so that the 
        agent minimizes wave drag'''
        
        # create sensory buffer
        l = (self.length/1000.) * 4
        
        # create wedge looking in front of fish 
        theta = np.radians(np.linspace(-15,15,100))
        arc_x = self.pos[0] + l * np.cos(theta)
        arc_y = self.pos[1] + l * np.sin(theta)
        arc_x = np.insert(arc_x,0,self.pos[0])
        arc_y = np.insert(arc_y,0,self.pos[1])
        arc = np.column_stack([arc_x, arc_y])
        arc = Polygon(arc)
        arc_rot = affinity.rotate(arc,np.degrees(self.heading), origin = (self.pos[0],self.pos[1]))
        
        arc_gdf = gpd.GeoDataFrame(index = [0],
                                   crs = depth_rast.crs,
                                   geometry = [arc_rot])
         
        # perform mask
        masked = mask(depth_rast,
                      arc_gdf.loc[0],
                      all_touched = True,
                      crop = True,
                      nodata = np.nan,
                      pad = True)
        
        # get mask origin
        mask_x = masked[1][2]
        mask_y = masked[1][5]
        
        '''where is the ideal location - the location that minimizes wave drag
        we are looking for the cell that has the smallest difference between 
        itself and the minimum depth'''
        
        idx = np.where(masked[0] - self.opt_wat_depth == np.nanmin(np.abs(masked[0] - self.opt_wat_depth)))
        
        # compute position of max value
        try:
            min_x = mask_x + idx[1][-1] * masked[1][0]
            min_y = mask_y + idx[2][-1] * masked[1][4]
                
            # vector of max velocity position relative to position of fish
            v = np.array([min_x,min_y]) - np.array([self.pos[0],self.pos[1]]) 
             
            # unit vector                               
            v_hat = v/np.linalg.norm(v) 
            
            # calculate attractive potential towards deepest point
            attArr = (weight * v_hat)/((5 * self.length/1000.)**2)
            
            return attArr
        
        except:
            return [0., 0.]


        
    def school_cue(self, weight):
        
        # make a functio to calculate direction from each fish to centroid of nearby fish
        def direction (xi, yi, x, y):
            '''function that calculates a unit vector from mental map cell to agent'''
            
            # vector of nearby fish centroid position relative to position of fish
            v = np.array([x, y]) - np.array([xi, yi])
            
            # unit vector
            vhat = v / np.linalg.norm(v)
            
            return np.arctan2(vhat[1],vhat[0])
        
        if not self.neighbors.empty:
            # find average speed and heading of nearby fish
            self.sog = self.neighbors.mean(numeric_only = True)['vel']
            
            # find centroid of nearby fish
            centroid = self.neighbors.dissolve().centroid
            
            # distance to centroid of nearby fish
            cent_dist = Point(self.pos).distance(centroid)[0]
            
            # direction = direction(self.pos[0], self.pos[1], centroid.x[0], centroid.y[0])
            
            v = np.array([centroid.x[0], centroid.y[0]]) - np.array([self.pos[0], self.pos[1]])
    
            v_hat = v/np.linalg.norm(v)
            school_cue = (weight * v_hat)/(cent_dist)
            
            return school_cue
        else:
            return [0.,0.]
        
    def collision_cue(self, weight):
        
        if not self.neighbors.empty:
            
            # calculate distance to each nearby fish normalized by self length
            self.neighbors['distance'] = np.array([Point(self.pos).distance(x)/self.length for x in self.neighbors['loc']])
            
            if self.neighbors['distance'].min() == 0:
                print('FISH BOINKED')
                return [0.,0.]
            
            else:
                # find closest nearby fish
                closest_fish = self.neighbors[self.neighbors['distance'] == self.neighbors['distance'].min()]
                
                # vector of closest fish position relative to position of fish
                c = np.array([closest_fish['loc'].x.values[0], closest_fish['loc'].y.values[0]]) - np.array(self.pos)
                
                # unit vector
                c_hat = -1*c/np.linalg.norm(c)
                print(c_hat)
                collision_cue = (weight * c_hat)/((closest_fish['distance'].values[0]*1000))
            
                return collision_cue
        else:
            return [0.,0.]
        
    def arbitrate(self,vel_mag_rast, depth_rast, vel_dir_rast, t):
        '''method arbitrates heading commands returning a new heading
        
        Depending on overall behavioral mode, fish cares about different inputs'''
                
        rheotaxis = self.rheo_cue(vel_dir_rast,10000)
        shallow = self.shallow_cue(depth_rast,5000)
        wave_drag = self.wave_drag_cue(depth_rast,8000)
        low_speed = self.vel_cue(vel_mag_rast,9000)
        avoid = self.already_been_here(depth_rast,6000, t)

        avoid = self.already_been_here(depth_rast,8000, t)
                
        school = self.school_cue(2000)
        collision = self.collision_cue(10000)
        
        # calculate the norm of some important behavioral cues
        shallow_n = np.linalg.norm(shallow)
        avoid_n = np.linalg.norm(avoid)
        school_n = np.linalg.norm(school)
        collision_n = np.linalg.norm(collision)
        
        # the fish only has so many fucks - aka prioritized acceleration - Reynolds 1987
        # if fish is actively migrating
        if self.swim_behav == 'migratory':
            # #most important cue is shallow - we can't hav ea fish out of water
            if shallow_n > 0.0:
                # create a heading vector - based on input from sensory cues
                head_vec = shallow
            
            elif collision_n > 100:
                # create a heading vector - based on input from sensory cues
                head_vec = collision
                    
            elif avoid_n > 0.0:
                # create a heading vector - based on input from sensory cue
                head_vec = rheotaxis + avoid
                
            elif school_n > 0.0:
                # create a heading vector - based on input from sensory cue
                head_vec = rheotaxis + school
            else:
                # create a heading vector - based on input from sensory cues
                head_vec = rheotaxis + low_speed + wave_drag + shallow + school
            
        # else if fish is seeking refugia
        elif self.swim_behav == 'refugia':
            # create a heading vector - based on input from sensory cues
            head_vec = shallow + low_speed
        
        # otherwise we are station holding
        else:
            # create a heading vector - based on input from sensory cues
            head_vec = rheotaxis
        
        # convert into preferred heading for timestep
        heading = np.arctan2(head_vec[1],head_vec[0])
        
        # change heading
        self.heading = heading
        
        # log this!        
        print('''Fish %s heading arbitration:
        rheotaxis:        %s
        too shallow:      %s
        wave drag:        %s
        lowest velocity:  %s
        place response:   %s
        schooling:        %s
        collision:        %s
        final heading:    %s'''%(self.ID,
        np.round(rheotaxis,2),
        np.round(shallow,2),
        np.round(wave_drag,2),
        np.round(low_speed,2),
        np.round(avoid,2),
        np.round(school,2),
        np.round(collision, 2),
        np.round(np.degrees(self.heading),2)))
    
        row_dict = {'ID':[self.ID],
                    'timestamp':[t],
                    'rheotaxis':['{:50}'.format(str(tuple(np.round(rheotaxis,2))))],
                    'shallow':['{:50}'.format(str(tuple(np.round(shallow,2))))],
                    'wave_drag':['{:50}'.format(str(tuple(np.round(wave_drag,2))))],
                    'low_speed':['{:50}'.format(str(tuple(np.round(low_speed,2))))],
                    'place_response':['{:50}'.format(str(tuple(np.round(avoid,2))))],
                    'schooling':['{:50}'.format(str(tuple(np.round(school,2))))],
                    'collision':['{:50}'.format(str(tuple(np.round(collision,2))))],                    
                    'final_heading':[np.round(np.degrees(self.heading),2)]}
        
        arb_df = pd.DataFrame.from_dict(row_dict, orient = 'columns')
        arb_df.to_hdf(self.hdf,
                      key = 'arbitrate',
                      mode = 'a',
                      format = 'table',
                      min_itemsize = {'rheotaxis':50,
                                       'shallow':50,
                                       'wave_drag':50,
                                       'low_speed':50,
                                       'place_response':50,
                                       'schooling':50,
                                       'collision':50},
                      append = True,
                      data_columns = True)
        self.hdf.flush()        

    def thrust_fun (self):
        '''Lighthill 1970 thrust equation. '''
        # density of freshwater assumed to be 1
        rho = 1.0 
        
        # theta that produces cos(theta) = 0.85
        theta = 32.
        
        # convert to units required for model
        length_cm = self.length/1000 * 100.
        
        # calculate swim sweed
        water_vel = np.array([self.x_vel,self.y_vel])
        
        ideal_vel_vec = np.array([self.ideal_sog * np.cos(self.heading),
                                  self.ideal_sog * np.sin(self.heading)]) #meters/sec
        
        ideal_swim_speed = np.linalg.norm(ideal_vel_vec - water_vel)
        swim_speed_cms = ideal_swim_speed * 100.
        
        # sockeye parameters (Webb 1975, Table 20) units in CM!!! FUCK
        length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.]) 
        speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.]) 
        amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4]) 
        wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3]) 
        edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.]) 
        
        # fit univariate spline
        amplitude = UnivariateSpline(length_dat,amp_dat,k = 2) 
        wave = UnivariateSpline(speed_dat,wave_dat,k = 1) 
        trail = UnivariateSpline(length_dat,edge_dat,k = 1) 
        
        # interpolate A, V, B
        A = amplitude(length_cm)
        V = wave(swim_speed_cms)
        B = trail(length_cm) 
        
        # Calculate thrust
        m = (np.pi * rho * B**2)/4.
        W = (self.Hz * A * np.pi)/1.414
        w = W * (1 - swim_speed_cms/V)
        
        # calculate thrust 
        thrust_erg_s = m * W * w * swim_speed_cms - (m * w**2 * swim_speed_cms)/(2. * np.cos(np.radians(theta)))
        thrust_Nm = thrust_erg_s / 10000000.
        thrust_N = thrust_Nm / (self.length/1000.)
        self.thrust = np.array([thrust_N * np.cos(self.heading), 
                                thrust_N * np.sin(self.heading)]) #meters/sec
        
    def frequency (self):
        ''' Function for tailbeat frequency.  By setting Lighthill (1970) equations 
        equal to drag, we can solve for tailbeat frequency (Hz).  
        
        Density of water (rho) is assumed to be 1
        
        Input parameters for this function include:
            U = speed over ground (or swim speed?) (cm/s)
            _lambda = length of the propulsive wave
            L = length, converted to trailing edge span (cm) = 0.2L
            D = force of drag'''
            
        # density of freshwater assumed to be 1
        rho = 1.0 
        
        # theta that produces cos(theta) = 0.85
        theta = 32.
        
        # convert to units required for model
        length_cm = self.length/1000 * 100.
        
        # get ideal swim speed
        water_vel = np.array([self.x_vel,self.y_vel])
        fish_vel = np.array([self.ideal_sog * np.cos(self.heading), 
                              self.ideal_sog * np.sin(self.heading)]) #meters/sec
        
        ideal_swim_speed  = np.linalg.norm(fish_vel - water_vel) 
        swim_speed_cms = ideal_swim_speed * 100.
        
        # sockeye parameters (Webb 1975, Table 20)
        length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
        speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
        amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
        wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
        edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
        
        # fit univariate spline
        amplitude = UnivariateSpline(length_dat,amp_dat,k = 2) 
        wave = UnivariateSpline(speed_dat,wave_dat,k = 1) 
        trail = UnivariateSpline(length_dat,edge_dat,k = 1) 
        
        # interpolate A, V, B
        A = amplitude(length_cm)
        V = wave(swim_speed_cms)
        B = trail(length_cm)  
        
        # convert vector drag to scalar in units of erg/s        
        drag = np.linalg.norm(self.ideal_drag_fun()) * (self.length/1000) * 10000000.
        
        # now that we have all variables, solve for f
        
        if self.swim_behav == 'station holding':
            self.Hz = 1.
        else:
            self.Hz = np.sqrt(drag*V**2*np.cos(np.radians(theta))/\
                              (A**2*B**2*swim_speed_cms*np.pi**3*rho*(swim_speed_cms - V)*\
                               (-0.062518880701972*swim_speed_cms - 0.125037761403944*V*\
                                np.cos(np.radians(theta)) + 0.062518880701972*V)))

    def kin_visc(self,temp):
        '''kinematic viscocity as a function of temperature
        author: Isha Deo
        '''

        # read databases for kinematic viscosity and density from Engineering Toolbox
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
        
        # create function 
        f_kinvisc = interp1d(kin_temp, kin_visc)
        
        return f_kinvisc(temp)
        
    def wat_dens(self,temp):
        '''water density g/ml3 as a function of water temperature
        author: Isha Deo'''
        
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
        
        # create function 
        f_density = interp1d(dens_temp, density)   
        
        return f_density(temp)
    
    def calc_Reynolds(self, length, visc, water_vel):
        '''function to calculate Reynolds Number
        author: Isha Deo'''
        
        length_m = length / 1000.
        
        return water_vel * length_m / visc
        
    def calc_surface_area(self, length, species):
        '''# calculate surface area of fish; assuming salmon for now'''
        
        if species == 'sockeye': # add other salmon species if we get there
            # uses length based method for salmon - better estimates with weight
            a = -0.143
            b = 1.881
            
            return 10 ** (a + b*np.log10(length))
        
        else:
            return np.nan
        
    def drag_coeff(self, reynolds):
        '''function to calculate the drag coefficient given the current Reynolds
        number
        author: Isha Deo'''
        
        # set up drag coefficient vs Reynolds number dataframe
        drag_coeffs_df = pd.DataFrame(data = {'Reynolds Number': [2.5e4, 5.0e4, 
                                                                  7.4e4, 9.9e4, 
                                                                  1.2e5, 1.5e5, 
                                                                  1.7e5, 2.0e5],
                                      'Drag Coefficient': [0.23,0.19,0.15,0.14,
                                                           0.12,0.12,0.11,0.10]}).\
            set_index('Reynolds Number')
        
        # fit drag coefficient vs Reynolds number to functio
        
        def fit_dragcoeffs(reynolds, a, b):
            return np.log(reynolds)*a + b
        
        dragf_popt, dragf_pcov = curve_fit(f = fit_dragcoeffs, 
                                           xdata = drag_coeffs_df.index, 
                                           ydata = drag_coeffs_df['Drag Coefficient'])
        
        # determine drag coefficient for calculated Reynolds number
        drag_coeff = np.abs(fit_dragcoeffs(reynolds, dragf_popt[0], dragf_popt[1]))
        
        return drag_coeff
        
        
    def drag_fun (self):
        """
        Created on Mon May 29 10:16:26 2023

        @author: Isha Deo

        Calculate drag force on a sockeye salmon swimming upstream 
        given the fish length, water & fish velocities, and water temperature.

        The drag on a fish is calculated using the drag force equation given in Mirzaei 2017:
            
            drag = -0.5 * density * surface area * drag coefficient * |fish velocity - water velocity|^2 * (fish velocity / |fish velocity|)
            
        The surface area of the fish is determined using an empirical relationship 
        developed in Oshea 2006 for salmon

        The drag coefficient is determined using measured values from Brett 1963 
        (adapted from Webb 1975), fitted to a log function
        """
        
        # define import values - note units!!
        
        water_vel = np.array([self.x_vel,self.y_vel])
        
        fish_vel = np.array([self.sog * np.cos(self.heading), 
                              self.sog * np.sin(self.heading)]) #meters/sec
        
        # fish_vel = np.array([self.ideal_sog * np.cos(self.heading), 
        #                       self.ideal_sog * np.sin(self.heading)]) #meters/sec
        
        if np.linalg.norm(fish_vel) == 0.0:
            fish_vel = [0.0001,0.0001]
                        
        # determine kinematic viscosity based on water temperature
        visc = self.kin_visc(self.water_temp)
        
        # determine density based on water temperature
        dens = self.wat_dens(self.water_temp)
        
        #TODO check to make sure length in cm correct
        reynolds = self.calc_Reynolds(self.length, visc, np.linalg.norm(water_vel))
        
        # Calculate surface area
        surface_area = self.calc_surface_area((self.length/1000.) * 100., 'sockeye')
        
        # calculate the drag coefficient
        drag_coeff = self.drag_coeff(reynolds)
        
        # calculate drag!
        self.drag = -0.5 * (dens * 1000) *\
            (surface_area / 100**2) *\
                drag_coeff *\
                    (np.linalg.norm(fish_vel - water_vel)**2) *\
                        (fish_vel/np.linalg.norm(fish_vel)) *\
                            self.wave_drag

    def ideal_drag_fun (self):
        """
        Created on Mon May 29 10:16:26 2023

        @author: Isha Deo

        Calculate drag force on a sockeye salmon swimming upstream 
        given the fish length, water & fish velocities, and water temperature.

        The drag on a fish is calculated using the drag force equation given in Mirzaei 2017:
            
            drag = -0.5 * density * surface area * drag coefficient * |fish velocity - water velocity|^2 * (fish velocity / |fish velocity|)
            
        The surface area of the fish is determined using an empirical relationship 
        developed in Oshea 2006 for salmon

        The drag coefficient is determined using measured values from Brett 1963 
        (adapted from Webb 1975), fitted to a log function
        """
        
        # get vector components of water velocity and speed over ground
        water_vel = np.array([self.x_vel,self.y_vel])
        fish_vel = np.array([self.ideal_sog * np.cos(self.heading), 
                             self.ideal_sog * np.sin(self.heading)]) #meters/sec
        
        # calculate ideal swim speed
        ideal_swim_speed = np.linalg.norm(fish_vel - water_vel)
       
        # make sure this fish isn't swimming faster than it can
        if self.swim_behav == 'refugia' or self.swim_behav == 'station holding':
            if ideal_swim_speed > self.max_s_U:
                fish_vel = fish_vel * (ideal_swim_speed/self.max_s_U)
        
                # convert back to vector form
                ideal_swim_speed_vec = np.array([ideal_swim_speed * np.cos(self.heading), 
                                                  ideal_swim_speed * np.sin(self.heading)]) #meters/sec
                
                # calculate the maximum practical sog that this fish can attain
                max_practical_sog = ideal_swim_speed_vec + water_vel
                
                # we will use this later for thrust - save as property
                self.max_practical_sog = max_practical_sog
            else:
                self.max_practical_sog = fish_vel
        else:
            self.max_practical_sog = fish_vel
        
        if np.linalg.norm(self.max_practical_sog) == 0.0:
            self.max_practical_sog = [0.0001,0.0001]
                
        # determine kinematic viscosity based on water temperature
        visc = self.kin_visc(self.water_temp)
        
        # determine density based on water temperature
        dens = self.wat_dens(self.water_temp)
        
        #TODO check to make sure length in cm correct
        reynolds = self.calc_Reynolds(self.length, visc, np.linalg.norm(water_vel))
        
        # Calculate surface area
        surface_area = self.calc_surface_area((self.length/1000.) * 100., 'sockeye')
        
        # calculate the drag coefficient
        drag_coeff = self.drag_coeff(reynolds)   
        
        # calculate drag!
        ideal_drag = -0.5 * (dens * 1000.) *\
            (surface_area / 100**2) *\
                drag_coeff *\
                    (np.linalg.norm(self.max_practical_sog - water_vel)**2) *\
                        (self.max_practical_sog/np.linalg.norm(self.max_practical_sog)) *\
                            self.wave_drag

        return ideal_drag
    
    def initial_swim_speed (self):
        '''Function calculates swim speed required to overcome current water 
        velocities and maintain speed over ground'''
        
        # get vector components of water velocity
        water_vel = np.linalg.norm(np.array([self.x_vel,self.y_vel]))
        
        # get vector components of speed over ground
        ideal_vel = np.array([self.ideal_sog * np.cos(self.heading), 
                             self.ideal_sog * np.sin(self.heading)]) #meters/sec

        # calculate swim speed
        self.swim_speed = np.linalg.norm(ideal_vel - water_vel)

    def swim(self, dt, t):
        '''Method propels a fish forward'''
        # get fish velocity in vector form        
        fish_vel_0 = np.array([self.sog * np.cos(self.heading), 
                             self.sog * np.sin(self.heading)]) #meters/sec
        
        # calculate surge
        surge = np.round(self.thrust,2) + np.round(self.drag,2)
        acc = np.round(surge/self.weight,2)
        
        # dampen that acceleration
        acc_mag = np.linalg.norm(acc)
        if acc_mag > 0.0:
            damp = (-0.067 * np.log(acc_mag) + 0.3718)
            
            if damp < 0:
                damp = 0.0000001
            #damp = acc * 0.90
            
            # TODO
            '''this is a tricky bit, the procedure is correct but because drag is 
            proportional to velocity squared - our thrust change will always over 
            react.  Until I figure this out - we are applying a massive dampener''' 

            acc = acc * damp 
            
            # if np.linalg.norm(acc) > 5.0:
            #     print ('fuck') 
        
        # what will velocity be at end of time step
        fish_vel_1 = fish_vel_0 + acc * dt
            
        self.sog = np.round(np.linalg.norm(fish_vel_1),6)
        
        if np.isnan(self.sog):
            print ('fuck - something aint right, think divide by zero somewhere')
            
        if self.sog > 5:
           print ('fuck - Im swimming fast, why?')
            
        if np.linalg.norm(self.thrust) / np.linalg.norm(self.drag) > 10:
            print ('fuck - holy hell thrust is way bigger than drag - we flying')
        
        # start movement
        self.prevPos = self.pos  
        
        print ('''Fish %s movement summary: 
        speed over ground:  %s m/s, 
        swim speed:         %s m/s,
        water velocity:     %s m/s,
        water depth:        %s m,
        swim depth:         %s m,
        wave drag:          %s,
        caudal fin:         %s Hz,
        thrust:             %s N,
        drag:               %s N,
        surge:              %s N,
        acceleration:       %s m/s/s'''%(self.ID,
        np.round(self.sog,2),
        np.round(self.swim_speed,2),
        np.round(np.linalg.norm([self.x_vel, self.y_vel]),2),
        np.round(self.depth,2),
        np.round(self.z,2),
        np.round(self.wave_drag,2),
        np.round(self.Hz,2),
        np.round(np.linalg.norm(self.thrust),2),
        np.round(np.linalg.norm(self.drag),2),
        np.round(np.linalg.norm(surge),2),
        np.round(np.linalg.norm(acc),2)))
    
        swim_dict = {'ID':[self.ID],
                     'timestep':[t],
                     'speed_over_ground':[np.round(self.sog,2)],
                     'swim_speed':[np.round(self.swim_speed,2)],
                     'water_velocity':[np.round(np.linalg.norm([self.x_vel, self.y_vel]),2)],
                     'water_depth':[np.round(self.depth,2)],
                     'swim_depth':[np.round(self.z,2)],
                     'wave_drag':[np.round(self.wave_drag,2)],
                     'tail_beats_Hz':[np.round(self.Hz,2)],
                     'thrust':[np.round(np.linalg.norm(self.thrust),2)],
                     'drag':[np.round(np.linalg.norm(self.drag),2)],
                     'surge':[np.round(np.linalg.norm(surge),2)],
                     'acceleration':[np.round(np.linalg.norm(acc),2)]}
    
        swim_df = pd.DataFrame.from_dict(swim_dict, orient = 'columns')
        swim_df.to_hdf(self.hdf,'swim',mode = 'a',format = 'table', append = True)
        self.hdf.flush()  
   
        # set new position
        self.pos = self.prevPos + fish_vel_1 
        
        print ('Fish %s is at %s'%(self.ID,np.round(self.pos,3)))
        
    def jump (self,t):
        '''Method that simulates fish jumping using ballistic trajectory.
        
        We make simplifying assumption that a fish can accelerate to ucrit by 
        the time it leaves the water column regardless of water depth.  
        
        Returns
        -------
        None.

        '''

        # reset jump time
        self.time_of_jump = t
        
        # get jump angle
        jump_angle = np.random.choice([np.radians(45),np.radians(60)])
        
        # calculate time airborne
        time_airborne = (2 * self.ucrit * np.sin(jump_angle))/g
        
        # calculate displacement 
        displacement = self.ucrit * time_airborne * np.cos(jump_angle)
        
        # set speed over ground to ucrit
        self.sog = self.ucrit
        
        # calculate new heading angle - based soley on flow direction
        self.heading = np.arctan2(self.y_vel,self.x_vel) - np.radians(180)
        
        # calculate the new position 
        new_pos = np.array(self.pos) + displacement * np.array([np.cos(self.heading),np.sin(self.heading)])
        
        # set new position
        self.pos = new_pos
        
        print ('''Fish %s jump report:
            jump angle:      %s
            time airborne:   %s
            displacement:    %s'''%(self.ID,
            np.degrees(jump_angle),
            np.round(time_airborne,2),
            np.round(displacement,2)))
    
        jump_dict = {'ID':[self.ID],
                     'timestep':[t],
                     'jump_angle':[np.degrees(jump_angle)],
                     'time_airborne':[np.round(time_airborne,2)],
                     'displacement':[np.round(displacement,2)]}
    
        jump_df = pd.DataFrame.from_dict(jump_dict, orient = 'columns')
        jump_df.to_hdf(self.hdf,'jump',mode = 'a',format = 'table', append = True)
        self.hdf.flush()  
                                     
        print ("Fish %s is at %s"%(self.ID,self.pos))
        
    def fatigue(self,t):    
        '''
        Method tracks battery levels and assigns swimming modes

        Returns
        -------
        None.

        '''
        
        # Values for Sockeye digitized from Bret 1964
        dt = 1.
        
        # get vector components of water velocity and speed over ground
        water_vel = np.array([self.x_vel,self.y_vel])
        fish_vel = np.array([self.sog * np.cos(self.heading), 
                             self.sog * np.sin(self.heading)]) #meters/sec
        
        # calculate swim speed
        self.swim_speed = np.linalg.norm(fish_vel - water_vel)
        
        # fish is not station holding and battery above critical state
        if self.swim_behav != 'station holding':
            # add distance travelled to bout odometer
            dist_travelled = np.linalg.norm(np.array(self.pos) - np.array(self.prevPos))
            self.dist_per_bout = self.dist_per_bout + dist_travelled
            self.bout_dur = self.bout_dur + 1

            # calculate ttf at current swim speed
            if self.max_s_U < self.swim_speed <= self.max_p_U:
                # reset stopwatch
                self.recover_stopwatch = 0.0
                
                # set swimming mode
                self.swim_mode = 'prolonged'
                
                # calculate time to fatigue at current swim speed - Brett 1964 was minutes!
                ttf = 10. ** (self.a_p + self.swim_speed * self.b_p) * 60.
        
            elif self.swim_speed > self.max_p_U:
                # reset stopwatch
                self.recover_stopwatch = 0.0
                
                # set swimming mode
                self.swim_mode = 'sprint'
                
                # calculate time to fatigue at current swim speed
                ttf = 10. ** (self.a_s + self.swim_speed * self.b_s) * 60.
                
            else:
                self.swim_mode = 'sustained'
                
                # calculate recovery % at beginning of time step
                rec0 = self.recovery(self.recover_stopwatch) / 100.
                
                # make sure realistic value - also can't divide by 0
                if rec0 < 0.0:
                    rec0 = 0.0
                    
                # calculate recovery % at end of time step
                rec1 = self.recovery(self.recover_stopwatch + dt) / 100. 
                
                if rec1 > 1.0:
                    rec1 = 1.0
                
                # calculate percent increase
                per_rec = rec1 - rec0
                
                # stopwatch
                self.recover_stopwatch = self.recover_stopwatch + dt
                
                ttf = np.nan                
                
                # calculate battery level
                self.battery = self.battery + per_rec    
                
                # make sure battery recharge is reasonable
                if self.battery > 1.0:
                    self.battery = 1.0
            
            if self.swim_mode != 'sustained':
                # take into account the time that has already past by multiplying by the batery %
                ttf0 = ttf * self.battery
                
                # calculate time to fatigue at end of time step
                ttf1 = ttf0 - dt
                
                # calculate battery level
                self.battery = self.battery * ttf1/ttf0
                
                # make sure battery drain is reasonable
                if self.battery < 0.0:
                    self.battery = 0.0            
            
            # set swimming behavior based on battery level
            if self.battery <= 0.1:
                self.swim_behav = 'station holding'
                self.swim_mode = 'sustained'
                self.ideal_sog = 0.
                self.ttfr = 0.
                self.bout_no = self.bout_no + 1
                
            elif 0.1 < self.battery <= 0.3:
                self.swim_behav = 'refugia'
                self.ideal_sog = 0.02
   
            else:
                self.swim_behav = 'migratory'
                if self.battery == 1.0:
                    self.ideal_sog = self.length/1000.
                else:
                    ideal_bls = 0.0075 * np.exp(4.89 * self.battery)
                    self.ideal_sog = np.round(ideal_bls * (self.length/1000.),2)          
                      
            # check distance travelled per bout - is it enough to warrant keeping going?
            if self.bout_dur > 300:
                 # if m/s is averaging below 10 cm/s after 5 min - let's give up
                if self.dist_per_bout/self.bout_dur < 0.1:
                    self.swim_behav = 'station holding'
                    self.swim_mode = 'sustained'
                    self.ideal_sog = 0.
                    self.ttfr = 0
                    self.bout_no = self.bout_no + 1
                    
        # fish is station holding and recovering    
        else:
            ''' recovery is based on how long a fish has been in a recovery state'''
            self.bout_dur = 0.
            self.dist_per_bout = 0.
            # calculate recovery % at beginning of time step
            rec0 = self.recovery(self.recover_stopwatch) / 100.
            
            # make sure realistic value - also can't divide by 0
            if rec0 < 0.0:
                rec0 = 0.0
                
            # calculate recovery % at end of time step
            rec1 = self.recovery(self.recover_stopwatch + dt) / 100. 
            
            if rec1 > 1.0:
                rec1 = 1.0
            
            # calculate percent increase
            per_rec = rec1 - rec0
            
            # calculate battery level
            self.battery = self.battery + per_rec
            
            # stopwatch
            self.recover_stopwatch = self.recover_stopwatch + dt
            
            # battery is charged enough to start moving again
            if self.battery >= 0.85:
                # reset recharge stopwatch
                self.recover_stopwatch = 0.0
                self.swim_behav = 'migratory'
                self.swim_mode = 'sustained'

            else:
                self.swim_behav = 'station holding'
                self.swim_mode = 'sustained'  
        
        # log this!
        print ('''Fish %s battery summary: 
        battery:       %s   
        swimming mode: %s
        behavior:      %s
        bout number:   %s
        dist per bout: %s m
        bout duration: %s s'''%(self.ID,
        np.round(self.battery,4),
        self.swim_mode,
        self.swim_behav,
        self.bout_no,
        np.round(self.dist_per_bout,2),
        self.bout_dur)) 
    
        batt_dict = {'ID':[self.ID],
                     'timestep':[t],
                     'battery':[np.round(self.battery,4)],
                     'swim_mode':['{:50}'.format(self.swim_mode)],
                     'swim_behavior':['{:50}'.format(self.swim_behav)],
                     'bout_no':[self.bout_no],
                     'dist_per_bout':[np.round(self.dist_per_bout,2)],
                     'bout_duration':[self.bout_dur]}
        
        batt_df = pd.DataFrame.from_dict(batt_dict)
        batt_df.to_hdf(self.hdf,
                       key = 'battery',
                       mode = 'a',
                       format = 'table', 
                       min_itemsize = {'swim_mode':50,
                                        'swim_behavior':50},
                       append = True,)
        self.hdf.flush()      
    
    def odometer(self):
        '''Created on Thu May 18 20:17:28 2023
    
        @author: KNebiolo
    
        Method that accounts for the amount of kCal consumed during a simulation.  
        An odometer if you will. 
    
        The intent of this method is to keep a running counter of the amount of kCal 
        consumed by converting the amount of oxygen resprired into calories with standard
        metabolic equations.  
    
        Brett (1964) provides active metabolic rates or oxygen consumption (O2/kg/hr) 
        as a function of water temperature and swimming speed (body lengths/second), 
        while Brett and Glass (1973) provide standard metabolic rate as a function of 
        water temperature and weight.  '''
        
        # calculate active standard metabolic rate using Table 2 from Brett and Glass (1973)
        # 02_rate in units of mg O2/hr
        if self.water_temp <= 5.3:
            sr_o2_rate = np.exp(0.0565 * np.power(np.log(self.weight*1000),0.9141)) 
            ar_o2_rate = np.exp(0.4667 * np.power(np.log(self.weight*1000),0.9989))
            
        elif 5.3 < self.water_temp <= 15:
            sr_o2_rate = np.exp(0.1498 * np.power(np.log(self.weight*1000),0.8465))
            ar_o2_rate = np.exp(0.9513 * np.power(np.log(self.weight*1000),0.9632))
    
        elif self.water_temp > 15:
            sr_o2_rate = np.exp(0.1987 * np.power(np.log(self.weight*1000),0.8844))
            ar_o2_rate = np.exp(0.8237 * np.power(np.log(self.weight*1000),0.9947))
    
        # calculate total metabolic rate
        swim_cost = sr_o2_rate + self.wave_drag * (np.exp(np.log(sr_o2_rate) + self.swim_speed * ((np.log(ar_o2_rate) - np.log(sr_o2_rate))/self.ucrit))-sr_o2_rate)
        
        self.kcal = self.kcal + swim_cost
      
class simulation():
    '''Python class object that implements an Agent Based Model of adult upstream
    migrating sockeye salmon through a modeled riffle cascade complex'''
    
    def __init__(self, model_dir, model_name, crs):
        '''initialization function that sets up a simulation.'''
        # model directory and model name
        self.model_dir = model_dir
        self.model_name = model_name
        
        # first step creates a project directory if it doesn't already exist
        if not os.path.exists(os.path.join(self.model_dir,self.model_name)):
            os.makedirs(os.path.join(self.model_dir,self.model_name))

        
        # coordinate reference system for the model
        self.crs = crs
        
        # create empty geodataframe of agents
        self.agents = gpd.GeoDataFrame(columns=['id', 'loc', 'vel', 'dir'], geometry='loc', crs= crs) 
        
        # create an empty hdf file for results
        self.hdf = pd.HDFStore(os.path.join(self.model_dir,self.model_name,'%s.hdf'%(self.model_name)))
        
        
    
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
            
    def vel_surf(self):
        '''Function calculates velocity magnitude and direction'''
        # calculate velocity magnitude
        vel_mag = np.sqrt((np.power(self.vel_x_rast.read(1),2)+np.power(self.vel_y_rast.read(1),2)))
        
        # calculate velocity direction in radians
        vel_dir = np.arctan2(self.vel_y_rast.read(1),self.vel_x_rast.read(1))            
        
        # create raster properties
        driver = 'GTiff'
        width = vel_mag.shape[1]
        height = vel_mag.shape[0]
        count = 1
        crs = self.crs
        transform = Affine.translation(self.vel_x_rast.bounds[0] - 0.5, self.vel_x_rast.bounds[3] - 0.5) * Affine.scale(1,-1)

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
            
        # write velocity mag raster
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
        
    def HECRAS (self,HECRAS_model):
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
        xint = np.arange(xmin,xmax,1)
        yint = np.arange(ymax,ymin,-1)
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
        transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)

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
        
    def create_agents(self, numb_agnts, model_dir, starting_box, water_temp):
        '''method that creates a set of agents for simulation'''
        
        agents_list = []
        
        for i in np.arange(0,numb_agnts,1):
            # create a fish
            fishy = fish(i,model_dir, starting_box, water_temp)
            
            # set initial parameters
            fishy.initial_heading(self.vel_dir_rast)
            fishy.initial_swim_speed()
            fishy.mental_map(self.depth_rast)
            
            # add it to the output list
            agents_list.append(fishy)
            
            # create a dataframe for this agent 
            df = pd.DataFrame.from_dict(data = {'id':[i],
                                                'loc':[Point(fishy.pos)],
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
        timestep['loc'] = self.agents['loc'].astype(str)
        timestep['vel'] = self.agents['vel']
        timestep['dir'] = self.agents['dir']
        timestep.to_hdf(self.hdf,'TS',mode = 'a',format = 'table', append = True)
        self.hdf.flush()        
        
        # second step creates a new agents geo dataframe
        self.agents = gpd.GeoDataFrame(columns=['id', 'loc', 'vel', 'dir'],
                                       geometry='loc',
                                       crs= self.crs) 

        agent_dict = {'id':[],'loc':[],'vel':[],'dir':[]}
        
        for i in self.agents_list: 
            agent_dict['id'].append(i.ID)
            agent_dict['loc'].append(Point(i.pos))
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
        
        agent_pts, = plt.plot([], [], marker = 'o', ms = 2, ls = '', color = 'red')
        
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        
        # Update the frames for the movie
        with writer.saving(fig, os.path.join(self.model_dir,self.model_name,'%s.mp4'%(model_name)), 300):
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
                    agent.odometer()
                
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
        

 
    
