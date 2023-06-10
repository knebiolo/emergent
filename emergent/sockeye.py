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
space as our environment is a depth and time averaged 2d model.  Movement in 
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
from rasterstats import zonal_stats
from shapely import Point, Polygon
from shapely import affinity
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline
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
    
    def __init__(self, ID, model_dir, starting_block):
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
        self.length = 250.
        self.weight = 2.2
        self.body_depth = 6.2
        
        # initialize internal states
        self.swim_mode = 'M' # swimming modes, M = migratory, R = refugia, S = station holding
        self.battery = 1.
        
        # initialize movement parameters
        self.swim_speed = 0.
        self.sog = self.length/1000                                            # sog = speed over ground - assume fish maintain 1 body length per second
        self.heading = 0.                                                      # direction of travel in radians
        self.drag_coef = 0.                                                    # drag coefficient 
        self.drag = 0                                                          # computed theoretical drag
        self.thrust = 0                                                        # computed theoretical thrust Lighthill 
        self.Hz = 0.                                                           # tail beats per second
        
        # initialize the odometer
        self.kcal = 0.
        
        #position the fish within the starting block
        x = np.random.uniform(starting_block[0],starting_block[1])
        y = np.random.uniform(starting_block[2],starting_block[3])
        self.pos = (x,y,0.)
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
        
    def vel_comm (self, vel_mag_rast):
        '''Function that returns a lowest velocity heading command - 
        the way upstream within this narrow arc in front of me - looking for 
        lowest velocity'''

        ##########################################
        # Step 1: Create Sensory Buffer
        #########################################        

        # create sensory buffer
        l = (self.length/1000.) * 10
        
        # create wedge looking in front of fish 
        theta = np.radians(np.linspace(-10,10,100))
        arc_x = self.pos[0] + l * np.cos(theta)
        arc_y = self.pos[1] + l * np.sin(theta)
        arc_x = np.insert(arc_x,0,self.pos[0])
        arc_y = np.insert(arc_y,0,self.pos[1])
        arc = np.column_stack([arc_x, arc_y])
        arc = Polygon(arc)
        arc_rot = affinity.rotate(arc,np.degrees(self.heading))
        
        arc_gdf = gpd.GeoDataFrame(index = [0],
                                   crs = vel_mag_rast.crs,
                                   geometry = [arc_rot])
        
        ##########################################
        # Step 2: Mask Velocity Magnitude Surface
        #########################################
        
        # perform mask
        masked = mask(vel_mag_rast,
                      arc_gdf.loc[0],
                      all_touched = True,
                      crop = True)
        
        ##############################################
        # Step 3: Get Cell Center of Highest Velocity
        ##############################################
        
        # get mask origin
        mask_x = masked[1][2]
        mask_y = masked[1][5]
        
        # get indices of cell in mask with highest elevation
        zs = zonal_stats(arc_gdf,
                         vel_mag_rast.read(1),
                         affine = vel_mag_rast.transform,
                         stats=['min'],
                         all_touched = True)
        
        idx = np.where(masked[0] == zs[0]['min'])
        
        # compute position of max value
        min_x = mask_x + idx[1][-1] * masked[1][0]
        min_y = mask_y + idx[2][-1] * masked[1][4]
        
        ##################################################
        # Step 4: Get Direction (unit vector) To Goal Cell
        ##################################################
        
        # vector of max velocity position relative to position of fish 
        v = (np.array([min_x,min_y]) - np.array([self.pos[0],self.pos[1]]))   
         
        # unit vector                               
        v_hat = v/np.linalg.norm(v)         
        
        self.min_velocity_heading = np.arctan2(v_hat[1],v_hat[0])
        
    def rheo_comm (self, vel_dir_rast):
        '''function rheotactic heading command.  
        
        Use spatial indexing to find current direction, heading is -180 degrees.'''
        
        # get the x, y position of the agent 
        x, y = (self.pos[0], self.pos[1])
        
        # find the row and column in the direction raster
        row, col = vel_dir_rast.index(x, y)
        
        # get velocity direction
        vel_dir = vel_dir_rast.read(1)[row, col] - np.radians(180)
        
        # if vel_dir < 0:
        #     vel_dir = (np.radians(360) + vel_dir) - np.radians(180)
        # else:
        #     vel_dir = vel_dir - np.radians(180)
        
        v_hat = np.array([np.cos(vel_dir), np.sin(vel_dir)])
        
        # create unit vector        
        # v_hat = v/np.linalg.norm(v)
    
        # calculate attractive force
        #rheotaxis = np.negative((1000 * v_hat)/2**2)
        rheotaxis = (self.weight * 1000 * v_hat)/2**2
        
        
        return rheotaxis
        
    def shallow_comm(self, depth_rast):
        '''

        Function finds all cells that are too shallow within the sensory buffer
        and then calculates their inverse gravitational potential.  Then adds up 
        all forces to produce the sum total repulsive force.
        '''
        fish = Point(self.pos)
        
        # create sensory buffer
        l = (self.length/1000.)
        
        # create a sensory buffer that is 2 fish lengths
        sensory = fish.buffer(10 * l)
        
        # make a geopandas geodataframe of sensory buffer
        sense_gdf = gpd.GeoDataFrame(index = [0],
                                     crs = depth_rast.crs,
                                     geometry = [sensory])
    
        # perform mask
        masked = mask(depth_rast,
                      sense_gdf.loc[0],
                      all_touched = True,
                      crop = True,
                      nodata = 9999.,
                      pad = True)
        
        # get mask origin
        mask_x = masked[1][2]
        mask_y = masked[1][5]

        # calculate max depth 
        min_depth = (self.body_depth * 2.) / 100.        
        
        # get indices of those cells shallower than min depth
        idxs = np.where(masked[0] < min_depth)
        
        # get position of fish
        fpos = np.array(self.pos[:2])
        
        repArr = []
        
        # if there are shallow cells, loop over indices and grab point
        if len(idxs[1]) > 0:
            for i in idxs:
                # get position of cell 
                idx_x = mask_x + idxs[1][-1] * masked[1][0]
                idx_y = mask_y + idxs[2][-1] * masked[1][4]
                
                x, y = depth_rast.index(idx_x,idx_y)
                dpt = depth_rast.read(1)[x, y]
                
                #if dpt > 0.
                
                # vector of index relative to position of fish 
                v = np.array([self.pos[0],self.pos[1]]) - np.array([idx_x,idx_y])
                 
                # unit vector                               
                v_hat = v/np.linalg.norm(v)  
                
                # calculate the repulsive force generated by the current cell
                rep = (1000 * v_hat)/(np.linalg.norm(v)**2)
                
                repArr.append(rep)
                    
        
                    
            # sum all repulsive forces
            repArr = np.sum(np.nan_to_num(np.array(repArr)),axis = 0)
            #return np.arctan2(repArr[1],repArr[0])
            return repArr
        
        else:
            return np.array([0., 0.])
        
        
       
    def arbitrate(self,vel_mag_rast, depth_rast, vel_dir_rast):
        '''method arbitrates heading commands returning a new heading
        
        we adding up forces now'''
        
        # TODO - obv. this has to get filled out, for now can the fish get upstream
        
        rheotaxis = self.rheo_comm(vel_dir_rast)
        shallow = self.shallow_comm(depth_rast)
        
        # calculate the change in current heading the result of rheotaxis and shallow 
        curr_heading = self.heading

        # create a heading vector - based on input from sensory cues
        head_vec = rheotaxis + shallow
        
        # convert into preferred heading
        heading = np.arctan2(head_vec[1],head_vec[0])
        
        # change heading
        self.heading = heading
        
        # if np.isnan(shallow):
        #     shallow_weight = 0.
        #     rheo_weight = 1.
            
        #     self.heading = rheotaxis
            
        # else:
        #     if delta_shallow > max_delta_heading:
        #         shallow_weight = 1.
        #         rheo_weight = 0.
        #     elif delta_rheo + delta_shallow > max_delta_heading:
        #         shallow_weight = delta_shallow/max_delta_heading
        #         rheo_weight = (max_delta_heading - delta_shallow)/max_delta_heading
        #     else:
        #         shallow_weight = 1.
        #         rheo_weight = 1.
            
        #     self.heading = np.sum([shallow * shallow_weight, rheotaxis * rheo_weight])/np.sum([shallow_weight,rheo_weight])
        # if len(shallow) > 0:
        #     self.heading = shallow
        # else:
        #     self.heading = rheotaxis 
        
        #self.heading = self.rheotaxis
        
    def thrust (self):
        '''Lighthill 1970 thrust equation. '''
        # density of freshwater assumed to be 1
        rho = 1.0 
        
        # theta that produces cos(theta) = 0.85
        theta = 32.
        
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
        A = amplitude(self.length)
        V = wave(self.swim_speed)
        B = trail(self.length) 
        
        # Calculate thrust
        m = (np.pi * rho * B**2)/4.
        W = (self.Hz * A * np.pi)/1.414
        w = W * (1 - self.swim_speed/V)
            
        self.thrhust = m * W * w * self.swim_speed - (m * w**2 * self.swim_speed)/(2. * np.cos(np.radians(theta)))
    
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
        A = amplitude(self.length)
        V = wave(self.swim_speed)
        B = trail(self.length)    
        
        # now that we have all variables, solve for f
        self.Hz = np.sqrt(self.drag*V**2*np.cos(np.radians(theta))/(A**2*B**2*self.swim_speed*np.pi**3*rho*(self.swim_speed - V)*(-0.062518880701972*self.swim_speed - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))

    def swim(self):
        '''

        '''
        # TODO incorporate drag and thrust
        
        # start movement
        self.prevPos = self.pos                                          # previous position is now equal to the current position


        newX = np.array([self.pos[0] + self.sog * np.cos(self.heading)])       # calculate New X
        newY = np.array([self.pos[1] + self.sog * np.sin(self.heading)])       # calculate New Y
        self.pos = np.zeros(3)
        self.pos = np.array([newX[0],newY[0],0])                         # set current position
        

class simulation():
    '''Python class object that implements an Agent Basded Model of adult upstream
    migrating sockeye salmon through a modeled riffle cascade complex'''
    
    def __init__(self, model_dir, model_name, crs):
        '''initialization function that sets up a simulation.'''
        # model directory and model name
        self.model_dir = model_dir
        self.model_name = model_name
        
        # coordinate reference system for the model
        self.crs = crs
        
        # create empty geodataframe of agents
        self.agents = gpd.GeoDataFrame(columns=['id', 'loc', 'vel', 'dir'], geometry='loc', crs= crs) 
        
        # create an empty hdf file for results
        self.hdf = pd.HDFStore(os.path.join(self.model_dir,'%s.hdf'%(self.model_name)))
    
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

        # write velocity x raster
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
            
        # write velocity y raster
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

        # write velocity x raster
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
            
        # write velocity y raster
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
        
    def create_agents(self, numb_agnts, model_dir, starting_box):
        '''method that creates a set of agents for simulation'''
        
        agents_list = []
        
        for i in np.arange(0,numb_agnts,1):
            # create a fish
            fishy = fish(i,model_dir, starting_box)
            
            # set it's initial heading
            fishy.initial_heading(self.vel_dir_rast)
            
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
        '''method that writes to log at end of timestep'''
    
        # first step writes current positions to the projec database
        #TODO fix this
        # self.agents['ts'] = np.repeat(ts,len(self.agents))
        # self.agents.astype(dtype = {'id':np.str}, copy = False)
        # self.agents.to_hdf(self.hdf,'TS',mode = 'a',format = 'table', append = True)
        # self.hdf.flush()        
        
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
    
    def run(self, model_name, agents, n):
        
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
        
        agent_pts, = plt.plot([], [], marker = 'o', ms = 5, ls = '', color = 'red')
        
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        
        # Update the frames for the movie
        with writer.saving(fig, os.path.join(self.model_dir,'%s.mp4'%(model_name)), 300):
            for i in range(n):
                for agent in agents:
                    agent.arbitrate(vel_mag_rast = self.vel_mag_rast, 
                                    depth_rast = self.depth_rast,
                                    vel_dir_rast = self.vel_dir_rast)
                    agent.swim()
                
                # write frame
                agent_pts.set_data(self.agents.geometry.x.values, self.agents.geometry.y.values)
                writer.grab_frame() 
                
                # log data to hdf
                self.ts_log(i)
                
                print ('Time Step %s complete'%(i))
        writer.finish()

# create a simulation function     
    
