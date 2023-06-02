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
from shapely import Point, Polygon
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline
    
# create a sockeye agent 
class fish():
    ''' Python class object for a sockeye agent. 
    
    Class object contains all of the sockeye's attributes, while the methods
    control how the sockeye agent senses its environment, reacts with changing 
    flow conditions, and interacts with other agents.  These methods, which are 
    in reality simple behavioral rules, will lead to complex, self organizing
    behavior.
    '''
    
    def __init__(self, ID, model_dir, basin, starting_box):
        '''initialization function for a sockeye agent.  this function creates
        an agent and parameterizes morphometric parameters from basin specific
        distributions
        
        Units
            length = mm
            weight = kg
            body depth = mm
            velocity = cms SOG'''
            
        # initialization methods
        def sex(self, basin):
            '''function simulates a sex for a given basin'''
            
        def length(self, basin):
            '''function simulates a fish length out of the user provided basin and 
            sex of fish'''
            
        def weight(self, basin):
            '''function simulates a fish weight out of the user provided basin and 
            sex of fish'''
            
        def body_depth(self, basin):
            '''function simulates a fish body depth out of the user provided basin and 
            sex of fish'''    
        # initialize morphometric paramters
        self.ID = ID
        self.sex = sex(basin)
        self.length = length(basin)
        self.weight = weight(basin)
        self.body_depth = body_depth(basin)
        
        # initialize internal states
        self.swim_mode = 'M' # swimming modes, M = migratory, R = refugia, S = station holding
        self.battery = 1.
        
        # initialize movement parameters
        self.swim_speed = 0.
        self.sog = 0.        # sog = speed over ground
        self.heading = 0.    # direction of travel in radians
        self.drag_coef = 0.
        self.drag = 0.
        self.thrust = 0.
        
        # initialize the odometer
        self.kcal = 0.
        
        # position the fish
        self.pos = (0.,0.,0.)
        
        # create agent database and write agent parameters 
        self.hdf = pd.HDFStore(os.path.join(model_dir,'%s.h5'%('agent_%s.h5'%(ID))))
        self.hdf['agent'] = pd.DataFrame.from_dict({'ID':self.ID,
                                                    'sex':self.sex,
                                                    'length':self.length,
                                                    'weight':self.weight,
                                                    'body_depth':self.body_depth})
        self.hdf.flush()
    
    def thrust (U,L,f):
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
        A = amplitude(L)
        V = wave(U)
        B = trail(L) 
        
        # Calculate thrust
        m = (np.pi * rho * B**2)/4.
        W = (f * A * np.pi)/1.414
        w = W * (1 - U/V)
            
        thrust = m * W * w * U - (m * w**2 * U)/(2. * np.cos(np.radians(theta)))
        
        return (thrust)
    
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
        #sol1 = -1 * np.sqrt(D*V**2*np.cos(np.radians(theta))/(A**2*B**2*U*np.pi**3*rho*(U - V)*(-0.062518880701972*U - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
        Hz = np.sqrt(self.swim_speed*V**2*np.cos(np.radians(theta))/(A**2*B**2*self.swim_speed*np.pi**3*rho*(self.swim_speed - V)*(-0.062518880701972*self.swim_speed - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
        
        return Hz

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
        self.hdf = pd.HDFStore(os.path.join(self.model_dir,self.model_name))
        
       
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
        
        print ("Exporting Rasters")

        # create raster properties
        driver = 'GTiff'
        width = elev_new.shape[1]
        height = elev_new.shape[0]
        count = 1
        crs = self.crs
        transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)
        #Affine.translation(np.min(pts[:,0]),np.max(pts[:,1])) * Affine.scale(1,1)

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

            
    # def agents(n, model_dir, basin):
    #     '''method that creates an agent'''
    #     agents = np.array([])
    #     for i in np.arange(0,n,1):
            
        
        
        
    
    
        

# create a simulation function     
    
