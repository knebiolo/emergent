# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:30:21 2023

@author: KNebiolo, Isha Deo

Python software for an Agent Based Model of migrating adult Sockeye salmon (spp.)  
with intent of understanding the potential ramifications of river discharge 
changes on ability of fish to succesffuly pass upstream through a riffle - 
cascade complex.  

"""
# import dependencies
import h5py
import pandas as pd
import os
import geopandas as gpd
import shapely
import rasterio
import numpy as np
from scipy import interpolate
    
# create a sockeye agent 
class sockeye():
    ''' Python class object for a sockeye agent. 
    
    Class object contains all of the sockeye's attributes, while the methods
    control how the sockeye agent senses its environment, reacts with changing 
    flow conditions, and interacts with other agents.  These methods, which are 
    in reality simple behavioral rules, will lead to complex, self organizing
    behavior.
    '''
    
    def __init__(self, ID, ws, basin):
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
        self.heading = 0.
        self.drag_coef = 0.
        self.drag = 0.
        self.thrust = 0.
        
        # initialize the odometer
        self.kcal = 0.
        
        # position the fish
        self.pos = (0.,0.,0.)
        
        # create agent database and write agent parameters 
        self.hdf = pd.HDFStore(os.path.join(ws,'%s.h5'%('agent_%s.h5'%(ID))))
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
        amplitude = interpolate.UnivariateSpline(length_dat,amp_dat,k = 2) 
        wave = interpolate.UnivariateSpline(speed_dat,wave_dat,k = 1) 
        trail = interpolate.UnivariateSpline(length_dat,edge_dat,k = 1) 
        
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
    
    def frequency (U,L,D):
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
        amplitude = interpolate.UnivariateSpline(length_dat,amp_dat,k = 2) 
        wave = interpolate.UnivariateSpline(speed_dat,wave_dat,k = 1) 
        trail = interpolate.UnivariateSpline(length_dat,edge_dat,k = 1) 
        
        # interpolate A, V, B
        A = amplitude(L)
        V = wave(U)
        B = trail(L)    
        
        # now that we have all variables, solve for f
        #sol1 = -1 * np.sqrt(D*V**2*np.cos(np.radians(theta))/(A**2*B**2*U*np.pi**3*rho*(U - V)*(-0.062518880701972*U - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
        Hz = np.sqrt(D*V**2*np.cos(np.radians(theta))/(A**2*B**2*U*np.pi**3*rho*(U - V)*(-0.062518880701972*U - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
        
        return Hz


    
    
        

# create a simulation function     
    
