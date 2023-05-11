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
    
# create a sockeye agent 
def sockeye():
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
        self.hdf = pd.HDFStore(os.path.join(outputWS,'%s.h5'%('agent_%s.h5'%(ID))))
        self.hdf['agent'] = pd.DataFrame.from_dict({'ID':self.ID,
                                                    'sex':self.sex,
                                                    'length':self.length,
                                                    'weight':self.weight,
                                                    'body_depth':self.body_depth})
        self.hdf.flush()
        
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
        
    # 


    
    
        

# create a simulation function     
    
