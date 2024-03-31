# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:46:19 2024

@author: Kevin.Nebiolo
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.constants
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import LinearRing
import networkx as nx
from scipy.integrate import odeint
from scipy import interpolate
import sqlite3
import os
from osgeo import ogr
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon as matplotPolygon
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.collections import PatchCollection
import scipy.constants
import time
import pickle
import copy
from scipy import interpolate
import networkx as nx
from osgeo import ogr
import fiona
import h5py
import tables

G = scipy.constants.G

class simulation:
    """Python class object for an agent-based model for navigation."""

    def __init__(self, n_agents, vessels, starting_locs, ending_locs, obstacles):
        self.n_agents = n_agents
        self.vessels = [self.Container(vessel, starting_loc, ending_loc) for vessel, starting_loc, ending_loc in zip(vessels, starting_locs, ending_locs)]
        # You might need to include logic to handle obstacles as well

    class container:
        """Class object for a container vessel."""

        def __init__(self, simulation, mask, starting_loc, ending_loc, t, dt, x, ui):
            self.mask = mask
            self.location = starting_loc
            self.destination = ending_loc
            self.t = t
            self.dt = dt

            #x and ui are state variables that change with time, these are initial values
            
            # Normalization variables
            self.L = 175                     # length of ship (m)
            self.U = np.sqrt(x[0]^2 + x[1]^2)   # service speed (m/s)
            
            # Check service speed
            if self.U <= 0:
                raise ValueError("The ship must have speed greater than zero.")

            if x(10) <= 0:
                raise ValueError('The propeller rpm must be greater than zero')
            
            self.delta_max = np.repeat(np.radians(10.),simulation.n_agents) # max rudder angle (deg)
            self.Ddelta_max = np.repeat(np.radians(5.),simulation.n_agents) # max rudder rate (deg/s)
            self.n_max = np.repeat(np.radians(160.),simulation.n_agents) # max shaft velocity (rpm)
            
            # Non-dimensional states and inputs
            self.delta_c = ui[0] 
            self.n_c = ui[1]/60*self.L/self.U
                        
            # Parameters, hydrodynamic derivatives and main dimensions
            self.m = np.repeat(0.00792, simulation.n_agents)    
            self.mx = np.repeat(0.000238, simulation.n_agents)     
            self.my = np.repeat(0.007049, simulation.n_agents)  
            self.Ix = np.repeat(0.0000176, simulation.n_agents)  
            self.alphay = np.repeat(0.05, simulation.n_agents)        
            self.lx = np.repeat(0.0313, simulation.n_agents)  
            self.ly = np.repeat(0.0313, simulation.n_agents)   
            self.Ix = np.repeat(0.0000176, simulation.n_agents)  
            self.Iz = np.repeat(0.000456, simulation.n_agents)  
            self.Jx = np.repeat(0.0000034, simulation.n_agents)  
            self.Jz = np.repeat(0.000419, simulation.n_agents)  
            self.xG = np.repeat(0, simulation.n_agents)  

            self.B = np.repeat(25.40, simulation.n_agents)  
            self.dF = np.repeat(8.00, simulation.n_agents)  
            self.g = np.repeat(9.81, simulation.n_agents)  
            self.dA = np.repeat(9.00, simulation.n_agents)  
            self.d = np.repeat(8.50, simulation.n_agents)   
            self.nabla = np.repeat(21222, simulation.n_agents)  
            self.KM = np.repeat(10.39, simulation.n_agents)     
            self.KB = np.repeat(4.6154, simulation.n_agents)    
            self.AR = np.repeat(33.0376, simulation.n_agents)  
            self.Delta = np.repeat(1.8219, simulation.n_agents)    
            self.D  = np.repeat(6.533, simulation.n_agents)     
            self.GM = np.repeat(0.3/self.L, simulation.n_agents)  
            self.rho = np.repeat(1025, simulation.n_agents)     
            self.t = np.repeat(0.175, simulation.n_agents)     
            self.T = np.repeat(0.0005, simulation.n_agents)   
             
            self.W = np.repeat(self.rho*self.g*self.nabla/(self.rho*self.L^2*self.U^2/2), simulation.n_agents)  

            self.Xuu = np.repeat(-0.0004226, simulation.n_agents)    
            self.Xvr = np.repeat(-0.00311, simulation.n_agents)      
            self.Xrr = np.repeat(0.00020, simulation.n_agents)   
            self.Xphiphi = np.repeat(-0.00020, simulation.n_agents)      
            self.Xvv = np.repeat(-0.00386, simulation.n_agents)  

            self.Kv = np.repeat(0.0003026, simulation.n_agents)    
            self.Kr = np.repeat(-0.000063, simulation.n_agents)   
            self.Kp = np.repeat(-0.0000075, simulation.n_agents)  
            self.Kphi = np.repeat(-0.000021, simulation.n_agents)   
            self.Kvvv = np.repeat( 0.002843, simulation.n_agents)  
            self.Krrr = np.repeat(-0.0000462, simulation.n_agents)  
            self.Kvvr = np.repeat(-0.000588, simulation.n_agents)   
            self.Kvrr = np.repeat(0.0010565, simulation.n_agents)  
            self.Kvvphi = np.repeat(-0.0012012, simulation.n_agents)  
            self.Kvphiphi = np.repeat(-0.0000793, simulation.n_agents)  
            self.Krrphi = np.repeat(-0.000243, simulation.n_agents)  
            self.Krphiphi = np.repeat(0.00003569, simulation.n_agents)  

            self.Yv = np.repeat(-0.0116, simulation.n_agents)  
            self.Yr = np.repeat(0.00242, simulation.n_agents)  
            self.Yp = np.repeat(0, simulation.n_agents)  
            self.Yphi = np.repeat(-0.000063, simulation.n_agents)  
            self.Yvvv = np.repeat(-0.109, simulation.n_agents)  
            self.Yrrr = np.repeat(0.00177, simulation.n_agents)  
            self.Yvvr = np.repeat(0.0214, simulation.n_agents)  
            self.Yvrr = np.repeat(-0.0405, simulation.n_agents)  
            self.Yvvphi = np.repeat(0.04605, simulation.n_agents)  
            self.Yvphiphi = np.repeat(0.00304, simulation.n_agents)   
            self.Yrrphi = np.repeat(0.009325, simulation.n_agents)  
    
            self.Yrphiphi = np.repeat(-0.001368, simulation.n_agents)  
            self.Nv = np.repeat(-0.0038545, simulation.n_agents)   
            self.Nr = np.repeat(-0.00222, simulation.n_agents)     
            self.Np = np.repeat(0.000213, simulation.n_agents)  
            self.Nphi = np.repeat(-0.0001424, simulation.n_agents)   
            self.Nvvv = np.repeat(0.001492, simulation.n_agents)  
            self.Nrrr = np.repeat(-0.00229, simulation.n_agents)  
            self.Nvvr = np.repeat(-0.0424, simulation.n_agents)  
            self.Nvrr  = np.repeat(0.00156, simulation.n_agents)   
            self.Nvvphi = np.repeat(-0.019058, simulation.n_agents)  
            self.Nvphiphi = np.repeat(-0.0053766, simulation.n_agents)   
            self.Nrrphi = np.repeat(-0.0038592, simulation.n_agents)  
            self.Nrphiphi = np.repeat(0.0024195, simulation.n_agents)  

            self.kk = np.repeat(0.631, simulation.n_agents)  
            self.epsilon = np.repeat(0.921, simulation.n_agents)  
            self.xR = np.repeat(-0.5, simulation.n_agents)  
            self.wp = np.repeat(0.184, simulation.n_agents)  
            self.tau = np.repeat(1.09 , simulation.n_agents)  
            self.xp = np.repeat(-0.526, simulation.n_agents)  
            self.cpv = np.repeat(0.0, simulation.n_agents)  
            self.cpr = np.repeat(0.0, simulation.n_agents)  
            self.ga =  np.repeat(0.088, simulation.n_agents)  
            self.cRr = np.repeat(-0.156, simulation.n_agents)  
            self.cRrrr = np.repeat(-0.275, simulation.n_agents)  
            self.cRrrv = np.repeat(1.96, simulation.n_agents)  
            self.cRX = np.repeat(0.71, simulation.n_agents)  
            self.aH = np.repeat(0.237, simulation.n_agents)  
            self.zR = np.repeat(0.033, simulation.n_agents)  
            self.xH = np.repeat(-0.48, simulation.n_agents)
                                
            # Masses and moments of inertia
            self.m11 = (self.m+self.mx)
            self.m22 = (self.m+self.my)
            self.m32 = -1 * self.my*self.ly
            self.m42 = self.my*self.alphay
            self.m33 = (self.Ix+self.Jx)
            self.m44 = (self.Iz+self.Jz)

        def update_position(self, x, ui):
            if (len(x) != 10):
                raise ValueError('x-vector must have dimension 10 !')
            
            if (len(ui) != 2):
                raise ValueError('u-vector must have dimension  2 !')
            
            # Include additional initializations as needed
            u = x[0]/self.U            # surge velocity          (m/s)
            v = x[1]/self.U            # sway velocity           (m/s)
            r = x[2]*self.L/self.U     # yaw velocity            (rad/s)
            x = x[3]                   # position in x-direction (m)
            y = x[4]                   # position in y-direction (m)
            psi = x[5]                  # yaw angle               (rad)
            p = x[6]*self.L/self.U     # roll velocity           (rad/s)
            phi = x[7]                 # roll angle              (rad)
            delta = x[8]               # actual rudder angle     (rad)
            n = x[9]/60.*self.L/self.U # actual shaft velocity   (rpm)
            
            # Rudder dynamics
            delta_c = self.delta_c  # Commanded rudder angle from the class attribute
            # Calculate the rate of change of the rudder angle, with a limit on the maximum rate
            delta_dot = np.clip(delta_c - delta, -self.Ddelta_max, self.Ddelta_max)
            
            # Shaft velocity saturation and dynamics
            self.n_c = self.n_c*self.U/self.L
            n = n*self.U/self.L
            self.n_c = np.where(np.abs(self.n_c) >= self.n_max/60.,
                                np.sign(self.n_c)*self.n_max/60,
                                self.n_c)
            self.Tm = np.where(n > 0.3,
                               5.65/n,
                               18.83)
            self.n_dot = 1/self.Tm*(self.n_c-n)*60

            # Calculation of state derivatives
            vR = self.ga*v + self.cRr*self.r + self.cRrrr*r^3 + self.cRrrv*r^2*v
            uP = u * ( (1 - self.wp) + self.tau*((v + self.xp*r)^2 + self.cpv*v + self.cpr*r) )
            J = uP*self.U/(n*self.D)
            KT = 0.527 - 0.455*J 
            uR = uP*self.epsilon*np.sqrt(1 + 8*self.kk*KT/(psi*J^2))
            alphaR = delta + np.atan(self.vR/self.uR)
            FN = - ((6.13*self.Delta)/(self.Delta + 2.25))*(self.AR/self.L^2)*(uR^2 + vR^2)*np.sin(alphaR)
            T = 2*self.rho*self.D^4/(self.U^2*self.L^2*self.rho)*self.KT*n*np.abs(n)

            # Forces and moments
            X = self.Xuu*u^2 + (1-self.t)*T + self.Xvr*v*r + self.Xvv*v^2 + \
                self.Xrr*r^2 + self.Xphiphi*phi^2 + self.cRX*self.FN*np.sin(delta) +\
                    (self.m + self.my)*v*r
              
            Y = self.Yv*v + self.Yr*r + self.Yp*p + self.Yphi*phi + self.Yvvv*v^3 +\
                self.Yrrr*r^3 + self.Yvvr*v^2*r + self.Yvrr*v*r^2 + self.Yvvphi*v^2*phi +\
                    self.Yvphiphi*v*phi^2 + self.Yrrphi*r^2*phi + \
                        self.Yrphiphi*r*phi^2 + (1 + self.aH)*FN*np.cos(delta) -\
                            (self.m + self.mx)*u*r

            K = self.Kv*v + self.Kr*r + self.Kp*p + self.Kphi*phi + self.Kvvv*v^3 + \
                self.Krrr*r^3 + self.Kvvr*v^2*r + self.Kvrr*v*r^2 + self.Kvvphi*v^2*phi +\
                    self.Kvphiphi*v*phi^2 + self.Krrphi*r^2*phi + self.Krphiphi*r*phi^2 -\
                        (1 + self.aH)*self.zR*FN*np.cos(delta) + self.mx*self.lx*u*r - self.W*self.GM*phi

            N = self.Nv*v + self.Nr*r + self.Np*p + self.Nphi*phi + self.Nvvv*v^3 + \
                self.Nrrr*r^3 + self.Nvvr*v^2*r + self.Nvrr*v*r^2 + self.Nvvphi*v^2*phi +\
                    self.Nvphiphi*v*phi^2 + self.Nrrphi*r^2*phi + self.Nrphiphi*r*phi^2 +\
                        (self.xR + self.aH*self.xH)*FN*np.cos(delta)

            # Dimensional state derivatives  xdot = [ u v r x y psi p phi delta n ]'
            detM = self.m22*self.m33*self.m44-self.m32^2*self.m44-self.m42^2*self.m33

            xdot =np.array([X*(self.U^2/self.L)/self.m11,
                            -((-self.m33*self.m44*Y+self.m32*self.m44*K+self.m42*self.m33*N)/detM)*(self.U^2/self.L),
                            ((-self.m42*self.m33*Y+self.m32*self.m42*K+N*self.m22*self.m33-N*self.m32^2)/detM)*(self.U^2/self.L^2),
                            (np.cos(psi)*u-np.sin(psi)*np.cos(phi)*v)*self.U,
                            (np.sin(psi)*u+np.cos(psi)*np.cos(phi)*v)*self.U,
                            np.cos(phi)*r*(self.U/self.L),
                            ((-self.m32*self.m44*Y+K*self.m22*self.m44-K*self.m42^2+self.m32*self.m42*N)/detM)*(self.U^2/self.L^2),
                            p*(self.U/self.L),
                            delta_dot,
                            self.n_dot])
    
    def create_ship_polygons(ship_positions, ship_orientations, ship_shape):
        """
        Create polygons for multiple ships using vectorized operations.
        
        :param ship_positions: An Nx2 numpy array of ship positions, where each row is (x, y).
        :param ship_orientations: An N-element numpy array of ship orientations in radians.
        :param ship_shape: An Mx2 numpy array of the ship's shape points relative to the center.
        :return: A list of Shapely Polygon objects representing the ships.
        """
        # Number of ships
        n_ships = ship_positions.shape[0]
        
        # Create rotation matrices for each ship
        cos_angles = np.cos(ship_orientations)
        sin_angles = np.sin(ship_orientations)
        rotation_matrices = np.array([[cos_angles, -sin_angles], [sin_angles, cos_angles]]).transpose(2, 0, 1)
        
        # Rotate ship shapes
        rotated_shapes = np.einsum('ijk,kl->ijl', rotation_matrices, ship_shape)
        
        # Translate shapes to ship positions
        final_shapes = rotated_shapes + ship_positions[:, np.newaxis, :]
        
        # Create polygons
        polygons = [Polygon(shape) for shape in final_shapes]
        
        return polygons

    def check_collision(ship_polygons):
        
        # Assuming `ship_polygons` is your list of Shapely Polygon objects
        gdf = gpd.GeoDataFrame(geometry=ship_polygons)
        
        # The spatial index is automatically created. Now, let's use it for a broad-phase collision check.
        # We'll create a list to store pairs of indices that might be intersecting
        potential_intersections = []
        
        for i, ship in gdf.iterrows():
            # Possible candidates for intersection are identified using the spatial index
            possible_matches_index = gdf.sindex.query(ship.geometry, predicate="intersects")
            possible_matches = gdf.iloc[possible_matches_index]
            # Detailed intersection checks are performed only on the candidates
            precise_matches = possible_matches[possible_matches.intersects(ship.geometry)]
            for j, _ in precise_matches.iterrows():
                if i != j:  # Ensure we don't compare the ship with itself
                    potential_intersections.append((i, j))
        
        # Remove duplicates from the list, as (i, j) and (j, i) are the same for our purposes
        unique_intersections = list(set(tuple(sorted(pair)) for pair in potential_intersections))
        
        return unique_intersections
    
