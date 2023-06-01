# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:08:28 2023

@author: KNebiolo

Script Intent: can rasterio perform efficient search and masking of a large high 
resolution raster?  Let's find out
"""
# import dependencies
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import h5py
import pandas as pd
from shapely import Polygon
from shapely import Point
import geopandas as gpd
import os
import fiona
from rasterstats import zonal_stats
import numpy as np

# define workspace
inputWS = r"J:\2819\005\Calcs\ABM\Data"
outputWS = r"J:\2819\005\Calcs\ABM\Output"

#%% opening and visualizing data
rast = rasterio.open(os.path.join(outputWS,'depth.tif'))
show(rast)

#%% spatial indexing 
bounds = rast.bounds
height = bounds[3] - bounds[1]
width = bounds[2] - bounds[0]
ll = (bounds[0],bounds[1])
#x, y = (rast.bounds.left + width / 2., rast.bounds.bottom + height / 4.)
x, y = (549500,6641600)
row, col = rast.index(x, y)
elev = rast.read(1)[row, col]

#%% masking 
'''Fish sense the surrounding flow field using their lateral line, which consists
of a bimodal, distributed velocity and pressure sensitive array.  The active
sensory space of both submodalities is limited to a range of 2+ body lengths around 
the fish's position (Tuhtan 2018).

Therefore, the active flow field sensory space is a 2 body length buffer around 
the position of the fish.'''

#######################################
# Step 1: Develop Sensory Buffer      #
#######################################
# current position of fish
fish = Point(x, y)

# create sensory buffer
l = 75
sensory = fish.buffer(2 * l)

# make a geopandas geodataframe of sensory buffer
sense_gdf = gpd.GeoDataFrame(index = [0],crs = rast.crs, geometry = [sensory])

# create wedge looking in front of fish 
theta = np.radians(np.linspace(0,45,100))
arc_x = x + l * np.cos(theta)
arc_y = y + l * np.sin(theta)
arc_x = np.insert(arc_x,0,x)
arc_y = np.insert(arc_y,0,y)

arc = np.column_stack([arc_x, arc_y])
arc = Polygon(arc)

arc_gdf = gpd.GeoDataFrame(index = [0],crs = rast.crs, geometry = [arc])
##########################################
# Step 2: Mask Velocity Magnitude Surface
#########################################

# perform mask
masked = mask(rast,arc_gdf.loc[0],all_touched = True, crop = True)

##############################################
# Step 3: Get Cell Center of Highest Velocity
##############################################

# get mask origin
mask_x = masked[1][2]
mask_y = masked[1][5]

# get indices of cell in mask with highest elevation
zs = zonal_stats(arc_gdf, rast.read(1), affine = rast.transform, stats=['min', 'max'], all_touched = True)
idx = np.where(masked[0] == zs[0]['max'])

# compute position of max value
max_x = mask_x + idx[1][-1] * masked[1][0]
max_y = mask_y + idx[2][-1] * masked[1][4]
# create a point
max_elev = Point([max_x,max_y])   

##################################################
# Step 4: Get Direction (unit vector) To Goal Cell
##################################################

# vector of max velocity position relative to position of fish 
v = (np.array([max_x,max_y]) - np.array([x, y]))   
 
# unit vector                               
v_hat = v/np.linalg.norm(v) 

# visualize and check 
# make geopandas dataframe and plot
gdf_max = gpd.GeoDataFrame(index = [0],crs = rast.crs, geometry = [max_elev])
gdf_current = gpd.GeoDataFrame(index = [0],crs = rast.crs, geometry = [fish])

# Plot the Polygons on top of the DEM
base = arc_gdf.plot(facecolor='None', edgecolor='red', linewidth=1)
max_elev = gdf_max.plot(ax = base, marker = 'o', color = 'magenta', markersize = 12)
current = gdf_current.plot(ax = base, marker = 'o', color = 'green', markersize = 12)

# Plot DEM
show(rast, ax=base) 


#%% zonal statistics
zs = zonal_stats(sense_gdf, rast.read(1), affine = rast.transform, stats=['min', 'max', 'mean', 'median', 'majority'])



#%% writing to and reading from HDF5