# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:15:24 2023

@author: KNebiolo

Script Intent: Extract raster grid of values from 2D HECRAS Model
"""
# declare dependencies
import h5py
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# where are we going to put these rasters when we are done with them?
outputWS = r"J:\2819\005\Calcs\ABM\Output"

#%% Connect to HECRAS Model and Extract Data

# connect to a HECRAS Model
print ("Connecting to HECRAS Models")
inputWS = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"
name = 'NuyakukABM2D.p02.hdf'
hdf = h5py.File(os.path.join(inputWS,name),'r')

# Extract Data from HECRAS HDF model
print ("Extracting Model Geometry and Results")
pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))

vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-2]
vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-2]
wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-2]
elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))

#%% Create shapefile of observations and export

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
gdf = gpd.GeoDataFrame(df,crs = 'EPSG:32604')

# remove the tuple column cuz shapefiles are babies
gdf.drop(axis = 1, columns = 'geom_tup', inplace = True)

# export for science
gdf.to_file(os.path.join(outputWS,'model_output.shp'))

#%% Create Interpolators for velocity, wsel and elevation
print ("Create multidimensional interpolator functions for velocity, wsel, elev")
vel_x_interp = LinearNDInterpolator(pts,gdf.vel_x)
vel_y_interp = LinearNDInterpolator(pts,gdf.vel_y)
wsel_interp = LinearNDInterpolator(pts,gdf.wsel)
elev_interp = LinearNDInterpolator(pts,gdf.elev)

#%% Interpolate Images

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
#wsel_new = wsel_interp(xnew, ynew)
print ("Interpolate bathymetry")
#elev_new = elev_interp(xnew, ynew)

# create a depth raster
#depth = wsel_new - elev_new

# calculate velocity magnitude
vel_mag = np.sqrt((np.power(vel_x_new,2)+np.power(vel_y_new,2)))

vel_dir = np.arctan2(vel_y_new,vel_x_new)


#%% Write Raster Files
print ("Exporting Rasters")

# create raster properties
driver = 'GTiff'
width = elev_new.shape[1]
height = elev_new.shape[0]
count = 1
dtype = 'float64'
crs = 'EPSG:32604'
transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)
#Affine.translation(np.min(pts[:,0]),np.max(pts[:,1])) * Affine.scale(1,1)

# write elev raster
with rasterio.open(os.path.join(outputWS,'elev.tif'),
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
with rasterio.open(os.path.join(outputWS,'wsel.tif'),
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
with rasterio.open(os.path.join(outputWS,'depth.tif'),
                   mode = 'w',
                   driver = driver,
                   width = width,
                   height = height,
                   count = count,
                   dtype = 'float64',
                   crs = crs,
                   transform = transform) as depth_rast:
    depth_rast.write(depth,1)

# write velocity x raster
with rasterio.open(os.path.join(outputWS,'vel_x.tif'),
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
with rasterio.open(os.path.join(outputWS,'vel_y.tif'),
                   mode = 'w',
                   driver = driver,
                   width = width,
                   height = height,
                   count = count,
                   dtype = 'float64',
                   crs = crs,
                   transform = transform) as vel_y_rast:
    vel_y_rast.write(vel_y_new,1)