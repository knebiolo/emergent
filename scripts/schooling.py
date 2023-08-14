# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:47:53 2023

@author: Isha Deo

This script determines the best vector response to maintain a fish's position within a school. 

"""

# assuming fish dataframe similar to the following:
    # build data frame
    # dataframe = {'profileNo':profiles,
    #              'shipTypes':shipTypes,
    #              'Tprime':Tprime,
    #              'Kprime':Kprime,
    #              'L':L,
    #              'B':B,
    #              'T':T,
    #              'DWT':DWT,
    #              'V_des':V_des,
    #              'v0':v0,
    #              'change':change,
    #              't-start':t_start,
    #              'origin':origin,
    #              'destination':destination}
    # df = pd.DataFrame.from_dict(dataframe,orient = 'columns')
    
#%% import packages

import pandas as pd
import numpy as np
import rasterio
import rasterio.plot
from shapely.geometry import Point, Polygon
import shapely.affinity as affinity
import geopandas
import matplotlib.pyplot as plt
import os
import math

#%% set up inputs
# workspaces
inputWS = r'J:\2819\005\Calcs\ABM\Data'
outputWS = r'J:\2819\005\Calcs\ABM\Output'

plt.rcParams['figure.dpi'] = 300
#%% misc functions

def calc_bearing(lat1, long1, lat2, long2):
  # Convert latitude and longitude to radians
  lat1 = np.radians(lat1)
  long1 = np.radians(long1)
  lat2 = np.radians(lat2)
  long2 = np.radians(long2)
  
  # Calculate the bearing
  bearing = np.arctan2(
      np.sin(long2 - long1) * np.cos(lat2),
      np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long2 - long1)
  )
  
  # Convert the bearing to degrees
  bearing = math.degrees(bearing)
  
  # Make sure the bearing is positive
  bearing = (bearing + 360) % 360
  
  return bearing

bearing_vec = np.vectorize(calc_bearing)

# code to update fish position
def one_timestep():
    fish_xs = fish_gdf['Longitude'] + (fish_gdf['speed'] * np.cos(fish_gdf['heading']))
    fish_ys = fish_gdf['Latitude'] + (fish_gdf['speed'] * np.sin(fish_gdf['heading']))
    
    fish_gdf.set_geometry(geopandas.points_from_xy(x=fish_xs, y=fish_ys))

#%% build dummy fish dataframe
n = 100
fish_lengths = np.random.uniform(low = 0.1, high = 1, size = n)
fish_xs = np.random.uniform(low = 549398, high = 549521, size = n)
fish_ys = np.random.uniform(low = 6641780, high = 6641807, size = n)

fish_speed = np.random.uniform(low = 0, high = 3, size = n)
fish_heading = np.random.uniform(low = 0, high = 2*math.pi, size = n)

fish_df_dict = {'length': fish_lengths,
                'Longitude': fish_xs,
                'Latitude': fish_ys,
                'speed': fish_speed,
                'heading': fish_heading}

# create regular dataframe
fish_df = pd.DataFrame.from_dict(fish_df_dict)

# create geodataframe and plot it
fish_gdf = geopandas.GeoDataFrame(fish_df, geometry=geopandas.points_from_xy(x=fish_df['Longitude'], y=fish_df['Latitude']))
fish_gdf.set_crs(epsg='3473',inplace=True)

fig, ax = plt.subplots(figsize = (10,10))

fish_gdf.plot(ax=ax, color = 'white', zorder = 0)
fish_xmin, fish_xmax = ax.get_xlim()
fish_ymin, fish_ymax = ax.get_ylim()

elev_raster = rasterio.open(os.path.join(inputWS, 'elev.tif'))
rasterio.plot.show(elev_raster, ax = ax, zorder = 1, cmap = 'gist_earth')
ax.set_xlim(fish_xmin, fish_xmax)
ax.set_ylim(fish_ymin, fish_ymax)
 
for i, row in fish_gdf.iterrows():
    ax.arrow(x = row['Longitude'], y = row['Latitude'], dx = row['length'] * math.cos(row['heading']), dy = row['length'] * math.sin(row['heading']), width = 0.1, head_width = 0.4, zorder = 2, fc = 'white', ec = 'k',length_includes_head = True)

#%% plot and run the schooling analysis

# plot for this iteration
fig2, ax2 = plt.subplots(figsize = (10,10))

fish_gdf.plot(ax=ax2, color = 'white', zorder = 0)
fish_xmin, fish_xmax = ax2.get_xlim()
fish_ymin, fish_ymax = ax2.get_ylim()

elev_raster = rasterio.open(os.path.join(inputWS, 'elev.tif'))
rasterio.plot.show(elev_raster, ax = ax2, zorder = 0, cmap = 'gist_earth')
 
for i, row in fish_gdf.iterrows():
    ax2.arrow(x = row['Longitude'], y = row['Latitude'], dx = row['length'] * math.cos(row['heading']), dy = row['length'] * math.sin(row['heading']), width = 0.1, head_width = 0.4, zorder = 3, fc = 'white', ec = 'k',length_includes_head = True)

# and run the schooling analysis
for fish_i in range(0,100):
    # find local agents within 2 body lengths
    buffer_poly = fish_gdf.at[fish_i,'geometry'].buffer(distance = 2*fish_gdf.at[fish_i,'length'])
    
    ax2.plot(*buffer_poly.exterior.xy, color = 'g', zorder = 3)
    
    nearbyfish_ser = fish_gdf.intersection(buffer_poly)
    nearbyfish_df = fish_gdf[~nearbyfish_ser.is_empty].drop(fish_i)
    
    
    # find average speed and heading of nearby fish
    fish_gdf.at[fish_i, 'speed'] = nearbyfish_df.mean(numeric_only = True)['speed']
    
    # find centroid of nearby fish
    cent_x, cent_y = nearbyfish_df[['Longitude','Latitude']].mean(axis = 0)
    
    # if fish is TOO close to another fish, MOVE
    # need to figure out what timestep to do this on - once actually boinked or when boinking in next time step?
    
    
    
    # if fish is within 1 body length of nearby fish centroid, just move in same direction as nearby fish
    if fish_gdf.at[fish_i, 'geometry'].distance(shapely.geometry.Point(cent_x, cent_y)) < fish_gdf.at[fish_i, 'length']:
        fish_gdf.at[fish_i, 'heading'] = nearbyfish_df.mean(numeric_only = True)['heading']
        
    # else, move towards the centroid of the fish
    else:
        fish_gdf.at[fish_i, 'heading'] = calc_bearing(lat1 = cent_y, long1 = cent_x, lat2 = fish_gdf.at[fish_i, 'geometry'].y, long2 = fish_gdf.at[fish_i, 'geometry'].x)
    
    #nearbyfish_poly = shapely.geometry.Polygon()
    ax2.arrow(x = fish_gdf.at[fish_i,'Longitude'], y = fish_gdf.at[fish_i,'Latitude'], dx = fish_gdf.at[fish_i,'length'] * math.cos(fish_gdf.at[fish_i,'heading']), dy = fish_gdf.at[fish_i,'length'] * math.sin(fish_gdf.at[fish_i,'heading']), width = 0.1, head_width = 0.4, zorder = 3, fc = 'red', ec = 'red',length_includes_head = True)

ax2.set_xlim(fish_xmin, fish_xmax)
ax2.set_ylim(fish_ymin, fish_ymax)


#%% schooling take two with collision avoidance, velocity matching, flock centering

repul_dist = 0.5 # x length
max_att_dist = 4 # x length

for fish_i in range(0,10):
    # collision avoidance
    
    buffer_poly = fish_gdf.at[fish_i,'geometry'].buffer(distance = max_att_dist*fish_gdf.at[fish_i,'length'])
    
    nearbyfish_ser = fish_gdf.intersection(buffer_poly)
    nearbyfish_df = fish_gdf[~nearbyfish_ser.is_empty].drop(fish_i)
    
    if not nearbyfish_df.empty:
        
        # calculate distance from fish to each nearby fish & attractive and repulsive force weights
        # distance is weighted by fish length
        nearbyfish_df['distance'] = [fish_gdf.at[fish_i, 'geometry'].distance(x)/fish_gdf.at[fish_i,'length'] for x in nearbyfish_df['geometry']]
        nearbyfish_df['bearing'] = calc_bearing(lat1 = fish_gdf.at[fish_i, 'geometry'].y, long1 = fish_gdf.at[fish_i, 'geometry'].x, lat2 = nearbyfish_df['geometry'].y, long2 = nearbyfish_df['geometry'].x)
        
        nearbyfish_df['attractive force strength factor'] = 1/(nearbyfish_df['distance'])
        nearbyfish_df['repulsive force strength factor'] = -1/(nearbyfish_df['distance']**2) # need to adjust weights based on sensitivity
    
        nearbyfish_df['net strength factor'] = nearbyfish_df['bearing']*nearbyfish_df['attractive force strength factor'] 
        + ((nearbyfish_df['bearing'] + 180) % 360)*nearbyfish_df['repulsive force strength factor']
    
    


    print('done')
    
#%% arbitration function try

def school_cue(fish_i, weight, buffer_poly):
    
    nearbyfish_ser = fish_gdf.intersection(buffer_poly)
    nearbyfish_df = fish_gdf[~nearbyfish_ser.is_empty].drop(fish_i)
    
    if not nearbyfish_df.empty:
        # find average speed and heading of nearby fish
        fish_gdf.at[fish_i, 'speed'] = nearbyfish_df.mean(numeric_only = True)['speed']
        
        # find centroid of nearby fish
        cent_x, cent_y = nearbyfish_df[['Longitude','Latitude']].mean(axis = 0)
        
        # distance to centroid of nearby fish
        cent_dist = fish_gdf.at[fish_i, 'geometry'].distance(shapely.geometry.Point(cent_x, cent_y))
        
        # vector of nearby fish centroid position relative to position of fish
        v = np.array([cent_x, cent_y] - np.array(fish_gdf.loc[fish_i, ['Longitude','Latitude']]))
        
        # unit vector
        v_hat = v/np.linalg.norm(v)
        
        school_cue = (weight * v_hat)/(cent_dist)
        
        return school_cue
    else:
        return [0.,0.]

def collision_cue(fish_i, weight, buffer_poly):
    

    nearbyfish_ser = fish_gdf.intersection(buffer_poly)
    nearbyfish_df = fish_gdf[~nearbyfish_ser.is_empty].drop(fish_i)
    
    nearbyfish_df['distance'] = [fish_gdf.at[fish_i, 'geometry'].distance(x)/fish_gdf.at[fish_i,'length'] for x in nearbyfish_df['geometry']]
    
    if not nearbyfish_df.empty:
        # find closest nearby fish
        closest_fish = nearbyfish_df[nearbyfish_df['distance'] == nearbyfish_df['distance'].min()]
        
        # vector of closest fish position relative to position of fish
        c = np.array(np.array([closest_fish['Longitude'].values[0], closest_fish['Latitude'].values[0]]) - np.array(fish_gdf.loc[fish_i, ['Longitude','Latitude']]))
        
        # unit vector
        c_hat = -1*c/np.linalg.norm(c)
        
        collision_cue = (weight * c_hat)/(closest_fish['distance'].values[0]**2)
    
        return collision_cue
    else:
        return [0.,0.]
    
def one_timestep():
    fish_xs = fish_gdf['Longitude'] + (fish_gdf['speed'] * np.cos(fish_gdf['heading']))
    fish_ys = fish_gdf['Latitude'] + (fish_gdf['speed'] * np.sin(fish_gdf['heading']))
    
    fish_gdf.set_geometry(geopandas.points_from_xy(x=fish_xs, y=fish_ys))

def arbitrate(fish_i, buffer):
    
    collision = collision_cue(fish_i, 2000, buffer)
    school = school_cue(fish_i, 1000, buffer)
    
    collision_n = np.linalg.norm(collision)
    school_n = np.linalg.norm(school)
    
    if collision_n > 5000:
        head_vec = collision
    else:
        head_vec = school

    return head_vec


#%% run arbitration schooling

fig2, ax2 = plt.subplots(figsize = (10,10))

fish_gdf.plot(ax=ax2, color = 'white', zorder = 0)
fish_xmin, fish_xmax = ax2.get_xlim()
fish_ymin, fish_ymax = ax2.get_ylim()

elev_raster = rasterio.open(os.path.join(inputWS, 'elev.tif'))
rasterio.plot.show(elev_raster, ax = ax2, zorder = 0, cmap = 'gist_earth')
 
for i, row in fish_gdf.iterrows():
    ax2.arrow(x = row['Longitude'], y = row['Latitude'], dx = row['length'] * math.cos(row['heading']), dy = row['length'] * math.sin(row['heading']), width = 0.1, head_width = 0.4, zorder = 3, fc = 'white', ec = 'k',length_includes_head = True)

# and run the schooling analysis
for fish_i in range(0,100):
    
    #buffer = fish_gdf.at[fish_i,'geometry'].buffer(distance = 2*fish_gdf.at[fish_i,'length'])
    
    # create sensory buffer
    l = (fish_gdf.at[fish_i,'length']) * 2.
            
    # create wedge looking in front of fish 
    theta = np.radians(np.linspace(-120,120,100))
    arc_x = fish_gdf.at[fish_i,'Longitude'] + l * np.cos(theta)
    arc_y = fish_gdf.at[fish_i,'Latitude'] + l * np.sin(theta)
    arc_x = np.insert(arc_x,0,fish_gdf.at[fish_i,'Longitude'])
    arc_y = np.insert(arc_y,0,fish_gdf.at[fish_i,'Latitude'])
    arc = np.column_stack([arc_x, arc_y])
    arc = Polygon(arc)
    arc_rot = affinity.rotate(arc,np.degrees(fish_gdf.at[fish_i,'heading']), origin = (fish_gdf.at[fish_i,'Longitude'],fish_gdf.at[fish_i,'Latitude']))    
        
    arc_gdf = geopandas.GeoDataFrame(index = [0],
                               crs = 'EPSG:3473',
                               geometry = [arc_rot])
    
    ax2.plot(*arc.exterior.xy, color = 'g', zorder = 3)
    
    head_vec = arbitrate(fish_i, arc)
    fish_gdf.at[fish_i, 'heading'] = np.arctan2(head_vec[1],head_vec[0])
   
one_timestep()
for i, row in fish_gdf.iterrows():
    ax2.arrow(x = row['Longitude'], y = row['Latitude'], dx = row['length'] * math.cos(row['heading']), dy = row['length'] * math.sin(row['heading']), width = 0.1, head_width = 0.4, zorder = 2, fc = 'r', ec = 'r',length_includes_head = True)

ax2.set_xlim(fish_xmin, fish_xmax)
ax2.set_ylim(fish_ymin, fish_ymax)


        
    
    