"""
Created on Tue August 15 09:13:30 2023

@author: EMuhlestein

Script intent: Organize the .h5 files into a list to seperate the .h5 files from the hdf file.
Create histograms of the agents length, weight, and body depths as well as the maximum, minimum,
and standard deviation of all the lengths, weight, and body depths of all agents. Bout_no and Bout_durations
are calculated as well as plots of agent locations, individual agent locations, agent jump locations, 
and heat maps of agent locations and agent jump locations.
"""

#Import Dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.wkt import loads as loads
from matplotlib.patches import Rectangle
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform


#TODO initialization
#Identify the directory path
directory_path = r"J:\2819\005\Calcs\ABM\Output\test_29"
hdf_path = r"J:\2819\005\Calcs\ABM\Output\test_29\test_29.hdf"
tif_path=r'J:\2819\005\Calcs\ABM\Output\elev.tif'



#TODO initialization 
class Initialization:
    def __init__(self, directory_path, hdf_path, tif_path, hdf5_files, h5_files):
        self.directory_path = directory_path
        self.hdf_path = hdf_path
        self.tif_path = tif_path
        self.hdf5_files = []
        self.h5_files = []

    # Separate the .h5 and .hdf5 files into 2 separate lists
    def list_hdf5_h5_files(self):
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.hdf5'):
                self.hdf5_files.append(filename)
            elif filename.endswith('.h5'):
                self.h5_files.append(filename)

    # List the keys in .h5 files
    def list_keys_h5_files(self):
        for filename in self.h5_files:
            full_path_keys = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path_keys, mode='r') as store:
                keys = store.keys()
                print(f"Keys in the .h5 file '{filename}':")
                for key in keys:
                    print(key)

    # Make .h5 list unique
    def unique_h5_index(self):
        self.h5_files = list(set(self.h5_files))

    # Process HDF5 data (keys and their contents)
    def process_hdf5_data(self):
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            keys = store.keys()
            for key in keys:
                print(f"Contents of key '{key}':")
                data = pd.read_hdf(self.hdf_path, key)
                print(data)

    # Process the 'battery' key in .h5 files
    def process_h5_battery(self):
        for filename in self.h5_files:
            full_path_battery = os.path.join(self.directory_path, filename)
            try:
                with pd.HDFStore(full_path_battery, mode='r') as store:
                    battery_key = '/battery'
                    if battery_key in store:
                        print(f"Contents of '/battery' in '{filename}':")
                        battery_data = pd.read_hdf(full_path_battery, battery_key)
                        print(battery_data)
                    else:
                        print(f"Key '/battery' not found in '{filename}'")
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")



#TODO class Lengths
class lengths:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.h5_files = []
        self.weights = []
        
    #Find the .h5 files
    def find_h5_files(self):
        for filename in os.listdir(self.directory_path):
            if filename.endswith ('.h5'):
                self.h5_files.append(filename)
                
    #Collect length values
    def length_values(self):
        for filename in self.h5_files:
            full_path=os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key='/agent'
                if agent_key in store:
                    agent_data=pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)
                    if 'length' in agent_data.columns:
                        print(f"/agent keys in '{filename}':{agent_data.keys()}")
                        self.lengths.extend(agent_data['length'])
                        
    #Plot lengths
    def plot_lengths(self):
        if self.lengths:
            plt.hist(self.lengths, bins=20, edgecolor='black')
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            plt.title('Agent Lengths')
            plt.show()
        else:
            print("No 'length' values found for plotting.")
            
    #Calculate Statistics
    def length_statistics(self):
        if self.lengths:
            mean_length=sum(self.lengths) / len(self.lengths)
            median_length = sorted(self.lengths)[len(self.lengths) // 2]
            std_dev_length = (sum((x - mean_length) ** 2 for x in self.lengths) / len(self.lengths)) ** 0.5
            
            print(f"Average (Mean) Length: {mean_length}")
            print(f"Median Length: {median_length}")
            print(f"Standard Deviation of Length: {std_dev_length}")
        else:
            print("No 'length' calues found for statistics")
            
           

#TODO class weights
class weights:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.h5_files = []
        self.body_depths = []
        
    #Collect weights
    def collect_weights(self):
        for filename in self.h5_files:
            full_path=os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key= '/agent'
                if agent_key in store:
                    agent_data=pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)
                    if 'weight' in agent_data.columns:
                        self.weights.extend(agent_data['weight'])
                    else:
                        print(f"'weight' not found in '{filename}'")
                else:
                    print(f"Key '/agent' not found in '{filename}'")
                    
    #Plot weights
    def plot_weights(self):
        if self.weights:
            plt.hist(self.weights, bins=20, edgecolor='black')
            plt.xlabel=('Weight')
            plt.ylabel=('Frequency')
            plt.title=('Weights for All Agents')
            plt.show()
        else:
            print("No 'weight' values found for plotting")
            
            
    #Calculate Statistics
    def weight_statistics(self):
        if self.weights:
            mean_weight=np.mean(self.weights)
            median_weight=np.median(self.weights)
            std_dev_weight=np.std(self.weights)
            
            print(f"Average (Mean) Weight: {mean_weight}")
            print(f"Median Weight: {median_weight}")
            print(f"Standard Deviation of Weight: {std_dev_weight}")
        else:
            print("No 'weight' values found for statistics.")
            


#TODO class body_depths
class body_depths:
    def __init__(self, directory_path):
        self.body_depths=[]
        
    #Collect body_depths data
    def collect_body_depths(self):
        for filename in h5_files:
            full_path=os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key='/agent'
                if agent_key in store:
                    agent_data=pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)
                    if 'body_depth' in agent_data.columns:
                        self.body_depths.extend(agent_data['body_depth'])
                    else:
                        print(f"'body_depth' not found in '{filename}'")
                else:
                    print(f"Key '/agent' not found in '{filename}'")
                    
    #Plot body_depths
    def plot_body_depths(self):
        if self.body_depths:
            plt.hist(self.body_depths, bins=20, edgecolor='black')
            plt.xlabel=('Body Depth')
            plt.ylabel=('Frequency')
            plt.title=('Body Depths for All Agents')
        else:
            print("No 'body_depth' values found for plotting.")
            
    #Calculate Statistics
    def body_depth_statistics(self):
        if self.body_depths:
            mean_body_depth=np.mean(self.body_depths)
            median_body_depth=np.median(self.body_depths)
            std_dev_body_depth=np.std(self.body_depths)
            
            print(f"Average (Mean) Body Depth: {mean_body_depth}")
            print(f"Median Body Depth: {median_body_depth}")
            print(f"Standard Deviation of Body Depth: {std_dev_body_depth}")
        else:
            print("No 'body_depth' values found for statistics.")
    
 

#TODO Max Bout duration Class
class Max_Bout_Duration:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.h5_files = [filename for filename in os.listdir(self.directory_path) if filename.endswith('.h5')]
        
    #Find max bout duration
    def max_bout_duration(self):
        for filename in self.h5_files:
            full_path=os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                if '/battery' in store:
                    battery_data=pd.read_hdf(full_path, '/battery')
                    if 'bout_duration' in battery_data.columns:
                        max_bout_duration=battery_data['bout_duration'].max()
                        print(f"Maximum bout duration in '/battery' key for file {filename}: {max_bout_duration}")
                    else:
                        print(f"Column 'bout_duration' not found in '/battery' key for file {filename}.")
                else:
                    print(f"Key '/battery' not found in file {filename}.")



#TODO Max bout_no
class Max_Bout_No:
    def __init__(self, directory_path):
        self.directory_path=directory_path
        self.h5_files=[filename for filename in os.listdir(self.directory_path)if filename.endswith('.h5')]
        
    def max_bout_no(self):
        for filename in self.h5_files:
            full_path=os.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                if '/battery' in store:
                    battery_data=pd.read_hdf(full_path, '/battery')
                    if 'bout_no' in battery_data.columns:
                        max_bout_no=battery_data['bout_no'].max()
                        print(f"Maximum bout no in '/battery' key for file {filename}: {max_bout_no}")
                    else:
                        print(f"Column 'bout_no' not found in '/battery' key for file {filename}.")
                else:
                    print(f"Key '/battery' not found in file {filename}.")
                        


#TODO Max bout_no and max bout_duration DF
class Max_bout_no_duration_df:
    def __init__(self, directory_path):
        self.directory_path=directory_path
        self.h5_files = [f for f in os.listdir(self.directory_path) if f.endswith('.h5')]
        self.data_list=[]
        
    def max_bout_noduration(self):
        for filename in self.h5_files:
            full_path=os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                if '/battery' in store:
                    battery_data=pd.read_hdf(full_path, '/battery')
                    if 'bout_no' in battery_data.columns and 'bout_duration' in battery_data.columns and 'ID' in battery_data.columns:
                        max_values=battery_data.groupby('ID')[['bout_no', 'bout_duration']].max().reset_index()
                        for index, row in max_values.iterrows():
                            self.data_list.append({'ID': row['ID'], 'max_bout_no': row['bout_no'], 'max_bout_duration': row['bout_duration']})
                    else:
                        print(f"Columns 'ID', 'bout_no', or 'bout_duration' not found in '/battery' key for file {filename}.")
                else:
                    print(f"Key '/battery' not found in file {filename}.")
        return pd.DataFrame(self.data_list)

        

#TODO AgentBoutDF
class AgentBoutDF:
    def __init__(self, directory_path):
        self.directory_path=directory_path
        self.h5_files = [f for f in os.listdir(self.directory_path) if f.endswith('.h5')]
        self.bout_duration_list = []
    
    def agent_bout_df(self):
        for filename in self.h5_files:
            full_path = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                if '/battery' in store:
                    battery_data = pd.read_hdf(full_path, '/battery')
                    if 'bout_no' in battery_data.columns and 'bout_duration' in battery_data.columns and 'ID' in battery_data.columns:
                        extracted_data = battery_data[['ID', 'bout_no', 'bout_duration']]
                        self.bout_duration_list.extend(extracted_data.to_dict(orient='records'))
                    else:
                        print(f"Columns 'ID', 'bout_no', or 'bout_duration' not found in '/battery' key for file {filename}.")
                else:
                    print(f"Key '/battery' not found in file {filename}.")
        return pd.DataFrame(self.bout_duration_list)
    

    
#TODO Plot Agent w/ Rectangle
class Plot_Agent_W_Rectangle:
    def __init__(self, hdf_path):
        self.hdf_path = hdf_path
        self.ids_in_rectangle = set()
        
    def Agent_Plot_Rectangle(self):
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            keys = store.keys()
            for key in keys:
                print(f"Contents of key '{key}':")
                data = pd.read_hdf(self.hdf_path, key)

                # Grab coordinates from the dataframe
                data['geometry'] = data['loc'].apply(loads)

                # Create a GeoDataFrame
                gdf = gpd.GeoDataFrame(data, geometry='geometry')

                # Plot the GeoDataFrame
                plt.figure(figsize=(10, 8))
                ax = gdf.plot(marker='o', markersize=1, color='blue', alpha=0.5)

                # Calculate the y-coordinate range for the top 10% of the plot
                y_max = gdf.geometry.bounds['maxy'].max()
                y_min_top_10 = y_max - 0.1 * (y_max - gdf.geometry.bounds['miny'].min())

                # Create a rectangle to cover the top 10% of the plot
                rect = Rectangle((gdf.geometry.bounds['minx'].min(), y_min_top_10),
                                 gdf.geometry.bounds['maxx'].max() - gdf.geometry.bounds['minx'].min(),
                                 y_max - y_min_top_10, linewidth=2, edgecolor='red', fill=False)

                # Add the rectangle to the plot
                ax.add_patch(rect)

                # Filter the data within the rectangle for each unique ID
                unique_ids = data['id'].unique()
                for id in unique_ids:
                    data_for_id = data[data['id'] == id]
                    gdf_id = gpd.GeoDataFrame(data_for_id, geometry='geometry')

                    # Filter the data within the rectangle for the current ID
                    data_in_rectangle = gdf_id.cx[gdf_id.geometry.bounds['minx'].min():gdf_id.geometry.bounds['maxx'].max(),
                                                  y_min_top_10:y_max]

                    # Get the first timestep when the ID enters the rectangle
                    first_timestep = data_in_rectangle.head(1)
                    if not first_timestep.empty:
                        # Add the IDs of the data points within the rectangle to the set
                        self.ids_in_rectangle.add(id)

                        # Display the filtered data for the current ID and timestep
                        print(f"First timestep within the rectangle for ID {id}:")
                        print(first_timestep)

                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title("Agents Passing Through Falls")
                plt.show()

        # Print the IDs of data points within the rectangle
        print("Agents with data points within the rectangle:")
        print(self.ids_in_rectangle)
        


#TODO AgentVisualizer
class AgentLocationPlotter:
    def __init__(self, hdf_path, tif_path):
        self.hdf_path = hdf_path
        self.tif_path = tif_path

    def plot_agent_locations(self):
        # Read the contents of the '/TS' key
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            ts_key = '/TS'
            if ts_key in store:
                ts_data = pd.read_hdf(self.hdf_path, ts_key)
            else:
                raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

        # Extract 'id' and 'ts' columns
        id_ts_data = ts_data[['id', 'ts']]

        # Filter out duplicate rows based on 'id' to get unique IDs
        unique_ids = id_ts_data['id'].unique()

        # Plot unique ID timesteps based on 'loc' data
        for unique_id in unique_ids:
            # Filter data for the current unique ID
            filtered_data = id_ts_data[id_ts_data['id'] == unique_id]

            # Extract 'ts' values for the unique ID
            timesteps = filtered_data['ts'].tolist()

            # Extract 'loc' data for the unique ID
            loc_data = ts_data[ts_data['id'] == unique_id]['loc'].apply(lambda x: loads(x))
            x_coords = [loc.x for loc in loc_data]
            y_coords = [loc.y for loc in loc_data]

            # Plot the timesteps for the unique ID
            plt.plot(x_coords, y_coords, label=f'ID {unique_id}')

        # Add labels, legend, and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.title('Agent Locations Based Off Timesteps')

        # Show the plot
        plt.show()

    def reproject_tiff_image(self, output_path, desired_crs_epsg):
        # Define the desired CRS (e.g., EPSG:32604)
        desired_crs = CRS.from_epsg(desired_crs_epsg)

        # Open the TIFF file with rasterio
        with rasterio.open(self.tif_path) as tiff_dataset:
            # Calculate the transformation parameters for reprojecting
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)

            # Reproject the TIFF image to the desired CRS
            image_data, _ = reproject(
                source=tiff_dataset.read(1),
                src_crs=tiff_dataset.crs,
                src_transform=tiff_dataset.transform,
                dst_crs=desired_crs,
                resampling=rasterio.enums.Resampling.bilinear)

            # Update the extent based on the reprojected data
            tiff_extent = rasterio.transform.array_bounds(height, width, transform)

        # Save the reprojected TIFF image to the output path
        with rasterio.open(output_path, 'w', driver='GTiff', crs=desired_crs, transform=transform, width=width, height=height, count=1, dtype=image_data.dtype) as dst:
            dst.write(image_data, 1)

    def plot_agent_locations_on_tiff(self):
        # Read the contents of the HDF file
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            keys = store.keys()
            for key in keys:
                print(f"Contents of key '{key}':")
                data = pd.read_hdf(self.hdf_path, key)

                # Grab coordinates from the dataframe
                data['geometry'] = data['loc'].apply(loads)

                # Create a GeoDataFrame
                gdf = gpd.GeoDataFrame(data, geometry='geometry')

                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(10, 8))

                # Display the reprojected TIFF image using imshow with the correct extent and CRS
                ax.imshow(image_data[0], cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')

                # Plot the agent points on top of the TIFF image as points
                gdf.plot(ax=ax, marker='o', markersize=1, color='blue', alpha=0.5)

                # Add labels, legend, and title as needed
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title("Agent Locations")

                # Show the plot
                plt.show()


#TODO Individual Agents
class Agent_Individual:
    def __init__(self, hdf_path, tif_path):
        self.hdf_path = hdf_path
        self.tif_path = tif_path

    def visualize_agent_locations(self):
        # Read the contents of the '/TS' key
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            ts_key = '/TS'
            if ts_key in store:
                ts_data = pd.read_hdf(self.hdf_path, ts_key)
            else:
                raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

        # Get unique IDs in the data
        unique_ids = ts_data['id'].unique()

        # Loop through each unique ID and plot points for the agent's locations
        for plot_id in unique_ids:
            # Filter data for the specified ID
            filtered_data = ts_data[ts_data['id'] == plot_id]

            # Extract 'ts' values for the specified ID
            timesteps = filtered_data['ts'].tolist()

            # Extract 'loc' data for the specified ID
            loc_data = filtered_data['loc'].apply(lambda x: loads(x))
            x_coords = [loc.x for loc in loc_data if loc is not None]  # Check for None values
            y_coords = [loc.y for loc in loc_data if loc is not None]  # Check for None values

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))

            # Display the reprojected TIFF image using imshow with the correct extent and CRS
            ax.imshow(image_data[0], cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')

            # Plot points for the agent's locations
            ax.scatter(x_coords, y_coords, label=f'ID {plot_id}', marker='o', s=5, color='red')

            # Add labels, legend, and title
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend()
            ax.set_title(f'Agent {plot_id} Timestep Locations (Points)')
            ax.grid(True)

            # Show the plot for the current unique ID
            plt.show()

    def load_tiff_image(self, crs_epsg=32604):
        # Define the desired CRS
        desired_crs = CRS.from_epsg(crs_epsg)

        # Open the TIFF file with rasterio
        with rasterio.open(self.tif_path) as tiff_dataset:
            # Calculate the transformation parameters for reprojecting
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)

            # Reproject the TIFF image to the desired CRS
            image_data, _ = reproject(
                source=tiff_dataset.read(1),
                src_crs=tiff_dataset.crs,
                src_transform=tiff_dataset.transform,
                dst_crs=desired_crs,
                resampling=rasterio.enums.Resampling.bilinear)

            # Update the extent based on the reprojected data
            tiff_extent = rasterio.transform.array_bounds(height, width, transform)
        
        return image_data, tiff_extent
    
    

#TODO Jump DF
class Jump_DF:
    def __init__(self, directory_path, hdf_path, tif_path):
        self.directory_path = directory_path
        self.hdf_path = hdf_path
        self.tif_path = tif_path

    def read_jump_data(self):
        jump_df_all = pd.DataFrame()
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.h5'):
                h5_file_path = os.path.join(self.directory_path, filename)
                try:
                    with pd.HDFStore(h5_file_path, mode='r') as store:
                        if '/jump' in store:
                            jump_data = pd.read_hdf(h5_file_path, '/jump')
                            jump_df_all = pd.concat([jump_df_all, jump_data], ignore_index=True)
                except Exception as e:
                    print(f"Error reading file '{h5_file_path}': {e}")
        return jump_df_all

    def read_ts_loc_data(self):
        ts_loc_df = pd.DataFrame()
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                if '/TS' in store:
                    ts_data = pd.read_hdf(self.hdf_path, '/TS')
                    ts_loc_df = pd.concat([ts_loc_df, ts_data], ignore_index=True)
        except Exception as e:
            print(f"Error reading file '{self.hdf_path}': {e}")
        return ts_loc_df

    def plot_agent_locations(self):
        jump_df_all = self.read_jump_data()
        ts_loc_df = self.read_ts_loc_data()

        ts_loc_df['id'] = ts_loc_df['id'].astype('int32')
        merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')
        merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
        merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)

        plt.figure(figsize=(10, 6))
        plt.scatter(merged_df['x'], merged_df['y'], marker='o')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Agents Plotted Based on Timesteps')
        plt.show()

    def plot_agent_locations_with_colors(self):
        jump_df_all = self.read_jump_data()
        ts_loc_df = self.read_ts_loc_data()

        ts_loc_df['id'] = ts_loc_df['id'].astype('int32')
        merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')
        merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
        merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)
        unique_ids = merged_df['ID'].unique()
        num_agents = len(unique_ids)
        color_map = plt.get_cmap('viridis', num_agents)

        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_id, color in zip(unique_ids, color_map(range(num_agents))):
            agent_data = merged_df[merged_df['ID'] == agent_id]
            x_coordinates = agent_data['x']
            y_coordinates = agent_data['y']
            ax.scatter(x_coordinates, y_coordinates, marker='o', label=f'Agent {agent_id}', color=color)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Agents Plotted with Different Colors')
        ax.legend()
        plt.show()

    def plot_aerial_jump_locations(self):
        desired_crs = CRS.from_epsg(32604)

        with rasterio.open(self.tif_path) as tiff_dataset:
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)
            image_data, _ = reproject(
                source=tiff_dataset.read(1),
                src_crs=tiff_dataset.crs,
                src_transform=tiff_dataset.transform,
                dst_crs=desired_crs,
                resampling=rasterio.enums.Resampling.bilinear)
            tiff_extent = rasterio.transform.array_bounds(height, width, transform)

        gdf = gpd.GeoDataFrame(merged_df, geometry=gpd.points_from_xy(merged_df['x'], merged_df['y']))
        gdf.crs = desired_crs

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_data[0], cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')
        gdf.plot(ax=ax, marker='o', markersize=10, color='red', label='Agent Locations')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        plt.show()



#TODO Jump_Data_Statistics
class JumpDataStatistics:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.jump_angles = []
        self.time_airborne_values = []
        self.displacement_values = []

    def read_jump_data(self):
        self.jump_angles = []
        self.time_airborne_values = []
        self.displacement_values = []

        for filename in os.listdir(self.directory_path):
            if filename.endswith('.h5'):
                h5_file_path = os.path.join(self.directory_path, filename)
                try:
                    with pd.HDFStore(h5_file_path, mode='r') as store:
                        if '/jump' in store:
                            jump_data = pd.read_hdf(h5_file_path, '/jump')
                            self.jump_angles.extend(jump_data['jump_angle'])
                            self.time_airborne_values.extend(jump_data['time_airborne'])
                            self.displacement_values.extend(jump_data['displacement'])
                except Exception as e:
                    print(f"Error reading file '{h5_file_path}': {e}")

    def calculate_statistics(self):
        self.read_jump_data()
        
        if not self.jump_angles:
            print("No jump data found.")
            return

        average_jump_angle = sum(self.jump_angles) / len(self.jump_angles)
        min_jump_angle = min(self.jump_angles)
        max_jump_angle = max(self.jump_angles)

        average_time_airborne = sum(self.time_airborne_values) / len(self.time_airborne_values)
        min_time_airborne = min(self.time_airborne_values)
        max_time_airborne = max(self.time_airborne_values)

        average_displacement = sum(self.displacement_values) / len(self.displacement_values)
        min_displacement = min(self.displacement_values)
        max_displacement = max(self.displacement_values)

        return {
            "Average Jump Angle": average_jump_angle,
            "Minimum Jump Angle": min_jump_angle,
            "Maximum Jump Angle": max_jump_angle,
            "Average Time Airborne": average_time_airborne,
            "Minimum Time Airborne": min_time_airborne,
            "Maximum Time Airborne": max_time_airborne,
            "Average Displacement": average_displacement,
            "Minimum Displacement": min_displacement,
            "Maximum Displacement": max_displacement
        }




#TODO Heatmap of ALL Agents
class AgentLocationHeatmap:
    def __init__(self, ts_data, bins=100, cmap='viridis', title='Heatmap of Agent Location Data'):
        self.ts_data = ts_data
        self.bins = bins
        self.cmap = cmap
        self.title = title

    def create_heatmap(self):
        # Extract 'x' and 'y' coordinates from the 'loc' column
        loc_data = self.ts_data['loc'].apply(lambda x: loads(x))
        x_coords = np.array([loc.x for loc in loc_data])
        y_coords = np.array([loc.y for loc in loc_data])

        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=self.bins)

        # Create a heatmap plot
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap=self.cmap)

        # Add labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(self.title)

        # Show the colorbar for reference
        plt.colorbar(label='Frequency')

        # Show the plot
        plt.show()



#TODO Agent Jump Location Heat Map
class AgentJumpLocationsHeatMap:
    def __init__(self, x_values, y_values, bins=(100, 100), cmap='viridis', title='Heatmap of Agent Locations'):
        self.x_values = x_values
        self.y_values = y_values
        self.bins = bins
        self.cmap = cmap
        self.title = title

    def create_heatmap(self):
        # Create a 2D histogram for agent locations
        plt.figure(figsize=(10, 8))
        plt.hist2d(self.x_values, self.y_values, bins=self.bins, cmap=self.cmap)

        # Add labels and a title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(self.title)

        # Add a colorbar to the heatmap
        colorbar = plt.colorbar()
        colorbar.set_label('Frequency')

        # Show the heatmap
        plt.show()
