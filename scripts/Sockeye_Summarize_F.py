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
from rasterio.plot import show



#TODO identify what needs to be written to class object (should be promoted to class object instead of being just printed out on screen)
#identify workspace
inputws=r"J:\2819\005\Calcs\ABM\Output\test_29"



#TODO initialization
#Identify the directory path
directory_path = r"J:\2819\005\Calcs\ABM\Output\test_29"
hdf_path = r"J:\2819\005\Calcs\ABM\Output\test_29/test_29.hdf"
tif_path=r'J:\2819\005\Calcs\ABM\Data\Nuyakuk_Area.tif'


# Separate the .h5 and .hdf5 files into 2 separate lists
hdf5_files = []
h5_files = []
for filename in os.listdir(directory_path):
    if filename.endswith('.hdf5'):
        hdf5_files.append(filename)
    elif filename.endswith('.h5'):
        h5_files.append(filename)

# Read the keys of all the .h5 file
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        keys = store.keys()
        print(f"Keys in the .h5 file '{filename}':")
        for key in keys:
            print(key)
        

# Create a unique index of the .h5 list
h5_files = list(set(h5_files))



#TODO seperate method
# Create a list to store the 'length' values
lengths = []

# Look in the /agent key for 'length', print the /agent keys, and collect the 'length' values
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        agent_key = '/agent'
        if agent_key in store:
            agent_data = pd.read_hdf(full_path, agent_key)
            agent_data.reset_index(inplace=True)
            if 'length' in agent_data.columns:
                print(f"/agent keys in '{filename}': {agent_data.keys()}")
                lengths.extend(agent_data['length'])
            
   

# Create a histogram of the 'length' values for all .h5 files
plt.hist(lengths, bins=20, edgecolor='black')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Lengths for All Agents')
plt.show()

# Calculate the average (mean), median, and standard deviation of the 'length' values
if lengths:
    mean_length = sum(lengths) / len(lengths)
    median_length = sorted(lengths)[len(lengths) // 2]
    std_dev_length = (sum((x - mean_length) ** 2 for x in lengths) / len(lengths)) ** 0.5

    print(f"Average (Mean) Length: {mean_length}")
    print(f"Median Length: {median_length}")
    print(f"Standard Deviation of Length: {std_dev_length}")
else:
    print("No 'length' values found.")



#TODO seperate method
#Create a list to store the 'weight' values
weights=[]
# Look in the /agent key for 'weight', and collect the 'weight' values
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        agent_key = '/agent'
        if agent_key in store:
            agent_data = pd.read_hdf(full_path, agent_key)
            agent_data.reset_index(inplace=True)  # Reset the index to ensure it's unique
            if 'weight' in agent_data.columns:
                weights.extend(agent_data['weight'])
            else:
                print(f"'weight' not found in '{filename}'")
        else:
            print(f"Key '/agent' not found in '{filename}'")

# Create a histogram of the 'weight' values for all .h5 files
plt.hist(weights, bins=20, edgecolor='black')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Histogram of Weights for All Agents')
plt.show()

# Calculate the average (mean) and median of the 'weight' values
if weights:
    mean_weight = np.mean(weights)
    median_weight = np.median(weights)
    std_dev_weight = (sum((x - mean_weight) ** 2 for x in weights) / len(weights)) ** 0.5

    print(f"Average (Mean) Weight: {mean_weight}")
    print(f"Median Weight: {median_weight}")
    print(f"Standard Deviation of Weight: {std_dev_weight}")
else:
    print("No 'weight' values found.")



#TODO seperate method
# Create a list to store the 'body_depth' values
body_depths = []

# Look in the /agent key for 'body_depth', and collect the 'body_depth' values
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        agent_key = '/agent'
        if agent_key in store:
            agent_data = pd.read_hdf(full_path, agent_key)
            agent_data.reset_index(inplace=True)  # Reset the index to ensure it's unique
            if 'body_depth' in agent_data.columns:
                body_depths.extend(agent_data['body_depth'])
            else:
                print(f"'body_depth' not found in '{filename}'")
        else:
            print(f"Key '/agent' not found in '{filename}'")

# Create a histogram of the 'body_depth' values for all .h5 files
plt.hist(body_depths, bins=20, edgecolor='black')
plt.xlabel('Body Depth')
plt.ylabel('Frequency')
plt.title('Histogram of Body Depths for All Agents')
plt.show()

# Calculate the average (mean) and median of the 'body_depth' values
if body_depths:
    mean_body_depth = np.mean(body_depths)
    median_body_depth = np.median(body_depths)
    std_dev_body_depth = (sum((x - mean_body_depth) ** 2 for x in body_depths) / len(body_depths)) ** 0.5
    print(f"Average (Mean) Body Depth: {mean_body_depth}")
    print(f"Median Body Depth: {median_body_depth}")
    print(f"Standard Deviation of Body Depth: {std_dev_body_depth}")
else:
    print("No 'body_depth' values found.")
 
    
 
#TODO initialization    
#Open up the /battery key and look at its contents
for filename in os.listdir(directory_path):
    if filename.endswith('.h5'):
        full_path = os.path.join(directory_path, filename)
        try:
            with pd.HDFStore(full_path, mode='r') as store:
                battery_key = '/battery'
                if battery_key in store:
                    print(f"Contents of '/battery' in '{filename}':")
                    battery_data = pd.read_hdf(full_path, battery_key)
                    print(battery_data)
                else:
                    print(f"Key '/battery' not found in '{filename}'")
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")

        

#TODO seperate method
#Look within all the .h5 files /battery key, find bout_duration and give the maximum number
h5_files = [filename for filename in os.listdir(directory_path) if filename.endswith('.h5')]

for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        if '/battery' in store:
            battery_data = pd.read_hdf(full_path, '/battery')
            if 'bout_duration' in battery_data.columns:
                max_bout_duration = battery_data['bout_duration'].max()
                print(f"Maximum bout duration in '/battery' key for file {filename}: {max_bout_duration}")
            else:
                print(f"Column 'bout_duration' not found in '/battery' key for file {filename}.")
        else:
            print(f"Key '/battery' not found in file {filename}.")

        

#TODO seperate method            
#Look within all the .h5 files /battery key, find bout_no and give the maximum number
h5_files = [filename for filename in os.listdir(directory_path) if filename.endswith('.h5')]

for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        if '/battery' in store:
            battery_data = pd.read_hdf(full_path, '/battery')
            if 'bout_no' in battery_data.columns:
                max_bout_no = battery_data['bout_no'].max()
                print(f"Maximum bout no in '/battery' key for file {filename}: {max_bout_no}")
            else:
                print(f"Column 'bout_no' not found in '/battery' key for file {filename}.")
        else:
            print(f"Key '/battery' not found in file {filename}.")
        #break if you want to only show the first agent
        
        
      
#TODO Seperate Method        
#Create a dataframe that hosts all the agents ID, maximum bout_no and maximum bout_duration
data_list = []
h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        if '/battery' in store:
            battery_data = pd.read_hdf(full_path, '/battery')
            if 'bout_no' in battery_data.columns and 'bout_duration' in battery_data.columns and 'ID' in battery_data.columns:
                # Get the maximum bout_no and bout_duration for each ID
                max_values = battery_data.groupby('ID')[['bout_no', 'bout_duration']].max().reset_index()
                for index, row in max_values.iterrows():
                    data_list.append({'ID': row['ID'], 'max_bout_no': row['bout_no'], 'max_bout_duration': row['bout_duration']})
            else:
                print(f"Columns 'ID', 'bout_no', or 'bout_duration' not found in '/battery' key for file {filename}.")
        else:
            print(f"Key '/battery' not found in file {filename}.")
battery_df = pd.DataFrame(data_list)
specific_ids = range(0, 25)  # This includes IDs from 0 to 24
filtered_battery_df = battery_df[battery_df['ID'].isin(specific_ids)]
print(filtered_battery_df)



#TODO seperate method
#Create a dataframe that hosts all the agents ID, bout_no and bout_duration
bout_duration_list = []
h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
for filename in h5_files:
    full_path = os.path.join(directory_path, filename)
    with pd.HDFStore(full_path, mode='r') as store:
        if '/battery' in store:
            battery_data = pd.read_hdf(full_path, '/battery')
            if 'bout_no' in battery_data.columns and 'bout_duration' in battery_data.columns and 'ID' in battery_data.columns:
                extracted_data = battery_data[['ID', 'bout_no', 'bout_duration']]
                bout_duration_list.extend(extracted_data.to_dict(orient='records'))
            else:
                print(f"Columns 'ID', 'bout_no', or 'bout_duration' not found in '/battery' key for file {filename}.")
        else:
            print(f"Key '/battery' not found in file {filename}.")
bout_duration_df = pd.DataFrame(bout_duration_list)
print(bout_duration_df)



#TODO initialzation
#Read hdf file and read the keys
with pd.HDFStore(hdf_path, mode='r') as store:
    keys = store.keys()
    for key in keys:
        print(f"Contents of key '{key}':")
        data = pd.read_hdf(hdf_path, key)
        print(data)
    
    
    

#TODO seperate method
#Create a plot to look at agents within a defined area
ids_in_rectangle = set()

#Create Dataframe from HDF file
with pd.HDFStore(hdf_path, mode='r') as store:
    keys = store.keys()
    for key in keys:
        print(f"Contents of key '{key}':")
        data = pd.read_hdf(hdf_path, key)
        
        # Grab coordinates from dataframe
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
                ids_in_rectangle.add(id)
                
                # Display the filtered data for the current ID and timestep
                print(f"First timestep within the rectangle for ID {id}:")
                print(first_timestep)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title("Agents Passing Through Falls")
        plt.show()

# Print the IDs of data points within the rectangle
print("Agents with data points within the rectangle:")
print(ids_in_rectangle)



#TODO method
# Read the contents of the '/TS' key
with pd.HDFStore(hdf_path, mode='r') as store:
    ts_key = '/TS'
    if ts_key in store:
        ts_data = pd.read_hdf(hdf_path, ts_key)
    else:
        raise ValueError(f"Key '{ts_key}' not found in '{hdf_path}'")

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
plt.grid(True)

# Show the plot
plt.show()




#TODO method
# Read the contents of the '/TS' key
with pd.HDFStore(hdf_path, mode='r') as store:
    ts_key = '/TS'
    if ts_key in store:
        ts_data = pd.read_hdf(hdf_path, ts_key)
    else:
        raise ValueError(f"Key '{ts_key}' not found in '{hdf_path}'")

# Get unique IDs in the data
unique_ids = ts_data['id'].unique()

# Loop through each unique ID and plot individually
for plot_id in unique_ids:
    # Filter data for the specified ID
    filtered_data = ts_data[ts_data['id'] == plot_id]

    # Extract 'ts' values for the specified ID
    timesteps = filtered_data['ts'].tolist()

    # Extract 'loc' data for the specified ID
    loc_data = filtered_data['loc'].apply(lambda x: loads(x))
    x_coords = [loc.x for loc in loc_data if loc is not None]  # Check for None values
    y_coords = [loc.y for loc in loc_data if loc is not None]  # Check for None values

    # Create a line plot for the specified ID
    plt.figure()
    plt.plot(x_coords, y_coords, label=f'ID {plot_id}')

    # Add labels, legend, and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title(f'Agent {plot_id} Timestep Locations (Lines)')
    plt.grid(True)

# Show all the individual plots
plt.show()



#TODOALL Initialization
#Create plot of jump locaions of all agents

# Create an empty dataframe for 'jump_df_all'
jump_df_all = pd.DataFrame()

# Iterate through each HDF5 file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.h5'):
        h5_file_path = os.path.join(directory_path, filename)
        
        try:
            # Open the HDF5 file
            with pd.HDFStore(h5_file_path, mode='r') as store:
                # Check if '/jump' key exists in the file
                if '/jump' in store:
                    jump_data = pd.read_hdf(h5_file_path, '/jump')
                    jump_df_all = pd.concat([jump_df_all, jump_data], ignore_index=True)
        
        except Exception as e:
            print(f"Error reading file '{h5_file_path}': {e}")

# 'jump_df_all' now contains all the concatenated '/jump' data from all HDF5 files
print("Jump DataFrame (All Files):")
print(jump_df_all)



# Create an empty dataframe for 'ts_loc_df'
ts_loc_df = pd.DataFrame()

try:
    # Open the specified HDF5 file
    with pd.HDFStore(hdf_path, mode='r') as store:
        # Check if '/TS' key exists in the file
        if '/TS' in store:
            ts_data = pd.read_hdf(hdf_path, '/TS')
            ts_loc_df = pd.concat([ts_loc_df, ts_data], ignore_index=True)

except Exception as e:
    print(f"Error reading file '{hdf_path}': {e}")

# 'ts_loc_df' now contains the data from the '/TS' key in the specified HDF5 file
print("TS Location DataFrame:")
print(ts_loc_df)



#TODO seperate Method
# Convert the 'id' column in ts_loc_df to int32
ts_loc_df['id'] = ts_loc_df['id'].astype('int32')

# Merge the jump_df_all and ts_loc_df DataFrames based on 'ID' and 'timestep'
merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')

# Extract the x and y coordinates from the 'loc' column
merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)
print(merged_df)

# Plot the agents based on their positions
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['x'], merged_df['y'], marker='o')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Agents Plotted Based on Timesteps')
plt.grid(True)
plt.show()


#TODO Seperate Method
# Convert the 'id' column in ts_loc_df to int32
ts_loc_df['id'] = ts_loc_df['id'].astype('int32')

# Merge the jump_df_all and ts_loc_df DataFrames based on 'ID' and 'timestep'
merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')

# Extract the x and y coordinates from the 'loc' column and add them to merged_df
merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)

# Get unique agent IDs from the merged_df DataFrame
unique_ids = merged_df['ID'].unique()

# Define a colormap with enough colors for all agents
num_agents = len(unique_ids)
color_map = plt.get_cmap('viridis', num_agents)  # You can choose any colormap you like

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over unique agent IDs and assign a unique color to each agent
for agent_id, color in zip(unique_ids, color_map(range(num_agents))):
    # Filter the DataFrame for data related to the current agent
    agent_data = merged_df[merged_df['ID'] == agent_id]
    
    # Extract x and y coordinates for the current agent
    x_coordinates = agent_data['x']
    y_coordinates = agent_data['y']
    
    # Create a scatter plot for the current agent with a unique color
    ax.scatter(x_coordinates, y_coordinates, marker='o', label=f'Agent {agent_id}', color=color)

# Set labels, title, and legend
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Agents Plotted with Different Colors')
ax.legend()

# Show the plot with all agents
plt.grid(True)
plt.show()



#TODO method
#Maximum, Minimum, and Averages of jumpe angles, time airborne, and displacement

# Create empty lists to store values for all files
jump_angles = []
time_airborne_values = []
displacement_values = []

# Iterate through each HDF5 file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.h5'):
        h5_file_path = os.path.join(directory_path, filename)

        try:
            # Open the HDF5 file
            with pd.HDFStore(h5_file_path, mode='r') as store:
                # Check if '/jump' key exists in the file
                if '/jump' in store:
                    jump_data = pd.read_hdf(h5_file_path, '/jump')
                    # Append values to lists
                    jump_angles.extend(jump_data['jump_angle'])
                    time_airborne_values.extend(jump_data['time_airborne'])
                    displacement_values.extend(jump_data['displacement'])

        except Exception as e:
            print(f"Error reading file '{h5_file_path}': {e}")

# Calculate average, minimum, and maximum values
average_jump_angle = sum(jump_angles) / len(jump_angles)
min_jump_angle = min(jump_angles)
max_jump_angle = max(jump_angles)

average_time_airborne = sum(time_airborne_values) / len(time_airborne_values)
min_time_airborne = min(time_airborne_values)
max_time_airborne = max(time_airborne_values)

average_displacement = sum(displacement_values) / len(displacement_values)
min_displacement = min(displacement_values)
max_displacement = max(displacement_values)

# Print the calculated values
print("Average Jump Angle:", average_jump_angle)
print("Minimum Jump Angle:", min_jump_angle)
print("Maximum Jump Angle:", max_jump_angle)

print("Average Time Airborne:", average_time_airborne)
print("Minimum Time Airborne:", min_time_airborne)
print("Maximum Time Airborne:", max_time_airborne)

print("Average Displacement:", average_displacement)
print("Minimum Displacement:", min_displacement)
print("Maximum Displacement:", max_displacement)



#TODO method
#Create a heat map of agent locations and frequency of points through timesteps
loc_data = ts_data['loc'].apply(lambda x: loads(x))
x_coords = np.array([loc.x for loc in loc_data])
y_coords = np.array([loc.y for loc in loc_data])

# Create a 2D histogram
heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=100)

# Create a heatmap plot
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='viridis')

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Agent Location Data')

# Show the colorbar for reference
plt.colorbar(label='Frequency')

# Show the plot
plt.show()



#TODO Seperate Method
#TODO download rasterio and work with raster and missing data
#Heat map of jump locations for agents based off frequency of timestep points

# Extract 'x' and 'y' coordinates from your merged_df DataFrame
x_values = merged_df['x']
y_values = merged_df['y']

# Create a 2D histogram for agent locations
plt.figure(figsize=(10, 8))
plt.hist2d(x_values, y_values, bins=(100, 100), cmap='viridis')

# Add labels and a title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Agent Locations')

# Add a colorbar to the heatmap
colorbar = plt.colorbar()
colorbar.set_label('Frequency')

# Show the heatmap
plt.show()





















tif_path = r'J:\2819\005\Calcs\ABM\Data\Nuyakuk_Area.tif'

# Open the TIFF file
with rasterio.open(tif_path) as dataset:
    # You can now work with the dataset using rasterio functions and methods
    # For example, you can access the metadata like this:
    metadata = dataset.meta
    print("Metadata:", metadata)
    
    # You can also read the image data as a numpy array:
    image_data = dataset.read()
    print("Image Data Shape:", image_data.shape)
    



# Open the TIFF file
with rasterio.open(tif_path) as dataset:
    # Show the georeference information
    print("Spatial Extent (Bounding Box):", dataset.bounds)
    print("Coordinate Reference System (CRS):", dataset.crs)
    
    # You can also display the image with its georeference using the show function
    show((dataset, 1), cmap='viridis')  # Assuming you want to show the first band with the 'viridis' colormap

# Display the plot
plt.show()























