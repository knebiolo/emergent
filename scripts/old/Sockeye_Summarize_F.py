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
from matplotlib.backends.backend_pdf import PdfPages



#Identify the directory path
directory_path = r"J:\2819\005\Calcs\ABM\Output\test_29"
hdf_path = r"J:\2819\005\Calcs\ABM\Output\test_29\test_29.hdf"
tif_path=r'J:\2819\005\Calcs\ABM\Data\elev.tif'
Data_output=r'J:\2819\005\Calcs\ABM\Data\Agent_data_files\Test_29'



class Summarization:
    def __init__(self, directory_path, hdf_path, tif_path, Data_output):
        self.directory_path = directory_path
        self.hdf_path = hdf_path
        self.tif_path = tif_path
        self.Data_output = Data_output
        self.h5_files = self.find_h5_files()
        self.lengths = []
        self.weights = []
        self.body_depths = []
        self.data_list = []  # Initialize data_list
        self.bout_duration_list = []  # Initialize bout_duration_list
        self.ids_in_rectangle = set()
       

    def h5_agent_list(self):
        # Create an empty list to store the .h5 files
        h5_files = []

        # Iterate through the files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.h5'):
                h5_files.append(filename)

        # Return the list of .h5 files
        return h5_files
    

    def list_keys_h5_files(self):
        for filename in self.h5_files:
            full_path_keys = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path_keys, mode='r') as store:
                keys = store.keys()
                print(f"Keys in the .h5 file '{filename}':")
                for key in keys:
                    print(key)

    def unique_h5_index(self):
        self.h5_files = list(set(self.h5_files))

    def process_hdf5_data(self):
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            keys = store.keys()
            for key in keys:
                print(f"Contents of key '{key}':")
                data = pd.read_hdf(self.hdf_path, key)
                print(data)

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



        
    # Find the .h5 files
    def find_h5_files(self):
        h5_files=[]
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.h5'):
                h5_files.append(filename)
        return h5_files

    # Collect length values
    def length_values(self):
        self.lengths = []  # Initialize the 'lengths' list

        # Iterate through each HDF5 file in the specified directory
        for filename in self.h5_files:
            full_path = os.path.join(self.directory_path, filename)

            # Open the HDF5 file in read-only mode
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key = '/agent'

                # Check if the '/agent' key exists in the HDF5 file
                if agent_key in store:
                    # Read the agent data from the '/agent' key
                    agent_data = pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)

                    # Check if the 'length' column exists in the agent data
                    if 'length' in agent_data.columns:
                        # Extend the 'lengths' list with the 'length' column values
                        self.lengths.extend(agent_data['length'])
                    else:
                        # Handle the case where 'length' column is not found in the agent data
                        pass
                else:
                    # Handle the case where '/agent' key is not found in the HDF5 file
                    pass

    # Plot lengths
    def plot_lengths(self):
        if self.lengths:
            # Create a PDF booklet
            pdf_filepath = os.path.join(self.Data_output, 'Agent_Lengths.pdf')
            with PdfPages(pdf_filepath) as pdf:
                # Plot histogram
                plt.hist(self.lengths, bins=20, edgecolor='black')
                plt.xlabel('Length')
                plt.ylabel('Frequency')
                plt.title('Agent Lengths')
                pdf.savefig()
                plt.close()

            print(f"Results written to: {pdf_filepath}")
        else:
            print("No 'length' values found for plotting")
    # Calculate Statistics
    def length_statistics(self):
        length_stats_file_path = os.path.join(Data_output, 'length_statistics_results.txt')

        with open(length_stats_file_path, 'w') as output_file:
            if self.lengths:
                mean_length = sum(self.lengths) / len(self.lengths)
                median_length = sorted(self.lengths)[len(self.lengths) // 2]
                std_dev_length = (sum((x - mean_length) ** 2 for x in self.lengths) / len(self.lengths)) ** 0.5

                result_line = f"Average (Mean) Length: {mean_length}\n"
                output_file.write(result_line)
                result_line = f"Median Length: {median_length}\n"
                output_file.write(result_line)
                result_line = f"Standard Deviation of Length: {std_dev_length}\n"
                output_file.write(result_line)
            else:
                result_line = "No 'length' values found for statistics\n"
                output_file.write(result_line)

        print(f"Results written to: {length_stats_file_path}")



    # class weights


    # Collect weights
    def collect_weights(self):
        self.weights = []  # Initialize the 'weights' list
        for filename in self.h5_files:
            full_path = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key = '/agent'
                if agent_key in store:
                    agent_data = pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)
                    if 'weight' in agent_data.columns:
                        self.weights.extend(agent_data['weight'])
                    else:
                        print(f"'weight' not found in '{filename}'")
                else:
                    print(f"Key '/agent' not found in '{filename}'")

    # Plot weights
    def plot_weights(self):
        if self.weights:
            # Create a PDF booklet
            pdf_filepath = os.path.join(self.Data_output, 'Agent_Weights.pdf')
            with PdfPages(pdf_filepath) as pdf:
                # Plot histogram
                plt.hist(self.weights, bins=20, edgecolor='black')
                plt.xlabel('Weight')
                plt.ylabel('Frequency')
                plt.title('Weights for All Agents')
                pdf.savefig()
                plt.close()

            print(f"Results written to: {pdf_filepath}")
        else:
            print("No 'weight' values found for plotting")

    # Calculate Statistics
    def weight_statistics(self):
        weight_stats_file_path = os.path.join(Data_output, 'weight_statistics_results.txt')

        with open(weight_stats_file_path, 'w') as output_file:
            if self.weights:
                mean_weight = np.mean(self.weights)
                median_weight = np.median(self.weights)
                std_dev_weight = np.std(self.weights)

                result_line = f"Average (Mean) Weight: {mean_weight}\n"
                output_file.write(result_line)
                result_line = f"Median Weight: {median_weight}\n"
                output_file.write(result_line)
                result_line = f"Standard Deviation of Weight: {std_dev_weight}\n"
                output_file.write(result_line)
            else:
                result_line = "No 'weight' values found for statistics\n"
                output_file.write(result_line)

        print(f"Results written to: {weight_stats_file_path}")


    # Collect body_depths data
    def collect_body_depths(self):
        self.body_depths = []  # Initialize the 'body_depths' list
        for filename in self.h5_files:
            full_path = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                agent_key = '/agent'
                if agent_key in store:
                    agent_data = pd.read_hdf(full_path, agent_key)
                    agent_data.reset_index(inplace=True)
                    if 'body_depth' in agent_data.columns:
                        self.body_depths.extend(agent_data['body_depth'])
                    else:
                        print(f"'body_depth' not found in '{filename}'")
                else:
                    print(f"Key '/agent' not found in '{filename}'")

    # Plot body_depths
    def plot_body_depths(self):
        if self.body_depths:
            # Create a PDF booklet
            pdf_filepath = os.path.join(self.Data_output, 'Agent_Body_Depths.pdf')
            with PdfPages(pdf_filepath) as pdf:
                # Plot histogram
                plt.hist(self.body_depths, bins=20, edgecolor='black')
                plt.xlabel('Body Depth')
                plt.ylabel('Frequency')
                plt.title('Body Depths for All Agents')
                pdf.savefig()
                plt.close()

            print(f"Results written to: {pdf_filepath}")
        else:
            print("No 'body_depth' values found for plotting.")

    # Calculate Statistics
    def body_depth_statistics(self):
        body_depth_stats_file_path = os.path.join(Data_output, 'body_depth_statistics_results.txt')

        with open(body_depth_stats_file_path, 'w') as output_file:
            if self.body_depths:
                mean_body_depth = np.mean(self.body_depths)
                median_body_depth = np.median(self.body_depths)
                std_dev_body_depth = np.std(self.body_depths)

                result_line = f"Average (Mean) Body Depth: {mean_body_depth}\n"
                output_file.write(result_line)
                result_line = f"Median Body Depth: {median_body_depth}\n"
                output_file.write(result_line)
                result_line = f"Standard Deviation of Body Depth: {std_dev_body_depth}\n"
                output_file.write(result_line)
            else:
                result_line = "No 'body_depth' values found for statistics\n"
                output_file.write(result_line)

        print(f"Results written to: {body_depth_stats_file_path}")
    
 
        
    # Find max bout duration
    def max_bout_duration(self):
        max_bout_file_path = os.path.join(Data_output, 'max_bout_duration_results.txt')

        with open(max_bout_file_path, 'w') as output_file:
            for filename in self.h5_files:
                full_path = os.path.join(self.directory_path, filename)
                with pd.HDFStore(full_path, mode='r') as store:
                    if '/battery' in store:
                        battery_data = pd.read_hdf(full_path, '/battery')
                        if 'bout_duration' in battery_data.columns:
                            max_bout_duration = battery_data['bout_duration'].max()
                            result_line = f"Maximum bout duration in '/battery' key for file {filename}: {max_bout_duration}\n"
                            output_file.write(result_line)
                        else:
                            result_line = f"Column 'bout_duration' not found in '/battery' key for file {filename}.\n"
                            output_file.write(result_line)
                    else:
                        result_line = f"Key '/battery' not found in file {filename}.\n"
                        output_file.write(result_line)

        print(f"Results written to: {max_bout_file_path}")


            
    def max_bout_no(self):
        max_bout_no_file_path = os.path.join(Data_output, 'max_bout_no_results.txt')

        with open(max_bout_no_file_path, 'w') as output_file:
            for filename in self.h5_files:
                full_path = os.path.join(self.directory_path, filename)
                with pd.HDFStore(full_path, mode='r') as store:
                    if '/battery' in store:
                        battery_data = pd.read_hdf(full_path, '/battery')
                        if 'bout_no' in battery_data.columns:
                            max_bout_no = battery_data['bout_no'].max()
                            result_line = f"Maximum bout no in '/battery' key for file {filename}: {max_bout_no}\n"
                            output_file.write(result_line)
                        else:
                            result_line = f"Column 'bout_no' not found in '/battery' key for file {filename}.\n"
                            output_file.write(result_line)
                    else:
                        result_line = f"Key '/battery' not found in file {filename}.\n"
                        output_file.write(result_line)

        print(f"Results written to: {max_bout_no_file_path}")



    def max_bout_noduration(self):
        for filename in self.h5_files:
            full_path = os.path.join(self.directory_path, filename)
            with pd.HDFStore(full_path, mode='r') as store:
                if '/battery' in store:
                    battery_data = pd.read_hdf(full_path, '/battery')
                    if 'bout_no' in battery_data.columns and 'bout_duration' in battery_data.columns and 'ID' in battery_data.columns:
                        max_values = battery_data.groupby('ID')[['bout_no', 'bout_duration']].max().reset_index()
                        for index, row in max_values.iterrows():
                            self.data_list.append({'ID': row['ID'], 'max_bout_no': row['bout_no'], 'max_bout_duration': row['bout_duration']})
                    else:
                        print(f"Columns 'ID', 'bout_no', or 'bout_duration' not found in '/battery' key for file {filename}.")
                else:
                    print(f"Key '/battery' not found in file {filename}.")
        return pd.DataFrame(self.data_list)
    
    
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

    # Plot Agent w/ Rectangle
    def Agent_Plot_Rectangle(self):
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Rectangle_Plots.pdf')
        txt_filepath = os.path.join(self.Data_output, 'Agents_In_Rectangle.txt')

        with PdfPages(pdf_filepath) as pdf, open(txt_filepath, 'w') as txt_file:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                keys = store.keys()
                for key in keys:
                    print(f"Contents of key '{key}':")
                    data = pd.read_hdf(self.hdf_path, key)

                    data['geometry'] = data['loc'].apply(loads)
                    gdf = gpd.GeoDataFrame(data, geometry='geometry')

                    plt.figure(figsize=(10, 8))
                    ax = gdf.plot(marker='o', markersize=1, color='blue', alpha=0.5)

                    y_max = gdf.geometry.bounds['maxy'].max()
                    y_min_top_10 = y_max - 0.1 * (y_max - gdf.geometry.bounds['miny'].min())

                    rect = Rectangle((gdf.geometry.bounds['minx'].min(), y_min_top_10),
                                     gdf.geometry.bounds['maxx'].max() - gdf.geometry.bounds['minx'].min(),
                                     y_max - y_min_top_10, linewidth=2, edgecolor='red', fill=False)

                    ax.add_patch(rect)

                    unique_ids = data['id'].unique()
                    for id in unique_ids:
                        data_for_id = data[data['id'] == id]
                        gdf_id = gpd.GeoDataFrame(data_for_id, geometry='geometry')

                        data_in_rectangle = gdf_id.cx[gdf_id.geometry.bounds['minx'].min():gdf_id.geometry.bounds['maxx'].max(),
                                                      y_min_top_10:y_max]

                        first_timestep = data_in_rectangle.head(1)
                        if not first_timestep.empty:
                            self.ids_in_rectangle.add(id)

                            print(f"First timestep within the rectangle for ID {id}:")
                            print(first_timestep)

                            # Write ID and relevant information to the text file
                            txt_file.write(f"First timestep within the rectangle for ID {id}:\n")
                            txt_file.write(first_timestep.to_string(index=False) + '\n\n')

                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.title("Agents Passing Through Falls")
                    pdf.savefig()
                    plt.close()

        print(f"Results written to: {pdf_filepath}")
        print(f"IDs of agents within the rectangle written to: {txt_filepath}")

    #  AgentVisualizer     
    def plot_agent_locations(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Locations_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
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

            # Loop through each unique ID and plot individually
            for unique_id in unique_ids:
                # Filter data for the current unique ID
                filtered_data = id_ts_data[id_ts_data['id'] == unique_id]

                # Extract 'ts' values for the unique ID
                timesteps = filtered_data['ts'].tolist()

                # Extract 'loc' data for the unique ID
                loc_data = ts_data[ts_data['id'] == unique_id]['loc'].apply(lambda x: loads(x))
                x_coords = [loc.x for loc in loc_data]
                y_coords = [loc.y for loc in loc_data]

                # Create a figure and axis for the plot
                plt.figure(figsize=(10, 8))

                # Plot the timesteps for the unique ID
                plt.plot(x_coords, y_coords, label=f'ID {unique_id}')

                # Add labels, legend, and title as needed
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.title(f'Agent {unique_id} Locations Based Off Timesteps')

                # Save the plot to the PDF
                pdf.savefig()
                plt.close()

        print(f"Results written to: {pdf_filepath}")




    def plot_agent_locations_on_tiff(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Locations_on_TIFF_Plots.pdf')
        tiff_filepath = os.path.join(self.Data_output, 'Agent_Locations_on_TIFF.tif')

        with PdfPages(pdf_filepath) as pdf:
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

                    # Open the TIFF file with rasterio
                    with rasterio.open(self.tif_path) as tiff_dataset:
                        # Calculate the transformation parameters for reprojecting
                        transform, width, height = calculate_default_transform(
                            tiff_dataset.crs, tiff_dataset.crs, tiff_dataset.width, tiff_dataset.height,
                            *tiff_dataset.bounds)

                        # Reproject the TIFF image to the same CRS for display
                        image_data, _ = reproject(
                            source=tiff_dataset.read(1),
                            src_crs=tiff_dataset.crs,
                            src_transform=tiff_dataset.transform,
                            dst_crs=tiff_dataset.crs,
                            resampling=rasterio.enums.Resampling.bilinear)

                        # Update the extent based on the reprojected data
                        tiff_extent = rasterio.transform.array_bounds(height, width, transform)

                    # Display the reprojected TIFF image using imshow with the correct extent and CRS
                    ax.imshow(image_data[0], cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')

                    # Plot the agent points on top of the TIFF image as points
                    gdf.plot(ax=ax, marker='o', markersize=1, color='blue', alpha=0.5)

                    # Add labels, legend, and title as needed
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.title("Agent Locations on TIFF Background")

                    # Save the plot to the PDF
                    pdf.savefig()
                    plt.close()

            # Save the plot as a TIFF image
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image_data[0], cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')
            gdf.plot(ax=ax, marker='o', markersize=1, color='blue', alpha=0.5)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title("Agent Locations on TIFF Background")
            plt.savefig(tiff_filepath)
            plt.close()

        print(f"Results written to: {pdf_filepath} and {tiff_filepath}")
                
                
    def plot_agent_timestep_locations_on_tiff(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Timestep_Locations_on_TIFF_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
            # Create a folder to store individual TIFF files
            tiff_folder = os.path.join(self.Data_output, 'Agent_TIFFs')
            os.makedirs(tiff_folder, exist_ok=True)

            # Load the TIFF image
            tiff_image, tiff_extent = self.load_tiff_image()

            # Check if tiff_image is a 3D array
            if len(tiff_image.shape) == 3 and tiff_image.shape[0] == 1:
                # If it's a 3D array, select the first channel (single band)
                tiff_image = tiff_image[0]

            # Read agent timestep data
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                ts_key = '/TS'
                if ts_key in store:
                    ts_data = pd.read_hdf(self.hdf_path, ts_key)
                else:
                    raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

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

                # Create a figure and axis for the plot
                fig, ax = plt.subplots(figsize=(10, 8))

                # Display the TIFF image using imshow with the correct extent and CRS
                ax.imshow(tiff_image, cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')

                # Plot the agent's path on top of the TIFF image
                ax.plot(x_coords, y_coords, label=f'ID {plot_id}')

                # Add labels, legend, and title as needed
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Agent {plot_id} Timestep Locations on TIFF Background')
                plt.legend()

                # Save the plot to the PDF
                pdf.savefig()
                plt.close()

                # Save the TIFF file for the current agent
                agent_tiff_filepath = os.path.join(tiff_folder, f'Agent_{plot_id}_Locations.tif')
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(tiff_image, cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')
                ax.plot(x_coords, y_coords, label=f'ID {plot_id}', color='blue', markersize=1, linestyle='', marker='o', alpha=0.5)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Agent {plot_id} Timestep Locations on TIFF Background')
                plt.savefig(agent_tiff_filepath)
                plt.close()

        print(f"Results written to: {pdf_filepath} and TIFFs in {tiff_folder}")


    # Individual Agents
         
    def plot_agent_timestep_locations(self, pdf_filename='Individual_Agent_Timesteps.pdf'):
        pdf_filepath = os.path.join(self.Data_output, pdf_filename)

        with pd.HDFStore(self.hdf_path, mode='r') as store:
            ts_key = '/TS'
            if ts_key in store:
                ts_data = pd.read_hdf(self.hdf_path, ts_key)
            else:
                raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

        # Get unique IDs in the data
        unique_ids = ts_data['id'].unique()

        # Create a PDF file for the plots
        with PdfPages(pdf_filepath) as pdf:
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

                # Save the current plot to the PDF file
                pdf.savefig()
                plt.close()

        print(f"PDF booklet created: {pdf_filepath}")


    def plot_individual_agent_timestep_locations_on_tiff(self):
        # Load the TIFF image
        tiff_image, tiff_extent = self.load_tiff_image()

        # Check if tiff_image is a 3D array
        if len(tiff_image.shape) == 3 and tiff_image.shape[0] == 1:
            # If it's a 3D array, select the first channel (single band)
            tiff_image = tiff_image[0]

        # Read agent timestep data from the HDF5 file
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            ts_key = '/TS'
            if ts_key in store:
                ts_data = pd.read_hdf(self.hdf_path, ts_key)
            else:
                raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

        # Get unique IDs in the data
        unique_ids = ts_data['id'].unique()

        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Individual_Agent_Timesteps_on_TIFF.pdf')
        with PdfPages(pdf_filepath) as pdf:
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

                # Create a figure and axis for the plot
                fig, ax = plt.subplots(figsize=(10, 8))

                # Display the TIFF image using imshow with the correct extent and CRS
                ax.imshow(tiff_image, cmap='viridis', extent=(tiff_extent[0], tiff_extent[2], tiff_extent[1], tiff_extent[3]), aspect='equal')

                # Plot the agent's path on top of the TIFF image
                ax.plot(x_coords, y_coords, label=f'ID {plot_id}')

                # Add labels, legend, and title as needed
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Agent {plot_id} Timestep Locations (Lines) on TIFF Background')

                # Save the plot to the PDF
                pdf.savefig()
                plt.close()

        print(f"Results written to: {pdf_filepath}")
        

    #Jump DF

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


    def plot_agent_location_jump(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Location_Jump_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
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

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

        print(f"Results written to: {pdf_filepath}")

    def plot_agent_locations_with_colors(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Locations_with_Colors_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
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

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

        print(f"Results written to: {pdf_filepath}")



    def plot_agent_jump_locations_tif(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Jump_Locations_TIF_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
            jump_df_all = self.read_jump_data()
            ts_loc_df = self.read_ts_loc_data()

            ts_loc_df['id'] = ts_loc_df['id'].astype('int32')
            merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')
            merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
            merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)

            # Create a folder to store individual TIFF files
            tiff_folder = os.path.join(self.Data_output, 'Agent_Jump_TIFFs')
            os.makedirs(tiff_folder, exist_ok=True)

            # Open the GeoTIFF file for the background
            with rasterio.open(self.tif_path) as src:
                tiff_image = src.read(1)  # Read the first band of the GeoTIFF
                tiff_extent = src.bounds

            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Display the TIFF image using imshow with the correct extent and CRS
            ax.imshow(tiff_image, cmap='viridis', extent=(tiff_extent.left, tiff_extent.right, tiff_extent.bottom, tiff_extent.top), aspect='equal')

            # Plot the agent locations on top of the TIFF image
            ax.scatter(merged_df['x'], merged_df['y'], marker='o', c='red', label='Agent Locations')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Agents Plotted Based on Timesteps')
            plt.legend()

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

            # Save the TIFF file for the current jump
            jump_tiff_filepath = os.path.join(tiff_folder, 'Agent_Jump_Locations.tif')
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(tiff_image, cmap='viridis', extent=(tiff_extent.left, tiff_extent.right, tiff_extent.bottom, tiff_extent.top), aspect='equal')
            ax.scatter(merged_df['x'], merged_df['y'], marker='o', c='red', label='Agent Locations')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Agents Plotted Based on Timesteps')
            plt.legend()
            plt.savefig(jump_tiff_filepath)
            plt.close()

        print(f"Results written to: {pdf_filepath} and TIFF in {tiff_folder}")


    #Jump_Data_Statistics
    def read_jump_data_1(self):
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

        # Create a dictionary with the statistics
        statistics = {
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

        # Save the statistics to a text file
        stats_filepath = os.path.join(self.Data_output, 'jump_statistics.txt')
        with open(stats_filepath, 'w') as stats_file:
            for stat_name, stat_value in statistics.items():
                stats_file.write(f"{stat_name}: {stat_value}\n")

        print(f"Statistics written to: {stats_filepath}")

        return statistics


    # Heatmap of ALL Agents
    
    def plot_agent_timestep_heatmap(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Timestep_Heatmap.pdf')
        with PdfPages(pdf_filepath) as pdf:
            # Read the contents of the '/TS' key
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                ts_key = '/TS'
                if ts_key in store:
                    ts_data = pd.read_hdf(self.hdf_path, ts_key)
                else:
                    raise ValueError(f"Key '{ts_key}' not found in '{self.hdf_path}'")

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

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

        print(f"Results written to: {pdf_filepath}")

    # Agent Jump Location Heat Map

    def plot_agent_jump_heatmap(self):
        # Create a PDF booklet
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Jump_Heatmap.pdf')
        with PdfPages(pdf_filepath) as pdf:
            jump_df_all = self.read_jump_data()
            ts_loc_df = self.read_ts_loc_data()

            ts_loc_df['id'] = ts_loc_df['id'].astype('int32')
            merged_df = jump_df_all.merge(ts_loc_df, left_on=['ID', 'timestep'], right_on=['id', 'ts'], how='inner')
            merged_df['x'] = merged_df['loc'].apply(lambda point: loads(point).x)
            merged_df['y'] = merged_df['loc'].apply(lambda point: loads(point).y)

            # Create a 2D histogram for agent locations
            plt.figure(figsize=(10, 8))
            x_values = merged_df['x']
            y_values = merged_df['y']
            plt.hist2d(x_values, y_values, bins=(100, 100), cmap='viridis')

            # Add labels and a title
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Heatmap of Agent Jump Locations')

            # Add a colorbar to the heatmap
            colorbar = plt.colorbar()
            colorbar.set_label('Frequency')

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

        print(f"Results written to: {pdf_filepath}")

#Use these to run code

summarization=Summarization(directory_path, hdf_path, tif_path, Data_output)
#summarization.h5_agent_list()
#summarization.list_keys_h5_files()
#summarization.unique_h5_index()
#summarization.process_hdf5_data()
#summarization.process_h5_battery()
tiff_image, tiff_extent = summarization.load_tiff_image()
#summarization.read_ts_loc_data()
#h5_files = summarization.find_h5_files()
summarization.length_values()
summarization.plot_lengths()
summarization.length_statistics()
summarization.collect_weights()
summarization.plot_weights()
summarization.weight_statistics()
summarization.collect_body_depths()
summarization.plot_body_depths()
summarization.body_depth_statistics()
summarization.max_bout_duration()
summarization.max_bout_no()
#result_df = summarization.max_bout_noduration()
#bout_duration_df = summarization.agent_bout_df()
summarization.Agent_Plot_Rectangle()
summarization.plot_agent_locations()
summarization.plot_agent_locations_on_tiff()
summarization.plot_agent_timestep_locations_on_tiff()
summarization.plot_agent_timestep_locations()
summarization.plot_individual_agent_timestep_locations_on_tiff()
summarization.read_jump_data()
summarization.plot_agent_location_jump()
summarization.plot_agent_locations_with_colors()
summarization.plot_agent_jump_locations_tif()
summarization.read_jump_data_1()
summarization.calculate_statistics()
summarization.plot_agent_timestep_heatmap()
summarization.plot_agent_jump_heatmap()