
"""
Created on Tue Nov 27 14:31:15 2023

@author: EMuhlestein

Script intent: Multi model summary is to be used to summarize multiple models at once. Once a model
has finished running, this script will take said model as well as others and complete the summarization
without having to run the class Summarization on multiple occasions.
"""

#Import Dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform
from matplotlib.backends.backend_pdf import PdfPages
from rasterio.plot import show
import h5py


#Identify the directory path
directory_path = r"J:\2819\005\Calcs\ABM\Data\Agent_data_files\Multi_Model_Summarize\Validation_Test"
hdf_path = r"J:\2819\005\Calcs\ABM\Data\Agent_data_files\Multi_Model_Summarize\Validation_Test\validation_testing.h5"
exports_path=r'J:\2819\005\Calcs\ABM\Data\Agent_data_files\Multi_Model_Summarize\Validation_Test'
Data_output=r'J:\2819\005\Calcs\ABM\Data\Agent_data_files\Multi_Model_Summarize\Validation_Test'
tif_path=r'J:\2819\005\Calcs\ABM\Data\Agent_data_files\Multi_Model_Summarize\Validation_Test\elev.tif'

#Rectangle boundary information
x_min = 548500
y_min = 6641036
width = 549000 - x_min
height = 6641968 - y_min
rect_position = (x_min, y_min, width, height)


class Multi_Summarization:
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
        self.exports_path=exports_path
       

    def h5_agent_list(self):
        agent_keys = []
        # Open the main HDF5 file in read-only mode using pandas.HDFStore
        with pd.HDFStore(self.hdf_path, 'r') as store:
            # Iterate through all keys in the store
            for key in store.keys():
                # Check if the key starts with the specified prefix '/agent_data/'
                # Note: HDFStore keys always start with '/', no need to add it
                if key.startswith('agent_data'):
                    # Append the key to the list
                    agent_keys.append(key)

        return agent_keys
    
    

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

        
    def find_h5_files(self):
        h5_agent_keys = []

        # Open the main HDF5 file in read-only mode
        with pd.HDFStore(self.hdf_path, mode='r') as store:
            # Iterate through all keys in the store
            for key in store.keys():
                # Check if the key starts with the specified prefix for agents
                if key.startswith('agent_data'):
                    # Append the key (representing an individual agent's data) to the list
                    h5_agent_keys.append(key)

        return h5_agent_keys
    
    
    

    

    def plot_lengths(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        pdf_filename = f"{base_name}_Lengths_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(self.Data_output, pdf_filename)
        
        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(self.hdf_path, 'r') as file:
                if 'agent_data' in file:
                    lengths = file['/agent_data/length'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        lengths_by_sex = lengths[sex_mask]

                        if lengths_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # Dynamically calculate the number of bins
                            q75, q25 = np.percentile(lengths_by_sex, [75, 25])
                            bin_width = 2 * (q75 - q25) * len(lengths_by_sex) ** (-1/3)  # Freedman-Diaconis rule
                            bins = round((max(lengths_by_sex) - min(lengths_by_sex)) / bin_width)
                            ax.hist(lengths_by_sex, bins=bins, alpha=0.7, color='blue' if sex == 0 else 'pink')
                            ax.set_title(f'{base_name} - {sex_label} Agent Lengths')
                            ax.set_xlabel('Length')
                            ax.set_ylabel('Frequency')
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                        else:
                            print(f"No length values found for {sex_label}.")
        
        print(f"Plots saved to: {pdf_filepath}")
        
            
        
    # Calculate Statistics
    def length_statistics(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        stats_file_name = f"{base_name}_length_statistics_by_sex.txt"
        stats_file_path = os.path.join(self.Data_output, stats_file_name)

        with h5py.File(self.hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                lengths = file['/agent_data/length'][:]
                sexes = file['/agent_data/sex'][:]  # This assumes 0 and 1 encoding for sexes

                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    lengths_by_sex = lengths[sex_mask]
                    if lengths_by_sex.size > 0:
                        mean_length = np.mean(lengths_by_sex)
                        median_length = np.median(lengths_by_sex)
                        std_dev_length = np.std(lengths_by_sex, ddof=1)
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\n")
                        output_file.write(f"  Median Length: {median_length:.2f}\n")
                        output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\n\n")
                    else:
                        output_file.write(f"No length values found for {sex_label}.\n\n")

        print(f"Results written to: {stats_file_path}")


    # class weights


    # Plot weights
    def plot_weights(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        pdf_filename = f"{base_name}_Weights_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(self.Data_output, pdf_filename)

        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(self.hdf_path, 'r') as file:
                if 'agent_data' in file:
                    weights = file['/agent_data/weight'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        weights_by_sex = weights[sex_mask]

                        if weights_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # Dynamically calculate the number of bins
                            q75, q25 = np.percentile(weights_by_sex, [75, 25])
                            bin_width = 2 * (q75 - q25) * len(weights_by_sex) ** (-1/3)  # Freedman-Diaconis rule
                            bins = round((max(weights_by_sex) - min(weights_by_sex)) / bin_width)
                            ax.hist(weights_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                            ax.set_title(f'{base_name} - {sex_label} Agent Weights')
                            ax.set_xlabel('Weight')
                            ax.set_ylabel('Frequency')
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                            print(f"Plot for {sex_label} in {base_name} added to PDF.")
                        else:
                            print(f"No weight values found for {sex_label} in {base_name}.")

        print(f"Plots saved to: {pdf_filepath}")
    
    

    # Calculate Statistics
    def weight_statistics(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        stats_file_name = f"{base_name}_weight_statistics_by_sex.txt"
        stats_file_path = os.path.join(self.Data_output, stats_file_name)

        with h5py.File(self.hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                weights = file['/agent_data/weight'][:]
                sexes = file['/agent_data/sex'][:]  # This assumes 0 and 1 encoding for sexes

                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    weights_by_sex = weights[sex_mask]
                    if weights_by_sex.size > 0:
                        mean_weight = np.mean(weights_by_sex)
                        median_weight = np.median(weights_by_sex)
                        std_dev_weight = np.std(weights_by_sex, ddof=1)  # Use ddof=1 for sample standard deviation
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Weight: {mean_weight:.2f}\n")
                        output_file.write(f"  Median Weight: {median_weight:.2f}\n")
                        output_file.write(f"  Standard Deviation of Weight: {std_dev_weight:.2f}\n\n")
                    else:
                        output_file.write(f"No weight values found for {sex_label} in {base_name}.\n\n")

        print(f"Results written to: {stats_file_path}")




    def plot_body_depths(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        pdf_filename = f"{base_name}_Body_Depths_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(self.Data_output, pdf_filename)

        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(self.hdf_path, 'r') as file:
                if 'agent_data' in file:
                    body_depths = file['/agent_data/body_depth'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        body_depths_by_sex = body_depths[sex_mask]

                        if body_depths_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # Dynamically calculate the number of bins
                            q75, q25 = np.percentile(body_depths_by_sex, [75, 25])
                            bin_width = 2 * (q75 - q25) * len(body_depths_by_sex) ** (-1/3)  # Freedman-Diaconis rule
                            bins = round((max(body_depths_by_sex) - min(body_depths_by_sex)) / bin_width)
                            ax.hist(body_depths_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                            ax.set_title(f'{base_name} - {sex_label} Body Depths')
                            ax.set_xlabel('Body Depth')
                            ax.set_ylabel('Frequency')
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                            print(f"Plot for {sex_label} in {base_name} added to PDF.")
                        else:
                            print(f"No body depth values found for {sex_label} in {base_name}.")

        print(f"Plots saved to: {pdf_filepath}")
    

    # Calculate Statistics
    def body_depth_statistics(self):
        # Construct the output filename based on the HDF5 filename
        base_name = os.path.splitext(os.path.basename(self.hdf_path))[0]
        stats_file_name = f"{base_name}_body_depth_statistics_by_sex.txt"
        stats_file_path = os.path.join(self.Data_output, stats_file_name)

        with h5py.File(self.hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                body_depths = file['/agent_data/body_depth'][:]
                sexes = file['/agent_data/sex'][:]  # This assumes 0 and 1 encoding for sexes

                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    body_depths_by_sex = body_depths[sex_mask]
                    if body_depths_by_sex.size > 0:
                        mean_body_depth = np.mean(body_depths_by_sex)
                        median_body_depth = np.median(body_depths_by_sex)
                        std_dev_body_depth = np.std(body_depths_by_sex, ddof=1)  # Use ddof=1 for sample standard deviation
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Body Depth: {mean_body_depth:.2f}\n")
                        output_file.write(f"  Median Body Depth: {median_body_depth:.2f}\n")
                        output_file.write(f"  Standard Deviation of Body Depth: {std_dev_body_depth:.2f}\n\n")
                    else:
                        output_file.write(f"No body depth values found for {sex_label} in {base_name}.\n\n")

        print(f"Results written to: {stats_file_path}")
    
 
        
    # Find max bout duration
    def max_bout_duration(self):
        max_bout_file_path = os.path.join(self.Data_output, 'max_bout_duration_by_agent.txt')

        # Open the HDF5 file in read-only mode
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_dur' exist
            if 'agent_data' in hdf.keys() and 'bout_dur' in hdf['agent_data'].keys():
                # Access the 'bout_dur' data
                bout_dur_data = hdf['agent_data']['bout_dur'][:]

                # Dictionary to store the maximum bout duration for each agent
                max_durations_by_agent = {}

                # Iterate over each row (agent) in the 'bout_dur' dataset
                for agent_id, durations in enumerate(bout_dur_data):
                    # Calculate the maximum duration for this agent
                    max_duration = np.max(durations)
                    max_durations_by_agent[agent_id] = max_duration

                # Write the maximum bout duration for each agent to the file
                if max_durations_by_agent:
                    with open(max_bout_file_path, 'w') as output_file:
                        for agent_id, max_duration in max_durations_by_agent.items():
                            output_file.write(f"Maximum bout duration for agent {agent_id}: {max_duration}\n")
                        print(f"Results written to: {max_bout_file_path}")
                else:
                    print("No 'bout_dur' data found for any agent.")
            else:
                print("No 'bout_dur' data found in 'agent_data'.")
                    
                    

    def max_bout_duration_statistics(self):
        # File path for the output text file
        stats_file_path = os.path.join(self.Data_output, 'bout_duration_statistics.txt')

        # Open the HDF5 file in read-only mode
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_dur' exist
            if 'agent_data' in hdf.keys() and 'bout_dur' in hdf['agent_data'].keys():
                # Access the 'bout_dur' data
                bout_dur_data = hdf['agent_data']['bout_dur'][:]

                # Flatten the data if it's multidimensional to get all durations
                all_durations = bout_dur_data.flatten()

                # Calculate statistics
                mean_duration = np.mean(all_durations)
                median_duration = np.median(all_durations)
                std_dev_duration = np.std(all_durations, ddof=1)  # Use ddof=1 for sample standard deviation
                min_duration = np.min(all_durations)
                max_duration = np.max(all_durations)

                # Write the statistics to the file
                with open(stats_file_path, 'w') as output_file:
                    output_file.write("Bout Duration Statistics:\n")
                    output_file.write(f"Mean Bout Duration: {mean_duration}\n")
                    output_file.write(f"Median Bout Duration: {median_duration}\n")
                    output_file.write(f"Standard Deviation of Bout Duration: {std_dev_duration}\n")
                    output_file.write(f"Minimum Bout Duration: {min_duration}\n")
                    output_file.write(f"Maximum Bout Duration: {max_duration}\n")
                    print(f"Results written to: {stats_file_path}")
            else:
                print("No 'bout_dur' data found in 'agent_data'.")
                    
                    
                    
                                
    
    def max_bout_no(self):
        max_bout_file_path = os.path.join(self.Data_output, 'max_bout_no_by_agent.txt')

        # Open the HDF5 file in read-only mode
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_no' exist
            if 'agent_data' in hdf.keys() and 'bout_no' in hdf['agent_data'].keys():
                # Access the 'bout_no' data
                bout_no_data = hdf['agent_data']['bout_no'][:]

                # Dictionary to store the maximum bout number for each agent
                max_no_by_agent = {}

                # Iterate over each row (agent) in the 'bout_no' dataset
                for agent_id, bout_nos in enumerate(bout_no_data):
                    # Calculate the maximum bout number for this agent
                    max_no = np.max(bout_nos)
                    max_no_by_agent[agent_id] = max_no

                # Write the maximum bout number for each agent to the file
                if max_no_by_agent:
                    with open(max_bout_file_path, 'w') as output_file:
                        for agent_id, max_no in max_no_by_agent.items():
                            output_file.write(f"Maximum bout no for agent {agent_id}: {max_no}\n")
                        print(f"Results written to: {max_bout_file_path}")
                else:
                    print("No 'bout_no' data found for any agent.")
            else:
                print("No 'bout_no' data found in 'agent_data'.")
                    
                    
    def bout_no_statistics(self):
        stats_file_path = os.path.join(self.Data_output, 'bout_no_statistics.txt')

        # Open the HDF5 file in read-only mode
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_no' exist
            if 'agent_data' in hdf.keys() and 'bout_no' in hdf['agent_data'].keys():
                # Access the 'bout_no' data
                bout_no_data = hdf['agent_data']['bout_no'][:]

                # Flatten the data if it's multidimensional to get all bout numbers
                all_bout_nos = bout_no_data.flatten()

                # Calculate statistics
                mean_bout_no = np.mean(all_bout_nos)
                median_bout_no = np.median(all_bout_nos)
                std_dev_bout_no = np.std(all_bout_nos, ddof=1)  # Use ddof=1 for sample standard deviation
                min_bout_no = np.min(all_bout_nos)
                max_bout_no = np.max(all_bout_nos)

                # Write the statistics to the file
                with open(stats_file_path, 'w') as output_file:
                    output_file.write("Bout Number Statistics:\n")
                    output_file.write(f"Mean Bout Number: {mean_bout_no}\n")
                    output_file.write(f"Median Bout Number: {median_bout_no}\n")
                    output_file.write(f"Standard Deviation of Bout Number: {std_dev_bout_no}\n")
                    output_file.write(f"Minimum Bout Number: {min_bout_no}\n")
                    output_file.write(f"Maximum Bout Number: {max_bout_no}\n")
                    print(f"Results written to: {stats_file_path}")
            else:
                print("No 'bout_no' data found in 'agent_data'.")
                
                

    def kcal_statistics(self):
        # File path for the output text file
        stats_file_path = os.path.join(self.Data_output, 'kcal_statistics_by_sex.txt')
        
        # Open the HDF5 file in read-only mode
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Ensure 'agent_data', 'kcal', and 'sex' exist
            if 'agent_data' in hdf.keys() and 'kcal' in hdf['agent_data'].keys() and 'sex' in hdf['agent_data'].keys():
                # Access the 'kcal' data and convert it to float64 to prevent overflow
                kcal_data = hdf['agent_data']['kcal'][:].astype(np.float64)
                # Access the 'sex' data
                sex_data = hdf['agent_data']['sex'][:]
                
                # Separate kcal data by sex
                males_kcal = kcal_data[sex_data == 0]  # Assuming '0' represents males
                females_kcal = kcal_data[sex_data == 1]  # Assuming '1' represents females
                
                # Calculate statistics for males
                mean_kcal_males = np.mean(males_kcal)
                median_kcal_males = np.median(males_kcal)
                std_dev_kcal_males = np.std(males_kcal, ddof=1)  # Use ddof=1 for sample standard deviation
                min_kcal_males = np.min(males_kcal)
                max_kcal_males = np.max(males_kcal)
                
                # Calculate statistics for females
                mean_kcal_females = np.mean(females_kcal)
                median_kcal_females = np.median(females_kcal)
                std_dev_kcal_females = np.std(females_kcal, ddof=1)
                min_kcal_females = np.min(females_kcal)
                max_kcal_females = np.max(females_kcal)
                
                # Write the statistics to the file
                with open(stats_file_path, 'w') as output_file:
                    output_file.write("Kcal Statistics for Males:\n")
                    output_file.write(f"Mean Kcal: {mean_kcal_males}\n")
                    output_file.write(f"Median Kcal: {median_kcal_males}\n")
                    output_file.write(f"Standard Deviation of Kcal: {std_dev_kcal_males}\n")
                    output_file.write(f"Minimum Kcal: {min_kcal_males}\n")
                    output_file.write(f"Maximum Kcal: {max_kcal_males}\n\n")

                    output_file.write("Kcal Statistics for Females:\n")
                    output_file.write(f"Mean Kcal: {mean_kcal_females}\n")
                    output_file.write(f"Median Kcal: {median_kcal_females}\n")
                    output_file.write(f"Standard Deviation of Kcal: {std_dev_kcal_females}\n")
                    output_file.write(f"Minimum Kcal: {min_kcal_females}\n")
                    output_file.write(f"Maximum Kcal: {max_kcal_females}\n")
                    print(f"Results written to: {stats_file_path}")
            else:
                print("Required data ('kcal' and/or 'sex') not found in 'agent_data'.")
                
                
                


#TODO we need to find the FIRST agent based on the timesteps with the information and then go to the second one and so forth
    def Agent_Plot_Rectangle(self, rect_position):
        # rect_position should be a tuple or list like (x, y, width, height) for the rectangle's position and size

        hdf_filename = os.path.basename(self.hdf_path)
        
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Rectangle_Plots.pdf')
        txt_filepath = os.path.join(self.Data_output, 'Agents_In_Rectangle.txt')
        jpeg_filepath= os.path.join(self.Data_output, 'Agents_In_Rectangle.jpeg')

        with PdfPages(pdf_filepath) as pdf, open(txt_filepath, 'w') as txt_file, h5py.File(self.hdf_path, 'r') as hdf, rasterio.open(self.tif_path) as tif:
            x_coords = hdf['agent_data']['X'][:]
            y_coords = hdf['agent_data']['Y'][:]

            fig, ax = plt.subplots(figsize=(10, 8))
            show(tif.read(1), transform=tif.transform, ax=ax)  # Show the TIFF image as a background

            ax.scatter(x_coords.flatten(), y_coords.flatten(), alpha=0.5, s=1, c='orange')  # Agents plotted on top

            # Define the rectangle's position based on rect_position
            rect = Rectangle((rect_position[0], rect_position[1]), rect_position[2], rect_position[3], linewidth=2, edgecolor='red', fill=False)
            ax.add_patch(rect)
        
            ax.set_xlabel('Easting (E)')
            ax.set_ylabel('Northing (N)')
            ax.set_title("Agents in Rectangle")
            pdf.savefig(fig)
            fig.savefig(jpeg_filepath, format='jpeg')
            plt.close(fig)

            # Adjust the rectangle's logic to use the specific rect_position
            rect_x_min, rect_y_min, rect_width, rect_height = rect_position
            rect_x_max = rect_x_min + rect_width
            rect_y_max = rect_y_min + rect_height

            txt_file.write(f"Data from HDF5 file: {hdf_filename}\n\n")
            for agent_index in range(x_coords.shape[0]):
                entered = False
                for timestep in range(x_coords.shape[1]):
                    if rect_x_min <= x_coords[agent_index, timestep] <= rect_x_max and rect_y_min <= y_coords[agent_index, timestep] <= rect_y_max:
                        if not entered:  # Log the first timestep the agent enters the rectangle
                            entered = True
                            # Write detailed agent entry information
                            entry_info = f"Agent ID {agent_index} first timestep within the rectangle:\n  timestep: {timestep}\n  Easting: {x_coords[agent_index, timestep]}\n  Northing: {y_coords[agent_index, timestep]}\n\n"
                            txt_file.write(entry_info)
                            break  # Only log the first entry

        print(f"Plot saved to: {pdf_filepath}")
        print(f"IDs of agents within the rectangle written to: {txt_filepath}")
        print(f"JPEG image saved to: {jpeg_filepath}")
#TODO          
                   
            
    #  AgentVisualizer     
    def plot_agent_locations(self):
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Locations_Plots.pdf')
        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(self.hdf_path, 'r') as hdf:
                # Assuming 'X' and 'Y' are under 'agent_data'
                x_data = hdf['agent_data']['X'][:]
                y_data = hdf['agent_data']['Y'][:]

                # Assume each row in 'X' and 'Y' corresponds to a different agent
                for agent_index in range(x_data.shape[0]):
                    plt.figure(figsize=(10, 8))
                    plt.plot(x_data[agent_index], y_data[agent_index], marker='o', markersize=1)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title(f'Agent {agent_index} Locations')
                    pdf.savefig()
                    plt.close()

        print(f"Results written to: {pdf_filepath}")



    def plot_agent_locations_on_tiff(self):
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Locations_on_TIFF_Plots.pdf')
        tiff_output_path = os.path.join(self.Data_output, 'Agent_Locations_on_TIFF_Plots.tif')  # Corrected path for TIFF file

        with PdfPages(pdf_filepath) as pdf, h5py.File(self.hdf_path, 'r') as hdf:
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]

            fig, ax = plt.subplots(figsize=(10, 8))
            with rasterio.open(self.tif_path) as tiff:
                tiff_extent = tiff.bounds
                show(tiff, ax=ax, extent=tiff_extent)

                # Plot all agents' locations on the same figure
                ax.scatter(x_data.flatten(), y_data.flatten(), s=1, color='red', alpha=0.5)  # Flatten arrays for consistent scatter plot

                ax.set_title('All Agent Locations')
                plt.xlabel('Easting (E)')
                plt.ylabel('Northing (N)')
                pdf.savefig(fig)  # Save the current figure to the PDF

        # After saving to PDF, now save the same figure as a TIFF file
        plt.savefig(tiff_output_path, dpi=300)  # Specify dpi for higher resolution if needed
        plt.close(fig)

        print(f"Plot saved to {pdf_filepath} and {tiff_output_path}")
                
                
    def plot_agent_timestep_locations_on_tiff(self):
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Timestep_Locations_on_TIFF_Plots.pdf')
        tiff_folder = os.path.join(self.Data_output, 'Agent_TIFFs')
        os.makedirs(tiff_folder, exist_ok=True)  # Create the folder for individual TIFFs if it doesn't exist

        with PdfPages(pdf_filepath) as pdf, h5py.File(self.hdf_path, 'r') as hdf:
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]

            with rasterio.open(self.tif_path) as tiff:
                tiff_extent = tiff.bounds
                for agent_index in range(x_data.shape[0]):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    show(tiff, ax=ax, extent=tiff_extent)
                    ax.scatter(x_data[agent_index, :], y_data[agent_index, :], s=1, color='red', alpha=0.5)
                    ax.set_title(f'Agent {agent_index} Locations')

                    # Save the current figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Additionally, save the figure as an individual TIFF file
                    agent_tiff_filepath = os.path.join(tiff_folder, f'Agent_{agent_index}_Locations.tif')
                    fig.savefig(agent_tiff_filepath)  # This should be before plt.close(fig)
                    plt.close()

        print(f"Plots saved to {pdf_filepath} and individual TIFFs saved in {tiff_folder}")


    # Individual Agents
         
    def plot_agent_timestep_locations(self, pdf_filename='Individual_Agent_Timesteps_with_TIFF.pdf'):
        pdf_filepath = os.path.join(self.Data_output, pdf_filename)

        with PdfPages(pdf_filepath) as pdf, h5py.File(self.hdf_path, 'r') as hdf:
            # Assuming 'X' and 'Y' are under 'agent_data' and represent locations over time for each agent
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]

            with rasterio.open(self.tif_path) as src:
                tiff_extent = src.bounds
                for agent_index in range(x_data.shape[0]):
                    fig, ax = plt.subplots()
                    show(src, ax=ax, extent=tiff_extent)  # Show the TIFF image as the background
                    
                    # Plot the agent's location over time
                    ax.scatter(x_data[agent_index, :], y_data[agent_index, :], s=1, alpha=0.5, color='orange', label=f'Agent {agent_index}')
                    ax.set_xlabel('Easting (E)')
                    ax.set_ylabel('Northing (N)')
                    ax.set_title(f'Agent {agent_index} Timestep Locations')

                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"PDF booklet with TIFF background created: {pdf_filepath}")



    #Jump DF
    def plot_agent_jump_locations(self):
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Access the datasets
            x_coords = hdf['agent_data']['X'][:]
            y_coords = hdf['agent_data']['Y'][:]
            time_of_jump = hdf['agent_data']['time_of_jump'][:]

        # Setup PDF output
        pdf_path = os.path.join(self.Data_output, 'agent_jump_locations.pdf')

        # Plotting starts here
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all coordinates for context
        ax.scatter(x_coords.flatten(), y_coords.flatten(), color='lightgray', alpha=0.5, s=1, label='All Positions')

        # Filter and plot jump positions
        valid_jump_mask = (time_of_jump >= 0) & (time_of_jump < x_coords.shape[1]) & np.isfinite(time_of_jump)
        valid_jump_indices = np.nonzero(valid_jump_mask)
        jump_times = time_of_jump[valid_jump_mask].astype(int)

        jump_x_coords = x_coords[valid_jump_indices[0], jump_times]
        jump_y_coords = y_coords[valid_jump_indices[0], jump_times]
        ax.scatter(jump_x_coords, jump_y_coords, color='orange', marker='o', s=50, label='Jump Positions')

        ax.set_xlabel('Easting (E)')
        ax.set_ylabel('Northing (N)')
        ax.set_title('Agent Locations at Time of Jump')
        plt.legend()
        plt.tight_layout()

        # Save the current figure into the PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        print(f"Plot saved to: {pdf_path}")

            
            
    def plot_agent_jump_locations_tif(self):
        with h5py.File(self.hdf_path, 'r') as hdf:
            # Access the datasets
            x_coords = hdf['agent_data']['X'][:]
            y_coords = hdf['agent_data']['Y'][:]
            time_of_jump = hdf['agent_data']['time_of_jump'][:]

        # Setup PDF output
        pdf_path = os.path.join(self.Data_output, 'agent_jump_locations.pdf')

        # Plotting starts here
        fig, ax = plt.subplots(figsize=(10, 6))

        # Display TIFF as the background
        with rasterio.open(self.tif_path) as tiff:
            show(tiff, ax=ax, alpha=1)  # Adjust alpha as needed for background transparency
        
        # Plot all coordinates for context
        ax.scatter(x_coords.flatten(), y_coords.flatten(), color='lightgray', alpha=0.5, s=1, label='All Positions')

        # Filter and plot jump positions
        valid_jump_mask = (time_of_jump >= 0) & (time_of_jump < x_coords.shape[1]) & np.isfinite(time_of_jump)
        valid_jump_indices = np.nonzero(valid_jump_mask)
        jump_times = time_of_jump[valid_jump_mask].astype(int)

        jump_x_coords = x_coords[valid_jump_indices[0], jump_times]
        jump_y_coords = y_coords[valid_jump_indices[0], jump_times]
        ax.scatter(jump_x_coords, jump_y_coords, color='orange', marker='o', s=50, label='Jump Positions')

        ax.set_xlabel('Easting (E)')
        ax.set_ylabel('Northing (N)')
        ax.set_title('Agent Locations at Time of Jump')
        plt.legend()
        plt.tight_layout()
        # Save the current figure into the PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        print(f"Plot saved to: {pdf_path}")



    #Jump_Data_Statistics

    def calculate_statistics(self, hdf_path):
        stats_filepath = os.path.join(self.Data_output, 'jump_and_out_of_water_statistics.txt')

        with h5py.File(hdf_path, 'r') as hdf:
            if 'agent_data' in hdf:
                agent_data_group = hdf['agent_data']
                keys_of_interest = ['time_of_jump', 'time_out_of_water']

                with open(stats_filepath, 'w') as stats_file:
                    for key in keys_of_interest:
                        if key in agent_data_group:
                            dataset = np.array(agent_data_group[key])
                            # Filter out zeros to focus on non-zero values (assuming zeros indicate no event)
                            non_zero_values = dataset[dataset > 0]

                            if non_zero_values.size > 0:
                                mean_val = np.mean(non_zero_values)
                                min_val = np.min(non_zero_values)
                                max_val = np.max(non_zero_values)
                                std_dev_val = np.std(non_zero_values)

                                stats_file.write(f"Statistics for {key}:\n")
                                stats_file.write(f"Mean: {mean_val:.2f}\n")
                                stats_file.write(f"Minimum: {min_val:.2f}\n")
                                stats_file.write(f"Maximum: {max_val:.2f}\n")
                                stats_file.write(f"Standard Deviation: {std_dev_val:.2f}\n\n")
                            else:
                                stats_file.write(f"No non-zero values found for {key}.\n\n")
                    else:
                            stats_file.write(f"Key '{key}' does not exist in 'agent_data'.\n\n")

                print(f"Statistics written to: {stats_filepath}")
            else:
                print("'agent_data' key does not exist in the file.")


    # Heatmap of ALL Agents
    
    def plot_agent_timestep_heatmap(self):
        pdf_filepath = os.path.join(self.Data_output, 'Agent_Timestep_Frequency_Heatmap.pdf')

        with PdfPages(pdf_filepath) as pdf, h5py.File(self.hdf_path, 'r') as hdf:
            # Access 'X' and 'Y' datasets under 'agent_data'
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]
            
            # Flatten the arrays to get all x and y coordinates across all timesteps
            x_flat = x_data.flatten()
            y_flat = y_data.flatten()
            
            # Generate a 2D histogram with the flattened x and y data to visualize frequency
            heatmap, xedges, yedges = np.histogram2d(x_flat, y_flat, bins=100, density=False)
            heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
            
            with rasterio.open(self.tif_path) as src:
                fig, ax = plt.subplots(figsize=(10, 8))
                show(src, ax=ax, with_bounds=True)
                
                # Overlay the heatmap
                im = ax.pcolormesh(xedges, yedges, heatmap_masked.T, cmap='hot', alpha=0.7)
                
                plt.xlabel('Easting (E)')
                plt.ylabel('Northing (N)')
                plt.title('Heatmap of Agent Frequencies by Timestep')
                plt.colorbar(im, ax=ax, label='Frequency')
                
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"Results written to: {pdf_filepath}")

    # Agent Jump Location Heat Map

    def plot_agent_jump_heatmap(self, tif_path):
        with h5py.File(self.hdf_path, 'r') as hdf:
            if 'agent_data' in hdf:
                x_coords = hdf['agent_data']['X'][:]
                y_coords = hdf['agent_data']['Y'][:]
                time_of_jump = hdf['agent_data']['time_of_jump'][:]
            
                # Filter out the invalid jump times
                valid_mask = time_of_jump > 0
                valid_x = x_coords[valid_mask]
                valid_y = y_coords[valid_mask]

                if np.any(valid_mask):
                    plt.figure(figsize=(10, 6))
                    with rasterio.open(tif_path) as tiff:
                        show(tiff, alpha=0.5)  # Show the TIFF image as a background
                        
                    # Create a heatmap of jump locations
                    plt.hexbin(valid_x, valid_y, gridsize=50, cmap='hot', bins='log')
                    plt.colorbar(label='log10(N)')
                    plt.xlabel('Easting (E)')
                    plt.ylabel('Northing (N)')
                    plt.title('Heatmap of Agent Jump Locations')

                    # Save the plot
                    plot_path = os.path.join(self.Data_output, 'agent_jump_locations_heatmap.png')
                    plt.savefig(plot_path)
                    plt.show
                    plt.close()
                
                    print(f"Heatmap plot saved to: {plot_path}")
                else:
                    print("No valid jump locations found.")
            else:
                print("'agent_data' key does not exist in the HDF5 file.")
 

    #Use these to run code

multi_summarization=Multi_Summarization(directory_path, hdf_path, tif_path, Data_output)
tiff_image, tiff_extent = multi_summarization.load_tiff_image()
#TODO multi_summarization.plot_lengths()
#TODO multi_summarization.length_statistics()
#TODO multi_summarization.plot_weights()
#TODO multi_summarization.weight_statistics()
#TODO multi_summarization.plot_body_depths()
#TODO multi_summarization.body_depth_statistics()
#TODO multi_summarization.max_bout_duration()
#TODO multi_summarization.max_bout_duration_statistics()
#TODO multi_summarization.max_bout_no()
#TODO multi_summarization.bout_no_statistics()
#TODO multi_summarization.kcal_statistics()
#TODO multi_summarization.Agent_Plot_Rectangle(rect_position)
#TODO multi_summarization.plot_agent_locations()
#TODO multi_summarization.plot_agent_locations_on_tiff()
#multi_summarization.plot_agent_timestep_locations_on_tiff()    Optional to have, might not need with such large datasets
#multi_summarization.plot_agent_timestep_locations() Optional for code, we don't need unless asked for
#TODO multi_summarization.plot_agent_jump_locations()
#TODO multi_summarization.plot_agent_jump_locations_tif()
#TODO multi_summarization.calculate_statistics(hdf_path)
#TODO multi_summarization.plot_agent_timestep_heatmap()
multi_summarization.plot_agent_jump_heatmap(tif_path)






































