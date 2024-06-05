
"""
Created on Tue Nov 27 14:31:15 2023

@author: EMuhlestein

Script intent: Multi model summary is to be used to summarize multiple models at once. Once a model
has finished running, this script will take said model as well as others and complete the summarization
without having to run the class Summarization on multiple occasions.
"""

#Import Dependencies
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform
from matplotlib.backends.backend_pdf import PdfPages
from rasterio.plot import show
import h5py
import geopandas as gpd
from joblib import Parallel, delayed
import glob
import re




#Identify the directory path
directory_path = r"J:\2819\005\Calcs\ABM\Output\Multi_TEST"
exports_path=r'J:\2819\005\Calcs\ABM\Output\Multi_TEST'
Data_output=r'J:\2819\005\Calcs\ABM\Output\Multi_TEST'
tif_path=r'C:\Users\EMuhlestein\Documents\ABM_TEST\val_TEST\val_TEST\elev.tif'
shapefile_path=r'J:\2819\005\Calcs\ABM\Output\Multi_TEST\Rectangle.shp'

                                                                              #Cell width and cell height are measured in meters
class Multi_Summarization:
    def __init__(self, directory_path, tif_path, Data_output, shapefile_path, cell_width=5, cell_height=5):
        self.directory_path = directory_path
        self.tif_path = tif_path
        self.Data_output = Data_output
        self.shapefile_path = shapefile_path
        self.lengths = []
        self.weights = []
        self.body_depths = []
        self.data_list = []  # Initialize data_list
        self.bout_duration_list = []  # Initialize bout_duration_list
        self.ids_in_rectangle = set()
        self.exports_path = directory_path  # Make sure exports_path is properly set
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cumulative_avg_agent_counts = None
        self.cumulative_std_agent_counts = None
        self.cumulative_counts = None
       


    def find_hdf_files(self):
        # Find all .h5 files in directory and subdirectories
        pattern = os.path.join(self.directory_path, '**', '*.h5')
        hdf_files = glob.glob(pattern, recursive=True)
        return hdf_files
    

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


    def plot_lengths(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        pdf_filename = f"{base_name}_Lengths_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(output_folder, pdf_filename)
    
        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(hdf_path, 'r') as file:
                if 'agent_data' in file:
                    lengths = file['/agent_data/length'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
                    
                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        lengths_by_sex = lengths[sex_mask]
                        lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]  # Remove NaN values

                        if lengths_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            try:
                                q75, q25 = np.percentile(lengths_by_sex, [75, 25])
                                bin_width = 2 * (q75 - q25) * len(lengths_by_sex) ** (-1/3)  # Freedman-Diaconis rule

                                # Ensure bin_width is not zero or NaN
                                if bin_width <= 0 or np.isnan(bin_width):
                                    bin_width = max(lengths_by_sex) - min(lengths_by_sex) / 10  # Default to 10 bins if calculation fails

                                bins = max(1, round((max(lengths_by_sex) - min(lengths_by_sex)) / bin_width))
                                ax.hist(lengths_by_sex, bins=bins, alpha=0.7, color='blue' if sex == 0 else 'pink')
                            except Exception as e:
                                print(f"Error in calculating histogram for {sex_label}: {e}")
                                continue

                            ax.set_title(f'{base_name} - {sex_label} Agent Lengths')
                            ax.set_xlabel('Length (mm)')
                            ax.set_ylabel('Frequency')
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                        else:
                            print(f"No length values found for {sex_label}.")
    
        print(f"Plots saved to: {pdf_filepath}")
        
            
        
    # Calculate Statistics
    def length_statistics(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        stats_file_name = f"{base_name}_length_statistics_by_sex.txt"
        stats_file_path = os.path.join(output_folder, stats_file_name)

        with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                lengths = file['/agent_data/length'][:]
                sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    lengths_by_sex = lengths[sex_mask]
                    lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]  # Filter out NaN values

                    if lengths_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                        mean_length = np.mean(lengths_by_sex)
                        median_length = np.median(lengths_by_sex)
                        std_dev_length = np.std(lengths_by_sex, ddof=1)  # ddof=1 ensures the sample standard deviation
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\n")
                        output_file.write(f"  Median Length: {median_length:.2f}\n")
                        output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\n\n")
                    elif lengths_by_sex.size == 1:
                        # Handle single value case
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                        output_file.write(f"  Length: {lengths_by_sex[0]:.2f}\n\n")
                    else:
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"No valid length values found for {sex_label}.\n\n")

        print(f"Results written to: {stats_file_path}")


    # class weights


    # Plot weights
    def plot_weights(self, hdf_path):
        base_directory = os.path.dirname(hdf_path)
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        pdf_filename = f"{base_name}_Weights_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(base_directory, pdf_filename)

        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(hdf_path, 'r') as file:
                if 'agent_data' in file:
                    weights = file['/agent_data/weight'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
                    
                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        weights_by_sex = weights[sex_mask]
                        weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]  # Remove NaN values

                        if weights_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            try:
                                q75, q25 = np.percentile(weights_by_sex, [75, 25])
                                iqr = q75 - q25
                                if iqr > 0:
                                    bin_width = 2 * iqr * len(weights_by_sex) ** (-1/3)
                                    bins = max(1, round((max(weights_by_sex) - min(weights_by_sex)) / bin_width))
                                else:
                                    # Fallback to a default number of bins if IQR is zero
                                    bins = 10

                                ax.hist(weights_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                ax.set_title(f'{base_name} - {sex_label} Agent Weights')
                                ax.set_xlabel('Weight')
                                ax.set_ylabel('Frequency')
                                plt.tight_layout()
                                pdf.savefig(fig)
                                plt.close()
                            except Exception as e:
                                print(f"Error in calculating histogram for {sex_label}: {e}")
                                plt.close(fig)  # Ensure plot is closed on error
                            else:
                                print(f"No weight values found for {sex_label} in {base_name}.")

        print(f"Plots saved to: {pdf_filepath}")
    
    

    # Calculate Statistics
    def weight_statistics(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        stats_file_name = f"{base_name}_weight_statistics_by_sex.txt"
        stats_file_path = os.path.join(output_folder, stats_file_name)

        with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                weights = file['/agent_data/weight'][:]
                sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    weights_by_sex = weights[sex_mask]
                    weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]  # Filter out NaN values

                    if weights_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                        mean_weight = np.mean(weights_by_sex)
                        median_weight = np.median(weights_by_sex)
                        std_dev_weight = np.std(weights_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Weight: {mean_weight:.2f}\n")
                        output_file.write(f"  Median Weight: {median_weight:.2f}\n")
                        output_file.write(f"  Standard Deviation of Weight: {std_dev_weight:.2f}\n\n")
                    elif weights_by_sex.size == 1:
                        # Handle single value case
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                        output_file.write(f"  Weight: {weights_by_sex[0]:.2f}\n\n")
                    else:
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"No valid weight values found for {sex_label}.\n\n")

        print(f"Results written to: {stats_file_path}")



    def plot_body_depths(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        pdf_filename = f"{base_name}_Body_Depths_By_Sex_Comparison.pdf"
        pdf_filepath = os.path.join(output_folder, pdf_filename)
        
        with PdfPages(pdf_filepath) as pdf:
            with h5py.File(hdf_path, 'r') as file:
                if 'agent_data' in file:
                    body_depths = file['/agent_data/body_depth'][:]
                    sexes = file['/agent_data/sex'][:]
                    
                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        sex_mask = sexes == sex
                        body_depths_by_sex = body_depths[sex_mask]
                        body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]
                        
                        if body_depths_by_sex.size > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            try:
                                q75, q25 = np.percentile(body_depths_by_sex, [75, 25])
                                iqr = q75 - q25
                                if iqr > 0:
                                    bin_width = 2 * iqr * len(body_depths_by_sex) ** (-1/3)
                                else:
                                    bin_width = (max(body_depths_by_sex) - min(body_depths_by_sex)) / max(10, len(body_depths_by_sex))  # Avoid zero division
                            
                                bins = max(1, round((max(body_depths_by_sex) - min(body_depths_by_sex)) / bin_width))
                                ax.hist(body_depths_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                ax.set_title(f'{base_name} - {sex_label} Body Depths')
                                ax.set_xlabel('Body Depth')
                                ax.set_ylabel('Frequency')
                                plt.tight_layout()
                                pdf.savefig(fig)
                                plt.close()
                            except Exception as e:
                                print(f"Error in calculating histogram for {sex_label}: {e}")
                                plt.close(fig)
                            else:
                                print(f"No body depth values found for {sex_label} in {base_name}.")

        print(f"Plots saved to: {pdf_filepath}")
    

    # Calculate Statistics
    def body_depth_statistics(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        stats_file_name = f"{base_name}_body_depth_statistics_by_sex.txt"
        stats_file_path = os.path.join(output_folder, stats_file_name)
        
        with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                body_depths = file['/agent_data/body_depth'][:]
                sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
                
                for sex in np.unique(sexes):
                    sex_mask = sexes == sex
                    body_depths_by_sex = body_depths[sex_mask]
                    body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]  # Filter out NaN values
                    
                    if body_depths_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                        mean_body_depth = np.mean(body_depths_by_sex)
                        median_body_depth = np.median(body_depths_by_sex)
                        std_dev_body_depth = np.std(body_depths_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Body Depth: {mean_body_depth:.2f}\n")
                        output_file.write(f"  Median Body Depth: {median_body_depth:.2f}\n")
                        output_file.write(f"  Standard Deviation of Body Depth: {std_dev_body_depth:.2f}\n\n")
                    elif body_depths_by_sex.size == 1:
                        # Handle single value case
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                        output_file.write(f"  Body Depth: {body_depths_by_sex[0]:.2f}\n\n")
                    else:
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"No valid body depth values found for {sex_label}.\n\n")

        print(f"Results written to: {stats_file_path}")
    
 
                          
    
    def max_bout_no(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        max_bout_file_path = os.path.join(output_folder, f"{base_name}_max_bout_no_by_agent.txt")

        # Open the HDF5 file in read-only mode
        with h5py.File(hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_no' exist
            if 'agent_data' in hdf.keys() and 'bout_no' in hdf['agent_data'].keys():
                # Access the 'bout_no' data
                bout_no_data = hdf['agent_data']['bout_no'][:]
                
                # Dictionary to store the maximum bout number for each agent
                max_no_by_agent = {}
                
                # Iterate over each row (agent) in the 'bout_no' dataset
                for agent_id, bout_nos in enumerate(bout_no_data):
                    # Handle cases where bout numbers might be NaN or empty
                    valid_bout_nos = bout_nos[~np.isnan(bout_nos)]  # Remove NaN values
                    if valid_bout_nos.size > 0:
                        max_no = np.max(valid_bout_nos)
                        max_no_by_agent[agent_id] = max_no

                # Write the maximum bout number for each agent to the file
                if max_no_by_agent:
                    with open(max_bout_file_path, 'w') as output_file:
                        for agent_id, max_no in max_no_by_agent.items():
                            output_file.write(f"Maximum bout no for agent {agent_id}: {max_no}\n")
                            print(f"Results written to: {max_bout_file_path}")
                        else:
                            print("No valid 'bout_no' data found for any agent.")
                else:
                    print("No 'bout_no' data found in 'agent_data'.")
                    
                    
    def bout_no_statistics(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        stats_file_path = os.path.join(output_folder, f"{base_name}_bout_no_statistics.txt")
        
        # Open the HDF5 file in read-only mode
        with h5py.File(hdf_path, 'r') as hdf:
            # Ensure 'agent_data' and 'bout_no' exist
            if 'agent_data' in hdf.keys() and 'bout_no' in hdf['agent_data'].keys():
                # Access the 'bout_no' data
                bout_no_data = hdf['agent_data']['bout_no'][:]
                
                # Handle potential multidimensional array and NaN values
                bout_no_data = bout_no_data.flatten()  # Flatten the data
                bout_no_data = bout_no_data[~np.isnan(bout_no_data)]  # Remove NaN values
                
                if bout_no_data.size > 0:
                    # Calculate statistics
                    mean_bout_no = np.mean(bout_no_data)
                    median_bout_no = np.median(bout_no_data)
                    std_dev_bout_no = np.std(bout_no_data, ddof=1)  # Use ddof=1 for sample standard deviation
                    min_bout_no = np.min(bout_no_data)
                    max_bout_no = np.max(bout_no_data)
                    
                    # Write the statistics to the file
                    with open(stats_file_path, 'w') as output_file:
                        output_file.write("Bout Number Statistics:\n")
                        output_file.write(f"Mean Bout Number: {mean_bout_no:.2f}\n")
                        output_file.write(f"Median Bout Number: {median_bout_no:.2f}\n")
                        output_file.write(f"Standard Deviation of Bout Number: {std_dev_bout_no:.2f}\n")
                        output_file.write(f"Minimum Bout Number: {min_bout_no}\n")
                        output_file.write(f"Maximum Bout Number: {max_bout_no}\n")
                        print(f"Results written to: {stats_file_path}")
                else:
                    print("No valid 'bout_no' data found for any agent.")
            else:
                print("No 'bout_no' data found in 'agent_data'.")
                

    def kcal_statistics(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        stats_file_name = f"{base_name}_kcal_statistics_by_sex.txt"
        stats_file_path = os.path.join(output_folder, stats_file_name)

        with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
            if 'agent_data' in file:
                kcals = file['/agent_data/kcal'][:]
                sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
            
                for i, (kcal, sex) in enumerate(zip(kcals, sexes)):
                    kcal_values = kcal[~np.isnan(kcal)]  # Remove NaN values
                
                    if kcal_values.size > 0:
                        mean_kcal = np.mean(kcal_values)
                        median_kcal = np.median(kcal_values)
                        std_dev_kcal = np.std(kcal_values, ddof=1)  # Use ddof=1 for sample standard deviation
                        min_kcal = np.min(kcal_values)
                        max_kcal = np.max(kcal_values)
                        sex_label = 'Male' if sex == 0 else 'Female'
                    
                        output_file.write(f"Agent {i + 1} ({sex_label}):\n")
                        output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                        output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                        output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                        output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                        output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                    else:
                        sex_label = 'Male' if sex == 0 else 'Female'
                        output_file.write(f"No valid kcal values found for Agent {i + 1} ({sex_label}).\n\n")
                        
                        

                    
                    
  
    def kcal_statistics_directory(self):
        # Prepare to collect cumulative statistics
        cumulative_stats = {}
        
        # Walk through all files in the directory and its subdirectories
        for subdir, dirs, files in os.walk(self.directory_path):  # Use self.directory_path
            for file in files:
                if file.endswith('.h5'):
                    hdf_path = os.path.join(subdir, file)
                    with h5py.File(hdf_path, 'r') as hdf_file:
                        if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys():
                            kcals = hdf_file['/agent_data/kcal'][:]
                            sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
                            
                            for sex in np.unique(sexes):
                                sex_label = 'Male' if sex == 0 else 'Female'
                                if sex_label not in cumulative_stats:
                                    cumulative_stats[sex_label] = []
                                
                                sex_mask = sexes == sex
                                kcals_by_sex = kcals[sex_mask]
                                kcals_by_sex = kcals_by_sex[~np.isnan(kcals_by_sex)]  # Remove NaN values
                                
                                cumulative_stats[sex_label].extend(kcals_by_sex)
        
        # Compute and print cumulative statistics
        stats_file_path = os.path.join(self.directory_path, "kcal_statistics_directory.txt")
        with open(stats_file_path, 'w') as output_file:
            for sex_label, values in cumulative_stats.items():
                if values:
                    values = np.array(values)
                    mean_kcal = np.mean(values)
                    median_kcal = np.median(values)
                    std_dev_kcal = np.std(values, ddof=1)
                    min_kcal = np.min(values)
                    max_kcal = np.max(values)
                    
                    output_file.write(f"Cumulative Statistics for {sex_label}:\n")
                    output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                    output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                    output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                    output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                    output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                else:
                    output_file.write(f"No valid kcal values found for {sex_label}.\n\n")
        
        print(f"Cumulative results written to: {stats_file_path}")
           
        
    def kcal_histograms_directory(self):
        # Old functionality to calculate cumulative averages and plot histograms
        stats_file_path = os.path.join(self.directory_path, "kcal_statistics_directory.txt")

        # Dictionary to hold data for males and females
        kcal_data = {'Male': {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []},
                     'Female': {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []}}

        # Read the statistics file
        with open(stats_file_path, 'r') as file:
            lines = file.readlines()
            current_sex = None
            for line in lines:
                if "Statistics for Male" in line:
                    current_sex = 'Male'
                elif "Statistics for Female" in line:
                    current_sex = 'Female'
                elif "Average (Mean) Kcal" in line:
                    kcal_data[current_sex]['Mean'].append(float(re.search(r"Average \(Mean\) Kcal: ([\d.]+)", line).group(1)))
                elif "Median Kcal" in line:
                    kcal_data[current_sex]['Median'].append(float(re.search(r"Median Kcal: ([\d.]+)", line).group(1)))
                elif "Standard Deviation of Kcal" in line:
                    kcal_data[current_sex]['Std Dev'].append(float(re.search(r"Standard Deviation of Kcal: ([\d.]+)", line).group(1)))
                elif "Minimum Kcal" in line:
                    kcal_data[current_sex]['Min'].append(float(re.search(r"Minimum Kcal: ([\d.]+)", line).group(1)))
                elif "Maximum Kcal" in line:
                    kcal_data[current_sex]['Max'].append(float(re.search(r"Maximum Kcal: ([\d.]+)", line).group(1)))

        # Create a PDF to save the cumulative histograms
        pdf_filename = os.path.join(self.directory_path, "kcal_histograms.pdf")
        with PdfPages(pdf_filename) as pdf:
            for sex, data in kcal_data.items():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Colors for the different data types
                colors = {'Mean': 'blue', 'Median': 'green', 'Std Dev': 'orange', 'Min': 'red', 'Max': 'purple'}
                
                # Plot each data type
                for dtype, values in data.items():
                    if values:
                        ax.hist(values, bins=50, alpha=0.7, edgecolor='black', color=colors[dtype], label=dtype)
                
                ax.set_title(f"Kcal Distribution for {sex} Agents")
                ax.set_xlabel("Kcal")
                ax.set_ylabel("Frequency")
                ax.legend()

                # Add statistics as text on the plot
                mean_kcal = np.mean(data['Mean']) if data['Mean'] else 0
                median_kcal = np.median(data['Median']) if data['Median'] else 0
                std_dev_kcal = np.mean(data['Std Dev']) if data['Std Dev'] else 0
                min_kcal = np.min(data['Min']) if data['Min'] else 0
                max_kcal = np.max(data['Max']) if data['Max'] else 0
                
                stats_text = (
                    f"Mean: {mean_kcal:.2f}\n"
                    f"Median: {median_kcal:.2f}\n"
                    f"Std Dev: {std_dev_kcal:.2f}\n"
                    f"Min: {min_kcal:.2f}\n"
                    f"Max: {max_kcal:.2f}"
                )
                plt.figtext(0.15, 0.7, stats_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

                pdf.savefig(fig)
                plt.close(fig)

        print(f"Cumulative histograms saved to: {pdf_filename}")

        # New functionality to create individual histograms for each agent
        agent_data = {'Male': {}, 'Female': {}}

        # Walk through all files in the directory and its subdirectories
        for subdir, dirs, files in os.walk(self.directory_path):
            for file in files:
                if 'kcal_statistics_by_sex' in file:
                    txt_path = os.path.join(subdir, file)
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        current_sex = None
                        current_agent = None
                        for line in lines:
                            if "Agent" in line:
                                match = re.search(r'Agent (\d+) \((Male|Female)\):', line)
                                if match:
                                    current_agent = int(match.group(1))
                                    current_sex = match.group(2)
                                    if current_agent not in agent_data[current_sex]:
                                        agent_data[current_sex][current_agent] = {'Mean': [], 'Median': [], 'Std Dev': [], 'Min': [], 'Max': []}
                            elif "Average (Mean) Kcal" in line:
                                agent_data[current_sex][current_agent]['Mean'].append(float(re.search(r"Average \(Mean\) Kcal: ([\d.]+)", line).group(1)))
                            elif "Median Kcal" in line:
                                agent_data[current_sex][current_agent]['Median'].append(float(re.search(r"Median Kcal: ([\d.]+)", line).group(1)))
                            elif "Standard Deviation of Kcal" in line:
                                agent_data[current_sex][current_agent]['Std Dev'].append(float(re.search(r"Standard Deviation of Kcal: ([\d.]+)", line).group(1)))
                            elif "Minimum Kcal" in line:
                                agent_data[current_sex][current_agent]['Min'].append(float(re.search(r"Minimum Kcal: ([\d.]+)", line).group(1)))
                            elif "Maximum Kcal" in line:
                                agent_data[current_sex][current_agent]['Max'].append(float(re.search(r"Maximum Kcal: ([\d.]+)", line).group(1)))

        # Calculate the average values for each agent
        averaged_data = {'Male': {}, 'Female': {}}
        for sex, agents in agent_data.items():
            for agent, values in agents.items():
                averaged_data[sex][agent] = {
                    'Mean': np.mean(values['Mean']) if values['Mean'] else 0,
                    'Median': np.mean(values['Median']) if values['Median'] else 0,
                    'Std Dev': np.mean(values['Std Dev']) if values['Std Dev'] else 0,
                    'Min': np.mean(values['Min']) if values['Min'] else 0,
                    'Max': np.mean(values['Max']) if values['Max'] else 0
                }

        # Find the maximum recorded kcal average across all agents for the y-axis limit
        max_kcal_value = max(
            max(kcal_data['Male']['Max']) if kcal_data['Male']['Max'] else 0,
            max(kcal_data['Female']['Max']) if kcal_data['Female']['Max'] else 0
        )

        # Create a PDF to save the individual histograms
        pdf_filename_individual = os.path.join(self.directory_path, "individual_kcal_histograms.pdf")
        with PdfPages(pdf_filename_individual) as pdf:
            for sex, agents in averaged_data.items():
                for agent, values in agents.items():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Colors for the different data types
                    colors = ['blue', 'green', 'orange', 'red', 'purple']
                    labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
                    agent_values = [values[label] for label in labels]
                    
                    # Plot the agent's data
                    ax.bar(labels, agent_values, color=colors, alpha=0.7, edgecolor='black')
                    
                    ax.set_title(f"Kcal Distribution for Agent {agent} ({sex})")
                    ax.set_xlabel("Kcal Type")
                    ax.set_ylabel("Value")
                    ax.set_ylim([0, max_kcal_value])  # Set y-axis limit

                    # Add statistics as text on the plot
                    stats_text = (
                        f"Mean: {values['Mean']:.2f}\n"
                        f"Median: {values['Median']:.2f}\n"
                        f"Std Dev: {values['Std Dev']:.2f}\n"
                        f"Min: {values['Min']:.2f}\n"
                        f"Max: {values['Max']:.2f}"
                    )
                    plt.figtext(0.15, 0.7, stats_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"Individual histograms saved to: {pdf_filename_individual}")
        
        
        

                
                
    def process_timestep(timestep, x_coords, y_coords, gdf, tif_crs):
        # Generate GeoDataFrame for current timestep coordinates
        agent_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_coords, y_coords), crs=tif_crs)
        # Spatial join to find agents within the shapefile area
        agents_in_shapefile = gpd.sjoin(agent_points, gdf, how="inner", predicate='intersects')
        return [{'timestep': timestep, 'agent_id': idx, 'coordinates': (x_coords[idx], y_coords[idx])}
                for idx in agents_in_shapefile.index]    


#Agents that pass into rectangle

    def Agent_Plot_Rectangle(self, shapefile_path, hdf_path):
        hdf_filename = os.path.basename(hdf_path)
        output_folder = os.path.dirname(hdf_path)

        pdf_filepath = os.path.join(output_folder, f'{hdf_filename}_Agent_Shapefile_Plots.pdf')
        txt_filepath = os.path.join(output_folder, f'{hdf_filename}_Agents_In_Shapefile.txt')
        jpeg_filepath = os.path.join(output_folder, f'{hdf_filename}_Agents_In_Shapefile.jpeg')

        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Pre-read data from TIFF to avoid "Dataset is closed" error
        with rasterio.open(self.tif_path) as tif:
            tif_data = tif.read(1)
            tif_transform = tif.transform
            tif_crs = tif.crs

        if gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)

        entries = []

        with h5py.File(hdf_path, 'r') as hdf, PdfPages(pdf_filepath) as pdf:
            x_coords = hdf['agent_data']['X'][:]
            y_coords = hdf['agent_data']['Y'][:]

            fig, ax = plt.subplots(figsize=(10, 8))
            show(tif_data, transform=tif_transform, ax=ax)
            gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)

            for timestep in range(x_coords.shape[1]):
                agent_x = x_coords[:, timestep]
                agent_y = y_coords[:, timestep]
                agent_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(agent_x, agent_y), crs=tif_crs)
                agents_in_shapefile = gpd.sjoin(agent_points, gdf, how="inner", predicate='intersects')

                for idx in agents_in_shapefile.index:
                    if idx not in [entry['agent_id'] for entry in entries]:
                        entries.append({
                            'timestep': timestep,
                            'agent_id': idx,
                            'coordinates': (agent_x[idx], agent_y[idx])
                        })

                # Plot agents for each timestep
                ax.scatter(agent_x, agent_y, alpha=0.5, s=1, c='orange')

            pdf.savefig(fig)
            plt.close(fig)

            try:
                fig.savefig(jpeg_filepath, format='jpeg')
                print(f"JPEG image saved to: {jpeg_filepath}")
            except Exception as e:
                print(f"Error saving JPEG image: {e}")

        with open(txt_filepath, 'w') as txt_file:
            txt_file.write(f"Data from HDF5 file: {hdf_filename}\n\n")
            txt_file.write("Agents entering the shapefile area sequentially:\n")
            for entry in entries:
                txt_file.write(f"Agent ID {entry['agent_id']} entered at timestep {entry['timestep']} with coordinates {entry['coordinates']}\n")

        print(f"Plot saved to: {pdf_filepath}")
        print(f"Details of agents entering the shapefile area written to: {txt_filepath}")
                   
            
#  AgentVisualizer     
    def plot_agent_locations(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        pdf_filepath = os.path.join(output_folder, f'{base_name}_Agent_Locations_Plots_Efficient.pdf')

        with h5py.File(hdf_path, 'r') as hdf, PdfPages(pdf_filepath) as pdf:
            # Load all data at once
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]

            # Create a single plot for all agents
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.get_cmap('viridis', np.amax(x_data.shape[0]))
        
            # Plot each agent. Consider using a subset if there are too many agents.
            for agent_index in range(x_data.shape[0]):
                ax.plot(x_data[agent_index], y_data[agent_index], color=cmap(agent_index), alpha=0.5, marker='o', markersize=1, linestyle='')

            ax.set_xlabel('Easting (E)')
            ax.set_ylabel('Northing (N)')
            ax.set_title('All Agent Locations')
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Efficient plot saved to: {pdf_filepath}")



    def plot_agent_locations_on_tiff(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        pdf_filepath = os.path.join(output_folder, f'{base_name}_Optimized_Agent_Locations_on_TIFF_Plots.pdf')
        tiff_output_path = os.path.join(output_folder, f'{base_name}_Optimized_Agent_Locations_on_TIFF_Plots.tif')

        with h5py.File(hdf_path, 'r') as hdf, rasterio.open(self.tif_path) as tiff:
            x_data = hdf['agent_data']['X'][:]
            y_data = hdf['agent_data']['Y'][:]
        
            # Consider subsampling for very large datasets
            # For example, take every 10th point (adjust as needed)
            subsample_factor = 10
            x_data_subsampled = x_data[:, ::subsample_factor].flatten()
            y_data_subsampled = y_data[:, ::subsample_factor].flatten()

            fig, ax = plt.subplots(figsize=(10, 8))
            tiff_extent = tiff.bounds
            show(tiff, ax=ax, extent=tiff_extent)

            # Plot using subsampled data
            ax.scatter(x_data_subsampled, y_data_subsampled, s=1, color='red', alpha=0.5)

            ax.set_title('All Agent Locations')
            plt.xlabel('Easting (E)')
            plt.ylabel('Northing (N)')

            # Use PdfPages context manager to save the figure to PDF
            with PdfPages(pdf_filepath) as pdf:
                pdf.savefig(fig)

            # Save the same figure as a TIFF file with potentially lower dpi if high resolution is not needed
            plt.savefig(tiff_output_path, dpi=150)
            plt.close(fig)

            print(f"Optimized plot saved to {pdf_filepath} and {tiff_output_path}")
                

    # Heatmap of ALL Agents
    
    def plot_agent_timestep_heatmap(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        pdf_filepath = os.path.join(output_folder, f'{base_name}_Agent_Timestep_Frequency_Heatmap_Optimized.pdf')

        with PdfPages(pdf_filepath) as pdf, h5py.File(hdf_path, 'r') as hdf, rasterio.open(self.tif_path) as src:
            # Directly access and flatten the data arrays
            x_flat = hdf['agent_data']['X'][:].flatten()
            y_flat = hdf['agent_data']['Y'][:].flatten()
    
            # Generate the 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x_flat, y_flat, bins=100, density=False)
            heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            show(src, ax=ax, with_bounds=True)  # Display TIFF data
            im = ax.pcolormesh(xedges, yedges, heatmap_masked.T, cmap='hot', alpha=0.7)  # Overlay the heatmap

            plt.xlabel('Easting (E)')
            plt.ylabel('Northing (N)')
            plt.title('Heatmap of Agent Frequencies by Timestep')
            plt.colorbar(im, ax=ax, label='Frequency')
    
            pdf.savefig(fig)  # Consider specifying dpi if needed
            plt.close(fig)

        print(f"Optimized results written to: {pdf_filepath}")
        
        


    
    def process_agents(self, hdf_path):
        base_name = os.path.splitext(os.path.basename(hdf_path))[0]
        output_folder = os.path.dirname(hdf_path)
        tiff_output_path = os.path.join(output_folder, f'{base_name}_Agents_Locations.tif')
        txt_output_path = os.path.join(output_folder, f'{base_name}_Agents_Per_Cell_Stats.txt')
        combined_heatmap_tiff_output_path = os.path.join(output_folder, f'{base_name}_2-Band-Avg_STD.tif')
    
        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")
    
        # Load the TIFF image
        try:
            with rasterio.open(self.tif_path) as tiff:
                tiff_data = tiff.read(1)
                tiff_transform = tiff.transform
                tiff_bounds = tiff.bounds
                tiff_meta = tiff.meta.copy()
        except Exception as e:
            print(f"Error loading TIFF image: {e}")
            return
    
        # Load agent data from HDF5 file
        try:
            with h5py.File(hdf_path, 'r') as hdf:
                x_data = hdf['agent_data']['X'][:]
                y_data = hdf['agent_data']['Y'][:]
                num_timesteps = x_data.shape[1]
        except Exception as e:
            print(f"Error loading HDF5 data: {e}")
            return
    
        # Process grid lines and cells without plotting them
        x_min, y_min = tiff_bounds.left, tiff_bounds.bottom
        x_max, y_max = tiff_bounds.right, tiff_bounds.top
        x_ticks = np.arange(x_min, x_max, self.cell_width)
        y_ticks = np.arange(y_min, y_max, self.cell_height)
        num_cells_x = len(x_ticks) - 1
        num_cells_y = len(y_ticks) - 1
    
        # Create a new TIFF file for agent locations
        try:
            agent_data = np.zeros_like(tiff_data, dtype=np.uint8)
            agent_coords = np.vstack((x_data.flatten(), y_data.flatten())).T
            agent_pixel_coords = np.array([~tiff_transform * coord for coord in agent_coords]).astype(int)
            valid_idx = (
                (agent_pixel_coords[:, 0] >= 0) & (agent_pixel_coords[:, 0] < tiff_data.shape[1]) &
                (agent_pixel_coords[:, 1] >= 0) & (agent_pixel_coords[:, 1] < tiff_data.shape[0])
            )
            valid_coords = agent_pixel_coords[valid_idx]
            agent_data[valid_coords[:, 1], valid_coords[:, 0]] = 255
            tiff_meta.update({'count': 1, 'dtype': 'uint8', 'nodata': 0})
            with rasterio.open(tiff_output_path, 'w', **tiff_meta) as dst:
                dst.write(agent_data, 1)
            print(f"Agent locations TIFF saved to {tiff_output_path}")
        except Exception as e:
            print(f"Error creating agent locations TIFF: {e}")
            return
    
        # Calculate the number of agents per grid cell per timestep
        agent_counts = np.zeros((num_timesteps, num_cells_y, num_cells_x), dtype=int)
    
        for timestep in range(num_timesteps):
            x_coords = x_data[:, timestep]
            y_coords = y_data[:, timestep]
            for x, y in zip(x_coords, y_coords):
                if np.isnan(x) or np.isnan(y):
                    continue
                cell_x = int((x - x_min) / self.cell_width)
                cell_y = int((y - y_min) / self.cell_height)
                if 0 <= cell_x < num_cells_x and 0 <= cell_y < num_cells_y:
                    agent_counts[timestep, cell_y, cell_x] += 1
    
        # Calculate the average and standard deviation of agents per grid cell
        avg_agent_counts = np.mean(agent_counts, axis=0)
        std_agent_counts = np.std(agent_counts, axis=0)
    
        # Save the average and standard deviation statistics to a text file
        try:
            with open(txt_output_path, 'w') as txt_file:
                txt_file.write("\n\nAverage and Standard Deviation of Cell Counts across All Timesteps:\n\n")
                cell_id = 1
                for y in range(num_cells_y):
                    for x in range(num_cells_x):
                        if avg_agent_counts[y, x] > 0 or std_agent_counts[y, x] > 0:  # Only write cells with agents
                            txt_file.write(f"  Cell {cell_id:5}:    Avg Agents: {avg_agent_counts[y, x]:10.2f}    Std Agents: {std_agent_counts[y, x]:10.2f}\n")
                        cell_id += 1
            print(f"Agent per cell statistics saved to {txt_output_path}")
        except Exception as e:
            print(f"Error saving agent statistics to text file: {e}")
            return
    
        # Create a combined 2-band TIFF file for average and standard deviation of agent presence
        # Band 1 is average agents per cell and band 2 is standard deviation of agents per cell
        try:
            combined_heatmap_meta = tiff_meta.copy()
            combined_heatmap_meta.update({'count': 2, 'dtype': 'float32', 'nodata': 0})
            with rasterio.open(combined_heatmap_tiff_output_path, 'w', **combined_heatmap_meta) as dst:
                dst.write(avg_agent_counts.astype(np.float32), 1)
                dst.write(std_agent_counts.astype(np.float32), 2)
            with rasterio.open(combined_heatmap_tiff_output_path, 'r+') as dst:
                avg_band = dst.read(1)
                std_band = dst.read(2)
                flipped_avg_band = np.flipud(avg_band)
                flipped_std_band = np.flipud(std_band)
                dst.write(flipped_avg_band, 1)
                dst.write(flipped_std_band, 2)
            print(f"2-Band Avg_STD heatmap TIFF saved to {combined_heatmap_tiff_output_path}")
        except Exception as e:
            print(f"Error creating combined heatmap TIFF: {e}")
            return
    

    

    def Tiff_Directory(self):
        # Initialize accumulators for data
        cell_data = {}

        # Walk through all files in the directory and its subdirectories
        for subdir, dirs, files in os.walk(self.directory_path):
            for file in files:
                if 'Agents_Per_Cell_Stats' in file:
                    txt_path = os.path.join(subdir, file)
                    print(f"Processing file: {txt_path}")  # Debug print

                    # Load data from the text file
                    try:
                        with open(txt_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                match = re.match(r'\s*Cell\s*(\d+):\s*Avg Agents:\s*([\d.]+)\s*Std Agents:\s*([\d.]+)', line)
                                if match:
                                    cell_id = int(match.group(1))
                                    avg_agents = float(match.group(2))
                                    std_agents = float(match.group(3))
                                    if cell_id not in cell_data:
                                        cell_data[cell_id] = {'avg_agents': [], 'std_agents': []}
                                    cell_data[cell_id]['avg_agents'].append(avg_agents)
                                    cell_data[cell_id]['std_agents'].append(std_agents)
                                else:
                                    print(f"Skipping line due to unexpected format: {line.strip()}")
                    except Exception as e:
                        print(f"Error loading text file {txt_path}: {e}")
                        continue

        if not cell_data:
            print("No valid data found in any text file.")
            return

        print("Aggregating data for cells...")  # Debug print
        # Calculate the overall average and standard deviation for each cell
        aggregated_data = {}
        for cell_id, data in cell_data.items():
            avg_agents = np.mean(data['avg_agents'])
            std_agents = np.mean(data['std_agents'])
            aggregated_data[cell_id] = (avg_agents, std_agents)

        print(f"Aggregated data: {aggregated_data}")  # Debug print

        # Save the aggregated data to a new text file
        aggregated_txt_output_path = os.path.join(self.directory_path, 'Aggregated_Agents_Per_Cell_Stats.txt')
        try:
            with open(aggregated_txt_output_path, 'w') as txt_file:
                txt_file.write("\n\nAverage and Standard Deviation of Cell Counts across All Timesteps:\n\n")
                for cell_id, (avg_agents, std_agents) in aggregated_data.items():
                    txt_file.write(f"  Cell {cell_id:5}:    Avg Agents: {avg_agents:10.2f}    Std Agents: {std_agents:10.2f}\n")
            print(f"Aggregated data saved to {aggregated_txt_output_path}")
        except Exception as e:
            print(f"Error saving aggregated data to text file: {e}")
            return

        # Create the 2-band TIFF file for average and standard deviation of agent presence
        combined_heatmap_tiff_output_path = os.path.join(self.directory_path, 'Cumulative_2-Band-Avg_STD.tif')
        try:
            with rasterio.open(self.tif_path) as tiff:
                tiff_bounds = tiff.bounds
                tiff_meta = tiff.meta.copy()

            x_min, y_min = tiff_bounds.left, tiff_bounds.bottom
            x_max, y_max = tiff_bounds.right, tiff_bounds.top
            x_ticks = np.arange(x_min, x_max, self.cell_width)
            y_ticks = np.arange(y_min, y_max, self.cell_height)
            num_cells_x = len(x_ticks) - 1
            num_cells_y = len(y_ticks) - 1

            avg_agent_counts = np.zeros((num_cells_y, num_cells_x), dtype=np.float32)
            std_agent_counts = np.zeros((num_cells_y, num_cells_x), dtype=np.float32)

            for cell_id, (avg_agents, std_agents) in aggregated_data.items():
                y = (cell_id - 1) // num_cells_x
                x = (cell_id - 1) % num_cells_x
                avg_agent_counts[y, x] = avg_agents
                std_agent_counts[y, x] = std_agents

            combined_heatmap_meta = tiff_meta.copy()
            combined_heatmap_meta.update({'count': 2, 'dtype': 'float32', 'nodata': 0})
            with rasterio.open(combined_heatmap_tiff_output_path, 'w', **combined_heatmap_meta) as dst:
                dst.write(avg_agent_counts, 1)
                dst.write(std_agent_counts, 2)
            with rasterio.open(combined_heatmap_tiff_output_path, 'r+') as dst:
                avg_band = dst.read(1)
                std_band = dst.read(2)
                flipped_avg_band = np.flipud(avg_band)
                flipped_std_band = np.flipud(std_band)
                dst.write(flipped_avg_band, 1)
                dst.write(flipped_std_band, 2)
            print(f"Cumulative 2-Band Avg_STD heatmap TIFF saved to {combined_heatmap_tiff_output_path}")
        except Exception as e:
            print(f"Error creating cumulative heatmap TIFF: {e}")
            return








          


    def process_files(self):
        hdf_files = self.find_hdf_files()
        # Use joblib to parallelize the processing
        Parallel(n_jobs=-1)(
            delayed(self.perform_file_operations)(hdf_path) for hdf_path in hdf_files
        )


   #Comment and uncomment different functions to run the code
    def perform_file_operations(self, hdf_path):
       self.plot_lengths(hdf_path)
       self.length_statistics(hdf_path)
       self.plot_weights(hdf_path)
       self.weight_statistics(hdf_path)
       self.plot_body_depths(hdf_path)
       self.body_depth_statistics(hdf_path)
       self.max_bout_no(hdf_path)
       self.bout_no_statistics(hdf_path)
       self.kcal_statistics(hdf_path)
       self.Agent_Plot_Rectangle(self.shapefile_path, hdf_path)
       self.plot_agent_locations(hdf_path)
       self.plot_agent_locations_on_tiff(hdf_path)
       self.plot_agent_timestep_heatmap(hdf_path)
       self.process_agents(hdf_path)


multi_summarization=Multi_Summarization(directory_path, tif_path, Data_output, shapefile_path)
tiff_image, tiff_extent = multi_summarization.load_tiff_image()
multi_summarization.kcal_statistics_directory()
multi_summarization.process_files()
multi_summarization.kcal_histograms_directory()
multi_summarization.Tiff_Directory()