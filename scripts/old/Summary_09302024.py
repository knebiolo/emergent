# import dependencies
#import cupy as cp
import h5py
import dask.array as da
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
# from pysal.explore import esda
# from pysal.lib import weights
# from pysal.model import spreg
from affine import Affine as AffineTransform
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform
from rasterio.features import rasterize
from shapely import Point, Polygon, box
from shapely import affinity
from shapely.wkt import loads as loads
from scipy.interpolate import LinearNDInterpolator, UnivariateSpline, interp1d, CubicSpline, RectBivariateSpline
from scipy.optimize import curve_fit
from scipy.constants import g
from scipy.spatial import cKDTree#, cdist
from scipy.stats import beta
from scipy.ndimage import distance_transform_edt
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from datetime import datetime
import time
import warnings
import sys
import random
from collections import deque
from sksurv.nonparametric import kaplan_meier_estimator
from PyPDF2 import PdfReader, PdfWriter
from dask import delayed, compute




class summary:
    '''The power of an agent based model lies in its ability to produce emergent
    behavior of interest to managers.  novel self organized patterns that 
    only happen once are a consequence, predictable self organized patterns  
    are powerful.  Each Emergent simulation should be run no less than 30 times. 
    This summary class object is designed to iterate over a parent directory, 
    extract data from child directories, and compile statistics.  The parent 
    directory describes a single scenario (for sockeye these are discharge)
    while each child directory is an individual iteration.  
    
    The class object iterates over child directories, extracts and manipulates data,
    calculate basic descriptive statistics, manages information, and utilizes 
    Poisson kriging to produce a surface that depicts the average number of agents
    per cell per second.  High use corridors should be visible in the surface.
    These corridors are akin to the desire paths we see snaking through college
    campuses and urban parks the world over.
    
    '''
    def __init__(self, parent_directory, tif_path):
        # set the model directory path
        self.parent_directory = parent_directory
        
        # where are the background tiffs stored?
        self.tif_path = tif_path
        
        #set input WS as parent_directory for compatibility with methods
        self.inputWS = parent_directory
        
        
        # get h5 files associated with this model
        self.h5_files = self.find_h5_files()
        
        # create empty thigs to hold agent data
        self.ts = gpd.GeoDataFrame(columns = ['agent','timestep','X','Y','kcal','Hz','filename','geometry'])
        self.morphometrics = pd.DataFrame()
        self.success_rates = {}

    def load_tiff(self, crs):
        # Define the desired CRS
        desired_crs = CRS.from_epsg(crs)

        # Open the TIFF file with rasterio
        with rasterio.open(self.tif_path) as tiff_dataset:
            # Calculate the transformation parameters for reprojecting
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)
            
            cell_size = 2.
            # Calculate the new transform for 10x10 meter resolution
            new_transform = from_origin(transform.c, transform.f, cell_size, cell_size)
            
            # Calculate new width and height
            new_width = int((tiff_dataset.bounds.right - tiff_dataset.bounds.left) / cell_size)
            new_height = int((tiff_dataset.bounds.top - tiff_dataset.bounds.bottom) / cell_size)
            
            self.transform = new_transform
            self.width = new_width
            self.height = new_height

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
   
    # Find the .h5 files
    def find_h5_files(self):
        
        # create empty holders for all of the h5 files and child directories
        h5_files=[]
        child_dirs = []
        
        # first iterate over the parent diretory to find the children (iterations)
        for item in os.listdir(self.parent_directory):
            # create a full path object
            full_path = os.path.join(self.parent_directory, item)
            
            if full_path.endswith('.h5'):
                h5_files.append(full_path)
            
            # if full path is a directory and not a file - we found a child
            if os.path.isdir(full_path):
                child_dirs.append(full_path)
        
        # iterate over child directories and find the h5 files
        for child_dir in child_dirs:
            for filename in os.listdir(child_dir):
                if filename.endswith('.h5'):
                    h5_files.append(os.path.join(child_dir,filename))
        
        # we found our files
        return h5_files
        
    # Collect, rearrange, and manage data
    def get_data(self, h5_files):

        # Iterate through each HDF5 file in the specified directory and get data
        for filename in h5_files:

            with h5py.File(filename, 'r') as hdf:
                cell_center_x = pd.DataFrame(hdf['x_coords'][:])
                cell_center_x['row'] = np.arange(len(cell_center_x))
                cell_center_y = pd.DataFrame(hdf['y_coords'][:])
                cell_center_y['row'] = np.arange(len(cell_center_y))

                melted_center_x = pd.melt(cell_center_x, id_vars = ['row'], var_name = 'column', value_name = 'X')
                melted_center_y = pd.melt(cell_center_y, id_vars = ['row'], var_name = 'column', value_name = 'Y')
                melted_center = pd.merge(melted_center_x, melted_center_y, on = ['row','column'])
                self.melted_center = melted_center
                self.x_coords = hdf['x_coords'][:]
                self.y_coords = hdf['y_coords'][:]
                              
                if 'agent_data' in hdf:
                    # timestep data
                    X = pd.DataFrame(hdf['agent_data/X'][:])
                    X['agent'] = np.arange(X.shape[0])
                    Y = pd.DataFrame(hdf['agent_data/Y'][:])
                    Y['agent'] = np.arange(Y.shape[0])
                    Hz = pd.DataFrame(hdf['agent_data/Hz'][:])
                    Hz['agent'] = np.arange(Hz.shape[0])
                    kcal = pd.DataFrame(hdf['agent_data/kcal'][:])
                    kcal['agent'] = np.arange(kcal.shape[0])  
                    
                    # agent specific 
                    length = pd.DataFrame(hdf['agent_data/length'][:])
                    length['agent'] = np.arange(len(length))
                    length.rename(mapper = {0:'length'}, axis = 'columns', inplace = True)
                    
                    weight = pd.DataFrame(hdf['agent_data/weight'][:])
                    weight['agent'] = np.arange(len(weight))
                    weight.rename(mapper = {0:'weight'}, axis = 'columns', inplace = True)

                    body_depth = pd.DataFrame(hdf['agent_data/body_depth'][:])
                    body_depth['agent'] = np.arange(len(body_depth))
                    body_depth.rename(mapper = {0:'body_depth'}, axis = 'columns', inplace = True)

                    # melt time series data
                    melted_X = pd.melt(X, id_vars=['agent'], var_name='timestep', value_name='X')
                    melted_Y = pd.melt(Y, id_vars=['agent'], var_name='timestep', value_name='Y')
                    melted_kcal = pd.melt(kcal, id_vars=['agent'], var_name='timestep', value_name='kcal')
                    melted_Hz = pd.melt(Hz, id_vars=['agent'], var_name='timestep', value_name='Hz')
                    
                    # make one dataframe 
                    ts = pd.merge(melted_X, melted_Y, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_kcal, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_Hz, on = ['agent','timestep'])
                    ts['filename'] = filename
                    
                    print ('Data Imported ')
                    # turn ts into a geodataframe and find the fish that passed
                    geometry = [Point(xy) for xy in zip(ts['X'], ts['Y'])]
                    geo_ts = gpd.GeoDataFrame(ts, geometry=geometry)            
                    
                    # make one morphometric dataframe
                    morphometrics = pd.merge(length, weight, on = ['agent'])
                    morphometrics = pd.merge(morphometrics, body_depth, on = ['agent'])
                    
                    # add to summary data
                    self.ts = pd.concat([self.ts,geo_ts], ignore_index = True)
                    self.morphometrics = pd.concat([self.morphometrics,morphometrics], 
                                                  ignore_index = True) 
                    
                    print ('File %s imported'%(filename))
                    
                    
                    
    # Collect histograms of agent lengths
    def plot_lengths(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            pdf_filename = f"{base_name}_Lengths_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(output_folder, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        lengths = file['/agent_data/length'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            lengths_by_sex = lengths[sex_mask]
                            lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                            if lengths_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(lengths_by_sex, [75, 25])
                                    bin_width = 2 * (q75 - q25) * len(lengths_by_sex) ** (-1 / 3)

                                    if bin_width <= 0 or np.isnan(bin_width):
                                        bin_width = (max(lengths_by_sex) - min(lengths_by_sex)) / 10

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

    def length_statistics(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            stats_file_name = f"{base_name}_length_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(h5_file, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    lengths = file['/agent_data/length'][:]
                    sexes = file['/agent_data/sex'][:]

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        lengths_by_sex = lengths[sex_mask]
                        lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                        if lengths_by_sex.size > 1:
                            mean_length = np.mean(lengths_by_sex)
                            median_length = np.median(lengths_by_sex)
                            std_dev_length = np.std(lengths_by_sex, ddof=1)
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\n")
                            output_file.write(f"  Median Length: {median_length:.2f}\n")
                            output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\n\n")
                        elif lengths_by_sex.size == 1:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Length: {lengths_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid length values found for {sex_label}.\n\n")

    def plot_weights(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_directory = os.path.dirname(h5_file)
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            pdf_filename = f"{base_name}_Weights_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(base_directory, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        weights = file['/agent_data/weight'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            weights_by_sex = weights[sex_mask]
                            weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]

                            if weights_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(weights_by_sex, [75, 25])
                                    iqr = q75 - q25
                                    if iqr > 0:
                                        bin_width = 2 * iqr * len(weights_by_sex) ** (-1 / 3)
                                        bins = max(1, round((max(weights_by_sex) - min(weights_by_sex)) / bin_width))
                                    else:
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
                                    plt.close(fig)
                            else:
                                print(f"No weight values found for {sex_label} in {base_name}.")

    def weight_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
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

    def plot_body_depths(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
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
                                        bin_width = 2 * iqr * len(body_depths_by_sex) ** (-1 / 3)
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

    def body_depth_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
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
    def kcal_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_kcal_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)
    
            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    kcals = file['/agent_data/kcal'][:]  # Get kcal data for all agents and timesteps
                    sexes = file['/agent_data/sex'][:]    # Get sex data for all agents
    
                    # Sum kcal data across all timesteps for each agent
                    total_kcals_per_agent = np.nansum(kcals, axis=1)  # Sum along the timestep axis, ignoring NaNs
    
                    # Perform statistics on the total kcal data across all agents
                    mean_kcal = np.mean(total_kcals_per_agent)
                    median_kcal = np.median(total_kcals_per_agent)
                    std_dev_kcal = np.std(total_kcals_per_agent, ddof=1)  # Use ddof=1 for sample standard deviation
                    min_kcal = np.min(total_kcals_per_agent)
                    max_kcal = np.max(total_kcals_per_agent)
    
                    for i, (total_kcal, sex) in enumerate(zip(total_kcals_per_agent, sexes)):
                        if not np.isnan(total_kcal):  # Check if the total kcal is a valid number
                            sex_label = 'Male' if sex == 0 else 'Female'
    
                            output_file.write(f"Agent {i + 1} ({sex_label}):\n")
                            output_file.write(f"  Total Kcal: {total_kcal:.2f}\n")
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
    
        # Iterate through all HDF5 files in the directory
        for hdf_path in self.h5_files:
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
    
                        # Process each agent's kcal data, ignoring inf and NaN values at each timestep
                        total_kcals_by_sex = []
                        for kcal_values in kcals_by_sex:
                            valid_kcals = kcal_values[np.isfinite(kcal_values)]  # Filter out inf and NaN values
                            total_kcal = np.sum(valid_kcals)  # Sum the remaining valid kcal values
                            
                            # Ignore NaN, inf, or total kcal values of 0
                            if np.isfinite(total_kcal) and total_kcal > 0:
                                total_kcals_by_sex.append(total_kcal)
    
                        cumulative_stats[sex_label].extend(total_kcals_by_sex)
    
        # Compute and print cumulative statistics
        stats_file_path = os.path.join(self.inputWS, "kcal_statistics_directory.txt")
        with open(stats_file_path, 'w') as output_file:
            for sex_label, values in cumulative_stats.items():
                if values:
                    values = np.array(values)
                    
                    # Further ensure no inf or NaN values are in the array
                    values = values[np.isfinite(values)]
                    
                    if len(values) > 0:
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
                else:
                    output_file.write(f"No valid kcal values found for {sex_label}.\n\n")
           

        
    def Kcal_histogram_by_timestep_intervals_for_all_simulations(self):
        """
        Generates a single histogram for the total kcal consumed by all agents in the folder that
        inputWS is pointing to and saves them as two separate JPEGs: one for males and one for females.
        """
        import os
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Get the folder name from inputWS
        folder_name = os.path.basename(os.path.normpath(self.inputWS))
        
        # Determine if "left" or "right" should be included in the title
        direction = "left" if "left" in folder_name.lower() else "right" if "right" in folder_name.lower() else ""
    
        # Prepare the output paths for male and female histograms
        male_histogram_path = os.path.join(self.inputWS, f"{folder_name}_male_kcal_histogram.jpg")
        female_histogram_path = os.path.join(self.inputWS, f"{folder_name}_female_kcal_histogram.jpg")
        
        # Initialize lists to collect kcal data across all HDF5 files
        all_male_kcals = []
        all_female_kcals = []
        
        # Set common plot parameters
        plt.rcParams.update({
            'font.size': 6,
            'font.family': 'serif',
            'figure.figsize': (3, 2)  # width, height in inches
        })
        
        # Iterate through all HDF5 files in the directory
        for hdf_path in self.h5_files:
            try:
                with h5py.File(hdf_path, 'r') as hdf_file:
                    if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys() and 'sex' in hdf_file['agent_data'].keys():
                        kcals = hdf_file['/agent_data/kcal'][:]
                        sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
    
                        # Separate male and female data
                        male_kcals = kcals[sexes == 0]
                        female_kcals = kcals[sexes == 1]
    
                        # Calculate total kcal for each agent across all timesteps
                        male_total_kcal = np.nansum(male_kcals, axis=1)
                        female_total_kcal = np.nansum(female_kcals, axis=1)
    
                        # Replace inf and NaN values with zero
                        male_total_kcal = np.where(np.isfinite(male_total_kcal), male_total_kcal, 0)
                        female_total_kcal = np.where(np.isfinite(female_total_kcal), female_total_kcal, 0)
    
                        # Append the total kcal data to the lists
                        all_male_kcals.extend(male_total_kcal)
                        all_female_kcals.extend(female_total_kcal)
    
            except Exception as e:
                print(f"Failed to process {hdf_path}: {e}")
    
        # Convert lists to numpy arrays for histogram plotting
        all_male_kcals = np.array(all_male_kcals)
        all_female_kcals = np.array(all_female_kcals)
    
        # Create and save male histogram as JPEG
        plt.figure()
        plt.hist(all_male_kcals, bins=20, color='blue', alpha=0.7)
        plt.title(f"Male Total Kcal Usage ({direction})", fontsize=6)
        plt.xlabel("Total Kcal", fontsize=6)
        plt.ylabel("Frequency", fontsize=6)
        plt.tight_layout()
        plt.savefig(male_histogram_path, format='jpeg', dpi=300)
        plt.close()
    
        # Create and save female histogram as JPEG
        plt.figure()
        plt.hist(all_female_kcals, bins=20, color='red', alpha=0.7)
        plt.title(f"Female Total Kcal Usage ({direction})", fontsize=6)
        plt.xlabel("Total Kcal", fontsize=6)
        plt.ylabel("Probability Density", fontsize=6)
        plt.tight_layout()
        plt.savefig(female_histogram_path, format='jpeg', dpi=300)
        plt.close()
    
        print(f"Kcal histograms saved to {male_histogram_path} and {female_histogram_path}")
            


                    
    def kaplan_curve(self, shapefile_path, tiffWS, inputWS):
        # Specify the gate shapefiles in the order of priority
        gate_files = ['Gate_04.shp', 'Gate_03.shp', 'Gate_02.shp']
        
        # Traverse all subfolders in the inputWS to find .h5 files
        print(f"Traversing {inputWS} for .h5 files...")
        for root, dirs, files in os.walk(inputWS):
            for file in files:
                if file.endswith(".h5"):
                    h5_file = os.path.join(root, file)
                    print(f"Processing file: {h5_file}")
                    
                    # Filter self.ts once for this HDF5 file
                    ts_filtered = self.ts[self.ts['filename'] == h5_file]
                    base_name = os.path.splitext(os.path.basename(h5_file))[0]
                    output_folder = root  # Use the current directory where .h5 file is located
    
                    # If no data is present for this file, skip the file
                    if ts_filtered.empty:
                        print(f"No data found for {base_name}. Skipping...")
                        continue
    
                    print(f"Data found for {base_name}. Proceeding with processing.")
    
                    # Load TIFF data for coordinate reference
                    with rasterio.open(tiffWS) as tif:
                        tif_crs = tif.crs
    
                    # Loop through each gate shapefile
                    for gate_file in gate_files:
                        shapefile_full_path = os.path.join(shapefile_path, gate_file)
                        
                        # Check if the shapefile exists before proceeding
                        if not os.path.exists(shapefile_full_path):
                            print(f"Shapefile {shapefile_full_path} not found. Skipping...")
                            continue
                        
                        print(f"Processing shapefile: {shapefile_full_path}")
    
                        # Extract gate name (e.g., Gate_04) from the file name
                        gate_name = os.path.splitext(gate_file)[0]
                        
                        # Update filenames to include the gate name
                        jpeg_filename = f"{base_name}_{gate_name}_Kaplan_Meier_Curve.jpeg"
                        jpeg_filepath = os.path.join(output_folder, jpeg_filename)
                        txt_filename = f"{base_name}_{gate_name}_Kaplan_Meier_Statistics.txt"
                        txt_filepath = os.path.join(output_folder, txt_filename)
                        # Remove the CSV export paths for Kaplan-Meier curve data
                        fish_csv_filename = f"{base_name}_{gate_name}_Fish_Passage_Data.csv"
                        fish_csv_filepath = os.path.join(output_folder, fish_csv_filename)
    
                        # Load shapefile
                        gdf = gpd.read_file(shapefile_full_path)
                        if gdf.empty:
                            print(f"Shapefile {shapefile_full_path} is empty. Skipping...")
                            continue
    
                        # Adjust shapefile CRS if needed
                        if gdf.crs != tif_crs:
                            gdf = gdf.to_crs(tif_crs)
    
                        # Perform spatial intersection with the shapefile
                        intersection = gpd.overlay(ts_filtered, gdf, how='intersection')
                        if intersection.empty:
                            print(f"No intersection found for {gate_name}. Skipping...")
                            continue
    
                        # Get the list of all agent IDs in the filtered data
                        all_agents = ts_filtered['agent'].unique()
                        total_agents = len(all_agents)
                        max_timestep = ts_filtered['timestep'].max()  # Max timestep for non-passing fish
    
                        # Get the unique list of agents that are found within the rectangle
                        successful_agents = intersection['agent'].unique()
                        num_agents_in_rectangle = len(successful_agents)
    
                        # Prepare the first entry times for each successful agent
                        entry_times = {agent: intersection[intersection['agent'] == agent]['timestep'].min()
                                       for agent in successful_agents}
    
                        # Calculate total kcal consumed by each successful agent
                        total_kcal = {agent: ts_filtered[ts_filtered['agent'] == agent]['kcal'].sum()
                                      for agent in successful_agents}
    
                        # Convert to arrays for Kaplan-Meier analysis
                        entry_times_array = np.array(list(entry_times.values()))
    
                        # Check if we have successful entries before performing Kaplan-Meier estimation
                        if len(entry_times_array) > 0:
                            # Create the survival data array (True if entered the rectangle, False if not)
                            survival_data = np.array([(True, time) for time in entry_times_array], dtype=[('event', bool), ('time', int)])
    
                            # Perform Kaplan-Meier estimation
                            time, survival_prob = kaplan_meier_estimator(survival_data['event'], survival_data['time'])
    
                            # Calculate the standard error and confidence intervals (95% CI)
                            n = len(survival_data)
                            se = np.sqrt((survival_prob * (1 - survival_prob)) / n)
                            lower_ci = survival_prob - 1.96 * se
                            upper_ci = survival_prob + 1.96 * se
    
                            # Ensure CI bounds are within [0, 1]
                            lower_ci = np.maximum(lower_ci, 0)
                            upper_ci = np.minimum(upper_ci, 1)
    
                            # Store the Kaplan-Meier data for CSV export
                            kaplan_df = pd.DataFrame({
                                'time': time,
                                'survival_prob': survival_prob,
                                'lower_ci': lower_ci,
                                'upper_ci': upper_ci,
                                'gate': gate_name,
                                'h5_file': base_name,
                                'total_agents': total_agents,
                                'agents_in_rectangle': num_agents_in_rectangle
                            })
                            # Commenting out the CSV export line for Kaplan-Meier data
                            # kaplan_df.to_csv(kaplan_csv_filepath, index=False)
                            print(f"Kaplan-Meier data created but not exported for {base_name} at {gate_name}.")
                        else:
                            print(f"No successful entries for {base_name} at {gate_name}. Skipping Kaplan-Meier estimation.")
    
                        # Create Fish Passage DataFrame for the fish passage CSV
                        fish_data_list = []
                        for agent in all_agents:
                            if agent in successful_agents:
                                time_of_passage = entry_times[agent]
                                state = 1  # Successful
                            else:
                                time_of_passage = max_timestep  # Non-successful agents get max timestep
                                state = 0  # Not successful
                            
                            # Append the data to the fish_data_list
                            fish_data_list.append({
                                'Fish ID': agent,
                                'Time of Passage': time_of_passage,
                                'State': state
                            })
                        
                        # Create DataFrame and export to CSV
                        fish_df = pd.DataFrame(fish_data_list)
                        fish_df.to_csv(fish_csv_filepath, index=False)
                        print(f"Fish Passage data exported to: {fish_csv_filepath}")
    
                        # Write the statistics and kcal information to a text file (original functionality)
                        with open(txt_filepath, 'w') as txt_file:
                            txt_file.write(f"Kaplan-Meier Statistics for {base_name} at {gate_name}\n")
                            txt_file.write("-" * 33 + "\n")
    
                            # Check if survival_prob exists before printing
                            if len(entry_times_array) > 0:
                                completion_times = {
                                    10: time[np.where(survival_prob <= 0.9)[0][0]],
                                    30: time[np.where(survival_prob <= 0.7)[0][0]],
                                    50: time[np.where(survival_prob <= 0.5)[0][0]],
                                    70: time[np.where(survival_prob <= 0.3)[0][0]],
                                    90: time[np.where(survival_prob <= 0.1)[0][0]],
                                }
                                for perc, comp_time in completion_times.items():
                                    txt_file.write(f"Time at which {perc}% of agents completed passage: {comp_time}\n")
                                txt_file.write(f"Last timestep the final agent crosses into the rectangle: {time[-1]}\n")
                                txt_file.write(f"Percentage of agents that complete passage: {(num_agents_in_rectangle / total_agents) * 100:.2f}%\n\n")
                            
                            # Kilocalories consumed by successful agents
                            txt_file.write("Kilocalories Consumed by Successful Agents:\n")
                            txt_file.write("-" * 33 + "\n")
                            for agent, kcal in total_kcal.items():
                                txt_file.write(f"Agent {agent}: {kcal:.2f} kcal\n")
                            
                            # Kilocalories statistics
                            kcal_values = np.array(list(total_kcal.values()))
                            txt_file.write("\nKilocalories Statistics:\n")
                            txt_file.write(f"Minimum kcal: {np.min(kcal_values):.2f}\n")
                            txt_file.write(f"Maximum kcal: {np.max(kcal_values):.2f}\n")
                            txt_file.write(f"Average kcal: {np.mean(kcal_values):.2f}\n\n")
    
                            # Explanation of confidence interval
                            txt_file.write("Explanation of the Confidence Interval:\n")
                            txt_file.write("The confidence interval represents the range within which we expect the true proportion of agents\n")
                            txt_file.write("remaining outside the rectangle to fall, with 95% confidence. The interval is calculated based on\n")
                            txt_file.write("the observed data, and it provides a measure of uncertainty around the Kaplan-Meier estimate.\n")
                            txt_file.write("A narrower confidence interval indicates greater certainty in the estimate, while a wider interval\n")
                            txt_file.write("indicates more uncertainty. The shaded area around the Kaplan-Meier curve on the plot represents\n")
                            txt_file.write("this confidence interval.\n")
    
                            # Check if survival_prob exists before plotting
                            if len(entry_times_array) > 0:
                                # Plot the Kaplan-Meier survival curve with confidence intervals
                                plt.figure(figsize=(3.5, 2.5))
                                plt.step(time, survival_prob, where="post", label="Kaplan Meier Curve")
                                plt.fill_between(time, lower_ci, upper_ci, color='gray', alpha=0.3, step="post", label="Confidence Interval")
                                plt.xlabel("Time (Timesteps)", fontsize=6)
                                plt.ylabel("Proportion of Agents Remaining Outside the Rectangle", fontsize=6)
                                plt.title(f"Kaplan-Meier Curve\n{num_agents_in_rectangle} Agents Entered {gate_name} out of {total_agents}", fontsize=6)
                                plt.legend(loc="upper right", fontsize=6)
                                plt.tight_layout()
                                plt.savefig(jpeg_filepath, format='jpeg', dpi=300, bbox_inches='tight')
                                plt.close()
    
                        print(f"Kaplan-Meier curves and statistics generated for {gate_name}.")
                        print(f"Saving JPEG to: {jpeg_filepath}")
    
        print("All Kaplan-Meier curves and fish passage data have been processed.")




    def Shapefile_Final_Timestep_Agents(self, inputWS):
        # Traverse all subfolders in inputWS to find .h5 files
        print(f"Traversing {inputWS} for .h5 files...")
        for root, dirs, files in os.walk(inputWS):
            for file in files:
                if file.endswith(".h5"):
                    h5_file = os.path.join(root, file)
                    print(f"Processing file: {h5_file}")
                    
                    # Filter self.ts for this HDF5 file (getting data related to the current file)
                    ts_filtered = self.ts[self.ts['filename'] == h5_file]
                    
                    # Debugging: Check the available columns
                    print(f"Columns in ts_filtered: {ts_filtered.columns}")
                    
                    base_name = os.path.splitext(os.path.basename(h5_file))[0]
                    
                    # If no data is present for this file, skip it
                    if ts_filtered.empty:
                        print(f"No data found for {base_name}. Skipping...")
                        continue
                    
                    print(f"Data found for {base_name}. Proceeding with processing.")
                    
                    # Initialize an empty list to store each row (agent's data)
                    agent_data_list = []
                    
                    # Get unique agents
                    unique_agents = ts_filtered['agent'].unique()
                    
                    for agent in unique_agents:
                        # Get the last timestep for each agent
                        agent_data = ts_filtered[ts_filtered['agent'] == agent]
                        last_timestep = agent_data['timestep'].max()
                        
                        # Filter to the last timestep and extract the corresponding coordinates
                        last_timestep_data = agent_data[agent_data['timestep'] == last_timestep]
                        
                        # Use the correct column names 'X' and 'Y' for coordinates
                        x_coord = last_timestep_data['X'].values[0]
                        y_coord = last_timestep_data['Y'].values[0]
                        
                        # Create a Point geometry for the agent's last known coordinates
                        point_geometry = Point(x_coord, y_coord)
                        
                        # Add this row of data (agent, timestep, geometry) to the list
                        agent_data_list.append({
                            'agent': agent,
                            'timestep': last_timestep,
                            'geometry': point_geometry
                        })
                    
                    # Convert the list of rows into a GeoDataFrame
                    agent_gdf = gpd.GeoDataFrame(agent_data_list, crs="EPSG:32604")
                    
                    # Set the output shapefile path to the inputWS directory with the h5 file's base name
                    output_shapefile_path = os.path.join(root, f"{base_name}_Final_Timestep_Agents.shp")
                    
                    # Export the detected unique agents with their final timestep to a new shapefile
                    agent_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
                    
                    print(f"Shapefile of agents' final timesteps saved to: {output_shapefile_path}")
        
        print("All agent final timestep data has been processed and saved to shapefiles.")



    def Residence_calc(self, shapefile_path, tiffWS, inputWS):
        # Specify gate shapefiles
        gate_files = ['Gate_Residence.shp']
        
        # Define the complete agent list
        all_agents = list(range(1000))  # From 0 to 999
    
        # Traverse all subfolders in inputWS to find .h5 files
        for root, _, files in os.walk(inputWS):
            for file in files:
                if file.endswith(".h5"):
                    h5_file = os.path.join(root, file)
                    print(f"Processing file: {h5_file}")
                    
                    # Filter self.ts once for this HDF5 file
                    ts_filtered = self.ts[self.ts['filename'] == h5_file]
                    base_name = os.path.splitext(os.path.basename(h5_file))[0]
                    output_folder = root  # Use the current directory where .h5 file is located
    
                    # If no data is present for this file, skip the file
                    if ts_filtered.empty:
                        print(f"No data found for {base_name}. Skipping...")
                        continue
    
                    print(f"Data found for {base_name}. Proceeding with residence calculation.")
    
                    # Load TIFF data for coordinate reference
                    with rasterio.open(tiffWS) as tif:
                        tif_crs = tif.crs
    
                    # Loop through each gate shapefile
                    for gate_file in gate_files:
                        shapefile_full_path = os.path.join(shapefile_path, gate_file)
                        
                        # Check if the shapefile exists before proceeding
                        if not os.path.exists(shapefile_full_path):
                            print(f"Shapefile {shapefile_full_path} not found. Skipping...")
                            continue
                        
                        print(f"Processing shapefile: {shapefile_full_path}")
    
                        # Load shapefile
                        gdf = gpd.read_file(shapefile_full_path)
                        if gdf.empty:
                            print(f"Shapefile {shapefile_full_path} is empty. Skipping...")
                            continue
    
                        # Adjust shapefile CRS if needed
                        if gdf.crs != tif_crs:
                            gdf = gdf.to_crs(tif_crs)
    
                        # Perform spatial intersection with the shapefile
                        intersection = gpd.overlay(ts_filtered, gdf, how='intersection')
                        if intersection.empty:
                            print(f"No intersection found for {gate_file}. Skipping...")
                            continue
    
                        # Calculate residence time for each agent
                        residence_data = []
                        agents = intersection['agent'].unique()
                        for agent in agents:
                            agent_data = intersection[intersection['agent'] == agent]
                            
                            # Calculate the residence time as the number of unique timesteps
                            residence_time = agent_data['timestep'].nunique()
    
                            # Append to residence data list
                            residence_data.append({
                                'Agent': agent,
                                'Residence_Time': residence_time
                            })
                        
                        # Convert to DataFrame
                        residence_df = pd.DataFrame(residence_data)
    
                        # Ensure all agents from 0 to 999 are present
                        full_residence_df = pd.DataFrame({
                            'Agent': all_agents
                        })
                        
                        # Merge with calculated residence times and fill missing with 0
                        full_residence_df = full_residence_df.merge(
                            residence_df, 
                            on='Agent', 
                            how='left'
                        )
                        full_residence_df['Residence_Time'] = full_residence_df['Residence_Time'].fillna(0)
    
                        # Sort the DataFrame by Agent
                        full_residence_df = full_residence_df.sort_values(by='Agent')
    
                        # Save to CSV
                        csv_filename = f"{base_name}_{os.path.splitext(gate_file)[0]}_Residence_Times.csv"
                        csv_filepath = os.path.join(output_folder, csv_filename)
                        full_residence_df.to_csv(csv_filepath, index=False)
                        print(f"Residence data saved to: {csv_filepath}")
        
        print("All residence calculations have been processed.")