# -*- coding: utf-8 -*-
"""Summary utilities extracted from legacy sockeye monolith."""
import os
import h5py
import numpy as np
import pandas as pd
import dask.array as da
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform
from shapely import Point
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from .utils import geo_to_pixel


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
                            output_file.write(f"Statistics for {sex_label}:\\n")
                            output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\\n")
                            output_file.write(f"  Median Length: {median_length:.2f}\\n")
                            output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\\n\\n")
                        elif lengths_by_sex.size == 1:
                            output_file.write(f"Statistics for {sex_label}:\\n")
                            output_file.write(f"  Only one length value available: {lengths_by_sex[0]:.2f}\\n\\n")
                        else:
                            output_file.write(f"Statistics for {sex_label}: No valid length values found.\\n\\n")

    def summarize_failure(self, h5_files):
        # Pull out only fish who failed to make it into the boundary
        fail = self.ts.loc[self.ts['timestep'] < self.ts.timestep.max()]
        fail = pd.merge(fail, self.morphometrics, on = ['agent'])

        # if fish failed get a summary for each file
        for filename in fail.filename.unique():
            fail_mod = fail.loc[fail.filename == filename]
            print(fail_mod[['timestep','Hz','kcal']].describe())

        # Make a summary of the data
        return fail

    def summarize_success(self, h5_files):
        success = self.ts.loc[self.ts['timestep'] == self.ts.timestep.max()]
        success = pd.merge(success, self.morphometrics, on = ['agent'])

        # calculate the success rates and timesteps for each file
        for filename in self.ts.filename.unique():
            succ = success.loc[success.filename == filename]
            rate = succ.agent.nunique() / self.ts.agent.nunique()
            self.success_rates[filename] = rate

        return success

    def plot_pathways(self, raster, extent, percent, scenario, cell_size, crs):
        # Set background raster
        fig, ax = plt.subplots()
        ax.imshow(raster, extent=extent, cmap='Greys')

        # Calculate the confidence interval for agent pathways (Poisson distribution)
        for filename in self.ts.filename.unique():
            dat = self.ts[self.ts.filename == filename]
            num_timesteps = self.ts.timestep.max() + 1
            num_iterations = len(self.ts.filename.unique())
            # "Agent seconds" in a cell
            counts = dat.groupby(['X','Y']).size().reset_index(name='counts')
            # Filter points based on percentile
            threshold = np.percentile(counts['counts'], percent)
            filtered_counts = counts[counts['counts'] > threshold]
            ax.scatter(filtered_counts['X'], filtered_counts['Y'], s=0.1, c='red', alpha=0.7)

        fig.savefig(os.path.join(self.parent_directory, f"{scenario}_paths.png"))

    def plot_corridors(self, raster, extent, scenario, cell_size, crs):
        # Set background raster
        fig, ax = plt.subplots()
        ax.imshow(raster, extent=extent, cmap='Greys')

        # Use average_per_cell to highlight corridors
        if not hasattr(self, 'average_per_cell'):
            print('No average_per_cell available. Run summarize_corridors first.')
            return

        # Plot the corridor raster
        ax.imshow(self.average_per_cell, extent=extent, cmap='hot', alpha=0.6)
        fig.savefig(os.path.join(self.parent_directory, f"{scenario}_corridors.png"))

    def summarize_corridors(self, scenario, cell_size, crs):
        '''
        Summarize agent corridor usage by creating a raster of average agent counts per cell.

        Parameters:
        ----------
        scenario : str
            Scenario name.
        cell_size : float
            Grid resolution.
        crs : str
            Coordinate reference system.

        Returns:
        -------
        corridor raster surface.

        '''
        # Agent coordinates and rasterio affine transform
        x_coords = self.ts.X  # X coordinates of agents
        y_coords = self.ts.Y  # Y coordinates of agents
        transform = self.transform  # affine transform from your rasterio dataset

        hdf5_filename = 'intermediate_results.h5'

        with h5py.File(os.path.join(self.parent_directory,hdf5_filename), 'w') as hdf5_file:
            for filename in self.ts.filename.unique():
                dat = self.ts[self.ts.filename == filename]
                num_timesteps = self.ts.timestep.max() + 1
                num_iterations = len(self.ts.filename.unique())

                # Create a dataset for each filename
                data_over_time = hdf5_file.create_dataset(
                    filename,
                    shape=(num_timesteps, self.height, self.width),
                    dtype=np.float32,
                    chunks=(1, self.height, self.width),
                    compression="gzip"
                )

                for timestep in range(num_timesteps):
                    t_dat = dat[dat.timestep == timestep]

                    # Convert geographic coordinates to pixel indices using your function
                    rows, cols = geo_to_pixel(t_dat.X, t_dat.Y, transform)

                    # Combine row and column indices to get unique cell identifiers
                    cell_indices = np.stack((cols, rows), axis=1)

                    # Count unique cells
                    unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
                    valid_rows, valid_cols = unique_cells[:, 1], unique_cells[:, 0]  # Unpack the unique cell indices

                    # Initialize a 2D array with zeros
                    agent_counts_grid = np.zeros((self.height, self.width), dtype=int)

                    # Ensure the indices are within the grid bounds and update the agent_counts_grid
                    within_bounds = (valid_rows >= 0) & (valid_rows < self.height) & (valid_cols >= 0) & (valid_cols < self.width)
                    agent_counts_grid[valid_rows[within_bounds], valid_cols[within_bounds]] = counts[within_bounds]

                    # Insert the 2D array into the pre-allocated 3D array in HDF5
                    data_over_time[timestep, :, :] = agent_counts_grid
                    print(f'file {filename} timestep {timestep} complete')

            # Now aggregate results from HDF5
            all_data = []
            for filename in self.ts.filename.unique():
                data = da.from_array(hdf5_file[filename], chunks=(1, self.height, self.width))
                all_data.append(data)

            all_data = da.stack(all_data, axis=0)  # Stack along the new iteration axis

            # Calculate the average and standard deviation count per cell over all iterations and timesteps
            self.average_per_cell = da.mean(all_data, axis=(0, 1)).astype(np.float32).compute()
            self.sd_per_cell = da.std(all_data, axis=(0, 1)).astype(np.float32).compute()

        # Create dual band raster and write to output directory
        output_file = f'{scenario}_dual_band.tif'

        with rasterio.open(
            os.path.join(self.parent_directory,output_file),
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=2,  # Two bands
            dtype=self.average_per_cell.dtype,
            crs=crs,
            transform=self.transform
        ) as dst:
            dst.write(self.average_per_cell, 1)  # Write the average to the first band
            dst.write(self.sd_per_cell, 2)       # Write the standard deviation to the second band

        print(f'Dual band raster {output_file} created successfully.')
