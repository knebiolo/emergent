# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:32:52 2024

@author: Kevin.Nebiolo
"""

#%% Import emergent
# software directory
import sys
#sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")
sys.path.append(r"C:\Users\EMuhlestein\OneDrive - Kleinschmidt Associates, Inc\Software\emergent\emergent")

#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os

#%% Identify Workspace to Summarize
inputWS = r'Q:\Client_Data\Other\Nuyakuk\01_ProjData\Studies\ABM\production_output\5000\left'
tiffWS = r'Q:\Client_Data\Other\Nuyakuk\01_ProjData\Studies\ABM\production_flow_surfaces\5000\elev_mask_mosaic.tif'
shapefile_path = r'Q:\Internal_Data\Staff_Projects\ENM\Nuyakuk\test\top.shp'
parent_directory=inputWS
crs = 32604
filename = os.path.join(inputWS,'left')
scenario_name = 'cf5000'

#%% Summarize
model_summary = sockeye.summary(inputWS,tiffWS)
image_data, tiff_extent = model_summary.load_tiff(crs)
h5_files = model_summary.find_h5_files()
model_summary.get_data(h5_files)
model_summary.kcal_statistics_directory()
model_summary.Kcal_histogram_by_timestep_intervals_for_all_simulations()
model_summary.kaplan_curve(shapefile_path, tiffWS)
model_summary.plot_lengths()
model_summary.length_statistics()
model_summary.plot_weights()
model_summary.weight_statistics()
model_summary.plot_body_depths()
model_summary.body_depth_statistics()
model_summary.kcal_statistics()
#model_summary.emergence(h5_files,scenario_name,crs)
