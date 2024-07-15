# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:32:52 2024

@author: Kevin.Nebiolo
"""

#%% Import emergent
# software directory
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")


#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os

#%% Identify Workspace to Summarize
inputWS = r'C:\Users\knebiolo\Desktop\abm_simulations\sensitivity\sense_10'
tiffWS = r'C:\Users\knebiolo\Desktop\abm_simulations\sensitivity\sense_10\depth_mask_mosaic.tif'
crs = 32604
filename = os.path.join(inputWS,'sense_10')

#%% Summarize
model_summary = sockeye.summary(inputWS,tiffWS)
image_data, tiff_extent = model_summary.load_tiff(crs)
h5_files = model_summary.find_h5_files()
model_summary.get_data(h5_files)
model_summary.emergence(filename,crs)