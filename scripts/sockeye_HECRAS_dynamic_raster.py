# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
#%% Import emergent
# software directory
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")
#sys.path.append(r"C:\Users\Isha Deo\OneDrive - Kleinschmidt Associates, Inc\GitHub\emergent\emergent")
#sys.path.append(r"C:\Users\AYoder\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")


#%% Import dpeendencies
# import dependencies
import emergent as sockeye
import os
# import ga optimization script

# identify input and output model names
HECRAS_model = 'Nuyakuk_Production_p16.hdf'
model_name = '11619_1m'

# identify directories
#model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\simulations",model_name)
model_dir = os.path.join(r"C:\Users\knebiolo\Desktop\abm_simulations\hydrograph",model_name)
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS Results for Kevin\20240122"



#%% Set model parameters
# identify the coordinate reference system for the model
crs = 32604

#%% Read environmental data into model
# read HECRAS model and create environment rasters
sockeye.HECRAS(model_dir,
               os.path.join(HECRAS_dir,HECRAS_model),
               1.0,
               crs)

