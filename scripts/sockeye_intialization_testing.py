# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:31:19 2023

@author: KNebiolo

Script Intent: test out simulation intialization methods
"""
# software directory
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\emergent")

# import dependencies
from emergent import sockeye
import os

# identify directories
model_dir = r"J:\2819\005\Calcs\ABM\Output"
HECRAS_dir = r"J:\2819\276\Calcs\HEC-RAS 6.3.1"

# identify input and output model names
HECRAS_model = 'NuyakukABM2D.p01.hdf'
model_name = 'test.hdf'

# identify the coordinate reference system for the model
crs = 'EPSG:32604'

# test out sockeye
sim = sockeye.simulation(model_dir,model_name,crs)
sim.HECRAS(os.path.join(HECRAS_dir,HECRAS_model))

