# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:41:59 2024

@author: Kevin.Nebiolo

Intent of script is to test Emergent summary functions
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
import numpy as np

#%% Identify directories and Summarize
parent_dir = r"C:\Users\knebiolo\Desktop\abm_simulations\summ_test"
tif_dir = r"C:\Users\knebiolo\Desktop\abm_simulations\summ_test\depth.tif"

summary = sockeye.summary(parent_dir, tif_dir)
summary.get_data()
summary.load_tiff(32604)
summary.emergence()
