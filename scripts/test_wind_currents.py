# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 23:21:33 2025

@author: Kevin.Nebiolo
"""

# 1) Copy this into a new file (e.g. test_ofs.py) alongside your emergent repo:
from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
import numpy as np
from datetime import datetime

# 2) Pick the port you’re simulating (must match SIMULATION_BOUNDS)
port = "Galveston"
curr_fn = get_current_fn(port)
wind_fn = get_wind_fn(port)

# 3) Build a small lon/lat grid over your bounding box
lon_min, lon_max = -96.0, -94.0
lat_min, lat_max = 28.5, 30.0

n = 5
lons = np.linspace(lon_min, lon_max, n)
lats = np.linspace(lat_min, lat_max, n)
LON, LAT = np.meshgrid(lons, lats)
LON[LON < 0] += 360  # Convert to 0–360 to match RTOFS if needed

# 4) Sample at “now”
now = datetime.utcnow()
uvc = curr_fn(LON.ravel(), LAT.ravel(), now)
uvw = wind_fn(LON.ravel(), LAT.ravel(), now)

# 5) Print the arrays to confirm variation
print("Currents U:\n", uvc[:,0].reshape(n,n))
print("Currents V:\n", uvc[:,1].reshape(n,n))
print("Wind    U:\n", uvw[:,0].reshape(n,n))
print("Wind    V:\n", uvw[:,1].reshape(n,n))
