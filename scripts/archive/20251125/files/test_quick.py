"""Quick test with minimal output"""
import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

print("Starting test...")
start = time.time()

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np

print(f"Imports done ({time.time()-start:.1f}s)")

current_fn = get_current_fn(port="Baltimore", start=datetime.now())
print(f"Function built! ({time.time()-start:.1f}s)")

# Test sampling
lons = np.array([-76.45])
lats = np.array([39.30])
result = current_fn(lons, lats, datetime.now())
print(f"Sample: u={result[0,0]:.4f}, v={result[0,1]:.4f} m/s ({time.time()-start:.1f}s)")
print("SUCCESS!")
