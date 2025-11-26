"""Simple single test - just verify it loads and samples"""
import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

port = sys.argv[1] if len(sys.argv) > 1 else "Seattle"

print(f"Testing {port}...")
start = time.time()

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np

current_fn = get_current_fn(port=port, start=datetime.now())
print(f"Loaded in {time.time()-start:.1f}s")

# Sample one point
result = current_fn(np.array([-122.34]), np.array([47.60]), datetime.now())
print(f"Sample: u={result[0,0]:.4f}, v={result[0,1]:.4f} m/s")
print(f"Total time: {time.time()-start:.1f}s")
print("SUCCESS!")
