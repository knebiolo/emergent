"""
Final test: Use get_current_fn() and verify it stops at first success
"""
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

print("="*70)
print("TESTING get_current_fn() - SHOULD STOP AT FIRST SUCCESS")
print("="*70)

from emergent.ship_abm.ofs_loader import get_current_fn

print("\n1. Loading current function for Baltimore...")
current_fn = get_current_fn(port="Baltimore", start=datetime.now())
print("   [SUCCESS] Function returned!\n")

print("2. Sampling at 5 locations in Chesapeake Bay:")
locations = [
    (-76.60, 39.20, "SW"),
    (-76.45, 39.27, "S"),
    (-76.45, 39.35, "Mid"),
    (-76.45, 39.42, "N"),
    (-76.30, 39.48, "NE")
]

now = datetime.now()
for lon, lat, name in locations:
    lons = np.array([lon])
    lats = np.array([lat])
    result = current_fn(lons, lats, now)
    u, v = result[0, 0], result[0, 1]
    speed = np.sqrt(u**2 + v**2)
    print(f"   {name:4s} ({lon:6.2f}, {lat:5.2f}): u={u:7.4f}, v={v:7.4f} -> {speed:.4f} m/s")

print("\n" + "="*70)
