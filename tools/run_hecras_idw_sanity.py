"""Simple runner that exercises HECRASMap loader and IDW mapping.

Usage:
    python tools/run_hecras_idw_sanity.py

It will load the same plan used in the benchmarks and map 10 sample points.
"""
import numpy as np
from pathlib import Path
from emergent.salmon_abm.sockeye_SoA import HECRASMap

plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")

print('Loading HECRAS plan (read-only):', plan)
m = HECRASMap(str(plan), field_name='Cells Minimum Elevation')
print('n coords =', m.coords.shape[0])

# sample 10 random valid points within the KDTree coords (off-grid test)
rng = np.random.default_rng(42)
idx = rng.choice(m.coords.shape[0], size=10, replace=False)
query_pts = m.coords[idx] + rng.normal(scale=0.1, size=(10,2))

vals = m.map_idw(query_pts, k=8)
for i, (pt, v) in enumerate(zip(query_pts, vals)):
    print(f'{i:2d}: point=({pt[0]:.3f},{pt[1]:.3f}) -> mapped={v:.6f}')

print('Done')
