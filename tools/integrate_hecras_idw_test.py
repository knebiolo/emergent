"""Integration test: simulate caching and per-agent mapping using the new helpers.

Usage:
    python tools/integrate_hecras_idw_test.py
"""
import types
import numpy as np
from pathlib import Path
from emergent.salmon_abm.sockeye_SoA import load_hecras_plan_cached, map_hecras_for_agents

plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
print('Plan:', plan)

# fake simulation object
sim = types.SimpleNamespace()

# request multiple fields commonly used in the sim
fields = [
    'Cells Minimum Elevation',
    'Water Surface',
    'Cell Velocity - Velocity X',
    'Cell Velocity - Velocity Y'
]

# load and cache
m = load_hecras_plan_cached(sim, str(plan), field_names=fields)
print('Cached map coords:', m.coords.shape[0])

# sample 20 random agent points near valid coords
rng = np.random.default_rng(1)
idx = rng.choice(m.coords.shape[0], size=20, replace=False)
agent_xy = m.coords[idx] + rng.normal(scale=0.2, size=(20,2))
out = map_hecras_for_agents(sim, agent_xy, str(plan), field_names=fields, k=8)

for i, xy in enumerate(agent_xy):
    vals = [out[f][i] for f in fields]
    print(f"{i:2d}: {xy[0]:.3f},{xy[1]:.3f} -> " + ", ".join([f"{v:.6f}" for v in vals]))

print('Done')
