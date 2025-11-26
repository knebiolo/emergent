import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
sys.path.insert(0, "src")
from emergent.ship_abm.simulation_core import simulation

print("Testing collision detection...")
sim = simulation(port_name="Baltimore", dt=0.1, T=10, n_agents=1, load_enc=True, verbose=False)
bridge_gdf = gpd.read_file("data/ship_abm/fsk_bridge.geojson")
if "BRIDGE" in sim.enc_data:
    sim.enc_data["BRIDGE"] = gpd.GeoDataFrame(pd.concat([sim.enc_data["BRIDGE"], bridge_gdf], ignore_index=True), crs=sim.enc_data["BRIDGE"].crs)
else:
    sim.enc_data["BRIDGE"] = bridge_gdf
sim.waypoints = [[[-76.5297, 39.2156], [-76.5297, 39.2156]]]
sim.spawn_speed = 3.0
sim.spawn()
events = sim._check_allision(0.0)
print(f"Events: {len(events)}")
if events:
    print("SUCCESS!")
    for e in events: print(e)
else:
    print("FAILED")
