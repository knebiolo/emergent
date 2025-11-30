from emergent.salmon_abm.sockeye_SoA import simulation
import os

model_dir = r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent'
model_name = 'test_alongstream'
crs = 'EPSG:32605'
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = os.path.join(model_dir, 'data', 'salmon_abm', 'start_polygon.geojson')
# create a tiny start polygon if it doesn't exist
if not os.path.exists(start_polygon):
    import json
    poly = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}]}
    with open(start_polygon,'w') as f:
        json.dump(poly,f)

env_files = {}

# instantiate
sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=start_polygon, num_timesteps=10, num_agents=10)
print('HDF path:', sim.db)
try:
    env = sim.hdf5.get('environment')
    if env is None:
        print('No environment group present')
    else:
        print('environment keys:', list(env.keys()))
except Exception as e:
    print('Error listing environment:', e)
sim.hdf5.close()
