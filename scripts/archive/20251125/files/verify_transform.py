from emergent.ship_abm.simulation_core import simulation
from pyproj import Transformer

# create a minimal simulation object (no ENC load)
sim = simulation(port_name='Rosario Strait', dt=0.5, T=1, n_agents=1, load_enc=False)
ll2utm = Transformer.from_crs('EPSG:4326', sim.crs_utm, always_xy=True)

# sample lon/lat in the Rosario area
lon, lat = -122.7, 48.6
ux, uy = ll2utm.transform(lon, lat)

print('CRS:', sim.crs_utm)
print('LonLat:', lon, lat)
print('UTM:', ux, uy)
