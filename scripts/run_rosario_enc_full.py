"""Run a full Rosario Strait simulation with ENC preloading and save PID trace.
This will download ENC cells (if not cached) into ~/.emergent_cache/enc and then run
T=300s straight-line crosswind test using geographic waypoints from SIMULATION_BOUNDS.
"""
import os, math, numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, SIMULATION_BOUNDS

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1
T = 300.0
WIND_SPEED = 5.0

trace = os.path.join(OUT_DIR, 'pid_trace_rosario_enc.csv')
PID_TRACE['enabled'] = True
PID_TRACE['path'] = trace

# use Rosario bounds center as approximate start (lon, lat)
bounds = SIMULATION_BOUNDS['Rosario Strait']
lon0 = 0.5 * (bounds['minx'] + bounds['maxx'])
lat0 = 0.5 * (bounds['miny'] + bounds['maxy'])
start = np.array([lon0, lat0])
# move ~0.05 lon east (~4-5 km depending on latitude)
goal = start + np.array([0.05, 0.0])

# request ENC preloading by setting load_enc=True
sim = simulation(port_name='Rosario Strait', dt=DT, T=T, n_agents=1, load_enc=True, test_mode=None)
# verbose True so ENC progress messages print
sim.verbose = True
# assign waypoints before spawn so spawn() can initialize position/heading
sim.waypoints = [[start, goal]]

print('Preloading ENC and running full Rosario simulation...')
# load ENC (will cache to ~/.emergent_cache/enc)
sim.load_enc_features(simulation.__module__.split('.')[0] + '.config.xml_url' if False else __import__('emergent.ship_abm.config', fromlist=['xml_url']).xml_url, verbose=True)

# now spawn and run
sim.spawn()

psi0 = float(sim.psi[0])
cross_theta = psi0 + math.pi/2.0
wx = WIND_SPEED * math.cos(cross_theta)
wy = WIND_SPEED * math.sin(cross_theta)

sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

try:
    if os.path.exists(trace):
        os.remove(trace)
except Exception:
    pass

sim.run()
print('Trace written to', trace)
