import sys
import sys
import time
import numpy as np
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src')
from PyQt5 import QtWidgets
from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.simulation_core import simulation

print('TEST_SCRIPT: launching QApplication')
app = QtWidgets.QApplication([])
print('TEST_SCRIPT: creating simulation')
# use a port that exists in SIMULATION_BOUNDS to avoid KeyError
sim = simulation(port_name='Seattle', dt=0.1, T=10, n_agents=1, load_enc=False, verbose=False)
# mark env loaded to avoid background forcing load
sim._env_loaded = True
# trivial environment functions
sim.wind_fn = lambda lon, lat, now: np.array([[0.0, 0.0]])
sim.current_fn = lambda lon, lat, now: np.array([[0.0, 0.0]])
# Provide trivial waypoints so spawn() does not raise
try:
    # Only prepopulate waypoints when explicitly allowed via an env var.
    # This avoids accidentally injecting routes during normal interactive runs.
    import os
    if os.environ.get('EMERGENT_ALLOW_PREPOP', '0') == '1':
        cx = 0.5 * (sim.minx + sim.maxx)
        cy = 0.5 * (sim.miny + sim.maxy)
        sim.waypoints = [[(cx, cy), (cx + 100.0, cy)] for _ in range(sim.n)]
        print(f"TEST_SCRIPT: prepopulated sim.waypoints with center {(cx,cy)}")
    else:
        print('TEST_SCRIPT: skipping prepopulation of sim.waypoints (set EMERGENT_ALLOW_PREPOP=1 to enable)')
except Exception as e:
    print(f"TEST_SCRIPT: failed to set waypoints: {e}")

print('TEST_SCRIPT: instantiating viewer')
viewer = ship_viewer(port_name='TEST', xml_url='', dt=0.1, T=10, n_agents=1, load_enc=False, sim_instance=sim)
viewer.show()

print('TEST_SCRIPT: viewers attributes snapshot:')
print('  has label_items?', hasattr(viewer, 'label_items'))
print('  has ship_items?', hasattr(viewer, 'ship_items'))
print('  has timer?', hasattr(viewer, 'timer'))
print('  attrs sample:', [a for a in dir(viewer) if a.endswith('_items') or a.startswith('btn_')][:30])

print('TEST_SCRIPT: calling _start_simulation')
viewer._start_simulation()

print('TEST_SCRIPT: running event loop for 2s to allow ticks')
start = time.time()
while time.time() - start < 2.0:
    app.processEvents()
    time.sleep(0.05)

print('TEST_SCRIPT: done; quitting')
app.quit()
