"""
Headless test runner: create simulation with HECRAS plan and launch SalmonViewer briefly
Saves a screenshot to outputs/tin_screenshot.png created by the viewer when mesh is ready.
"""
import os
import sys
from time import sleep

# ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from emergent.salmon_abm.sockeye import simulation
    from emergent.salmon_abm.salmon_viewer import SalmonViewer
except Exception as e:
    print('Import error:', e)
    raise

# Use the default model_dir from repo structure
model_dir = os.path.abspath('.')
hecras_default = os.path.join(model_dir, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
if not os.path.exists(hecras_default):
    print('HECRAS plan not found at', hecras_default)
    sys.exit(1)

# create a minimal simulation instance with HECRAS enabled
sim = simulation(num_agents=10, num_timesteps=100, model_dir=model_dir, hecras_plan_path=hecras_default, use_hecras=True, hecras_write_rasters=False)

# create viewer (will save screenshot once mesh is ready)
viewer = SalmonViewer(sim, dt=0.1, T=1.0)

# run the viewer for a short time to allow mesh build and screenshot. On headless systems this may fail.
try:
    # show for a short period
    import threading
    t = threading.Thread(target=viewer.run, daemon=True)
    t.start()
    sleep(5)
    print('Viewer run attempted; check outputs/tin_screenshot.png')
except Exception as e:
    print('Viewer run failed:', e)
    raise
