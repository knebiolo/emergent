"""Run a short raster-write benchmark: hecras_write_rasters=True, defer_hdf=False
Writes canonical HDF to tools/option3_output.h5
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.profile_timestep_cprofile import build_sim
import h5py

out_h5 = os.path.join(ROOT, 'tools', 'option3_output.h5')
if os.path.exists(out_h5):
    try:
        os.remove(out_h5)
    except Exception:
        pass

# Build sim that will write rasters
sim = build_sim(num_agents=100, use_hecras=True, hecras_write_rasters=True)
# ensure sim writes to our output path if it creates HDF
try:
    sim.model_dir = os.path.join(ROOT, 'tools')
    sim.model_name = 'option3'
except Exception:
    pass
try:
    import numpy as np
    sim.heading = np.zeros(getattr(sim, 'num_agents', 100))
    if hasattr(sim, 'behavior') and sim.behavior is not None:
        sim.behavior.arbitrate = (lambda self_obj, tt: sim.heading)
except Exception:
    pass

from emergent.salmon_abm.sockeye_SoA import PID_controller
pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
try:
    pid.interp_PID()
except Exception:
    pass

for t in range(20):
    sim.timestep(t, 1.0, 9.81, pid)

# if sim has hdf5 writer, try to flush/close
try:
    if hasattr(sim, 'hdf5') and hasattr(sim.hdf5, 'close'):
        sim.hdf5.close()
except Exception:
    pass

print('Completed Option 3 run; check', out_h5)
