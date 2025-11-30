"""Run a short per-agent-only benchmark: use_hecras=True, hecras_write_rasters=False, defer_hdf=True
Outputs deferred logs to tools/defer_logs/option2
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.profile_timestep_cprofile import build_sim

outdir = os.path.join(ROOT, 'tools', 'defer_logs', 'option2')
os.makedirs(outdir, exist_ok=True)

# build a robust minimal sim configured for profiling
sim = build_sim(num_agents=200, use_hecras=True, hecras_write_rasters=False)
# attach our chosen defer_log_dir
sim.defer_hdf = True
try:
    sim._log_writer.out_dir = outdir
except Exception:
    pass

# Monkeypatch HDF-heavy behavior methods to keep run focused on per-agent cost
import numpy as np
try:
    sim.find_nearest_refuge = lambda radius: np.zeros(getattr(sim, 'num_agents', 200), dtype=float)
    sim.already_been_here = lambda *a, **k: np.zeros(getattr(sim, 'num_agents', 200), dtype=bool)
    sim.vel_cue = lambda *a, **k: np.zeros(getattr(sim, 'num_agents', 200), dtype=float)
except Exception:
    pass

# ensure heading exists and bypass arbitration
try:
    sim.heading = np.zeros(getattr(sim, 'num_agents', 200))
    if hasattr(sim, 'behavior') and sim.behavior is not None:
        sim.behavior.arbitrate = (lambda self_obj, tt: sim.heading)
except Exception:
    pass

# disable plotting hooks if present
try:
    sim.plotting = False
except Exception:
    pass

from emergent.salmon_abm.sockeye_SoA import PID_controller
pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
try:
    pid.interp_PID()
except Exception:
    pass

for t in range(50):
    sim.timestep(t, 1.0, 9.81, pid)

print('Completed Option 2 run; logs in', outdir)
