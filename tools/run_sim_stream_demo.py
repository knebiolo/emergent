#!/usr/bin/env python3
"""Run a tiny simulation instance and stream raw frames to the viewer for testing."""
import time
from emergent.salmon_abm.simulation import simulation

if __name__ == '__main__':
    # create a minimal sim with 1000 agents and a small number of steps
    sim = simulation(model_dir='.', model_name='demo', crs=None, basin=None, water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None, fish_length=50, num_timesteps=200, num_agents=1000)
    # Move agents in a simple radial swirl inside timestep for visibility
    def simple_timestep(simobj, i, dt, pid_controller=None):
        t = float(i)
        angles = (simobj.prev_X + simobj.prev_Y) * 0.001 + t * 0.01
        simobj.X = simobj.prev_X + 0.1 * np.cos(angles)
        simobj.Y = simobj.prev_Y + 0.1 * np.sin(angles)
        simobj.prev_X = simobj.X.copy()
        simobj.prev_Y = simobj.Y.copy()
    # monkeypatch the timestep method
    import numpy as np
    sim.timestep = lambda i, dt, pid_controller=None: simple_timestep(sim, i, dt)

    # run with live streaming enabled
    sim.run(n=200, dt=1.0, viewer_live=True, viewer_host='127.0.0.1', viewer_port=50007, viewer_stream_raw=True, viewer_fps=20)
