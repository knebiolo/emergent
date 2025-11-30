"""Small demo to run a short simulation and save an agent scatter PNG using Matplotlib Agg backend.

Usage: python tools/run_matplotlib_demo.py

This script builds a minimal simulation instance using the existing profiling builder
and runs 10 timesteps, then dumps a PNG of agent positions.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ensure repo root is importable
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emergent.salmon_abm.sockeye_SoA import simulation


def run_demo(out_png='tools/matplotlib_demo.png'):
    # Build a small sim using the profiling builder in tools.profile_timestep_cprofile
    from tools.profile_timestep_cprofile import build_sim
    sim = build_sim(num_agents=200, use_hecras=False, hecras_write_rasters=False)

    # Run a few timesteps
    from emergent.salmon_abm.sockeye_SoA import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    try:
        pid.interp_PID()
    except Exception:
        pass

    for t in range(10):
        sim.timestep(t, 1.0, 9.81, pid)

    # Plot agent positions
    plt.figure(figsize=(6,4), dpi=150)
    plt.scatter(sim.X, sim.Y, s=1, c='blue')
    plt.title('Demo: agent positions after 10 timesteps')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    print('Wrote', out_png)


if __name__ == '__main__':
    run_demo()
