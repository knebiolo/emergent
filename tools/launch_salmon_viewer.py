import sys, os
from types import SimpleNamespace

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.train_behavioral_weights_visual import setup_training_simulation
from emergent.salmon_abm.salmon_viewer_v2 import launch_viewer

if __name__ == '__main__':
    args = SimpleNamespace()
    args.fish_length = 450
    args.timesteps = 1800
    args.agents = 100
    args.dt = 0.1

    sim, trainer, hecras_plan = setup_training_simulation(args)

    total_time = args.timesteps * args.dt
    launch_viewer(simulation=sim, dt=args.dt, T=total_time, rl_trainer=trainer, show_depth=True)
