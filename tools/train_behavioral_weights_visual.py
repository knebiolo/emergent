"""Train behavioral weights with live PyQt5 visualization.

Allows you to watch the RL training unfold in real-time with interactive controls.

Usage:
    python tools/train_behavioral_weights_visual.py --episodes 10 --timesteps 100 --agents 100
"""
import argparse
import os
import sys
import json
from pathlib import Path

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import visualization and simulation
from emergent.salmon_abm.salmon_viewer_v2 import launch_viewer
from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights
import numpy as np


def setup_training_simulation(args):
    """Initialize a simulation for RL training with visualization."""
    
    # Auto-discover HECRAS plan
    hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
    hecras_plan = None
    for f in os.listdir(hecras_folder):
        if f.endswith('.p05.hdf'):
            hecras_plan = os.path.join(hecras_folder, f)
            break
            
    if not hecras_plan or not os.path.exists(hecras_plan):
        raise FileNotFoundError('HECRAS plan not found')
        
    # quiet: do not print HECRAS plan path to reduce console noise
    
    start_poly = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location', 'start_loc_river_right.shp')
    centerline_path = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'Longitudinal', 'longitudinal.shp')
    
    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'behavioral_training_visual',
        'crs': 'EPSG:26904',
        'basin': 'Nushagak River',
        'water_temp': 10.0,
        'start_polygon': start_poly,
        'centerline': centerline_path,
        'fish_length': args.fish_length,
        'num_timesteps': args.timesteps,
        'num_agents': args.agents,
        'use_gpu': False,
        'defer_hdf': True,
        'hecras_plan_path': hecras_plan,
        'use_hecras': True,
        'hecras_k': 1,
    }
    
    os.makedirs(config['model_dir'], exist_ok=True)
    
    sim = simulation(**config)
    trainer = RLTrainer(sim)
    try:
            sim.apply_behavioral_weights(trainer.behavioral_weights)
    except Exception as e:
        print(f'ERROR applying behavioral weights: {e}')
        import traceback
        traceback.print_exc()
        raise

    # Enforce simulation-owned PID: the simulation must attach `sim.pid_controller`.
    if not hasattr(sim, 'pid_controller') or getattr(sim, 'pid_controller', None) is None:
        raise RuntimeError('Simulation did not attach a pid_controller; ensure the sim creates it during init')
    
    return sim, trainer, hecras_plan


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights with visualization')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=1800, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    parser.add_argument('--dt', type=float, default=0.1, help='Timestep duration (s)')
    
    args = parser.parse_args()
    
    # Minimal startup logging to avoid hanging/log spam
    
    # Setup simulation and trainer, then hand control to the simulation run loop
    sim, trainer, hecras_plan = setup_training_simulation(args)

    model_name = 'behavioral_training_visual'
    n = args.timesteps
    dt = args.dt

    # Run the simulation directly. This keeps the script minimal and delegates
    # execution to the `simulation` class (which owns PID and the run loop).
    sim.run(model_name, n, dt, video=False, interactive=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise
