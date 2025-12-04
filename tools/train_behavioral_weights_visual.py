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
from src.emergent.salmon_abm.salmon_viewer import launch_viewer
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights
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
        
    print(f'Using HECRAS plan: {hecras_plan}')
    
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
    
    print(f'Initializing training simulation with {args.agents} agents...')
    sim = simulation(**config)
    
    print('Simulation initialized. Setting up RL trainer...')
    trainer = RLTrainer(sim)
    sim.apply_behavioral_weights(trainer.behavioral_weights)
    
    return sim, trainer, hecras_plan


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights with visualization')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=1800, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    parser.add_argument('--dt', type=float, default=0.1, help='Timestep duration (s)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RL BEHAVIORAL WEIGHT TRAINING (Visual Mode)')
    print('='*80)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Timesteps per episode: {args.timesteps}')
    print(f'  Agents: {args.agents}')
    print(f'  Fish length: {args.fish_length} mm')
    print(f'  Timestep: {args.dt} s')
    print('='*80)
    
    # Setup simulation and trainer
    sim, trainer, hecras_plan = setup_training_simulation(args)
    
    print(f"\nDEBUG: Simulation setup complete. sim type: {type(sim)}, trainer type: {type(trainer)}")
    print(f"DEBUG: About to launch viewer...")
    
    print("\n" + "="*80)
    print("LAUNCHING VIEWER...")
    print("="*80)
    
    # Launch viewer with RL trainer
    try:
        total_time = args.timesteps * args.dt
        launch_viewer(
            simulation=sim,
            dt=args.dt,
            T=total_time,
            rl_trainer=trainer,
            show_velocity_field=False,  # Too expensive for large grids
            show_depth=True
        )
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR launching viewer: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
