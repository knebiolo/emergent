"""RL training script for behavioral weight optimization.

Trains behavioral weights using reinforcement learning to maximize schooling quality
and migration success. Uses evolutionary strategy with Gaussian weight perturbation.

Usage:
    # Default training (50 episodes, 100 timesteps, 200 agents)
    python tools/train_behavioral_weights.py

    # Quick test run
    python tools/train_behavioral_weights.py --episodes 10 --timesteps 50 --agents 50

    # Extended training
    python tools/train_behavioral_weights.py --episodes 100 --timesteps 200 --agents 500

    # Custom output directory
    python tools/train_behavioral_weights.py --out outputs/rl_custom --episodes 50

Output:
    - best_weights.json: Optimized behavioral weights
    - training_history.csv: Reward progression over episodes
    - training_config.json: Training hyperparameters and metadata
"""
import os
import argparse
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer
from emergent.salmon_abm import io


def discover_env_files(base_dir):
    """Discover environment raster files in base directory."""
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def create_simulation_factory(
    model_dir,
    model_name,
    crs,
    basin,
    water_temp,
    start_polygon,
    env_files,
    num_agents,
    num_timesteps,
    fish_length=None,
    dt=1.0
):
    """Create a factory function that returns configured simulation instances.
    
    Args:
        model_dir: Path to HECRAS model directory
        model_name: Name of HECRAS model
        crs: Coordinate reference system
        basin: Basin name
        water_temp: Water temperature (°C)
        start_polygon: Path to starting polygon shapefile
        env_files: List of environment raster files
        num_agents: Number of agents per episode
        num_timesteps: Number of timesteps per episode
        fish_length: Fixed fish length (mm) or None for random
        dt: Timestep duration in seconds
        
    Returns:
        Callable that takes BehavioralWeights and returns a configured simulation object
    """
    def factory_func(weights: BehavioralWeights):
        """Create and configure simulation with given behavioral weights."""
        # Create simulation with minimal output writes (compute-only)
        sim = simulation(
            model_dir=model_dir,
            model_name=model_name,
            crs=crs,
            basin=basin,
            water_temp=water_temp,
            start_polygon=start_polygon,
            env_files=env_files,
            longitudinal_profile=None,
            fish_length=fish_length,
            num_timesteps=num_timesteps,
            num_agents=num_agents,
            use_gpu=False,
            pid_tuning=False,
            db_path=None,  # temporary DB (auto-cleanup)
            output_write_mode='none',  # skip all writes (compute-only)
            output_write_backend='sync'
        )
        
        # Load behavioral weights into simulation
        sim.load_behavioral_weights(weights_dict=weights.to_dict())
        
        # Enable neighbor sensing for schooling cues
        sensory_range = 2.0  # Fixed biological constant (body lengths)
        sim.neighbor_buffer_radius = sensory_range * (fish_length / 1000.0 if fish_length else 1.0)
        sim.neighbor_buffer_lengths = sensory_range
        
        return sim
    
    return factory_func


def save_training_results(out_dir, best_weights, history, config):
    """Save training results to output directory.
    
    Args:
        out_dir: Output directory path
        best_weights: BehavioralWeights instance with best parameters
        history: List of (episode, reward) tuples
        config: Training configuration dict
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Save best weights as JSON
    weights_path = os.path.join(out_dir, 'best_weights.json')
    best_weights.to_json(Path(weights_path))
    print(f'Saved best weights to {weights_path}')
    
    # Save training history as CSV
    history_path = os.path.join(out_dir, 'training_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        writer.writerows(history)
    print(f'Saved training history to {history_path}')
    
    # Save training config as JSON
    config_path = os.path.join(out_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Saved training config to {config_path}')
    
    # Print summary
    print('\n=== Training Summary ===')
    print(f'Episodes: {len(history)}')
    print(f'Initial reward: {history[0][1]:.2f}')
    print(f'Final reward: {history[-1][1]:.2f}')
    print(f'Best reward: {max(r for _, r in history):.2f}')
    print(f'Improvement: {history[-1][1] - history[0][1]:.2f}')
    print('\nTop 5 weights:')
    weights_dict = best_weights.to_dict()
    for k, v in sorted(weights_dict.items(), key=lambda x: -abs(x[1]) if isinstance(x[1], (int, float)) else 0)[:5]:
        print(f'  {k}: {v}')


def main():
    parser = argparse.ArgumentParser(
        description='Train behavioral weights using RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes (default: 50)')
    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of timesteps per episode (default: 100)')
    parser.add_argument('--agents', type=int, default=200,
                        help='Number of agents per episode (default: 200)')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                        help='Gaussian exploration noise stddev (default: 0.1)')
    parser.add_argument('--fish-length', type=float, default=None,
                        help='Fixed fish length in mm (default: random 400-600mm)')
    
    # Environment parameters
    parser.add_argument('--model-dir', type=str,
                        default='data/salmon_abm/nuyakuk',
                        help='HECRAS model directory (default: data/salmon_abm/nuyakuk)')
    parser.add_argument('--model-name', type=str, default='nuyakuk',
                        help='HECRAS model name (default: nuyakuk)')
    parser.add_argument('--basin', type=str, default='nuyakuk',
                        help='Basin name (default: nuyakuk)')
    parser.add_argument('--start-polygon', type=str,
                        default='data/salmon_abm/nuyakuk/start_polygon.shp',
                        help='Starting polygon shapefile (default: data/salmon_abm/nuyakuk/start_polygon.shp)')
    parser.add_argument('--crs', type=str, default='EPSG:3338',
                        help='Coordinate reference system (default: EPSG:3338)')
    parser.add_argument('--water-temp', type=float, default=10.0,
                        help='Water temperature in °C (default: 10.0)')
    
    # Output parameters
    parser.add_argument('--out', type=str,
                        default=f'outputs/rl_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Output directory (default: outputs/rl_training_TIMESTAMP)')
    parser.add_argument('--initial-weights', type=str, default=None,
                        help='Path to initial weights JSON (default: use defaults)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_dir):
        raise ValueError(f'Model directory not found: {args.model_dir}')
    if not os.path.exists(args.start_polygon):
        raise ValueError(f'Start polygon not found: {args.start_polygon}')
    
    # Discover environment files
    env_files = discover_env_files(args.model_dir)
    if not env_files:
        raise ValueError(f'No environment rasters found in {args.model_dir}')
    print(f'Found {len(env_files)} environment files:')
    for ef in env_files:
        print(f'  - {os.path.basename(ef)}')
    
    # Load or create initial weights
    if args.initial_weights:
        print(f'\nLoading initial weights from {args.initial_weights}')
        initial_weights = BehavioralWeights.from_json(args.initial_weights)
    else:
        print('\nUsing default initial weights')
        initial_weights = BehavioralWeights()
    
    # Create simulation factory
    print('\nCreating simulation factory...')
    factory = create_simulation_factory(
        model_dir=args.model_dir,
        model_name=args.model_name,
        crs=args.crs,
        basin=args.basin,
        water_temp=args.water_temp,
        start_polygon=args.start_polygon,
        env_files=env_files,
        num_agents=args.agents,
        num_timesteps=args.timesteps,
        fish_length=args.fish_length
    )
    
    # Configure RL trainer
    config = {
        'exploration_noise': args.exploration_noise,
        'body_length': (args.fish_length / 1000.0) if args.fish_length else 0.5,  # meters
        'dt': 1.0,  # seconds
        'num_episodes': args.episodes,
        'num_timesteps': args.timesteps,
        'num_agents': args.agents,
        'model_dir': args.model_dir,
        'model_name': args.model_name,
        'basin': args.basin,
        'start_polygon': args.start_polygon,
        'crs': args.crs,
        'water_temp': args.water_temp,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Create trainer
    print('\nInitializing RL trainer...')
    trainer = RLTrainer(
        simulation_factory=factory,
        initial_weights=initial_weights,
        config=config
    )
    
    # Run training
    print(f'\n=== Starting RL Training ===')
    print(f'Episodes: {args.episodes}')
    print(f'Timesteps per episode: {args.timesteps}')
    print(f'Agents per episode: {args.agents}')
    print(f'Exploration noise: {args.exploration_noise}')
    print(f'Output directory: {args.out}')
    print('\nTraining in progress...')
    
    start_time = time.time()
    best_weights, history = trainer.train(num_episodes=args.episodes)
    elapsed_time = time.time() - start_time
    
    # Save results
    print(f'\nTraining complete in {elapsed_time:.1f}s ({elapsed_time/args.episodes:.1f}s per episode)')
    save_training_results(args.out, best_weights, history, config)


if __name__ == '__main__':
    main()
