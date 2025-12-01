"""Train behavioral weights for sockeye salmon ABM using reinforcement learning.

This script pre-trains instinctual behavioral parameters (cohesion, separation,
rheotaxis, border avoidance, collision avoidance) that produce realistic schooling
and upstream migration behavior.

Once trained, these weights are frozen and reused across different river geometries.
Spatial knowledge (specific routes, obstacle memory) is reset each simulation.

Usage:
    python tools/train_behavioral_weights.py --episodes 50 --timesteps 100 --agents 200
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights
import numpy as np


def setup_training_simulation(args):
    """Initialize a simulation for RL training."""
    
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
    
    # Use pre-generated rasters from outputs/hecras_run
    out_dir = os.path.join(REPO_ROOT, 'outputs', 'hecras_run')
    
    env_files = {
        'elev': os.path.join(out_dir, 'elev.tif'),
        'depth': os.path.join(out_dir, 'depth.tif'),
        'wetted': os.path.join(out_dir, 'wetted.tif'),
        'distance_to': os.path.join(out_dir, 'distance_to.tif'),
        'x_vel': os.path.join(out_dir, 'x_vel.tif'),
        'y_vel': os.path.join(out_dir, 'y_vel.tif'),
        'vel_dir': os.path.join(out_dir, 'vel_dir.tif'),
        'vel_mag': os.path.join(out_dir, 'vel_mag.tif')
    }
    
    start_poly = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location', 'start_loc_river_right.shp')
    long_profile = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'Longitudinal', 'longitudinal.shp')
    
    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'behavioral_training',
        'crs': 'EPSG:26904',
        'basin': 'Nushagak River',
        'water_temp': 10.0,
        'start_polygon': start_poly,
        'env_files': env_files,
        'longitudinal_profile': long_profile,
        'fish_length': args.fish_length,
        'num_timesteps': args.timesteps,
        'num_agents': args.agents,
        'use_gpu': False,
        'defer_hdf': True,  # Fast logging for training
    }
    
    os.makedirs(config['model_dir'], exist_ok=True)
    
    print(f'Initializing training simulation with {args.agents} agents...')
    sim = simulation(**config)
    
    # Enable HECRAS node mapping
    import h5py
    hdf_path = os.path.splitext(hecras_plan)[0] + '.hdf'
    print('Loading HECRAS nodes for node-based mapping')
    hdf = h5py.File(hdf_path, 'r')
    try:
        pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    except Exception:
        pts = None
    
    node_fields = {}
    try:
        node_fields['depth'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1] - np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
    except Exception:
        pass
    try:
        node_fields['vel_x'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
        node_fields['vel_y'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
    except Exception:
        pass
    hdf.close()

    if pts is not None and node_fields:
        # Initialize required fields
        n = sim.num_agents
        for attr in ['depth', 'x_vel', 'y_vel', 'vel_mag', 'wet', 'distance_to']:
            if not hasattr(sim, attr):
                setattr(sim, attr, np.zeros(n, dtype=float))
        
        sim.enable_hecras(pts, node_fields, k=1)
        
        if 'depth' in node_fields:
            sim.depth = sim.apply_hecras_mapping(node_fields['depth'])
        if 'vel_x' in node_fields:
            sim.x_vel = sim.apply_hecras_mapping(node_fields['vel_x'])
        if 'vel_y' in node_fields:
            sim.y_vel = sim.apply_hecras_mapping(node_fields['vel_y'])
            sim.vel_mag = np.sqrt(sim.x_vel**2 + sim.y_vel**2)

        # Sample distance_to raster at HECRAS node locations
        try:
            import rasterio
            with rasterio.open(env_files['distance_to']) as src:
                coords = [(float(p[0]), float(p[1])) for p in pts]
                samples = np.fromiter((s[0] for s in src.sample(coords)), dtype=float)
                node_fields['distance_to'] = samples
                print(f"Sampled distance_to at {len(samples)} HECRAS nodes (range: {samples.min():.2f}-{samples.max():.2f}m)")
        except Exception as e:
            print(f"Warning: Failed to sample distance_to: {e}")

        if 'distance_to' in node_fields:
            sim.distance_to = sim.apply_hecras_mapping(node_fields['distance_to'])
            print(f"Applied distance_to to agents (range: {sim.distance_to.min():.2f}-{sim.distance_to.max():.2f}m)")
        
        # Re-initialize heading based on flow
        flow_direction = np.arctan2(sim.y_vel, sim.x_vel)
        sim.heading = flow_direction - np.pi  # Point upstream
        
        print('HECRAS node mapping enabled for training.')
    
    return sim


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights for salmon ABM using RL')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=200, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=500, help='Fish length in mm')
    parser.add_argument('--output', type=str, default='outputs/rl_training/behavioral_weights.json',
                        help='Path to save trained weights')
    args = parser.parse_args()
    
    # Setup simulation
    sim = setup_training_simulation(args)
    
    # Create trainer
    trainer = RLTrainer(sim)
    
    # Load existing weights if available
    weights_path = os.path.join(REPO_ROOT, args.output)
    if os.path.exists(weights_path):
        print(f"Loading existing weights from {weights_path}")
        trainer.behavioral_weights.load(weights_path)
    else:
        print("Starting from default behavioral weights")
    
    # Train
    print("\n" + "="*80)
    print("STARTING RL TRAINING")
    print("="*80)
    print(f"Episodes: {args.episodes}")
    print(f"Timesteps per episode: {args.timesteps}")
    print(f"Agents: {args.agents}")
    print(f"Output: {weights_path}")
    print("="*80 + "\n")
    
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    learned_weights = trainer.train(
        num_episodes=args.episodes,
        timesteps_per_episode=args.timesteps,
        save_path=weights_path
    )
    
    # Save final weights
    learned_weights.save(weights_path)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best weights saved to: {weights_path}")
    print("\nLearned behavioral parameters:")
    for key, value in learned_weights.to_dict().items():
        print(f"  {key}: {value:.4f}")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
