#!/usr/bin/env python3
"""
Simple RL training script using simulation.run() - no OpenGL visualization.
"""
import os
import sys
import json
import argparse
import numpy as np

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights


def setup_training_simulation(args):
    """Initialize simulation and RL trainer."""
    
    # HECRAS plan path
    hecras_plan = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    
    if not os.path.exists(hecras_plan):
        raise FileNotFoundError(f"HECRAS plan not found: {hecras_plan}")
    
    print(f'Using HECRAS plan: {hecras_plan}')
    print(f'Initializing training simulation with {args.agents} agents...')
    
    # Simulation configuration
    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'rl_training_run',
        'crs': 'EPSG:32605',
        'basin': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Bathymetry.shp'),
        'water_temp': 10.0,
        'start_polygon': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Start_Polygon.shp'),
        'longitudinal_profile': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'nuyakuk_centerline.shp'),
        'env_files': None,  # Using HECRAS instead
        'num_agents': args.agents,
        'fish_length': args.fish_length,
        'hecras_plan_path': hecras_plan,
        'use_hecras': True,
        'hecras_k': 1,
    }
    
    # Create output directory
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Create simulation
    sim = simulation(**config)
    print('Simulation initialized.')
    
    # Create RL trainer
    print('Setting up RL trainer...')
    trainer = RLTrainer(sim)
    
    # Apply initial behavioral weights
    trainer.apply_weights_to_simulation()
    print(f'Applied behavioral weights: cohesion={sim.cohesion_weight:.2f}, separation={sim.separation_weight:.2f}, border={sim.border_weight}')
    
    return sim, trainer, hecras_plan


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights (headless)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RL BEHAVIORAL WEIGHT TRAINING (Headless Mode)')
    print('='*80)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Timesteps per episode: {args.timesteps}')
    print(f'  Agents: {args.agents}')
    print(f'  Fish length: {args.fish_length} mm')
    print('='*80)
    print()
    
    # Setup simulation and trainer
    sim, trainer, hecras_plan = setup_training_simulation(args)
    
    # Training loop
    best_reward = -np.inf
    
    for episode in range(args.episodes):
        print(f'\n{"="*80}')
        print(f'EPISODE {episode + 1}/{args.episodes}')
        print(f'{"="*80}')
        
        # Reset simulation for new episode
        if episode > 0:
            # Re-create simulation to reset state
            config = {
                'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
                'model_name': f'rl_training_ep{episode}',
                'crs': 'EPSG:32605',
                'basin': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Bathymetry.shp'),
                'water_temp': 10.0,
                'start_polygon': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Start_Polygon.shp'),
                'longitudinal_profile': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'nuyakuk_centerline.shp'),
                'env_files': None,
                'num_agents': args.agents,
                'fish_length': args.fish_length,
                'hecras_plan_path': hecras_plan,
                'use_hecras': True,
                'hecras_k': 1,
            }
            sim = simulation(**config)
            trainer.sim = sim
            trainer.apply_weights_to_simulation()
        
        # Run simulation using built-in run() method
        print(f'Running simulation for {args.timesteps} timesteps...')
        sim.run(
            model_name=f'rl_training_ep{episode}',
            n=args.timesteps,
            dt=1.0,
            video=False,
            k_p=1.0,
            k_i=0.0,
            k_d=0.0,
            interactive=False
        )
        
        # Extract final metrics and compute reward
        final_metrics = trainer.extract_state_metrics()
        
        # Simple reward: negative of final variance (want tight cohesion)
        reward = -final_metrics['variance']
        
        print(f'\nEpisode {episode + 1} Complete')
        print(f'  Final variance: {final_metrics["variance"]:.2f}')
        print(f'  Reward: {reward:.2f}')
        
        if reward > best_reward:
            best_reward = reward
            print(f'  🏆 NEW BEST REWARD: {best_reward:.2f}')
            
            # Save best weights
            save_path = os.path.join(REPO_ROOT, 'outputs', 'rl_training', 'best_weights.json')
            with open(save_path, 'w') as f:
                json.dump(trainer.behavioral_weights.to_dict(), f, indent=2)
            print(f'  Saved best weights to {save_path}')
        
        # Mutate weights for next episode
        if episode < args.episodes - 1:
            trainer.mutate_behavioral_weights()
            print(f'\nMutated weights for next episode:')
            print(f'  cohesion: {trainer.behavioral_weights.cohesion:.2f}')
            print(f'  separation: {trainer.behavioral_weights.separation:.2f}')
            print(f'  border: {trainer.behavioral_weights.border:.2f}')
    
    print('\n' + '='*80)
    print('TRAINING COMPLETE!')
    print(f'Best reward achieved: {best_reward:.2f}')
    print('='*80)


if __name__ == '__main__':
    main()
