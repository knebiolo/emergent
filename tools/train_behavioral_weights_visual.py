"""Train behavioral weights with live OpenGL visualization.

Allows you to watch the RL training unfold in real-time with directional agent indicators.

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

# Import must come after path setup
import moderngl_window as mglw
from tools.run_hecras_opengl import FishSimViewer
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights
import numpy as np


class RLTrainingViewer(FishSimViewer):
    """Extended viewer with RL training overlay."""
    
    def __init__(self, **kwargs):
        # Get training context from class attribute
        ctx = self.__class__.training_context
        self.trainer = ctx['trainer']
        self.current_episode = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.best_reward = -np.inf
        self.episode_complete = False
        
        # Call parent init (will set up sim from sim_context)
        super().__init__(**kwargs)
        
    def on_render(self, time, frametime):
        """Override render to add RL training logic."""
        # Call parent render (background + agents)
        super().on_render(time, frametime)
        
        # RL training update
        if not self.paused and self.current_timestep < self.timesteps:
            # Extract metrics and compute reward
            current_metrics = self.trainer.extract_state_metrics()
            
            if self.prev_metrics is not None:
                reward = self.trainer.compute_reward(self.prev_metrics, current_metrics)
                self.episode_reward += reward
                
                # Print reward every 10 steps
                if self.current_timestep % 10 == 0:
                    print(f'  Episode {self.current_episode} | t={self.current_timestep} | step_reward={reward:.2f} | cumulative={self.episode_reward:.2f}')
            
            self.prev_metrics = current_metrics
        
        # Episode completed
        if self.current_timestep >= self.timesteps and not self.episode_complete:
            self.episode_complete = True
            print(f'\n{"="*80}')
            print(f'EPISODE {self.current_episode} COMPLETE | Total Reward: {self.episode_reward:.2f}')
            
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                print(f'🏆 NEW BEST REWARD: {self.best_reward:.2f}')
                
                # Save best weights
                save_path = os.path.join(REPO_ROOT, 'outputs', 'rl_training', 'best_weights.json')
                with open(save_path, 'w') as f:
                    json.dump(self.trainer.behavioral_weights.to_dict(), f, indent=2)
                print(f'Saved best weights to {save_path}')
            
            print(f'{"="*80}\n')
            
            # Mutate weights for next episode
            self.trainer.behavioral_weights.mutate(scale=0.1)
            self.sim.apply_behavioral_weights(self.trainer.behavioral_weights)
            
            # Reset episode
            print(f'Starting Episode {self.current_episode + 1}...')
            self.sim.reset_spatial_state()
            self.current_episode += 1
            self.current_timestep = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
            self.episode_complete = False
        
        # Update window title with RL info
        self.wnd.title = (f"RL Training - Episode {self.current_episode} | "
                         f"t={self.current_timestep}/{self.timesteps} | "
                         f"reward={self.episode_reward:.1f} | best={self.best_reward:.1f}")


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
    long_profile = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'Longitudinal', 'longitudinal.shp')
    
    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'behavioral_training_visual',
        'crs': 'EPSG:26904',
        'basin': 'Nushagak River',
        'water_temp': 10.0,
        'start_polygon': start_poly,
        'longitudinal_profile': long_profile,
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
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    parser.add_argument('--window-size', type=str, default='1920,1080', help='Window size (width,height)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RL BEHAVIORAL WEIGHT TRAINING (Visual Mode)')
    print('='*80)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Timesteps per episode: {args.timesteps}')
    print(f'  Agents: {args.agents}')
    print(f'  Fish length: {args.fish_length} mm')
    print('='*80)
    
    # Parse window size
    width, height = map(int, args.window_size.split(','))
    
    # Setup simulation and trainer
    sim, trainer, hecras_plan = setup_training_simulation(args)
    
    # Skip background raster for now - just use bounds for visualization
    import h5py
    
    with h5py.File(hecras_plan, 'r') as hdf:
        pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
        # bounds format: [left, right, bottom, top]
        bounds = [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]
    
    background_extent = bounds
    depth_raster = None  # No background texture
    
    # Setup PID controller with fixed parameters (no interpolation needed)
    from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    
    # Store simulation context in class attributes (required by FishSimViewer)
    RLTrainingViewer.sim_context = {
        'sim': sim,
        'timesteps': args.timesteps,
        'pid': pid,
        'background_extent': background_extent,
        'depth_raster': depth_raster,
        'start_paused': True  # Start paused for RL training
    }
    
    # Store RL training context
    RLTrainingViewer.training_context = {
        'trainer': trainer
    }
    
    # Launch OpenGL viewer with RL training
    print('\nLaunching OpenGL viewer with RL training...')
    print('Controls:')
    print('  S = start/resume | P = pause | SPACE = toggle pause')
    print('  R = restart | Mouse wheel = zoom | ESC = quit\n')
    
    mglw.run_window_config(
        RLTrainingViewer,
        args=(
            '--window', 'pygame2',
            '--size', f'{width}x{height}',
        )
    )
    
    print('\nTraining complete!')


if __name__ == '__main__':
    main()
