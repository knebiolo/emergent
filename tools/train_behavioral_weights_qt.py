#!/usr/bin/env python3
"""
RL training with Qt/PyQtGraph visualization (matching ship_abm architecture).

Usage:
    python tools/train_behavioral_weights_qt.py --episodes 10 --timesteps 100 --agents 500
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights


class RLTrainingViewer(QtWidgets.QMainWindow):
    """Qt-based RL training viewer matching ship_abm architecture."""
    
    def __init__(self, sim, trainer, timesteps, pid, hecras_plan):
        super().__init__()
        self.sim = sim
        self.trainer = trainer
        self.timesteps = timesteps
        self.pid = pid
        self.hecras_plan = hecras_plan
        
        self.current_timestep = 0
        self.paused = True  # Start paused
        self.episode = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.best_reward = -np.inf
        
        self.setWindowTitle('RL Behavioral Weight Training')
        self.resize(1600, 900)
        
        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start')
        self.pause_btn = QtWidgets.QPushButton('Pause')
        self.reset_btn = QtWidgets.QPushButton('Reset')
        
        self.start_btn.clicked.connect(self.on_start)
        self.pause_btn.clicked.connect(self.on_pause)
        self.reset_btn.clicked.connect(self.on_reset)
        
        # Style buttons
        self.start_btn.setStyleSheet('background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;')
        self.pause_btn.setStyleSheet('background-color: #FF9800; color: white; font-weight: bold; padding: 8px;')
        self.reset_btn.setStyleSheet('background-color: #f44336; color: white; font-weight: bold; padding: 8px;')
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        
        # Status label
        self.status_label = QtWidgets.QLabel('Ready - Press Start')
        self.status_label.setStyleSheet('font-size: 14px; padding: 5px;')
        button_layout.addWidget(self.status_label)
        
        layout.addLayout(button_layout)
        
        # Graphics view
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)
        
        # Agent scatter plot
        self.agent_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 200))
        self.plot_widget.addItem(self.agent_scatter)
        
        # Agent direction arrows
        self.arrow_lines = []
        
        # Set view to agent extent
        import h5py
        with h5py.File(hecras_plan, 'r') as hdf:
            pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
            xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        
        self.plot_widget.setXRange(xmin, xmax)
        self.plot_widget.setYRange(ymin, ymax)
        # Create background raster from HECRAS HDF (translucent)
        try:
            self._create_background(hecras_plan, xmin, xmax, ymin, ymax)
        except Exception as e:
            print(f'Warning: could not create raster background: {e}')
        
        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(50)  # 20 FPS
        
        self.update_agents()

    def _create_background(self, hecras_plan, xmin, xmax, ymin, ymax):
        """Rasterize HECRAS cell values (depth) to a regular grid and display as ImageItem."""
        import h5py
        try:
            with h5py.File(hecras_plan, 'r') as hdf:
                pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
                # Try to read water surface and cell min elev to compute depth
                wsurf = None
                elev = None
                if 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface' in hdf:
                    wsurf = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'])
                if 'Geometry/2D Flow Areas/2D area/Cells Minimum Elevation' in hdf:
                    elev = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'])

                depth_vals = None
                if wsurf is not None and elev is not None:
                    # use last timestep water surface
                    depth_vals = wsurf[-1] - elev
                else:
                    # fallback: try common depth names
                    if 'Geometry/2D Flow Areas/2D area/Cell Hydraulic Depth' in hdf:
                        depth_vals = np.array(hdf['Geometry/2D Flow Areas/2D area/Cell Hydraulic Depth'])
                if depth_vals is None:
                    raise RuntimeError('No depth-related datasets found in HECRAS HDF')

            # Prefer smooth scattered-data interpolation using scipy.griddata to avoid bin artifacts
            nx, ny = 800, 600
            xs = pts[:, 0]
            ys = pts[:, 1]

            try:
                from scipy.interpolate import griddata
                # create grid coordinates (xi, yi) where xi varies in x, yi in y
                xi = np.linspace(xmin, xmax, nx)
                yi = np.linspace(ymin, ymax, ny)
                XI, YI = np.meshgrid(xi, yi)

                # Interpolate depth values to grid (linear yields piecewise-linear, similar to bilinear when sampled)
                grid_vals = griddata((xs, ys), depth_vals, (XI, YI), method='linear')

                # grid_vals shape is (ny, nx) with NaNs outside convex hull -> treat as no-data
                valid_grid = ~np.isnan(grid_vals)

                # Replace NaNs only for display normalization (don't fill them; keep mask)
                if np.any(valid_grid):
                    vmin = np.nanmin(grid_vals[valid_grid])
                    vmax = np.nanmax(grid_vals[valid_grid])
                else:
                    vmin, vmax = 0.0, 1.0

                if vmax - vmin < 1e-9:
                    norm = np.zeros_like(grid_vals, dtype=float)
                else:
                    norm = (grid_vals - vmin) / (vmax - vmin)
                    norm = np.clip(norm, 0.0, 1.0)

                # Create RGBA image: grayscale where deeper->darker; alpha 0 for no-data, 153 (~0.6) for data
                img_rgba = np.zeros((grid_vals.shape[0], grid_vals.shape[1], 4), dtype=np.uint8)
                gray = (255 * (1.0 - np.nan_to_num(norm, nan=0.0))).astype(np.uint8)
                img_rgba[..., 0] = gray
                img_rgba[..., 1] = gray
                img_rgba[..., 2] = gray
                img_rgba[..., 3] = (valid_grid * 153).astype(np.uint8)

                # The grid (XI,YI) has yi increasing; ImageItem expects image rows from top to bottom, so flip vertically
                img_rgba = np.flipud(img_rgba)

                img_item = pg.ImageItem(img_rgba)
                try:
                    img_item.setOpts(interpolate=True)
                except Exception:
                    try:
                        img_item.setInterpolation(True)
                    except Exception:
                        pass

                rect = QtCore.QRectF(float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin))
                img_item.setRect(rect)
                self.plot_widget.addItem(img_item)
                self._background_item = img_item
                print('Background raster created from HECRAS plan (griddata, interpolated, transparent no-data)')

            except Exception:
                # If SciPy not available, fall back to histogram2d approach
                nx_fallback, ny_fallback = 600, 400
                H, xedges, yedges = np.histogram2d(xs, ys, bins=[nx_fallback, ny_fallback], range=[[xmin, xmax], [ymin, ymax]], weights=depth_vals)
                C, _, _ = np.histogram2d(xs, ys, bins=[nx_fallback, ny_fallback], range=[[xmin, xmax], [ymin, ymax]])
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean_grid = H / C
                valid_mask = (C > 0)
                grid = np.flipud(mean_grid.T)
                valid_grid = np.flipud(valid_mask.T)
                if np.any(valid_grid):
                    vmin = np.nanmin(grid[valid_grid])
                    vmax = np.nanmax(grid[valid_grid])
                else:
                    vmin, vmax = 0.0, 1.0
                if vmax - vmin < 1e-9:
                    norm = np.zeros_like(grid, dtype=float)
                else:
                    norm = (grid - vmin) / (vmax - vmin)
                    norm = np.clip(norm, 0.0, 1.0)
                img_rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.uint8)
                gray = (255 * (1.0 - norm)).astype(np.uint8)
                img_rgba[..., 0] = gray
                img_rgba[..., 1] = gray
                img_rgba[..., 2] = gray
                img_rgba[..., 3] = (valid_grid * 153).astype(np.uint8)
                img_item = pg.ImageItem(img_rgba)
                try:
                    img_item.setOpts(interpolate=True)
                except Exception:
                    try:
                        img_item.setInterpolation(True)
                    except Exception:
                        pass
                rect = QtCore.QRectF(float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin))
                img_item.setRect(rect)
                self.plot_widget.addItem(img_item)
                self._background_item = img_item
                print('Background raster created from HECRAS plan (histogram2d fallback)')
        except Exception:
            raise
        
    def on_start(self):
        # Ensure simulation spatial state is reset at the start of an episode
        if self.current_timestep == 0:
            try:
                self.sim.reset_spatial_state()
                print('Reset spatial state before starting episode')
            except Exception:
                pass

            # Cluster agents into a tight group at the start so training begins from a school
            try:
                n = getattr(self.sim, 'num_agents', getattr(self.sim, 'X').size)
                cx = float(np.nanmean(self.sim.X))
                cy = float(np.nanmean(self.sim.Y))
                # Small Gaussian cluster: sigma = 1.5 meters
                sigma = 1.5
                self.sim.X = np.random.normal(loc=cx, scale=sigma, size=n)
                self.sim.Y = np.random.normal(loc=cy, scale=sigma, size=n)
                # Reset headings consistent with velocity if possible
                if hasattr(self.sim, 'x_vel') and hasattr(self.sim, 'y_vel'):
                    self.sim.heading = np.arctan2(self.sim.y_vel, self.sim.x_vel)
                print('Clustered agents into starting school')
            except Exception as e:
                print(f'Could not cluster agents at start: {e}')

        self.paused = False
        self.timer.start()
        self.status_label.setText(f'Running - Episode {self.episode} | t={self.current_timestep}/{self.timesteps}')
        print('Started/Resumed')
        
    def on_pause(self):
        self.paused = True
        self.timer.stop()
        self.status_label.setText(f'Paused - Episode {self.episode} | t={self.current_timestep}/{self.timesteps}')
        print('Paused')
        
    def on_reset(self):
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.update_agents()
        self.status_label.setText(f'Reset - Episode {self.episode} | t=0/{self.timesteps}')
        print('Reset to timestep 0')
        
    def update_agents(self):
        """Update agent positions and directions."""
        if not hasattr(self.sim, 'X') or not hasattr(self.sim, 'Y'):
            return
            
        positions = np.column_stack([self.sim.X.flatten(), self.sim.Y.flatten()])
        headings = self.sim.heading.flatten()
        
        # Filter valid agents
        mask = np.isfinite(positions).all(axis=1) & np.isfinite(headings)
        if hasattr(self.sim, 'dead'):
            mask &= (self.sim.dead.flatten() == 0)
        
        positions = positions[mask]
        headings = headings[mask]
        
        if len(positions) == 0:
            return
            
        # Update scatter plot
        self.agent_scatter.setData(positions[:, 0], positions[:, 1])
        
        # Remove previous shaft/head plot items
        if hasattr(self, 'shaft_plot') and self.shaft_plot is not None:
            try:
                self.plot_widget.removeItem(self.shaft_plot)
            except Exception:
                pass
        if hasattr(self, 'head_plot') and self.head_plot is not None:
            try:
                self.plot_widget.removeItem(self.head_plot)
            except Exception:
                pass

        # Build line segments for shafts (behind agents) and small forward ticks for heading
        shaft_length = 3.0  # meters behind agent
        head_len = 0.6      # small forward tick length

        # Vectorized computations
        xs = positions[:, 0]
        ys = positions[:, 1]
        hs = headings

        # Shafts: from (x,y) to (x - shaft_length*cos(h), y - shaft_length*sin(h))
        x_shaft = np.empty(len(xs) * 3)
        y_shaft = np.empty(len(xs) * 3)
        for i in range(len(xs)):
            x_shaft[i*3 + 0] = xs[i]
            y_shaft[i*3 + 0] = ys[i]
            x_shaft[i*3 + 1] = xs[i] - shaft_length * np.cos(hs[i])
            y_shaft[i*3 + 1] = ys[i] - shaft_length * np.sin(hs[i])
            x_shaft[i*3 + 2] = np.nan
            y_shaft[i*3 + 2] = np.nan

        # Heads: small line forward in heading direction
        x_head = np.empty(len(xs) * 3)
        y_head = np.empty(len(xs) * 3)
        for i in range(len(xs)):
            x_head[i*3 + 0] = xs[i]
            y_head[i*3 + 0] = ys[i]
            x_head[i*3 + 1] = xs[i] + head_len * np.cos(hs[i])
            y_head[i*3 + 1] = ys[i] + head_len * np.sin(hs[i])
            x_head[i*3 + 2] = np.nan
            y_head[i*3 + 2] = np.nan

        self.shaft_plot = pg.PlotDataItem(x_shaft, y_shaft, pen=pg.mkPen(color=(0,200,200), width=2))
        self.head_plot = pg.PlotDataItem(x_head, y_head, pen=pg.mkPen(color=(0,255,255), width=2))
        self.plot_widget.addItem(self.shaft_plot)
        self.plot_widget.addItem(self.head_plot)
        
    def update_simulation(self):
        """Run one simulation timestep."""
        if self.paused or self.current_timestep >= self.timesteps:
            return
            
        try:
            # Run timestep
            self.sim.timestep(self.current_timestep, 1.0, 9.81, self.pid)
            
            # RL metrics
            current_metrics = self.trainer.extract_state_metrics()
            if self.prev_metrics is not None:
                reward = self.trainer.compute_reward(self.prev_metrics, current_metrics)
                self.episode_reward += reward
            self.prev_metrics = current_metrics
            
            self.current_timestep += 1
            
            # Update display
            self.update_agents()
            self.status_label.setText(f'Running - Episode {self.episode} | t={self.current_timestep}/{self.timesteps} | Reward: {self.episode_reward:.1f}')
            
            # Episode complete
            if self.current_timestep >= self.timesteps:
                self.on_episode_complete()
                
        except Exception as e:
            print(f'Simulation error at t={self.current_timestep}: {e}')
            import traceback
            traceback.print_exc()
            self.on_pause()
            
    def on_episode_complete(self):
        """Handle episode completion."""
        print(f'\\n{"="*80}')
        print(f'EPISODE {self.episode} COMPLETE | Total Reward: {self.episode_reward:.2f}')
        
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            print(f'🏆 NEW BEST REWARD: {self.best_reward:.2f}')
            
            # Save best weights
            save_path = os.path.join(REPO_ROOT, 'outputs', 'rl_training', 'best_weights.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.trainer.behavioral_weights.to_dict(), f, indent=2)
            print(f'Saved best weights to {save_path}')
        
        print(f'{"="*80}\\n')
        
        # Prepare next episode
        self.episode += 1
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        
        # Mutate weights using trainer API and apply to simulation
        try:
            self.trainer.behavioral_weights.mutate(scale=0.05)
            self.sim.apply_behavioral_weights(self.trainer.behavioral_weights)
        except Exception as e:
            print(f'Warning: Could not mutate/apply behavioral weights: {e}')
        
        # Reset simulation spatial state
        try:
            self.sim.reset_spatial_state()
            print(f'Mutated weights for episode {self.episode}: cohesion={self.trainer.behavioral_weights.cohesion:.2f}, separation={self.trainer.behavioral_weights.separation:.2f}')
        except Exception as e:
            print(f'Warning: Could not reset spatial state: {e}')
        
        self.status_label.setText(f'Episode {self.episode - 1} complete! Reward: {self.episode_reward:.1f} - Press Start for Episode {self.episode}')
        self.on_pause()


def setup_training_simulation(args):
    """Initialize simulation and RL trainer."""
    
    # HECRAS plan path
    hecras_plan = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    
    if not os.path.exists(hecras_plan):
        raise FileNotFoundError(f"HECRAS plan not found: {hecras_plan}")
    
    print(f'Using HECRAS plan: {hecras_plan}')
    print(f'Initializing training simulation with {args.agents} agents...')
    
    # Simulation configuration
    # Auto-detect start polygon file with prefix 'start_loc_river_right' in starting_location
    start_dir = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location')
    start_polygon_path = None
    if os.path.exists(start_dir):
        for fname in os.listdir(start_dir):
            if fname.lower().startswith('start_loc_river_right') and fname.lower().endswith('.shp'):
                start_polygon_path = os.path.join(start_dir, fname)
                break
    # Fallback to legacy river_right.shp name
    if start_polygon_path is None:
        legacy = os.path.join(start_dir, 'river_right.shp')
        if os.path.exists(legacy):
            start_polygon_path = legacy

    if start_polygon_path is None:
        raise FileNotFoundError(f"Start polygon not found in {start_dir}. Expected file starting with 'start_loc_river_right' or 'river_right.shp'.")

    # Ensure longitudinal profile exists; synthesize if missing
    longitudinal_path = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'nuyakuk_centerline.shp')
    if not os.path.exists(longitudinal_path):
        try:
            import h5py
            import shapely.geometry as geom
            import fiona
            from fiona.crs import from_epsg

            with h5py.File(hecras_plan, 'r') as hdf:
                pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
                # sort points by x then y to make a crude centerline
                idx = np.argsort(pts[:, 0])
                line = geom.LineString(pts[idx, :2])

            synth_dir = os.path.join(REPO_ROOT, 'outputs', 'rl_training', 'synth')
            os.makedirs(synth_dir, exist_ok=True)
            synth_line_path = os.path.join(synth_dir, 'synth_centerline.shp')

            schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}
            with fiona.open(synth_line_path, 'w', driver='ESRI Shapefile', crs=from_epsg(32605), schema=schema) as dst:
                dst.write({'geometry': geom.mapping(line), 'properties': {'id': 1}})

            longitudinal_path = synth_line_path
            print(f'Warning: longitudinal profile missing; synthesized at {longitudinal_path}')
        except Exception as e:
            print(f'Error synthesizing longitudinal profile: {e}')
            raise

    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'rl_training_qt',
        'crs': 'EPSG:32605',
        'basin': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Bathymetry.shp'),
        'water_temp': 10.0,
        'start_polygon': start_polygon_path,
        'longitudinal_profile': longitudinal_path,
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
    
    # Create RL trainer and apply weights to simulation
    print('Setting up RL trainer...')
    trainer = RLTrainer(sim)
    # Apply trainer's behavioral weights to the simulation via simulation API
    sim.apply_behavioral_weights(trainer.behavioral_weights)
    print('Applied behavioral weights to simulation')
    
    # Setup PID controller
    from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    
    return sim, trainer, hecras_plan, pid


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights with Qt visualization')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=500, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RL BEHAVIORAL WEIGHT TRAINING (Qt Mode)')
    print('='*80)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Timesteps per episode: {args.timesteps}')
    print(f'  Agents: {args.agents}')
    print(f'  Fish length: {args.fish_length} mm')
    print('='*80)
    print()
    
    # Setup simulation
    sim, trainer, hecras_plan, pid = setup_training_simulation(args)
    
    # Launch Qt application
    app = QtWidgets.QApplication(sys.argv)
    viewer = RLTrainingViewer(sim, trainer, args.timesteps, pid, hecras_plan)
    viewer.show()
    
    print('\\nQt viewer launched!')
    print('Click the green "Start" button to begin training\\n')
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
