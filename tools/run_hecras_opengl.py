"""Real-time OpenGL viewer for HECRAS fish migration simulation.

Uses moderngl for efficient GPU rendering of thousands of agents with raster background.

Usage:
    python tools/run_hecras_opengl.py --timesteps 500 --agents 1000
    
Requirements:
    pip install moderngl moderngl-window pillow
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from PIL import Image
import rasterio

from src.emergent.salmon_abm.sockeye_SoA_OpenGL import simulation
from src.emergent.salmon_abm.sockeye_dynamic_environment import HECRAS


class FishSimViewer(mglw.WindowConfig):
    """OpenGL window for real-time fish simulation visualization."""
    
    gl_version = (3, 3)
    title = "Fish Migration - HECRAS Simulation"
    window_size = (1920, 1080)
    aspect_ratio = 16 / 9
    resizable = True
    sim_context = None  # Will be set before running
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get simulation context from class attribute
        ctx = self.sim_context or {}
        self.sim = ctx.get('sim')
        self.timesteps = ctx.get('timesteps', 500)
        self.pid = ctx.get('pid')
        self.current_timestep = 0
        self.paused = False
        self.background_extent = ctx.get('background_extent')
        self.num_agents = getattr(self.sim, 'num_agents', 0) if self.sim else 0
        self.depth_raster = ctx.get('depth_raster')
        
        # Setup shaders and geometry
        self._setup_background()
        self._setup_agents()
        
        # Load background texture
        if self.depth_raster:
            print(f'Loading background texture from {self.depth_raster}')
            if self.load_background_texture(self.depth_raster):
                print('Background texture loaded successfully')
            else:
                print('Failed to load background texture')
        
        # Initial agent update
        self.update_agents()
        print(f'Initial agent count: {self.num_agents}')
        
        # Camera/view setup
        self._setup_camera()
        
        self.fps_counter = 0
        self.fps_timer = 0
        self.last_timestep_time = 0
        self.hung_warning_shown = False
        
    def _setup_camera(self):
        """Setup orthographic projection for 2D view."""
        if self.background_extent:
            left, right, bottom, top = self.background_extent
        else:
            # Default extent based on agent positions
            left, right = 549000, 551000
            bottom, top = 6641000, 6642000
            
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        self._update_projection()
        
    def _update_projection(self):
        """Update projection matrix based on current view."""
        # Calculate view bounds with zoom and pan
        width = (self.right - self.left) / self.zoom
        height = (self.top - self.bottom) / self.zoom
        
        center_x = (self.left + self.right) / 2 + self.pan_x
        center_y = (self.bottom + self.top) / 2 + self.pan_y
        
        left = center_x - width / 2
        right = center_x + width / 2
        bottom = center_y - height / 2
        top = center_y + height / 2
        
        # Standard orthographic projection formula
        # This creates the matrix to transform world coordinates to clip space [-1, 1]
        rl = right - left
        tb = top - bottom
        
        # Column-major matrix for OpenGL
        self.projection = np.array([
            [2.0/rl, 0, 0, 0],
            [0, 2.0/tb, 0, 0],
            [0, 0, -1, 0],
            [-(right+left)/rl, -(top+bottom)/tb, 0, 1]
        ], dtype='f4')
        
    def _setup_background(self):
        """Setup background raster texture and shader."""
        # Vertex shader for background quad
        vertex_shader = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 uv;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        
        # Fragment shader for background
        fragment_shader = """
        #version 330
        in vec2 uv;
        out vec4 fragColor;
        uniform sampler2D background;
        uniform float alpha;
        
        void main() {
            vec4 color = texture(background, uv);
            fragColor = vec4(color.rgb, alpha);
        }
        """
        
        self.bg_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create background quad
        if self.background_extent:
            left, right, bottom, top = self.background_extent
            vertices = np.array([
                [left, bottom, 0, 1],
                [right, bottom, 1, 1],
                [right, top, 1, 0],
                [left, bottom, 0, 1],
                [right, top, 1, 0],
                [left, top, 0, 0],
            ], dtype='f4')
            
            print(f'Background quad vertices: left={left:.1f}, right={right:.1f}, bottom={bottom:.1f}, top={top:.1f}')
            print(f'First vertex: {vertices[0]}, Last vertex: {vertices[-1]}')
            
            self.bg_vbo = self.ctx.buffer(vertices.tobytes())
            self.bg_vao = self.ctx.vertex_array(
                self.bg_program,
                [(self.bg_vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )
        else:
            self.bg_vao = None
            
    def _setup_agents(self):
        """Setup agent rendering with point sprites."""
        # Vertex shader for agents
        vertex_shader = """
        #version 330
        in vec2 in_position;
        uniform mat4 projection;
        uniform float point_size;
        
        void main() {
            gl_Position = projection * vec4(in_position, 0.0, 1.0);
            gl_PointSize = point_size;
        }
        """
        
        # Fragment shader for agents (bright cyan dots)
        fragment_shader = """
        #version 330
        out vec4 fragColor;
        
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord) * 2.0;
            
            if (dist > 1.0) discard;
            
            // Bright cyan fill
            fragColor = vec4(0.0, 1.0, 1.0, 1.0);
        }
        """
        
        self.agent_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create empty VBO (will be updated each frame)
        max_agents = 10000
        self.agent_vbo = self.ctx.buffer(reserve=max_agents * 8)  # 2 floats per agent
        self.agent_vao = self.ctx.vertex_array(
            self.agent_program,
            [(self.agent_vbo, '2f', 'in_position')]
        )
        
    def load_background_texture(self, raster_path):
        """Load background raster as OpenGL texture."""
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                
                # Normalize to 0-255
                vmin, vmax = np.nanpercentile(data, [2, 98])
                data = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype('uint8')
                
                # Apply colormap (viridis-like)
                cmap = self._get_viridis_colormap()
                rgb = cmap[data]
                
                # Don't flip - raster is already in correct orientation
                # OpenGL texture coords will handle the coordinate system
                
                # Create texture
                self.bg_texture = self.ctx.texture(rgb.shape[:2][::-1], 3, rgb.tobytes())
                self.bg_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                
                return True
        except Exception as e:
            print(f'Failed to load background texture: {e}')
            return False
            
    def _get_viridis_colormap(self):
        """Generate viridis-like colormap."""
        cmap = np.zeros((256, 3), dtype='uint8')
        t = np.linspace(0, 1, 256)
        cmap[:, 0] = np.clip(255 * (0.267 + 0.095*t - 0.024*t**2 + 0.554*t**3), 0, 255)
        cmap[:, 1] = np.clip(255 * (0.005 + 0.380*t + 0.800*t**2 - 0.185*t**3), 0, 255)
        cmap[:, 2] = np.clip(255 * (0.329 + 0.685*t - 0.014*t**2), 0, 255)
        return cmap
        
    def update_agents(self):
        """Update agent positions from simulation."""
        if hasattr(self.sim, 'X') and hasattr(self.sim, 'Y'):
            positions = np.column_stack([self.sim.X.flatten(), self.sim.Y.flatten()])
            
            # Filter out NaN and dead agents
            mask = np.isfinite(positions).all(axis=1)
            if hasattr(self.sim, 'dead'):
                mask &= (self.sim.dead.flatten() == 0)
            positions = positions[mask].astype('f4')
            
            if self.current_timestep <= 1:
                print(f'update_agents: total={len(self.sim.X)}, valid={len(positions)}, first 3 positions: {positions[:3] if len(positions) > 0 else "none"}')
            
            if len(positions) > 0:
                self.agent_vbo.write(positions.tobytes())
                self.num_agents = len(positions)
            else:
                self.num_agents = 0
        else:
            self.num_agents = 0
            
    def on_render(self, time, frametime):
        """Render frame (required by moderngl-window)."""
        self.ctx.clear(0.1, 0.1, 0.15)  # Dark background
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        # Check if simulation appears to be hung
        if hasattr(self, 'last_timestep_time') and self.last_timestep_time > 0 and not self.paused:
            time_since_last = time - self.last_timestep_time
            if time_since_last > 30.0 and not self.hung_warning_shown:
                print(f'WARNING: No simulation progress for {time_since_last:.1f}s - may be hung at timestep {self.current_timestep}', flush=True)
                self.hung_warning_shown = True
        
        # Coordinate check on first frame only
        if self.current_timestep == 0 and self.sim:
            agent_x_range = (self.sim.X.min(), self.sim.X.max())
            agent_y_range = (self.sim.Y.min(), self.sim.Y.max())
            bg_x_range = (self.background_extent[0], self.background_extent[1])
            bg_y_range = (self.background_extent[2], self.background_extent[3])
            agents_in_bounds = (agent_x_range[0] >= bg_x_range[0] and agent_x_range[1] <= bg_x_range[1] and
                                agent_y_range[0] >= bg_y_range[0] and agent_y_range[1] <= bg_y_range[1])
            if not agents_in_bounds:
                print(f'WARNING: Agents [{agent_x_range[0]:.0f}-{agent_x_range[1]:.0f}, {agent_y_range[0]:.0f}-{agent_y_range[1]:.0f}] outside background [{bg_x_range[0]:.0f}-{bg_x_range[1]:.0f}, {bg_y_range[0]:.0f}-{bg_y_range[1]:.0f}]')
        
        # Render background
        if self.bg_vao and hasattr(self, 'bg_texture'):
            try:
                self.bg_program['projection'].write(self.projection.tobytes())
                self.bg_program['alpha'].value = 0.7  # Semi-transparent so we can see agents
                self.bg_texture.use(0)
                self.bg_program['background'].value = 0
                self.bg_vao.render()
            except Exception as e:
                print(f'Background render error: {e}')
                import traceback
                traceback.print_exc()
            
        # Update simulation
        if not self.paused and self.current_timestep < self.timesteps:
            if self.sim is None:
                print('ERROR: simulation object is None!')
                self.paused = True
                return
                
            try:
                import time
                import sys
                t_start = time.time()
                
                # Progress logging every 10 timesteps
                if self.current_timestep % 10 == 0:
                    print(f'Timestep {self.current_timestep}...', flush=True)
                    sys.stdout.flush()
                
                # Run simulation timestep
                self.sim.timestep(self.current_timestep, 1.0, 9.81, self.pid)
                t_sim = time.time() - t_start
                self.last_timestep_time = time.time()
                self.hung_warning_shown = False
                
                # Warn if timestep is taking too long
                if t_sim > 5.0:
                    print(f'WARNING: t={self.current_timestep} took {t_sim:.1f}s (very slow!)', flush=True)
                elif self.current_timestep < 3 or self.current_timestep % 10 == 0:
                    print(f't={self.current_timestep}: completed in {t_sim*1000:.1f}ms', flush=True)
                
                self.current_timestep += 1
                
                # Check for invalid agent positions before updating
                if hasattr(self.sim, 'X') and hasattr(self.sim, 'Y'):
                    invalid_x = (~np.isfinite(self.sim.X)).sum()
                    invalid_y = (~np.isfinite(self.sim.Y)).sum()
                    if invalid_x > 0 or invalid_y > 0:
                        print(f'ERROR t={self.current_timestep}: {invalid_x} agents with invalid X, {invalid_y} with invalid Y')
                        self.paused = True
                        return
                
                self.update_agents()
                
                if self.current_timestep <= 3:
                    # Check how many are in water
                    if hasattr(self.sim, 'depth'):
                        in_water = (self.sim.depth > 0.1).sum()
                        print(f'  agents in water: {in_water}/{len(self.sim.depth)}')
            except Exception as e:
                print('='*80)
                print(f'!!! SIMULATION ERROR at t={self.current_timestep}: {e}')
                print('='*80)
                import traceback
                traceback.print_exc()
                print('='*80)
                print('PAUSING simulation due to error')
                self.paused = True
                
        # Render agents
        if self.num_agents > 0:
            try:
                self.agent_program['projection'].write(self.projection.tobytes())
                self.agent_program['point_size'].value = 3.0
                self.agent_vao.render(moderngl.POINTS, vertices=self.num_agents)
            except Exception as e:
                print(f'Agent render error: {e}')
                import traceback
                traceback.print_exc()
            
        # FPS counter
        self.fps_counter += 1
        self.fps_timer += frametime
        if self.fps_timer > 1.0:
            fps = self.fps_counter / self.fps_timer
            self.wnd.title = f"Fish Migration - t={self.current_timestep}/{self.timesteps} agents={self.num_agents} FPS={fps:.1f}"
            self.fps_counter = 0
            self.fps_timer = 0
            
    def key_event(self, key, action, modifiers):
        """Handle keyboard events."""
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.paused = not self.paused
            elif key == self.wnd.keys.R:
                self.current_timestep = 0
            elif key == self.wnd.keys.Q or key == self.wnd.keys.ESCAPE:
                self.wnd.close()
                
    def mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll for zoom."""
        zoom_factor = 1.1 if y_offset > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = np.clip(self.zoom, 0.1, 10.0)
        self._update_projection()
        
    def mouse_drag_event(self, x, y, dx, dy):
        """Handle mouse drag for panning."""
        width = (self.right - self.left) / self.zoom
        height = (self.top - self.bottom) / self.zoom
        self.pan_x -= dx / self.wnd.width * width
        self.pan_y += dy / self.wnd.height * height
        self._update_projection()


def main():
    parser = argparse.ArgumentParser(description='OpenGL HECRAS fish simulation viewer')
    parser.add_argument('--timesteps', '-t', type=int, default=500)
    parser.add_argument('--agents', '-a', type=int, default=1000)
    parser.add_argument('--hecras-plan', '-p', type=str, default=None)
    parser.add_argument('--fast-raster', action='store_true', default=True, help='Use fast KDTree rasterization (default)')
    parser.add_argument('--fish-length', type=int, default=500, help='Fish length in mm')
    args = parser.parse_args()
    
    # Auto-discover HECRAS plan
    hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
    if args.hecras_plan is None:
        for f in os.listdir(hecras_folder):
            if f.endswith('.p05.hdf'):
                args.hecras_plan = os.path.join(hecras_folder, f)
                break
                
    if not args.hecras_plan or not os.path.exists(args.hecras_plan):
        print('HECRAS plan not found')
        return 1
        
    print(f'Using HECRAS plan: {args.hecras_plan}')
    
    # Setup HECRAS rasters
    out_dir = os.path.join(REPO_ROOT, 'outputs', 'hecras_run')
    os.makedirs(out_dir, exist_ok=True)
    
    print('Preparing HECRAS rasters...')
    if args.fast_raster:
        # Fast KDTree rasterization: read HDF datasets and rasterize all fields using nearest-neighbor.
        import h5py
        import rasterio
        from rasterio.transform import Affine
        from scipy.spatial import cKDTree

        print('Running fast KDTree rasterization from HECRAS HDF')
        hdf = h5py.File(os.path.splitext(args.hecras_plan)[0] + '.hdf', 'r')
        pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
        wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1]
        elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
        # Also extract velocities for behavioral cues
        try:
            vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
            vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
        except Exception:
            vel_x = None
            vel_y = None
        hdf.close()

        # build grid extents
        resolution = 1.0
        xmin = np.min(pts[:,0]); xmax = np.max(pts[:,0])
        ymin = np.min(pts[:,1]); ymax = np.max(pts[:,1])
        xint = np.arange(xmin, xmax, resolution)
        yint = np.arange(ymax, ymin, resolution * -1.)
        xnew, ynew = np.meshgrid(xint, yint)

        # Simple nearest-neighbor rasterization via KDTree lookup (fast)
        tree = cKDTree(pts)
        qx = np.column_stack((xnew.ravel(), ynew.ravel()))
        dists, inds = tree.query(qx, k=1)
        elev_rast = np.asarray(elev)[inds].reshape((len(yint), len(xint)))
        wsel_rast = np.asarray(wsel)[inds].reshape((len(yint), len(xint)))
        depth_rast = wsel_rast - elev_rast
        
        # Create wetted raster: 1 where depth > 0, 0 elsewhere
        wetted_rast = (depth_rast > 0.0).astype(np.float64)
        
        # Rasterize velocities if available
        if vel_x is not None and vel_y is not None:
            vel_x_rast = np.asarray(vel_x)[inds].reshape((len(yint), len(xint)))
            vel_y_rast = np.asarray(vel_y)[inds].reshape((len(yint), len(xint)))
            # Calculate velocity direction (radians) and magnitude
            vel_dir_rast = np.arctan2(vel_y_rast, vel_x_rast)
            vel_mag_rast = np.sqrt(vel_x_rast**2 + vel_y_rast**2)
        else:
            vel_x_rast = np.zeros_like(depth_rast)
            vel_y_rast = np.zeros_like(depth_rast)
            vel_dir_rast = np.zeros_like(depth_rast)
            vel_mag_rast = np.zeros_like(depth_rast)

        transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution) * Affine.scale(resolution, -1 * resolution)
        # write rasters
        def _write(name, arr):
            outp = os.path.join(out_dir, name)
            with rasterio.open(outp, mode='w', driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1, dtype='float64', crs='EPSG:26904', transform=transform) as dst:
                dst.write(arr, 1)
        _write('elev.tif', elev_rast)
        _write('depth.tif', depth_rast)
        _write('wetted.tif', wetted_rast)
        _write('x_vel.tif', vel_x_rast)
        _write('y_vel.tif', vel_y_rast)
        _write('vel_dir.tif', vel_dir_rast)
        _write('vel_mag.tif', vel_mag_rast)
        env_files = {
            'elev': os.path.join(out_dir, 'elev.tif'),
            'depth': os.path.join(out_dir, 'depth.tif'),
            'wetted': os.path.join(out_dir, 'wetted.tif'),
            'x_vel': os.path.join(out_dir, 'x_vel.tif'),
            'y_vel': os.path.join(out_dir, 'y_vel.tif'),
            'vel_dir': os.path.join(out_dir, 'vel_dir.tif'),
            'vel_mag': os.path.join(out_dir, 'vel_mag.tif')
        }
        print('Fast KDTree rasterization complete; rasters written to', out_dir)
    else:
        env_files = HECRAS.prepare_hecras_rasters(args.hecras_plan, out_dir, resolution=1.0)
        
    print(f'Rasters ready in {out_dir}')
    
    # Setup simulation
    start_poly = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location', 'start_loc_river_right.shp')
    
    config = {
        'model_dir': REPO_ROOT,
        'model_name': 'hecras_run',
        'crs': 'EPSG:26904',
        'basin': 'Nushagak River',
        'water_temp': 10.0,
        'start_polygon': start_poly,
        'env_files': env_files,
        'longitudinal_profile': None,
        'fish_length': args.fish_length,
        'num_timesteps': args.timesteps,
        'num_agents': args.agents,
        'use_gpu': False,
        'defer_hdf': False,
    }
    
    print(f'Initializing simulation with {args.agents} agents...')
    sim = simulation(**config)
    
    # Enable HECRAS node-based mapping
    import h5py
    print('Loading HECRAS nodes for node-based mapping (fast)')
    hdf = h5py.File(os.path.splitext(args.hecras_plan)[0] + '.hdf', 'r')
    try:
        pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    except Exception:
        pts = None
    node_fields = {}
    # depth and vel components if present
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
        # Ensure simulation has required sampled attributes expected by enable_hecras
        n = sim.num_agents
        if not hasattr(sim, 'depth'):
            sim.depth = np.zeros(n, dtype=float)
        if not hasattr(sim, 'x_vel'):
            sim.x_vel = np.zeros(n, dtype=float)
        if not hasattr(sim, 'y_vel'):
            sim.y_vel = np.zeros(n, dtype=float)
        if not hasattr(sim, 'vel_mag'):
            sim.vel_mag = np.zeros(n, dtype=float)
        if not hasattr(sim, 'wet'):
            sim.wet = np.ones(n, dtype=float)
        if not hasattr(sim, 'distance_to'):
            sim.distance_to = np.zeros(n, dtype=float)

        print('Enabling HECRAS node mapping on simulation (nearest neighbor k=1)')
        sim.enable_hecras(pts, node_fields, k=1)
        
        # Perform initial sampling of HECRAS values at agent positions
        if 'depth' in node_fields:
            sim.depth = sim.apply_hecras_mapping(node_fields['depth'])
        if 'vel_x' in node_fields:
            sim.x_vel = sim.apply_hecras_mapping(node_fields['vel_x'])
        if 'vel_y' in node_fields:
            sim.y_vel = sim.apply_hecras_mapping(node_fields['vel_y'])
            sim.vel_mag = np.sqrt(sim.x_vel**2 + sim.y_vel**2)
        
        # Re-initialize heading now that we have velocities from HECRAS
        flow_direction = np.arctan2(sim.y_vel, sim.x_vel)  # radians
        sim.heading = flow_direction - np.pi  # Point upstream
        sim.max_practical_sog = np.array([sim.sog * np.cos(sim.heading), 
                                           sim.sog * np.sin(sim.heading)])
        
        print('Node-based HECRAS mapping enabled; simulation will sample HECRAS per-agent directly.')
        print(f'Initial agent positions: X range [{sim.X.min():.1f}, {sim.X.max():.1f}], Y range [{sim.Y.min():.1f}, {sim.Y.max():.1f}]')
        print(f'Initial velocities: x_vel range [{sim.x_vel.min():.3f}, {sim.x_vel.max():.3f}], y_vel range [{sim.y_vel.min():.3f}, {sim.y_vel.max():.3f}]')
        print(f'Initial depth range: [{sim.depth.min():.3f}, {sim.depth.max():.3f}]')
        print(f'Initial heading range: [{np.degrees(sim.heading).min():.1f}°, {np.degrees(sim.heading).max():.1f}°]')
    
    # Setup PID controller
    try:
        from src.emergent.salmon_abm.sockeye_SoA import PID_controller
        pid = PID_controller(args.agents, k_p=1.0, k_i=0.0, k_d=0.0)
        try:
            pid.interp_PID()
        except Exception:
            pass
    except Exception:
        pid = None
    
    # Get background extent from depth raster
    depth_raster = env_files.get('depth')
    background_extent = None
    if depth_raster and os.path.exists(depth_raster):
        with rasterio.open(depth_raster) as src:
            bounds = src.bounds
            background_extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
            
    print('Starting OpenGL viewer...')
    print('Controls: SPACE=pause/resume, R=restart, Mouse wheel=zoom, Mouse drag=pan, ESC=quit')
    
    # Store simulation context in class attributes for the viewer
    FishSimViewer.sim_context = {
        'sim': sim,
        'timesteps': args.timesteps,
        'pid': pid,
        'background_extent': background_extent,
        'depth_raster': depth_raster
    }
    
    # Run OpenGL window
    mglw.run_window_config(
        FishSimViewer,
        args=(
            '--window', 'pygame2',
            '--size', '1920x1080',
        )
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
