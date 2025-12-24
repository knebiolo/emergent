"""Behavior helpers extracted from sockeye.py.

This includes perception and social cue calculations.
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import UnivariateSpline

from emergent.salmon_abm.utils import geo_to_pixel, pixel_to_geo, standardize_shape, calculate_front_masks, determine_slices_from_vectors, determine_slices_from_headings
from emergent.salmon_abm import hdf5_io


class behavior():
    def __init__(self, dt, simulation_object):
        self.dt = dt
        self.simulation = simulation_object

    def already_been_here(self, weight, t):
        x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)

        mental_map_rows, mental_map_cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)

        buff = 10
        row_min = np.clip(mental_map_rows - buff, 0, None)
        # use hdf5_io to support both h5py.File and dict-like mocks
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        memory0 = hdf5_io.read_dataset(h5, 'memory/0', default=np.zeros((1, 1)))
        row_max = np.clip(mental_map_rows + buff + 1, None, memory0.shape[0])
        col_min = np.clip(mental_map_cols - buff, 0, None)
        col_max = np.clip(mental_map_cols + buff + 1, None, memory0.shape[1])

        repulsive_forces_per_agent = np.array([
            self._calculate_repulsive_force(agent_idx, rmin, rmax, cmin, cmax, weight, t)
            for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
        ])

        return repulsive_forces_per_agent

    def _calculate_repulsive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight, t):
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        mmap = hdf5_io.read_dataset(h5, f'memory/{agent_idx}', default=np.zeros((1, 1)))
        mmap_section = mmap[row_min:row_max, col_min:col_max]
        t_since = mmap_section - t
        multiplier = np.where((t_since > 10) & (t_since < 7200), 1 - (t_since - 5) / (7195), 0)

        delta_x = self.simulation.X[agent_idx] - np.arange(col_min, col_max)
        delta_y = self.simulation.Y[agent_idx] - np.arange(row_min, row_max)[:, np.newaxis]
        magnitudes = np.sqrt(delta_x**2 + delta_y**2)
        magnitudes[magnitudes == 0] = 0.000001

        unit_vector_x = delta_x / magnitudes
        unit_vector_y = delta_y / magnitudes
        x_force = ((weight * unit_vector_x) / magnitudes) * multiplier
        y_force = ((weight * unit_vector_y) / magnitudes) * multiplier

        total_x_force = np.nansum(x_force)
        total_y_force = np.nansum(y_force)

        return np.array([total_x_force, total_y_force])

    def find_nearest_refuge(self, weight):
        x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)
        refugia_map_rows, refugia_map_cols = geo_to_pixel(x, y, self.simulation.refugia_map_transform)
        buff = 50
        # use hdf5_io to support both h5py.File and dict-like mocks
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        refugia0 = hdf5_io.read_dataset(h5, 'refugia/0', default=np.zeros((1, 1)))
        row_min = np.clip(refugia_map_rows - buff, 0, None)
        row_max = np.clip(refugia_map_rows + buff + 1, None, refugia0.shape[0])
        col_min = np.clip(refugia_map_cols - buff, 0, None)
        col_max = np.clip(refugia_map_cols + buff + 1, None, refugia0.shape[1])

        attractive_forces_per_agent = np.array([
            self._calculate_attractive_force(agent_idx, rmin, rmax, cmin, cmax, weight)
            for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
        ])

        return attractive_forces_per_agent

    def _calculate_attractive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight):
        # ensure we have the hdf5-like object available (works with h5py.File or dict-like mocks)
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        refugia = hdf5_io.read_dataset(h5, f'refugia/{agent_idx}', default=np.zeros((1, 1)))
        refugia_section = refugia[row_min:row_max, col_min:col_max]
        refuge_mask = (refugia_section == 1)
        if np.any(refuge_mask):
            distances = distance_transform_edt(~refuge_mask)
            nearest_refuge_coords = np.unravel_index(np.argmin(distances), distances.shape)
            ref_xy = pixel_to_geo(self.simulation.refugia_map_transform, nearest_refuge_coords[0], nearest_refuge_coords[1])
            delta_x = ref_xy[0] - self.simulation.X
            delta_y = ref_xy[1] - self.simulation.Y
            magnitudes = np.sqrt(delta_x**2 + delta_y**2)
            magnitudes[magnitudes == 0] = 0.000001
            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes
            x_force = (weight * unit_vector_x)
            y_force = (weight * unit_vector_y)
            attract_x = np.nansum(x_force)
            attract_y = np.nansum(y_force)
            return np.array([attract_x, attract_y])
        else:
            return np.array([0, 0])

    def vel_cue(self, weight):
        length_numpy = self.simulation.length
        buff = 2
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)

        xmin = cols - buff
        xmax = cols + buff + 1
        ymin = rows - buff
        ymax = rows + buff + 1

        # sanitize numeric arrays before casting to int to avoid invalid-cast runtime warnings
        xmin = np.nan_to_num(xmin, nan=0, posinf=0, neginf=0).astype(np.int32)
        xmax = np.nan_to_num(xmax, nan=0, posinf=0, neginf=0).astype(np.int32)
        ymin = np.nan_to_num(ymin, nan=0, posinf=0, neginf=0).astype(np.int32)
        ymax = np.nan_to_num(ymax, nan=0, posinf=0, neginf=0).astype(np.int32)

        slices = [(agent, slice(y0, y1), slice(x0, x1))
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                   ymin.flatten(),
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten()
                                                   )
                  ]
        # read datasets via hdf5_io so this works with h5py.File or dict-like mocks
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        vel_ds = hdf5_io.read_dataset(h5, 'environment/vel_mag', default=np.zeros((1, 1)))
        x_coords_ds = hdf5_io.read_dataset(h5, 'x_coords', default=np.zeros((1, 1)))
        y_coords_ds = hdf5_io.read_dataset(h5, 'y_coords', default=np.zeros((1, 1)))

        vel3d = np.stack([standardize_shape(vel_ds[sl[-2:]]) for sl in slices])
        x_coords = np.stack([standardize_shape(x_coords_ds[sl[-2:]]) for sl in slices])
        y_coords = np.stack([standardize_shape(y_coords_ds[sl[-2:]]) for sl in slices])

        vel3d_multiplier = calculate_front_masks(self.simulation.heading.flatten(),
                                                 x_coords,
                                                 y_coords,
                                                 np.nan_to_num(self.simulation.X.flatten()),
                                                 np.nan_to_num(self.simulation.Y.flatten()),
                                                 behind_value=999.9)

        vel3d = vel3d * vel3d_multiplier

        num_agents, rows, cols = vel3d.shape
        vel3d = vel3d.reshape(num_agents, rows * cols)

        flat_indices = np.argmin(vel3d, axis=1)
        min_row_indices = flat_indices // cols
        min_col_indices = flat_indices % cols

        min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform,
                                    min_row_indices + ymin,
                                    min_col_indices + xmin)

        delta_x = min_x - self.simulation.X
        delta_y = min_y - self.simulation.Y
        dist = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
        dist_safe = np.where(dist == 0, 1e-6, dist)
        attract_x = weight * delta_x / dist_safe
        attract_y = weight * delta_y / dist_safe
        # where distance was zero, set attraction to zero to avoid NaNs
        attract_x = np.where(dist == 0, 0.0, attract_x)
        attract_y = np.where(dist == 0, 0.0, attract_y)
        return np.column_stack((attract_x, attract_y))

    def rheo_cue(self, weight, downstream=False):
        length_numpy = self.simulation.length
        if not downstream:
            x_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform, 'vel_x') * -1
            y_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform, 'vel_y') * -1
        else:
            x_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform, 'vel_x')
            y_vel = self.simulation.sample_environment(self.simulation.vel_dir_rast_transform, 'vel_y')

        v = np.column_stack([x_vel, y_vel])
        v_hat = v / np.linalg.norm(v, axis=-1)[:, np.newaxis]
        rheotaxis = np.zeros_like(v)
        rheotaxis = weight * v_hat
        return rheotaxis

    def border_cue(self, weight, t):
        length_numpy = self.simulation.length
        buff = 2
        x, y = (np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y))
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)

        xmin = cols - buff
        xmax = cols + buff + 1
        ymin = rows - buff
        ymax = rows + buff + 1

        # ensure indices within dataset bounds using hdf5_io
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        dist_ds = hdf5_io.read_dataset(h5, 'environment/distance_to', default=np.zeros((1, 1)))
        xmin = np.clip(xmin, 0, dist_ds.shape[1] - 1)
        xmax = np.clip(xmax, 0, dist_ds.shape[1])
        ymin = np.clip(ymin, 0, dist_ds.shape[0] - 1)
        ymax = np.clip(ymax, 0, dist_ds.shape[0])

        slices = [(agent, slice(y0, y1), slice(x0, x1))
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                   ymin.flatten(),
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten()
                                                   )
                  ]

        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        x_coords_ds = hdf5_io.read_dataset(h5, 'x_coords', default=np.zeros((1, 1)))
        y_coords_ds = hdf5_io.read_dataset(h5, 'y_coords', default=np.zeros((1, 1)))
        x_coords = np.stack([standardize_shape(x_coords_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])
        y_coords = np.stack([standardize_shape(y_coords_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])

        # calculate_front_masks expects 1D headings and per-agent (n,H,W) coords
        front_multiplier = calculate_front_masks(np.asarray(self.simulation.heading).flatten(),
                             x_coords,
                             y_coords,
                             np.nan_to_num(np.asarray(self.simulation.X).flatten()),
                             np.nan_to_num(np.asarray(self.simulation.Y).flatten()))

        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        dist_ds = hdf5_io.read_dataset(h5, 'environment/distance_to', default=np.zeros((1, 1)))
        dist3d = np.stack([standardize_shape(dist_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])
        # ensure front_multiplier can broadcast to dist3d shape
        try:
            dist3d = dist3d * front_multiplier
        except Exception:
            try:
                front_multiplier_b = np.broadcast_to(front_multiplier, dist3d.shape)
                dist3d = dist3d * front_multiplier_b
            except Exception:
                # fallback: ignore front mask if broadcasting fails
                pass

        num_agents, rows, cols = dist3d.shape
        dist3d = dist3d.reshape(num_agents, rows * cols)
        flat_indices = np.argmax(dist3d, axis=1)
        max_row_indices = flat_indices // cols
        max_col_indices = flat_indices % cols

        max_x, max_y = pixel_to_geo(self.simulation.vel_mag_rast_transform,
                                    max_row_indices + ymin,
                                    max_col_indices + xmin)

        delta_x = max_x - self.simulation.X
        delta_y = max_y - self.simulation.Y
        dist = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

        current_distances = self.simulation.sample_environment(self.simulation.depth_rast_transform, 'distance_to')
        self.simulation.current_distances = current_distances

        too_close = np.where(current_distances <= 1 * (self.simulation.length / 1000.), 1, 0)
        too_close = np.where(self.simulation.in_eddy == 1, 1, too_close)

        repulse_x = np.where(too_close, weight * delta_x / dist, np.zeros_like(delta_x))
        repulse_y = np.where(too_close, weight * delta_y / dist, np.zeros_like(delta_y))

        return np.column_stack((repulse_x, repulse_y))

    def shallow_cue(self, weight):
        buff = 2
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)

        xmin = cols - buff
        xmax = cols + buff + 1
        ymin = rows - buff
        ymax = rows + buff + 1

        xmin = xmin.astype(np.int32)
        xmax = xmax.astype(np.int32)
        ymin = ymin.astype(np.int32)
        ymax = ymax.astype(np.int32)

        repulsive_forces = np.zeros((self.simulation.num_agents, 2), dtype=float)
        min_depth = self.simulation.too_shallow

        slices = [(agent, slice(y0, y1), slice(x0, x1))
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                   ymin.flatten(),
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten())
                  ]

        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=np.zeros((1, 1)))
        x_coords_ds = hdf5_io.read_dataset(h5, 'x_coords', default=np.zeros((1, 1)))
        y_coords_ds = hdf5_io.read_dataset(h5, 'y_coords', default=np.zeros((1, 1)))
        depths = np.stack([standardize_shape(depth_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])
        x_coords = np.stack([standardize_shape(x_coords_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])
        y_coords = np.stack([standardize_shape(y_coords_ds[sl[-2:]], target_shape=(2 * buff + 1, 2 * buff + 1)) for sl in slices])

        front_multiplier = calculate_front_masks(self.simulation.heading, x_coords, y_coords, self.simulation.X, self.simulation.Y)

        depth_multiplier = np.where(depths < min_depth[:, np.newaxis, np.newaxis], 1, 0)

        delta_x = self.simulation.X[:, np.newaxis, np.newaxis] - x_coords
        delta_y = self.simulation.Y[:, np.newaxis, np.newaxis] - y_coords
        magnitudes = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
        magnitudes = np.where(magnitudes == 0, 0.000001, magnitudes)

        unit_vector_x = delta_x / magnitudes
        unit_vector_y = delta_y / magnitudes

        x_force = ((weight * unit_vector_x) / magnitudes) * depth_multiplier * front_multiplier
        y_force = ((weight * unit_vector_y) / magnitudes) * depth_multiplier * front_multiplier

        if self.simulation.num_agents > 1:
            total_x_force = np.nansum(x_force, axis=(1, 2))
            total_y_force = np.nansum(y_force, axis=(1, 2))
        else:
            total_x_force = np.nansum(x_force)
            total_y_force = np.nansum(y_force)

        repulsive_forces = np.array([total_x_force, total_y_force]).T
        return repulsive_forces

    def wave_drag_multiplier(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/wave_drag_huges_2004_fig3.csv')
        hughes = pd.read_csv(data_dir)
        hughes.sort_values(by='body_depths_submerged', ascending=True, inplace=True)
        wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged, hughes.wave_drag_multiplier, k=3, ext=0)
        body_depths = self.simulation.z / (self.simulation.body_depth / 100.)
        self.simulation.wave_drag = np.where(body_depths >= 3, 1, wave_drag_fun(body_depths))

    def wave_drag_cue(self, weight):
        buff = 2.0
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)

        xmin = cols - buff
        xmax = cols + buff + 1
        ymin = rows - buff
        ymax = rows + buff + 1

        xmin = xmin.astype(np.int32)
        xmax = xmax.astype(np.int32)
        ymin = ymin.astype(np.int32)
        ymax = ymax.astype(np.int32)

        slices = [(agent, slice(y0, y1), slice(x0, x1))
                  for agent, y0, y1, x0, x1 in zip(np.arange(self.simulation.num_agents),
                                                   ymin.flatten(),
                                                   ymax.flatten(),
                                                   xmin.flatten(),
                                                   xmax.flatten())
                  ]

        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=np.zeros((1, 1)))
        x_coords_ds = hdf5_io.read_dataset(h5, 'x_coords', default=np.zeros((1, 1)))
        y_coords_ds = hdf5_io.read_dataset(h5, 'y_coords', default=np.zeros((1, 1)))
        dep3D = np.stack([standardize_shape(depth_ds[sl[-2:]]) for sl in slices])
        x_coords = np.stack([standardize_shape(x_coords_ds[sl[-2:]]) for sl in slices])
        y_coords = np.stack([standardize_shape(y_coords_ds[sl[-2:]]) for sl in slices])

        dep3D_multiplier = calculate_front_masks(self.simulation.heading.flatten(), x_coords, y_coords, self.simulation.X.flatten(), self.simulation.Y.flatten(), behind_value=99999.9)
        dep3D = dep3D * dep3D_multiplier

        num_agents, rows, cols = dep3D.shape
        reshaped_dep3D = dep3D.reshape(num_agents, rows * cols)
        optimal_depth_diff = np.abs(reshaped_dep3D - self.simulation.opt_wat_depth[:, np.newaxis])
        flat_indices = np.argmin(optimal_depth_diff, axis=1)
        min_row_indices = flat_indices // cols
        min_col_indices = flat_indices % cols

        min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, min_row_indices + ymin, min_col_indices + xmin)
        delta_x = min_x - self.simulation.X
        delta_y = min_y - self.simulation.Y
        dist = np.sqrt(delta_x**2 + delta_y**2)
        dist_safe = np.where(dist == 0, 1e-6, dist)
        attract_x = weight * delta_x / dist_safe
        attract_y = weight * delta_y / dist_safe
        attract_x = np.where(dist == 0, 0.0, attract_x)
        attract_y = np.where(dist == 0, 0.0, attract_y)
        return np.column_stack((attract_x, attract_y))

    def cohesion_cue(self, weight, consider_front_only=False):
        num_agents = self.simulation.num_agents
        neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
        agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
        x_neighbors = self.simulation.X[neighbor_indices]
        y_neighbors = self.simulation.Y[neighbor_indices]
        vectors_to_neighbors_x = x_neighbors - self.simulation.X[agent_indices]
        vectors_to_neighbors_y = y_neighbors - self.simulation.Y[agent_indices]

        if consider_front_only:
            agent_velocities_x = self.simulation.x_vel[agent_indices]
            agent_velocities_y = self.simulation.y_vel[agent_indices]
            dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
            valid_neighbors_mask = dot_products > 0
        else:
            valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)

        valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
        valid_agent_indices = agent_indices[valid_neighbors_mask]

        center_x = np.zeros(num_agents)
        center_y = np.zeros(num_agents)
        np.add.at(center_x, valid_agent_indices, x_neighbors[valid_neighbors_mask])
        np.add.at(center_y, valid_agent_indices, y_neighbors[valid_neighbors_mask])
        counts = np.bincount(valid_agent_indices, minlength=num_agents)
        center_x /= counts + (counts == 0)
        center_y /= counts + (counts == 0)

        vectors_to_center_x = center_x - self.simulation.X
        vectors_to_center_y = center_y - self.simulation.Y
        distances_to_center = np.sqrt(vectors_to_center_x**2 + vectors_to_center_y**2)
        epsilon = 1e-10
        v_hat_center_x = np.divide(vectors_to_center_x, distances_to_center + epsilon, out=np.zeros_like(self.simulation.x_vel), where=distances_to_center+epsilon != 0)
        v_hat_center_y = np.divide(vectors_to_center_y, distances_to_center + epsilon, out=np.zeros_like(self.simulation.y_vel), where=distances_to_center+epsilon != 0)
        cohesion_array = np.zeros((num_agents, 2))
        cohesion_array[:, 0] = weight * v_hat_center_x
        cohesion_array[:, 1] = weight * v_hat_center_y
        return np.nan_to_num(cohesion_array)

    def alignment_cue(self, weight, consider_front_only=False):
        num_agents = self.simulation.num_agents
        neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
        agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
        # capture raw neighbor headings (may be all zeros at init)
        # Unconditional diagnostic prints to trace execution and exceptions
        try:
            print('DBG alignment_cue ENTER: num_agents=', num_agents)
            print('DBG agents_within_buffers lengths=', [len(x) for x in self.simulation.agents_within_buffers])
            print('DBG neighbor_indices sample=', neighbor_indices[:20])
            print('DBG sim.heading sample=', np.asarray(self.simulation.heading)[:20])
        except Exception as e:
            try:
                import traceback
                print('DBG alignment_cue initial prints failed:', e)
                traceback.print_exc()
            except Exception:
                pass
        try:
            raw_headings_neighbors = np.asarray(self.simulation.heading)[neighbor_indices]
        except Exception as e:
            try:
                import traceback
                print('DBG alignment_cue: reading raw headings failed:', e)
                traceback.print_exc()
            except Exception:
                pass
            raw_headings_neighbors = np.array([], dtype=float)
        headings_neighbors = raw_headings_neighbors.copy()
        # If headings are all zero (common at initialization), fall back to neighbor velocity directions
        used_velocity_heading = False
        try:
            if headings_neighbors.size > 0 and np.allclose(headings_neighbors, 0.0):
                # compute neighbor velocities' headings where available
                vx = np.asarray(self.simulation.x_vel)[neighbor_indices]
                vy = np.asarray(self.simulation.y_vel)[neighbor_indices]
                vel_mag = np.sqrt(vx**2 + vy**2)
                if np.any(vel_mag > 0):
                    headings_neighbors = np.arctan2(vy, vx)
                    used_velocity_heading = True
                    if getattr(self.simulation, 'debug_behavior', False):
                        try:
                            print('alignment_cue: used velocity fallback; sample headings_neighbors=', headings_neighbors[:20])
                        except Exception:
                            pass
        except Exception:
            pass
        # store diagnostics for NPZ writer to include
        try:
            print('DBG alignment_cue: about to set _alignment_diag; sizes raw/headings=', getattr(raw_headings_neighbors, 'size', None), getattr(headings_neighbors, 'size', None))
            self.simulation._alignment_diag = {
                'raw_headings_neighbors': np.asarray(raw_headings_neighbors, dtype=float),
                'headings_neighbors_used': np.asarray(headings_neighbors, dtype=float),
                'used_velocity_heading': bool(used_velocity_heading),
                'neighbor_indices': np.asarray(neighbor_indices, dtype=np.int32),
                'agent_indices': np.asarray(agent_indices, dtype=np.int32),
            }
            print('DBG alignment_cue: _alignment_diag set successfully')
        except Exception as e:
            try:
                import traceback
                print('DBG alignment_cue: failed to set _alignment_diag:', e)
                traceback.print_exc()
            except Exception:
                pass
        vectors_to_neighbors_x = self.simulation.X[neighbor_indices] - self.simulation.X[agent_indices]
        vectors_to_neighbors_y = self.simulation.Y[neighbor_indices] - self.simulation.Y[agent_indices]

        if consider_front_only:
            agent_velocities_x = self.simulation.x_vel[agent_indices]
            agent_velocities_y = self.simulation.y_vel[agent_indices]
            dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
            valid_neighbors_mask = dot_products > 0
        else:
            valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)

        valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
        valid_agent_indices = agent_indices[valid_neighbors_mask]

        # compute circular mean of neighbor headings per-agent using sum of unit vectors
        sum_cos = np.zeros(num_agents)
        sum_sin = np.zeros(num_agents)
        np.add.at(sum_cos, valid_agent_indices, np.cos(headings_neighbors[valid_neighbors_mask]))
        np.add.at(sum_sin, valid_agent_indices, np.sin(headings_neighbors[valid_neighbors_mask]))
        counts = np.bincount(valid_agent_indices, minlength=num_agents)
        # avoid divide-by-zero
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        mean_cos = sum_cos / counts_safe
        mean_sin = sum_sin / counts_safe
        # resulting desired heading unit vector
        avg_vec_x = mean_cos
        avg_vec_y = mean_sin
        no_school = np.where(counts == 0, 0., 1.)

        # current heading unit vector (use stored heading angles for direction)
        cur_hat_x = np.cos(self.simulation.heading)
        cur_hat_y = np.sin(self.simulation.heading)

        # vector difference between desired heading unit vector and current heading unit vector
        vectors_to_heading_x = avg_vec_x - cur_hat_x
        vectors_to_heading_y = avg_vec_y - cur_hat_y
        distances = np.sqrt(vectors_to_heading_x**2 + vectors_to_heading_y**2)
        epsilon = 1e-10
        v_hat_align_x = np.divide(vectors_to_heading_x, distances + epsilon, out=np.zeros_like(cur_hat_x), where=distances+epsilon != 0)
        v_hat_align_y = np.divide(vectors_to_heading_y, distances + epsilon, out=np.zeros_like(cur_hat_y), where=distances+epsilon != 0)
        alignment_array = np.zeros((num_agents, 2))
        alignment_array[:, 0] = weight * v_hat_align_x * no_school
        alignment_array[:, 1] = weight * v_hat_align_y * no_school

        sogs = np.array([np.mean(self.simulation.sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])
        sogs = np.where(sogs < 0.5 * self.simulation.length / 1000,
                        0.5 * self.simulation.length / 1000,
                        sogs)
        self.simulation.school_sog = sogs
        # record whether alignment used velocity-derived headings for diagnostics
        try:
            self.simulation.alignment_used_velocity = used_velocity_heading
        except Exception:
            pass
        return np.nan_to_num(alignment_array)

    def collision_cue(self, weight):
        # ensure closest_agent and nearest_neighbor_distance are populated; reconstruct when missing
        try:
            closest_agent_arr = np.asarray(self.simulation.closest_agent, dtype=float).copy()
        except Exception:
            closest_agent_arr = np.full(self.simulation.num_agents, np.nan)
        try:
            nearest_d_arr = np.asarray(self.simulation.nearest_neighbor_distance, dtype=float).copy()
        except Exception:
            nearest_d_arr = np.full(self.simulation.num_agents, np.nan)

        # reconstruct missing entries from agents_within_buffers
        try:
            awb = getattr(self.simulation, 'agents_within_buffers', None)
            if awb is not None:
                for ag in range(self.simulation.num_agents):
                    if np.isnan(nearest_d_arr[ag]) or np.isnan(closest_agent_arr[ag]):
                        nbrs = awb[ag]
                        if nbrs is None or len(nbrs) == 0:
                            continue
                        # compute distances to neighbors
                        dx = self.simulation.X[nbrs] - self.simulation.X[ag]
                        dy = self.simulation.Y[nbrs] - self.simulation.Y[ag]
                        dists = np.sqrt(dx**2 + dy**2)
                        idx = int(np.argmin(dists))
                        closest_agent_arr[ag] = nbrs[idx]
                        nearest_d_arr[ag] = float(dists[idx])
        except Exception:
            pass

        # update simulation attributes so other code sees reconstructed values
        try:
            self.simulation.closest_agent = closest_agent_arr
            self.simulation.nearest_neighbor_distance = nearest_d_arr
        except Exception:
            pass

        valid_indices = ~np.isnan(closest_agent_arr)
        closest_X = np.full_like(self.simulation.X, np.nan)
        closest_Y = np.full_like(self.simulation.Y, np.nan)
        try:
            closest_X[valid_indices] = self.simulation.X[closest_agent_arr[valid_indices].astype(int)]
            closest_Y[valid_indices] = self.simulation.Y[closest_agent_arr[valid_indices].astype(int)]
        except Exception:
            # fallback: leave NaNs
            pass

        self_2_closest = np.column_stack((closest_X.flatten() - self.simulation.X.flatten(), closest_Y.flatten() - self.simulation.Y.flatten()))
        closest_2_self = np.column_stack((self.simulation.X.flatten() - closest_X.flatten(), self.simulation.Y.flatten() - closest_Y.flatten()))

        invalid_vectors = np.isnan(closest_2_self).any(axis=1)
        closest_2_self[invalid_vectors] = [np.nan, np.nan]
        closest_2_self = np.nan_to_num(closest_2_self)

        safe_distances = np.where(self.simulation.nearest_neighbor_distance > 0, self.simulation.nearest_neighbor_distance, np.nan)
        v_hat_x = np.divide(closest_2_self[:, 0], safe_distances, out=np.zeros_like(closest_2_self[:, 0]), where=safe_distances != 0)
        v_hat_y = np.divide(closest_2_self[:, 1], safe_distances, out=np.zeros_like(closest_2_self[:, 1]), where=safe_distances != 0)

        collision_cue_x = np.divide(weight * v_hat_x, safe_distances**2, out=np.zeros_like(v_hat_x), where=safe_distances != 0)
        collision_cue_y = np.divide(weight * v_hat_y, safe_distances**2, out=np.zeros_like(v_hat_y), where=safe_distances != 0)

        collision_cue_mm = np.column_stack((collision_cue_x, collision_cue_y))
        np.nan_to_num(collision_cue_mm, copy=False)
        return collision_cue_mm

    def is_in_eddy(self, t):
        linear_positions = self.simulation.compute_linear_positions(self.simulation.longitudinal)
        self.current_longitudes = linear_positions
        self.simulation.past_longitudes[:, :-1] = self.simulation.past_longitudes[:, 1:]
        self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]
        self.simulation.past_longitudes[:, -1] = linear_positions
        self.simulation.swim_speeds[:, -1] = self.simulation.sog
        valid_entries = ~np.isnan(self.simulation.swim_speeds[:, 0]) & ~np.isnan(self.simulation.swim_speeds[:, -1])

        avg_speeds = np.full(self.simulation.swim_speeds.shape[0], np.nan)
        avg_speeds[valid_entries] = np.max(self.simulation.swim_speeds[valid_entries], axis=-1)

        total_displacement = np.full(self.simulation.past_longitudes.shape[0], np.nan)
        total_displacement[valid_entries] = self.simulation.past_longitudes[valid_entries, -1] - self.simulation.past_longitudes[valid_entries, 0]

        delta = self.simulation.past_longitudes[valid_entries, 0] - self.simulation.past_longitudes[valid_entries, -1]
        dt = self.simulation.past_longitudes.shape[1]
        expected_displacement = avg_speeds * dt
        long_dir = self.simulation.past_longitudes[:, -2] - self.simulation.past_longitudes[:, -1]

        if delta.shape == total_displacement.shape and t >= 1800.:
            stuck_conditions = (expected_displacement >= 5. * np.abs(total_displacement)) & (self.simulation.swim_behav == 1)
        else:
            stuck_conditions = np.zeros_like(self.simulation.X)

        not_in_eddy_anymore = self.simulation.time_since_eddy_escape >= self.simulation.max_eddy_escape_seconds
        self.simulation.swim_speeds[not_in_eddy_anymore, :] = np.nan
        self.simulation.past_longitudes[not_in_eddy_anymore, :] = np.nan
        self.simulation.time_since_eddy_escape[not_in_eddy_anymore] = 0.0

        already_in_eddy = self.simulation.in_eddy == True
        self.simulation.in_eddy = np.where(np.logical_or(stuck_conditions, already_in_eddy), True, False)
        self.simulation.in_eddy[not_in_eddy_anymore] = False
        self.simulation.time_since_eddy_escape[self.simulation.in_eddy == True] += 1

    def arbitrate(self, t):
        # debug: print current simulation.heading at start of arbitration
        if getattr(self.simulation, 'debug_behavior', False):
            try:
                print('arbitrate: simulation.heading (start)=', np.asarray(self.simulation.heading))
            except Exception:
                pass
        if self.simulation.pid_tuning:
            rheotaxis = self.rheo_cue(50000)
        else:
            rheotaxis = self.rheo_cue(25000)
            try:
                print('DBG arbitrate: about to call alignment_cue')
            except Exception:
                pass
            alignment = self.alignment_cue(20500)
            cohesion = self.cohesion_cue(11000)
            low_speed = self.vel_cue(1500)
            wave_drag = self.wave_drag_cue(0)
            refugia = self.find_nearest_refuge(50000)
            border = self.border_cue(50000, t)
            shallow = self.shallow_cue(100000)
            avoid = self.already_been_here(25000, t)
            collision = self.collision_cue(50000)

        order_dict = {0: 'shallow', 1: 'border', 2: 'avoid', 3: 'collision', 4: 'alignment', 5: 'cohesion', 6: 'low_speed', 7: 'rheotaxis', 8: 'wave_drag'}

        cue_dict = {'rheotaxis': rheotaxis,
                'shallow': shallow,
                'border': border,
                'wave_drag': wave_drag,
                'low_speed': low_speed,
                'avoid': avoid,
                'alignment': alignment,
                'cohesion': cohesion,
                'collision': collision,
                'refugia': refugia}

        # Diagnostic: capture raw cue shapes to help find broadcasting issues
        try:
            shapes = {}
            for k, v in cue_dict.items():
                try:
                    arr = np.asarray(v)
                    shapes[k] = {'ndim': arr.ndim, 'shape': arr.shape}
                except Exception:
                    shapes[k] = {'error': 'cannot convert to array'}
            self.simulation.cue_shapes = shapes
            if getattr(self.simulation, 'debug_behavior', False):
                import json, os, time
                outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
                os.makedirs(outdir, exist_ok=True)
                fname = os.path.join(outdir, f'cue_shapes_{int(getattr(self.simulation, "current_step", t))}_{int(time.time())}.json')
                with open(fname, 'w', encoding='utf-8') as fh:
                    json.dump(shapes, fh, indent=2)
        except Exception:
            pass

        low_bat_cue_dict = {0: 'shallow', 1: 'border', 2: 'refugia'}
        try:
            self.is_in_eddy(t)
        except Exception:
            # If simulation lacks helpers during lightweight probes, skip eddy detection
            pass
        tolerance = 50000
        vec_sum_migratory = np.zeros_like(rheotaxis)
        vec_sum_tired = np.zeros_like(rheotaxis)

        cue_magnitudes = {}
        raw_vecs = {}

        # helper: coerce cue arrays to shape (num_agents, 2)
        def _ensure_agent_vec(vec):
            arr = np.asarray(vec)
            n = self.simulation.num_agents
            # common shapes: (n,2) -> OK
            try:
                if arr.shape == (n, 2):
                    return arr
            except Exception:
                pass
            # (2, n) -> transpose
            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n:
                return arr.T
            # 1-D vector of length 2 -> replicate for all agents
            if arr.ndim == 1 and arr.size == 2:
                return np.tile(arr, (n, 1))
            # 1-D per-agent scalar -> use as x component, zero y
            if arr.ndim == 1 and arr.size == n:
                return np.column_stack((arr, np.zeros(n)))
            # 3-D arrays (n, H, W) or similar: collapse spatial dims to scalar per agent
            if arr.ndim >= 2:
                # try to find axis equal to n (number of agents)
                axes = [i for i, s in enumerate(arr.shape) if s == n]
                if axes:
                    axis = axes[0]
                    # move axis to front
                    moved = np.moveaxis(arr, axis, 0)
                    # collapse remaining dims to a scalar per agent (sum)
                    collapsed = np.nan_to_num(moved).reshape(n, -1).sum(axis=1)
                    return np.column_stack((collapsed, np.zeros(n)))
            # final fallback: zeros
            return np.zeros((n, 2), dtype=float)
        for i in order_dict.keys():
            cue = order_dict[i]
            vec = cue_dict[cue]
            # coerce to (n_agents, 2) to avoid accidental broadcasting
            vec = _ensure_agent_vec(vec)
            # store coerced vector for debug dumps
            raw_vecs[cue] = vec
            # record L2 norm per agent for debugging
            try:
                cue_magnitudes[cue] = np.linalg.norm(vec, axis=1)
            except Exception:
                # scalar or different shape
                try:
                    cue_magnitudes[cue] = np.abs(vec)
                except Exception:
                    cue_magnitudes[cue] = np.zeros(self.simulation.num_agents)
            if cue != 'refugia':
                vec_sum_migratory = np.where(np.linalg.norm(vec_sum_migratory, axis=-1)[:, np.newaxis] < tolerance,
                                              vec_sum_migratory + vec,
                                              vec_sum_migratory)

        for i in np.arange(0, 3, 1):
            cue = low_bat_cue_dict[i]
            vec = cue_dict[cue]
            vec = _ensure_agent_vec(vec)
            vec_sum_tired = np.where(np.linalg.norm(vec_sum_tired, axis=-1)[:, np.newaxis] < tolerance,
                                     vec_sum_tired + vec,
                                     vec_sum_tired)

        head_vec = np.zeros_like(rheotaxis)
        head_vec = np.where(self.simulation.swim_behav[:, np.newaxis] == 1, vec_sum_migratory, head_vec)
        head_vec = np.where(self.simulation.swim_behav[:, np.newaxis] == 2, vec_sum_tired, head_vec)
        head_vec = np.where(self.simulation.swim_behav[:, np.newaxis] == 3, vec_sum_tired, head_vec)
        # ensure we use coerced (n,2) cue vectors for in-eddy override
        border_vec = _ensure_agent_vec(cue_dict['border'])
        shallow_vec = _ensure_agent_vec(cue_dict['shallow'])
        head_vec = np.where(self.simulation.in_eddy[:, np.newaxis] == 1, border_vec + shallow_vec, head_vec)

        if len(head_vec.shape) == 2:
            # debug print of cue magnitudes if requested
            if getattr(self.simulation, 'debug', False):
                try:
                    print('behavior cue magnitudes (summary min/max) at t=', t)
                    for k, v in cue_magnitudes.items():
                        arr = np.array(v)
                        print(f' - {k}: min={float(np.nanmin(arr)):.4g}, max={float(np.nanmax(arr)):.4g}')
                except Exception:
                    pass
            # store last head_vec and cue magnitudes on simulation for quick inspection
            try:
                self.simulation.last_head_vec = np.asarray(head_vec)
                self.simulation.last_cue_magnitudes = {k: np.asarray(v) for k, v in cue_magnitudes.items()}
            except Exception:
                pass

            # optional behavior debugging: dump cue snapshots
            try:
                if getattr(self.simulation, 'debug_behavior', False):
                    outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
                    os.makedirs(outdir, exist_ok=True)
                    import time
                    fname_npz = os.path.join(outdir, f'behavior_debug_step_{int(getattr(self.simulation, "current_step", t))}_{int(time.time())}.npz')
                    # safe serializable payload: head_vec (n,2) and cue_magnitudes (per-cue arrays)
                    # include both magnitudes and the coerced per-agent vectors for inspection
                    safe_cues = {f'{k}_mag': (np.asarray(v).astype(float) if getattr(v, 'size', 0) > 0 else np.array([])) for k, v in cue_magnitudes.items()}
                    safe_vecs = {f'{k}_vec': (np.asarray(v).astype(float) if getattr(v, 'size', 0) > 0 else np.zeros((self.simulation.num_agents, 2))) for k, v in raw_vecs.items()}
                    try:
                        # include battery and swim state fields when present for fatigue inspection
                        extra = {}
                        for bk in ('battery', 'swim_behav', 'swim_mode', 'ideal_sog', 'sog', 'bout_dur', 'dist_per_bout'):
                            if hasattr(self.simulation, bk):
                                val = getattr(self.simulation, bk)
                                extra[bk] = np.asarray(val).astype(float)
                        # diagnostic: alignment fallback flag
                        try:
                            extra['alignment_used_velocity'] = float(getattr(self.simulation, 'alignment_used_velocity', 0.0))
                        except Exception:
                            extra['alignment_used_velocity'] = 0.0
                        # debug: print neighbor diagnostics before writing NPZ
                        if getattr(self.simulation, 'debug_behavior', False):
                            try:
                                nc = neighbor_counts if 'neighbor_counts' in locals() else None
                                nc_sample = nc[:10] if nc is not None else None
                                print('NPZ write: neighbor_counts sample=', nc_sample)
                                print('NPZ write: simulation.heading sample=', np.asarray(self.simulation.heading)[:10])
                            except Exception:
                                pass

                        # sanitize safe_cues and safe_vecs: convert empty arrays or all-NaN arrays to zeros to avoid
                        # runtime warnings when consumers compute min/max/mean
                        def _sanitize(arr):
                            a = np.asarray(arr)
                            if a.size == 0:
                                return np.zeros((self.simulation.num_agents,)) if a.ndim == 1 else np.zeros((self.simulation.num_agents, 2))
                            if np.all(np.isnan(a)):
                                # replace all-NaN with zeros
                                return np.nan_to_num(a, nan=0.0)
                            return a

                        safe_cues_s = {k: _sanitize(v) for k, v in safe_cues.items()}
                        safe_vecs_s = {k: _sanitize(v) for k, v in safe_vecs.items()}

                        # Neighbor diagnostics: convert agents_within_buffers (list of arrays)
                        # into concise serializable arrays: counts per agent and concatenated indices.
                        neighbor_counts = None
                        neighbors_concat = None
                        neighbor_any_within_2bl = None
                        neighbor_mean_distance = None
                        try:
                            if hasattr(self.simulation, 'agents_within_buffers'):
                                awb = self.simulation.agents_within_buffers
                                neighbor_counts = np.array([len(x) for x in awb], dtype=np.int32)
                                if neighbor_counts.sum() > 0:
                                    neighbors_concat = np.concatenate(awb).astype(np.int32)
                                    # reconstruct agent indices for each entry in concat
                                    agent_idx_repeat = np.repeat(np.arange(self.simulation.num_agents), neighbor_counts)
                                    # compute distances for each neighbor entry
                                    X = np.asarray(self.simulation.X).flatten()
                                    Y = np.asarray(self.simulation.Y).flatten()
                                    nbr_X = X[neighbors_concat]
                                    nbr_Y = Y[neighbors_concat]
                                    dx = nbr_X - X[agent_idx_repeat]
                                    dy = nbr_Y - Y[agent_idx_repeat]
                                    dists = np.sqrt(dx**2 + dy**2)
                                    # per-agent mean distance (nan if no neighbors)
                                    # start from zeros, accumulate distances per-agent, then divide
                                    mean_per_agent = np.zeros(self.simulation.num_agents, dtype=float)
                                    np.add.at(mean_per_agent, agent_idx_repeat, dists)
                                    # divide by counts where >0; mark agents with zero neighbors as NaN
                                    nonzero = neighbor_counts > 0
                                    mean_per_agent[nonzero] = mean_per_agent[nonzero] / neighbor_counts[nonzero]
                                    mean_per_agent[~nonzero] = np.nan
                                    neighbor_mean_distance = mean_per_agent
                                    # per-agent minimum neighbor distance
                                    min_per_agent = np.full(self.simulation.num_agents, np.nan)
                                    if dists.size > 0:
                                        # compute min per agent
                                        # initialize accumulator with +inf
                                        acc_min = np.full(self.simulation.num_agents, np.inf)
                                        for idx, ag in enumerate(agent_idx_repeat):
                                            acc_min[ag] = min(acc_min[ag], dists[idx])
                                        acc_min[acc_min == np.inf] = np.nan
                                        min_per_agent = acc_min
                                    neighbor_min_distance = min_per_agent
                                    # any neighbor within two body lengths?
                                    two_bl = 2.0 * (self.simulation.length / 1000.0)
                                    within_mask = dists <= two_bl[agent_idx_repeat]
                                    any_within = np.zeros(self.simulation.num_agents, dtype=np.bool_)
                                    if within_mask.size > 0:
                                        np.logical_or.at(any_within, agent_idx_repeat[within_mask], True)
                                    neighbor_any_within_2bl = any_within
                                    # neighbor headings and relative headings per neighbor entry
                                    try:
                                        neighbor_headings = np.asarray(self.simulation.heading)[neighbors_concat]
                                        neighbor_rel_heading = neighbor_headings - np.asarray(self.simulation.heading)[agent_idx_repeat]
                                        # normalize to [-pi, pi]
                                        neighbor_rel_heading = (neighbor_rel_heading + np.pi) % (2 * np.pi) - np.pi
                                    except Exception:
                                        neighbor_headings = np.array([], dtype=float)
                                        neighbor_rel_heading = np.array([], dtype=float)
                                else:
                                    neighbors_concat = np.array([], dtype=np.int32)
                                    neighbor_mean_distance = np.full(self.simulation.num_agents, np.nan)
                                    neighbor_any_within_2bl = np.zeros(self.simulation.num_agents, dtype=np.bool_)
                        except Exception:
                            neighbor_counts = np.zeros(self.simulation.num_agents, dtype=np.int32)
                            neighbors_concat = np.array([], dtype=np.int32)
                            neighbor_mean_distance = np.full(self.simulation.num_agents, np.nan)
                            neighbor_any_within_2bl = np.zeros(self.simulation.num_agents, dtype=np.bool_)

                        try:
                            print('DBG NPZ write: neighbor_counts.shape=', None if neighbor_counts is None else getattr(neighbor_counts, 'shape', None))
                            print('DBG NPZ write: neighbors_concat.shape=', None if neighbors_concat is None else getattr(neighbors_concat, 'shape', None))
                            print('DBG NPZ write: agent_idx_repeat.shape=', None if 'agent_idx_repeat' not in locals() else getattr(agent_idx_repeat, 'shape', None))
                            if neighbors_concat is not None and getattr(neighbors_concat, 'size', 0) > 0:
                                print('DBG NPZ write: neighbors_concat sample=', neighbors_concat[:20])
                            if 'agent_idx_repeat' in locals() and getattr(agent_idx_repeat, 'size', 0) > 0:
                                print('DBG NPZ write: agent_idx_repeat sample=', agent_idx_repeat[:20])
                        except Exception:
                            pass

                        np.savez_compressed(
                            fname_npz,
                            head_vec=np.asarray(head_vec).astype(float),
                            **safe_cues_s,
                            **safe_vecs_s,
                            **extra,
                            neighbor_counts=neighbor_counts,
                            neighbors_concat=neighbors_concat,
                            neighbor_mean_distance=neighbor_mean_distance,
                            neighbor_any_within_2bl=neighbor_any_within_2bl,
                            neighbor_headings=(neighbor_headings if 'neighbor_headings' in locals() else np.array([])),
                            neighbor_rel_heading=(neighbor_rel_heading if 'neighbor_rel_heading' in locals() else np.array([])),
                            neighbors_owner=(agent_idx_repeat if 'agent_idx_repeat' in locals() else np.array([], dtype=np.int32)),
                            neighbor_min_distance=(neighbor_min_distance if 'neighbor_min_distance' in locals() else np.full(self.simulation.num_agents, np.nan)),
                            closest_agent=(np.asarray(getattr(self.simulation, 'closest_agent', np.array([]))).astype(np.float64) if hasattr(self.simulation, 'closest_agent') else np.array([])),
                            nearest_neighbor_distance=(np.asarray(getattr(self.simulation, 'nearest_neighbor_distance', np.array([]))).astype(np.float64) if hasattr(self.simulation, 'nearest_neighbor_distance') else np.array([])),
                            # include alignment diagnostics when available
                            **({'alignment_diag': self.simulation._alignment_diag} if hasattr(self.simulation, '_alignment_diag') else {}),
                        )
                        try:
                            print('Wrote behavior debug NPZ:', fname_npz)
                        except Exception:
                            pass
                    except Exception:
                        # fallback: write a small JSON containing head_vec and cue magnitudes
                        import json
                        fname_json = fname_npz.replace('.npz', '.json')
                        serial = {'head_vec': (np.asarray(head_vec)).astype(float).tolist(), 'cue_magnitudes': {k: (np.asarray(v).astype(float)).tolist() for k, v in cue_magnitudes.items()}}
                        with open(fname_json, 'w', encoding='utf-8') as fh:
                            json.dump(serial, fh)
            except Exception:
                pass
            return np.arctan2(head_vec[:, 1], head_vec[:, 0])
        else:
            # If head_vec has unexpected shape, try to sanitize: replace NaNs and zero-length vectors
            try:
                hv = np.asarray(head_vec)
                if hv.ndim == 3:
                    hv = hv.reshape(hv.shape[0], -1)
                # compute norms and replace NaNs/zeros
                norms = np.linalg.norm(hv, axis=-1)
                # fallback unit vectors from previous headings
                prev_hat_x = np.cos(self.simulation.heading)
                prev_hat_y = np.sin(self.simulation.heading)
                prev_hat = np.column_stack((prev_hat_x, prev_hat_y))
                # where norm is zero or nan, replace with prev_hat or rheotaxis
                safe_hv = np.where(np.isnan(norms)[:, np.newaxis] | (norms[:, np.newaxis] == 0), prev_hat, hv)
                return np.arctan2(safe_hv[:, 1], safe_hv[:, 0])
            except Exception:
                # ultimate fallback: return previous heading
                return np.asarray(self.simulation.heading)
        # end of arbitrate: handled 2D and attempted safe fallback above
