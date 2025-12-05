
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:30:21 2023

@author: KNebiolo, Isha Deo

Python software for an Agent Based Model of migrating adult Sockeye salmon (spp.)  
with intent of understanding the potential ramifications of river discharge 
changes on ability of fish to succesffuly pass upstream through a riffle - 
cascade complex.  

An agent is a goal-directed, autonomous, software-object that interacts with 
other agents in simulated space.  In the case of a fish passage agent, our fish 
are motivated to move upstream to spawn, thus their goal is simply to pass the 
impediment.   Their motivation is clear, they have an overriding instinct to 
migrate upstream to their natal reach and will do so at the cost of their own 
mortality.   

Our fish agents are python class objects with initialization methods, and 
methods for movement, behaviors, and perception.  Movement is continuous in 2d 
space as our environment is a depth averaged 2d model.  Movement in 
the Z direction is handled with logic.  We will use velocity distributions and 
the agent's position within the water column to understand the forces acting on 
the body of the fish (drag).  To maintain position, the agent must generate 
enough thrust to counteract drag.  The fish generates thrust by beating its tail.  
According to Castro-Santos (2006), fish tend to migrate at a specific speed over 
ground in body lengths per second depending upon the mode of swimming it is in.  
Therefore their tail beat per minute rate is dependent on the amount of drag 
and swimming mode.   
"""
# import dependencies
import h5py
import dask.array as da
import geopandas as gpd
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import random
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
import sys
import time
from datetime import datetime
from affine import Affine
from scipy.interpolate import CubicSpline, LinearNDInterpolator, UnivariateSpline
from scipy.ndimage import distance_transform_edt, label, gaussian_filter1d, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import beta
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry
from sklearn.decomposition import PCA
try:
    from sksurv.nonparametric import kaplan_meier_estimator
    _HAS_SKSURV = True
except ImportError:
    kaplan_meier_estimator = None
    _HAS_SKSURV = False
# from pysal.explore import esda
# from pysal.lib import weights
try:
    import numba
    _HAS_NUMBA = True
except Exception:
    numba = None
    _HAS_NUMBA = False
class BehavioralWeights:
    def __init__(self):
        # Alignment SOG augmentation
        self.use_sog = True            # Whether to consider neighbor SOG for alignment
        self.sog_weight = 0.5          # Relative weight (0..1) blending heading vs SOG alignment
        self.cohesion_weight = 1.0        # Attraction to group center
        self.alignment_weight = 1.0       # Tendency to match neighbors' heading
        self.separation_weight = 2.0      # Short-range repulsion from neighbors
        self.separation_radius = 3.0      # Distance (body lengths) for separation
        
        # Dynamic cohesion based on threat level
        self.threat_level = 0.0           # 0.0 = relaxed, 1.0 = high threat (predators)
        self.cohesion_radius_relaxed = 2.0   # Body lengths when relaxed
        self.cohesion_radius_threatened = 0.5 # Body lengths when threatened (tight ball)
        
        # Energy efficiency from drafting (swimming behind others)
        self.drafting_enabled = True
        self.drafting_distance = 2.0      # Body lengths behind to get benefit
        self.drafting_angle_tolerance = 30.0  # Degrees off-axis for drafting
        self.drag_reduction_single = 0.15  # 15% drag reduction behind one agent
        self.drag_reduction_dual = 0.25    # 25% drag reduction between two agents (V-formation)
        
        # Environmental responses
        self.rheotaxis_weight = 3.0       # Swim against flow
        self.border_cue_weight = 50000.0  # Avoid channel edges/banks
        self.border_threshold_multiplier = 2.0  # Threshold = this × body_length (min 1m)
        self.border_max_force = 5.0       # Cap border repulsion force
        
        # Collision avoidance
        self.collision_weight = 5.0       # Avoid other agents
        self.collision_radius = 2.0       # Distance (body lengths) for collision avoidance
        
        # Navigation priorities (higher = more important)
        self.upstream_priority = 1.0      # Weight for upstream progress
        self.energy_efficiency_priority = 0.5  # Prefer energy-efficient paths
        
        # Learning rates (for RL training)
        self.learning_rate = 0.001
        self.exploration_epsilon = 0.1
        # Alignment SOG augmentation
        self.use_sog = True            # Whether to consider neighbor SOG for alignment
        self.sog_weight = 0.5          # Relative weight (0..1) blending heading vs SOG alignment
        
    def to_dict(self):
        """Export weights as dictionary for saving."""
        return {
            'cohesion_weight': self.cohesion_weight,
            'alignment_weight': self.alignment_weight,
            'separation_weight': self.separation_weight,
            'separation_radius': self.separation_radius,
            'threat_level': self.threat_level,
            'cohesion_radius_relaxed': self.cohesion_radius_relaxed,
            'cohesion_radius_threatened': self.cohesion_radius_threatened,
            'drafting_enabled': self.drafting_enabled,
            'drafting_distance': self.drafting_distance,
            'drafting_angle_tolerance': self.drafting_angle_tolerance,
            'drag_reduction_single': self.drag_reduction_single,
            'drag_reduction_dual': self.drag_reduction_dual,
            'rheotaxis_weight': self.rheotaxis_weight,
            'border_cue_weight': self.border_cue_weight,
            'border_threshold_multiplier': self.border_threshold_multiplier,
            'border_max_force': self.border_max_force,
            'collision_weight': self.collision_weight,
            'collision_radius': self.collision_radius,
            'upstream_priority': self.upstream_priority,
            'energy_efficiency_priority': self.energy_efficiency_priority,
            'learning_rate': self.learning_rate,
            'exploration_epsilon': self.exploration_epsilon
            ,'use_sog': self.use_sog
            ,'sog_weight': self.sog_weight
        }
    
    def from_dict(self, data):
        """Load weights from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, filepath):
        """Save learned weights to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved behavioral weights to {filepath}")
    
    def load(self, filepath):
        """Load learned weights from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.from_dict(data)
        print(f"Loaded behavioral weights from {filepath}")
    
    def mutate(self, scale=0.1):
        """Apply small random perturbations for exploration during training."""
        for attr in ['cohesion_weight', 'alignment_weight', 'separation_weight',
                     'rheotaxis_weight', 'collision_weight']:
            current = getattr(self, attr)
            noise = np.random.normal(0, scale * abs(current))
            setattr(self, attr, max(0.01, current + noise))


# =============================================================================
# Schooling Metrics and Energy Efficiency Calculations
# =============================================================================

@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_schooling_loop(positions, headings, neighbor_data, neighbor_offsets, ideal_dist, mean_BL,
                           cohesion_scores, alignment_scores, N):
    """Numba-compiled loop for schooling metrics computation.
    
    Args:
        neighbor_data: Flattened array of all neighbor indices
        neighbor_offsets: Array where offsets[i]:offsets[i+1] gives neighbors of agent i
    """
    for i in numba.prange(N):
        start = neighbor_offsets[i]
        end = neighbor_offsets[i + 1]
        n_neighbors = 0
        
        # Count actual neighbors (excluding self)
        for idx in range(start, end):
            if neighbor_data[idx] != i:
                n_neighbors += 1
        
        if n_neighbors > 0:
            # COHESION: centroid and distance
            centroid_x = 0.0
            centroid_y = 0.0
            for idx in range(start, end):
                j = neighbor_data[idx]
                if j != i:
                    centroid_x += positions[j, 0]
                    centroid_y += positions[j, 1]
            centroid_x /= n_neighbors
            centroid_y /= n_neighbors
            
            dx = positions[i, 0] - centroid_x
            dy = positions[i, 1] - centroid_y
            dist_to_centroid = np.sqrt(dx*dx + dy*dy)
            
            cohesion_scores[i] = np.exp(-0.5 * ((dist_to_centroid - ideal_dist) / (0.5 * mean_BL))**2)
            
            # ALIGNMENT: circular mean of headings
            sin_sum = 0.0
            cos_sum = 0.0
            for idx in range(start, end):
                j = neighbor_data[idx]
                if j != i:
                    sin_sum += np.sin(headings[j])
                    cos_sum += np.cos(headings[j])
            sin_mean = sin_sum / n_neighbors
            cos_mean = cos_sum / n_neighbors
            mean_heading = np.arctan2(sin_mean, cos_mean)
            
            heading_diff = np.arctan2(np.sin(headings[i] - mean_heading), 
                                     np.cos(headings[i] - mean_heading))
            alignment_scores[i] = np.cos(heading_diff)


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_drafting_loop(positions, headings, neighbor_data, neighbor_offsets, angle_tol_rad,
                          drag_reduction_single, drag_reduction_dual, drag_reductions, N):
    """Numba-compiled loop for drafting benefit computation.
    
    Args:
        neighbor_data: Flattened array of all neighbor indices
        neighbor_offsets: Array where offsets[i]:offsets[i+1] gives neighbors of agent i
    """
    for i in numba.prange(N):
        start = neighbor_offsets[i]
        end = neighbor_offsets[i + 1]
        
        # Count agents ahead within cone
        left_count = 0
        right_count = 0
        total_ahead = 0
        
        for idx in range(start, end):
            j = neighbor_data[idx]
            if j == i:
                continue
            
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            angle_to_neighbor = np.arctan2(dy, dx)
            angle_diff = np.arctan2(np.sin(angle_to_neighbor - headings[i]),
                                   np.cos(angle_to_neighbor - headings[i]))
            
            if np.abs(angle_diff) < angle_tol_rad:
                total_ahead += 1
                if angle_diff < 0:
                    left_count += 1
                else:
                    right_count += 1
        
        # Determine drag reduction
        if total_ahead == 1:
            drag_reductions[i] = drag_reduction_single
        elif total_ahead >= 2:
            if left_count > 0 and right_count > 0:
                drag_reductions[i] = drag_reduction_dual
            else:
                drag_reductions[i] = drag_reduction_single


def compute_schooling_metrics_biological(positions, headings, body_lengths, behavioral_weights, alive_mask=None):
    """Compute biologically-grounded schooling quality metrics.
    
    Based on Boids model with sensory-range constraints (2 body lengths).
    
    Args:
        positions: (N, 2) array of agent positions
        headings: (N,) array of agent headings (radians)
        body_lengths: (N,) array of body lengths (meters)
        behavioral_weights: BehavioralWeights object
        alive_mask: (N,) boolean array (optional)
    
    Returns:
        dict with:
            'cohesion_score': 0-1 (proximity to ideal group spacing)
            'alignment_score': -1 to 1 (heading similarity with neighbors)
            'separation_penalty': -1 to 0 (crowding penalty)
            'overall_schooling': combined score
    """
    if alive_mask is not None:
        positions = positions[alive_mask]
        headings = headings[alive_mask]
        body_lengths = body_lengths[alive_mask]
    
    N = len(positions)
    if N == 0:
        return {'cohesion_score': 0, 'alignment_score': 0, 'separation_penalty': 0, 'overall_schooling': 0}
    
    # Dynamic cohesion radius based on threat level
    threat = behavioral_weights.threat_level
    cohesion_radius_BL = (
        behavioral_weights.cohesion_radius_relaxed * (1 - threat) +
        behavioral_weights.cohesion_radius_threatened * threat
    )
    
    # Build KDTree for efficient neighbor queries
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    
    # Fixed 1-meter radius for all agents (performance optimization)
    search_radius = 1.0
    
    # Get all neighbors for all agents at once
    neighbor_lists = tree.query_ball_point(positions, r=search_radius)
    
    # Preallocate arrays
    cohesion_scores = np.zeros(N)
    alignment_scores = np.full(N, -0.5)  # Default to isolation penalty
    separation_penalties = np.zeros(N)
    
    # Vectorized separation: find nearest neighbor for all agents at once
    nearest_dists, nearest_indices = tree.query(positions, k=2)  # k=2 includes self
    if N > 1:
        nearest_dists = nearest_dists[:, 1]  # Exclude self (index 0)
        mean_BL = np.mean(body_lengths)
        separation_mask = nearest_dists < mean_BL
        separation_penalties[separation_mask] = -(mean_BL - nearest_dists[separation_mask]) / mean_BL
        
        ideal_dist = 2.0 * mean_BL * (1 - 0.5 * threat)
        
        # Convert neighbor_lists to flattened arrays for numba
        neighbor_data = []
        neighbor_offsets = [0]
        for neighbors in neighbor_lists:
            neighbor_data.extend(neighbors)
            neighbor_offsets.append(len(neighbor_data))
        neighbor_data = np.array(neighbor_data, dtype=np.int32)
        neighbor_offsets = np.array(neighbor_offsets, dtype=np.int32)
        
        # Numba-compiled parallel loop for cohesion and alignment
        _compute_schooling_loop(positions, headings, neighbor_data, neighbor_offsets, ideal_dist, mean_BL, 
                                cohesion_scores, alignment_scores, N)
    
    # Average across all agents
    mean_cohesion = float(np.mean(cohesion_scores))
    mean_alignment = float(np.mean(alignment_scores))
    mean_separation = float(np.mean(separation_penalties))
    
    overall = mean_cohesion + mean_alignment + mean_separation
    
    return {
        'cohesion_score': mean_cohesion,
        'alignment_score': mean_alignment,
        'separation_penalty': mean_separation,
        'overall_schooling': overall
    }


def compute_drafting_benefits(positions, headings, velocities, body_lengths, behavioral_weights, alive_mask=None):
    """Calculate energy savings from drafting (swimming behind other agents).
    
    Drafting reduces drag when swimming in wake of others:
    - Single agent ahead: 15% drag reduction
    - Two agents flanking (V-formation): 25% drag reduction
    
    Args:
        positions: (N, 2) array
        headings: (N,) array (radians)
        velocities: (N, 2) array (velocity vectors)
        body_lengths: (N,) array (meters)
        behavioral_weights: BehavioralWeights object
        alive_mask: (N,) boolean array
    
    Returns:
        (N,) array of drag reduction factors (0.0 to 0.25)
    """
    if not behavioral_weights.drafting_enabled:
        return np.zeros(len(positions))
    
    if alive_mask is not None:
        positions = positions[alive_mask]
        headings = headings[alive_mask]
        velocities = velocities[alive_mask]
        body_lengths = body_lengths[alive_mask]
    
    N = len(positions)
    drag_reductions = np.zeros(N)
    
    if N < 2:
        return drag_reductions
    
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    
    # Fixed 1-meter drafting radius (performance optimization)
    drafting_radius = 1.0
    angle_tol_rad = np.radians(behavioral_weights.drafting_angle_tolerance)
    
    # Vectorized: query all agents at once with fixed radius
    neighbor_lists = tree.query_ball_point(positions, r=drafting_radius)
    
    # Convert neighbor_lists to flattened arrays for numba
    neighbor_data = []
    neighbor_offsets = [0]
    for neighbors in neighbor_lists:
        neighbor_data.extend(neighbors)
        neighbor_offsets.append(len(neighbor_data))
    neighbor_data = np.array(neighbor_data, dtype=np.int32)
    neighbor_offsets = np.array(neighbor_offsets, dtype=np.int32)
    
    # Numba-compiled parallel loop for drafting calculations
    drag_reduction_single = behavioral_weights.drag_reduction_single
    drag_reduction_dual = behavioral_weights.drag_reduction_dual
    _compute_drafting_loop(positions, headings, neighbor_data, neighbor_offsets, angle_tol_rad,
                          drag_reduction_single, drag_reduction_dual, drag_reductions, N)
    
    return drag_reductions


class RLTrainer:
    """Reinforcement learning trainer for behavioral weight optimization.
    
    Trains agents to exhibit realistic schooling and upstream migration through
    reward shaping based on:
    - Cohesive schooling (agents stay together)
    - Upstream progress (agents migrate against flow)
    - Energy efficiency (minimize thrashing/oscillation)
    - Collision avoidance (maintain personal space)
    - Boundary avoidance (stay in channel)
    """
    
    def __init__(self, simulation):
        self.sim = simulation
        self.behavioral_weights = BehavioralWeights()
        self.training_history = []
        # Configurable penalties (can be adjusted during training)
        self.collision_penalty_per_event = 100.0
        self.dry_penalty_per_agent = 500.0
        self.shallow_penalty_per_agent = 200.0
        # Initial SOG jitter controls (fraction of ideal SOG)
        # If strong_initial_sog_jitter is True, use strong_initial_sog_jitter_fraction
        self.initial_sog_jitter_fraction = 0.1
        self.strong_initial_sog_jitter = False
        self.strong_initial_sog_jitter_fraction = 0.3
        
    def compute_reward(self, prev_state, current_state):
        """Compute reward for current timestep based on biologically-grounded behavior quality.
        
        Args:
            prev_state: dict with previous timestep metrics
            current_state: dict with current timestep metrics
            
        Returns:
            float: Total reward (higher = better behavior)
        """
        reward = 0.0
        
        # 1. SCHOOLING QUALITY (Biological metrics: cohesion + alignment + separation)
        if 'schooling_metrics' in current_state:
            sm = current_state['schooling_metrics']
            # Cohesion: 0-1 (proximity to ideal group spacing)
            cohesion_reward = sm['cohesion_score'] * 10.0
            # Alignment: -1 to 1 (heading similarity)
            alignment_reward = (sm['alignment_score'] + 1.0) * 5.0  # Scale to 0-10
            # Separation: -1 to 0 (crowding penalty)
            separation_penalty = sm['separation_penalty'] * 5.0
            
            schooling_total = cohesion_reward + alignment_reward + separation_penalty
            reward += schooling_total
        
        # 2. UPSTREAM PROGRESS: Reward net forward movement
        if 'mean_upstream_progress' in current_state:
            # Positive centerline measure change = upstream progress = good
            upstream_reward = current_state['mean_upstream_progress'] * 0.5
            reward += upstream_reward
        
        # 3. ENERGY EFFICIENCY: Reward distance traveled per unit energy
        if 'energy_efficiency' in current_state:
            # meters per kcal (higher = better)
            efficiency_reward = current_state['energy_efficiency'] * 2.0
            reward += efficiency_reward
        
        # 4. DRAFTING UTILIZATION: Reward agents using energy-saving formations
        if 'mean_drafting_benefit' in current_state:
            # 0.0 - 0.25 (drag reduction)
            drafting_reward = current_state['mean_drafting_benefit'] * 20.0  # Scale to 0-5
            reward += drafting_reward
        
        # 5. BOUNDARY AVOIDANCE: Penalize agents near banks
        if 'agents_near_boundary' in current_state:
            boundary_penalty = -current_state['agents_near_boundary'] * 5.0
            reward += boundary_penalty
        
        # 6. SURVIVAL: Strong penalty for mortality
        if 'dead_count' in current_state:
            mortality_penalty = -current_state['dead_count'] * 50.0
            reward += mortality_penalty
        
        # 7. MOVEMENT SMOOTHNESS: Penalize erratic acceleration
        if 'mean_acceleration' in current_state and 'mean_acceleration' in prev_state:
            accel_change = abs(current_state['mean_acceleration'] - prev_state['mean_acceleration'])
            smoothness_reward = -accel_change * 0.2
            reward += smoothness_reward

        # 8. COLLISIONS: heavy penalty per collision event (pairwise contact)
        if 'collision_count' in current_state:
            # Penalize collisions strongly to encourage separation behavior
            collision_penalty = -float(current_state['collision_count']) * float(getattr(self, 'collision_penalty_per_event', 100.0))
            reward += collision_penalty

        # 9. SHALLOW / DRY PENALTIES: heavy penalties when agents are on very shallow water
        # current_state may include 'dry_count' (depth <= 0) and 'shallow_count' (depth < 0.5*body_depth)
        if 'dry_count' in current_state:
            # Severe penalty for being on dry ground (out of water)
            dry_penalty = -float(current_state['dry_count']) * float(getattr(self, 'dry_penalty_per_agent', 500.0))
            reward += dry_penalty
        if 'shallow_count' in current_state:
            # Heavy penalty for being in water shallower than half their body depth
            shallow_penalty = -float(current_state['shallow_count']) * float(getattr(self, 'shallow_penalty_per_agent', 200.0))
            reward += shallow_penalty
        
        return reward
    
    def extract_state_metrics(self):
        """Extract current simulation state metrics for biologically-grounded reward computation."""
        metrics = {}
        
        # Get alive agents
        alive_mask = (self.sim.dead == 0) if hasattr(self.sim, 'dead') else np.ones(len(self.sim.X), dtype=bool)
        
        # Cache expensive metrics (compute every N frames, not every frame)
        # These Python loops are slow with many agents
        current_time = getattr(self.sim, 'current_time', 0)
        metrics_update_interval = getattr(self, 'metrics_update_interval', 10)
        should_update_metrics = (int(current_time) % metrics_update_interval == 0)
        
        # 1. SCHOOLING METRICS (Biological: cohesion + alignment + separation)
        if hasattr(self.sim, 'X') and hasattr(self.sim, 'Y') and hasattr(self.sim, 'heading'):
            if should_update_metrics or not hasattr(self, '_cached_schooling_metrics'):
                positions = np.column_stack([self.sim.X, self.sim.Y])
                headings = self.sim.heading
                body_lengths = self.sim.length / 1000.0  # Convert mm to meters
                
                schooling_metrics = compute_schooling_metrics_biological(
                    positions, headings, body_lengths, 
                    self.behavioral_weights, alive_mask
                )
                self._cached_schooling_metrics = schooling_metrics
            metrics['schooling_metrics'] = self._cached_schooling_metrics

            # Count collision events (pairs closer than 0.25 * body length)
            try:
                from scipy.spatial import cKDTree
                positions = np.column_stack([self.sim.X, self.sim.Y])
                body_lengths = self.sim.length / 1000.0
                # per-agent threshold = 0.25 * body_length
                per_agent_thresh = 0.25 * body_lengths
                max_thresh = np.nanmax(per_agent_thresh)
                tree = cKDTree(positions)
                pairs = tree.query_pairs(r=max_thresh)
                collision_count = 0
                if pairs:
                    for i, j in pairs:
                        dist = np.hypot(self.sim.X[i] - self.sim.X[j], self.sim.Y[i] - self.sim.Y[j])
                        pair_thresh = 0.25 * (per_agent_thresh[i] + per_agent_thresh[j]) * 0.5
                        if dist < pair_thresh:
                            collision_count += 1
                metrics['collision_count'] = collision_count
            except Exception:
                metrics['collision_count'] = 0

            # Dry / shallow detection
            try:
                # self.sim.depth should be per-agent depth in meters
                depths = np.asarray(self.sim.depth)
                # body_depth stored in cm in sim; convert to meters
                if hasattr(self.sim, 'body_depth'):
                    body_depth_m = np.asarray(self.sim.body_depth) / 100.0
                else:
                    body_depth_m = np.full_like(depths, 0.1)

                # dry: depth <= 0 (or very near zero)
                dry_mask = depths <= 0.0
                dry_count = int(np.sum(dry_mask & alive_mask))

                # shallow: depth < 0.5 * body_depth
                shallow_mask = depths < (0.5 * body_depth_m)
                shallow_count = int(np.sum(shallow_mask & alive_mask))

                metrics['dry_count'] = dry_count
                metrics['shallow_count'] = shallow_count
            except Exception:
                metrics['dry_count'] = 0
                metrics['shallow_count'] = 0
        
        # 2. DRAFTING BENEFITS (Energy efficiency from swimming behind others)
        if hasattr(self.sim, 'X') and hasattr(self.sim, 'Y') and hasattr(self.sim, 'x_vel') and hasattr(self.sim, 'y_vel'):
            if should_update_metrics or not hasattr(self, '_cached_drag_reductions'):
                positions = np.column_stack([self.sim.X, self.sim.Y])
                velocities = np.column_stack([self.sim.x_vel, self.sim.y_vel])
                headings = self.sim.heading
                body_lengths = self.sim.length / 1000.0
                
                # Compute drafting once and cache for reuse below
                drag_reductions = compute_drafting_benefits(
                    positions, headings, velocities, body_lengths,
                    self.behavioral_weights, alive_mask
                )
                self._cached_drag_reductions = drag_reductions
            
            metrics['mean_drafting_benefit'] = float(np.mean(self._cached_drag_reductions))
            # store in sim for reuse in energy_efficiency
            self.sim._last_drag_reductions = self._cached_drag_reductions
        
        # 3. UPSTREAM PROGRESS (Change in centerline position)
        if hasattr(self.sim, 'centerline_meas'):
            prev_meas = getattr(self.sim, '_prev_centerline_meas', self.sim.centerline_meas)
            upstream_deltas = self.sim.centerline_meas - prev_meas
            inst_upstream = float(np.mean(upstream_deltas[alive_mask]) if np.any(alive_mask) else 0.0)
            # maintain a rolling buffer on sim for recent upstream progress (seconds)
            window = int(getattr(self, 'upstream_metric_window', 30))
            from collections import deque
            if not hasattr(self.sim, '_upstream_progress_buf'):
                self.sim._upstream_progress_buf = deque(maxlen=window)
            self.sim._upstream_progress_buf.append(inst_upstream)
            metrics['mean_upstream_progress'] = float(np.mean(self.sim._upstream_progress_buf))
            # also expose instantaneous upstream velocity component (positive upstream only)
            try:
                # compute upstream unit vector from centerline direction if available
                # fallback: use sim.flow_direction (radians) if present
                if hasattr(self.sim, 'flow_direction'):
                    flow_dir = float(self.sim.flow_direction)
                    # upstream direction is opposite the flow
                    ux, uy = -np.cos(flow_dir), -np.sin(flow_dir)
                else:
                    # fallback to unit x,y upstream
                    ux, uy = 1.0, 0.0

                # per-agent velocity vectors
                if hasattr(self.sim, 'x_vel') and hasattr(self.sim, 'y_vel'):
                    vel_proj = self.sim.x_vel * ux + self.sim.y_vel * uy
                    # positive values mean moving upstream
                    mean_upstream_vel_inst = float(np.mean(np.maximum(vel_proj[alive_mask], 0.0))) if np.any(alive_mask) else 0.0
                else:
                    mean_upstream_vel_inst = 0.0

                if not hasattr(self.sim, '_upstream_vel_buf'):
                    self.sim._upstream_vel_buf = deque(maxlen=window)
                self.sim._upstream_vel_buf.append(mean_upstream_vel_inst)
                metrics['mean_upstream_velocity'] = float(np.mean(self.sim._upstream_vel_buf))
            except Exception:
                metrics['mean_upstream_velocity'] = 0.0
            self.sim._prev_centerline_meas = self.sim.centerline_meas.copy()
        
        # 4. ENERGY EFFICIENCY (Distance per kcal, accounting for drafting)
        if hasattr(self.sim, 'x_vel') and hasattr(self.sim, 'y_vel'):
            velocities = np.column_stack([self.sim.x_vel, self.sim.y_vel])
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Base energy: proportional to speed^2 (drag force)
            base_energy = speeds ** 2
            
            # Apply drafting reduction
            if 'mean_drafting_benefit' in metrics:
                drag_reductions = getattr(self.sim, '_last_drag_reductions', None)
                if drag_reductions is None:
                    drag_reductions = compute_drafting_benefits(
                        np.column_stack([self.sim.X, self.sim.Y]),
                        self.sim.heading, velocities,
                        self.sim.length / 1000.0,
                        self.behavioral_weights, alive_mask
                    )
                adjusted_energy = base_energy * (1.0 - drag_reductions)
            else:
                adjusted_energy = base_energy
            
            # Efficiency = distance per energy
            total_distance = np.sum(speeds[alive_mask]) if np.any(alive_mask) else 0.0
            total_energy = np.sum(adjusted_energy[alive_mask]) if np.any(alive_mask) else 1.0
            metrics['energy_efficiency'] = total_distance / (total_energy + 1e-6)  # Avoid div by zero
        
        # 5. BOUNDARY PROXIMITY (Agents too close to banks)
        if hasattr(self.sim, 'distance_to'):
            threshold = 2.0  # meters
            metrics['agents_near_boundary'] = np.sum((self.sim.distance_to < threshold) & alive_mask)
        
        return metrics
        
        best_reward = -np.inf
        best_weights = None
        
        print(f"Starting RL training: {num_episodes} episodes {timesteps_per_episode} timesteps")
        
        for episode in range(num_episodes):
            # Run episode with current weights
            episode_reward = self.train_episode(timesteps_per_episode)
            
            # Track history
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'weights': self.behavioral_weights.to_dict()
            })
            
            # Update best weights
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_weights = self.behavioral_weights.to_dict()
                print(f"Episode {episode}: New best reward = {episode_reward:.2f}")
                
                if save_path:
                    self.behavioral_weights.save(save_path)
            
            # Mutate weights for next episode (exploration)
            self.behavioral_weights.mutate(scale=0.05)
            
            # Reset simulation spatial state (keep behavioral weights)
            self.sim.reset_spatial_state()
            # Force garbage collection and clear any cached metrics to avoid memory growth between episodes
            try:
                import gc
                # Clear cached metrics if present
                if hasattr(self, '_cached_schooling_metrics'):
                    delattr(self, '_cached_schooling_metrics') if hasattr(self, '_cached_schooling_metrics') else None
                if hasattr(self, '_cached_drag_reductions'):
                    delattr(self, '_cached_drag_reductions') if hasattr(self, '_cached_drag_reductions') else None
                if hasattr(self.sim, '_last_drag_reductions'):
                    try:
                        del self.sim._last_drag_reductions
                    except Exception:
                        pass
                gc.collect()
            except Exception:
                pass
        
        # Load best weights
        if best_weights:
            self.behavioral_weights.from_dict(best_weights)
            print(f"Training complete. Best reward: {best_reward:.2f}")
        
        return self.behavioral_weights


# --- HECRAS IDW mapping helpers (read-only, KDTree cached) -----------------
class HECRASMap:
    """Lightweight container for a HECRAS plan KDTree and field values.

    Usage:
      m = HECRASMap(plan_path, field_name='Cells Minimum Elevation')
      vals = m.map_idw(query_pts, k=8)
    """
    def __init__(self, plan_path, field_names=None):
        self.plan_path = plan_path
        if field_names is None:
            field_names = ['Cells Minimum Elevation']
        # store requested field names (list)
        self.field_names = list(field_names)
        self._load_plan()

    def _find_dataset_by_name(self, hdf, name_pattern):
        """Search HDF recursively for a dataset whose name contains name_pattern (case-insensitive).

        Returns the dataset path if found, otherwise None.
        """
        name_pattern = name_pattern.lower()
        candidates = []

        def visitor(path, obj):
            if isinstance(obj, h5py.Dataset):
                p = path.lower()
                if name_pattern in p or name_pattern in obj.name.lower():
                    try:
                        shape = obj.shape
                    except Exception:
                        shape = None
                    candidates.append((path, shape))

        hdf.visititems(visitor)
        if not candidates:
            return None

        # prefer candidates that contain the coords length as one axis
        # n_coords is unknown here; we'll return best available: prefer Results paths
        # First, look for any candidate with 'results' in path
        results_cands = [c for c in candidates if 'results/' in c[0].lower()]
        if results_cands:
            return results_cands[0][0]
        # otherwise return the first candidate
        return candidates[0][0]

    def _load_plan(self):
        with h5py.File(self.plan_path, 'r') as h:
            coords = h['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]

            # load each requested field, attempting Geometry first then Results
            fields = {}
            for fname in self.field_names:
                geom_path = f'/Geometry/2D Flow Areas/2D area/{fname}'
                if geom_path in h:
                    arr = h[geom_path][:]
                else:
                    # try to find a dataset anywhere that contains fname
                    ds_path = self._find_dataset_by_name(h, fname)
                    if ds_path is not None:
                        ds = h[ds_path]
                        # if time series (first dim > 1), take last timestep
                        if ds.ndim > 1:
                            arr = ds[-1]
                        else:
                            arr = ds[:]
                    else:
                        raise KeyError(f"Field '{fname}' not found in HECRAS HDF: {self.plan_path}")
                fields[fname] = np.asarray(arr)

        # normalize field arrays to align with coords length
        n_coords = coords.shape[0]
        def normalize_field_array(arr):
            arr = np.asarray(arr)
            # if already 1D and matches
            if arr.ndim == 1 and arr.shape[0] == n_coords:
                return arr
            # if total size matches, reshape
            if arr.size == n_coords:
                return arr.reshape(n_coords,)
            # try to find axis with length == n_coords
            for axis, dim in enumerate(arr.shape):
                if dim == n_coords:
                    # build index tuple: select last on other axes
                    idx = []
                    for i in range(arr.ndim):
                        if i == axis:
                            idx.append(slice(None))
                        else:
                            idx.append(-1)
                    sliced = arr[tuple(idx)]
                    return np.asarray(sliced).reshape(n_coords,)
            # As a last resort, return nan-filled array
            return np.full((n_coords,), np.nan)

        # choose primary field (for valid-cell masking) as first field
        primary = self.field_names[0]
        # normalize all fields first
        normed = {k: normalize_field_array(v) for k, v in fields.items()}
        mask = np.isfinite(normed[primary])

        self.coords = coords[mask].astype(np.float64)
        # store each field masked and casted
        self.fields = {k: np.asarray(v[mask], dtype=np.float64) for k, v in normed.items()}
        # build KDTree once
        self.tree = cKDTree(self.coords)

    def map_idw(self, query_pts, k=8, eps=1e-8):
        """Map `query_pts` (N x 2) to a dict of field_name -> mapped values via IDW.

        Returns: dict where each key is a field name and value is a (N,) array.
        """
        query = np.asarray(query_pts, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, 2)
        dists, inds = self.tree.query(query, k=k)
        if k == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        inv = 1.0 / (dists + eps)
        w = inv / np.sum(inv, axis=1)[:, None]
        out = {}
        for fname, arr in self.fields.items():
            vals = arr[inds]
            mapped = np.sum(vals * w, axis=1)
            out[fname] = mapped
        return out


def infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=0.05, timestep=0):
    """Infer wetted perimeter from HECRAS depth at specified timestep.
    
    Algorithm:
    1. Load depth at timestep (default t=0)
    2. Apply threshold: wetted = depth > threshold
    3. Remove islands: dry regions completely surrounded by wetted cells (not touching boundary)
    4. Extract boundary edges between wetted and dry cells
    
    Parameters
    ----------
    plan_path : str
        Path to HECRAS plan HDF5 file
    depth_threshold : float
        Depth threshold in meters (default 0.05m)
    timestep : int
        Timestep index to use for depth (default 0)
    
    Returns
    -------
    dict with:
        'wetted_mask': (N,) boolean array - True for wetted cells
        'wetted_coords': (M, 2) array - coordinates of wetted cell centers
        'dry_coords': (P, 2) array - coordinates of dry cell centers
        'perimeter_points': (Q, 2) array - boundary points between wetted/dry
        'perimeter_cells': (Q,) array - indices of wetted cells on perimeter
    """
    import h5py
    from scipy.spatial import cKDTree
    from scipy.ndimage import label
    from shapely.geometry import Point, MultiPoint, Polygon
    from shapely.ops import unary_union
    
    with h5py.File(plan_path, 'r') as hdf:
        # Load cell centers
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:], dtype=np.float64)
        
        # Load depth at specified timestep
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][timestep], dtype=np.float32)
        
        # Load perimeter info for boundary detection
        try:
            perimeter = np.array(hdf['Geometry/2D Flow Areas/2D area/Perimeter'][:], dtype=np.float64)
        except Exception:
            perimeter = None
        
        # Load FacePoints for boundary detection
        try:
            facepoints = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'][:], dtype=np.float64)
            is_perim = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter'][:], dtype=np.int32)
        except Exception:
            facepoints = None
            is_perim = None
    
    # Step 1: Apply depth threshold
    wetted_mask = depth > depth_threshold
    dry_mask = ~wetted_mask
    
    print(f"Initial wetted cells: {wetted_mask.sum():,} ({wetted_mask.sum()/len(wetted_mask)*100:.1f}%)")
    print(f"Initial dry cells: {dry_mask.sum():,} ({dry_mask.sum()/len(dry_mask)*100:.1f}%)")
    
    # Step 2: Remove islands (dry regions not touching boundary)
    # Build KDTree for dry cells
    dry_coords = coords[dry_mask]
    if len(dry_coords) > 0 and perimeter is not None:
        # Create spatial graph of dry cells using connectivity
        dry_tree = cKDTree(dry_coords)
        
        # Find which dry cells are near the domain perimeter (boundary)
        # Use median cell spacing to determine connectivity
        sample_size = min(1000, len(coords))
        sample_idx = np.random.choice(len(coords), size=sample_size, replace=False)
        sample_coords = coords[sample_idx]
        sample_tree = cKDTree(sample_coords)
        dists, _ = sample_tree.query(sample_coords, k=2)
        median_spacing = np.median(dists[:, 1])
        
        # Find dry cells within 2x median spacing of perimeter
        boundary_search_dist = median_spacing * 3.0
        touching_boundary_mask = np.zeros(len(dry_coords), dtype=bool)
        
        for perim_pt in perimeter:
            dists = np.sqrt((dry_coords[:, 0] - perim_pt[0])**2 + (dry_coords[:, 1] - perim_pt[1])**2)
            touching_boundary_mask |= (dists < boundary_search_dist)
        
        print(f"Dry cells touching boundary: {touching_boundary_mask.sum():,}")
        
        # OPTIMIZATION: Skip island removal for large meshes (> 500k dry cells)
        # Island removal via connected components is O(N^2) and very slow
        if n_dry > 500000:
            print(f"   Skipping island removal (too many dry cells: {n_dry:,})")
            print(f"   (Island removal disabled for meshes with >500k dry cells)")
        else:
            # Label connected components of dry cells
            # Build adjacency graph for dry cells
            print("   Building dry cell connectivity graph...")
            dry_pairs = dry_tree.query_pairs(r=median_spacing * 1.5)
            print(f"   Found {len(dry_pairs):,} dry cell connections")
        
            # Use connected components to find islands
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            
            if len(dry_pairs) > 0:
                pairs_arr = np.array(list(dry_pairs))
                row = pairs_arr[:, 0]
                col = pairs_arr[:, 1]
                data = np.ones(len(row))
                # Make symmetric
                row_sym = np.concatenate([row, col])
                col_sym = np.concatenate([col, row])
                data_sym = np.concatenate([data, data])
                adj_matrix = csr_matrix((data_sym, (row_sym, col_sym)), shape=(n_dry, n_dry))
                
                # Find connected components
                n_components, labels = connected_components(adj_matrix, directed=False)
                
                print(f"Dry regions (connected components): {n_components}")
                
                # Identify which components touch the boundary
                components_touching_boundary = set()
                for comp_id in range(n_components):
                    comp_mask = labels == comp_id
                    if np.any(touching_boundary_mask[comp_mask]):
                        components_touching_boundary.add(comp_id)
                
                # Islands are components NOT touching boundary
                island_mask = np.zeros(n_dry, dtype=bool)
                for comp_id in range(n_components):
                    if comp_id not in components_touching_boundary:
                        island_mask |= (labels == comp_id)
                
                print(f"Island cells (dry but surrounded): {island_mask.sum():,}")
                
                # Convert islands from dry to wetted
                if island_mask.sum() > 0:
                    # Map back to original indices
                    dry_indices = np.where(dry_mask)[0]
                    island_indices_global = dry_indices[island_mask]
                    wetted_mask[island_indices_global] = True
                    dry_mask = ~wetted_mask
                    print(f"Updated wetted cells (after removing islands): {wetted_mask.sum():,}")
    
    # Step 3: Extract perimeter (boundary between wetted and dry)
    wetted_coords = coords[wetted_mask]
    dry_coords = coords[dry_mask]
    
    # Find wetted cells adjacent to dry cells (these are on the perimeter)
    if len(wetted_coords) > 0 and len(dry_coords) > 0:
        wetted_tree = cKDTree(wetted_coords)
        dry_tree = cKDTree(dry_coords)
        
        # For each wetted cell, check if any dry cell is within connectivity distance
        dists, _ = dry_tree.query(wetted_coords, k=1)
        perimeter_mask = dists < (median_spacing * 1.5)
        
        perimeter_coords = wetted_coords[perimeter_mask]
        perimeter_indices = np.where(wetted_mask)[0][perimeter_mask]
        
        print(f"Perimeter cells (wetted adjacent to dry): {len(perimeter_coords):,}")
    else:
        perimeter_coords = np.zeros((0, 2), dtype=np.float64)
        perimeter_indices = np.zeros(0, dtype=np.int32)
    
    return {
        'wetted_mask': wetted_mask,
        'wetted_coords': wetted_coords,
        'dry_coords': dry_coords,
        'perimeter_points': perimeter_coords,
        'perimeter_cells': perimeter_indices,
        'median_spacing': median_spacing if 'median_spacing' in locals() else None
    }


def compute_distance_to_bank_hecras(wetted_info, coords, median_spacing=None):
    """Compute distance-to-bank for HECRAS irregular mesh cells.
    
    Uses Dijkstra's algorithm on the mesh connectivity graph to compute
    geodesic distance from each wetted cell to the nearest perimeter cell.
    
    Parameters
    ----------
    wetted_info : dict
        Output from infer_wetted_perimeter_from_hecras()
    coords : ndarray (N, 2)
        All cell center coordinates
    median_spacing : float, optional
        Median cell spacing (computed if not provided)
    
    Returns
    -------
    ndarray (N,)
        Distance to bank for each cell (NaN for dry cells, 0 for perimeter cells)
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    
    wetted_mask = wetted_info['wetted_mask']
    perimeter_indices = wetted_info['perimeter_cells']
    
    if median_spacing is None:
        median_spacing = wetted_info.get('median_spacing')
    
    if median_spacing is None:
        # Compute median spacing
        sample_size = min(1000, len(coords))
        sample_idx = np.random.choice(len(coords), size=sample_size, replace=False)
        sample_coords = coords[sample_idx]
        sample_tree = cKDTree(sample_coords)
        dists, _ = sample_tree.query(sample_coords, k=2)
        median_spacing = np.median(dists[:, 1])
    
    # Build connectivity graph for wetted cells only
    wetted_coords = coords[wetted_mask]
    wetted_indices = np.where(wetted_mask)[0]
    n_wetted = len(wetted_coords)
    
    print(f"Building connectivity graph for {n_wetted:,} wetted cells...")
    
    # Build KDTree for wetted cells
    wetted_tree = cKDTree(wetted_coords)
    
    # Find neighbors within connectivity radius
    connectivity_radius = median_spacing * 1.5
    pairs = wetted_tree.query_pairs(r=connectivity_radius, output_type='ndarray')
    
    # Build sparse adjacency matrix with edge weights (Euclidean distances)
    if len(pairs) > 0:
        row = pairs[:, 0]
        col = pairs[:, 1]
        
        # Compute edge weights (distances)
        edge_coords_i = wetted_coords[row]
        edge_coords_j = wetted_coords[col]
        edge_dists = np.sqrt(np.sum((edge_coords_i - edge_coords_j)**2, axis=1))
        
        # Make symmetric
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        data_sym = np.concatenate([edge_dists, edge_dists])
        
        graph = csr_matrix((data_sym, (row_sym, col_sym)), shape=(n_wetted, n_wetted))
        
        print(f"Graph edges: {len(pairs):,}")
        
        # Map perimeter cell indices from global to wetted-only indexing
        perimeter_wetted_indices = []
        for perim_idx in perimeter_indices:
            # Find position in wetted_indices
            pos = np.where(wetted_indices == perim_idx)[0]
            if len(pos) > 0:
                perimeter_wetted_indices.append(pos[0])
        
        perimeter_wetted_indices = np.array(perimeter_wetted_indices, dtype=np.int32)
        
        print(f"Running Dijkstra from {len(perimeter_wetted_indices):,} perimeter cells...")
        
        # Run Dijkstra from all perimeter cells
        if len(perimeter_wetted_indices) > 0:
            dist_matrix = dijkstra(csgraph=graph, directed=False, indices=perimeter_wetted_indices)
            
            # dist_matrix shape: (num_perimeter, n_wetted)
            # Take minimum distance to any perimeter cell
            if dist_matrix.ndim == 2:
                distances_wetted = np.min(dist_matrix, axis=0)
            else:
                distances_wetted = dist_matrix
        else:
            # No perimeter cells (shouldn't happen)
            distances_wetted = np.full(n_wetted, np.inf)
    else:
        # No edges (shouldn't happen with reasonable connectivity radius)
        distances_wetted = np.full(n_wetted, np.inf)
    
    # Map back to full array (all cells)
    distances_all = np.full(len(coords), np.nan, dtype=np.float32)
    distances_all[wetted_indices] = distances_wetted.astype(np.float32)
    
    # Set perimeter cells to distance 0
    distances_all[perimeter_indices] = 0.0
    
    print(f"Distance-to-bank computed. Min: {np.nanmin(distances_all):.2f}, Max: {np.nanmax(distances_all):.2f}")
    
    return distances_all


def derive_centerline_from_hecras_distance(coords, distances, wetted_mask, crs=None, 
                                           min_distance_threshold=None, min_length=50):
    """Derive centerline from distance-to-bank field on irregular HECRAS mesh.
    
    Strategy:
    1. Find cells with maximum distance-to-bank (ridge/channel center)
    2. Order them by connectivity to form a line
    3. Smooth and convert to LineString
    
    Parameters
    ----------
    coords : ndarray (N, 2)
        All cell center coordinates
    distances : ndarray (N,)
        Distance-to-bank for each cell
    wetted_mask : ndarray (N,) bool
        Wetted cell mask
    crs : CRS, optional
        Coordinate reference system for output
    min_distance_threshold : float, optional
        Minimum distance-to-bank to be considered centerline (auto: 75th percentile)
    min_length : float
        Minimum centerline length in meters
    
    Returns
    -------
    LineString or None
        Main centerline (longest connected path)
    """
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge
    from scipy.ndimage import gaussian_filter1d
    
    # Filter to wetted cells with valid distances
    valid_mask = wetted_mask & np.isfinite(distances)
    valid_coords = coords[valid_mask]
    valid_distances = distances[valid_mask]
    
    if len(valid_coords) == 0:
        print("No valid wetted cells for centerline extraction")
        return None
    
    # Find ridge cells (high distance-to-bank)
    if min_distance_threshold is None:
        min_distance_threshold = np.percentile(valid_distances, 75)
    
    ridge_mask = valid_distances >= min_distance_threshold
    ridge_coords = valid_coords[ridge_mask]
    ridge_distances = valid_distances[ridge_mask]
    
    print(f"Ridge cells (distance >= {min_distance_threshold:.2f}m): {len(ridge_coords):,}")
    
    if len(ridge_coords) < 10:
        print("Too few ridge cells for centerline extraction")
        return None
    
    # Order ridge cells by connectivity (greedy nearest-neighbor path)
    # Start from cell with maximum distance
    start_idx = np.argmax(ridge_distances)
    ordered_indices = [start_idx]
    remaining = set(range(len(ridge_coords))) - {start_idx}
    
    current_idx = start_idx
    ridge_tree = cKDTree(ridge_coords)
    
    while remaining:
        # Find nearest unvisited neighbor
        current_pt = ridge_coords[current_idx]
        dists, indices = ridge_tree.query(current_pt, k=len(ridge_coords))
        
        # Find first index in remaining
        next_idx = None
        for idx in indices:
            if idx in remaining:
                next_idx = idx
                break
        
        if next_idx is None:
            break
        
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)
        current_idx = next_idx
    
    ordered_coords = ridge_coords[ordered_indices]
    
    # Smooth the path
    if len(ordered_coords) > 5:
        sigma = max(1, len(ordered_coords) // 20)  # Adaptive smoothing
        smoothed_x = gaussian_filter1d(ordered_coords[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(ordered_coords[:, 1], sigma=sigma)
        ordered_coords = np.column_stack((smoothed_x, smoothed_y))
    
    # Create LineString
    centerline = LineString(ordered_coords)
    
    print(f"Centerline extracted: {centerline.length:.2f}m long")
    
    if centerline.length < min_length:
        print(f"Centerline too short ({centerline.length:.2f}m < {min_length}m)")
        return None
    
    return centerline


def extract_centerline_fast_hecras(plan_path, depth_threshold=0.05, sample_fraction=0.1, min_length=50):
    """Fast centerline extraction from HECRAS by sampling wetted cells.
    
    Instead of computing full distance-to-bank field, this:
    1. Samples wetted cells (faster than all cells)
    2. Finds the longitudinal axis via PCA
    3. Orders points along that axis
    4. Smooths to create centerline
    
    Much faster than distance-field approach for large meshes.
    
    Parameters
    ----------
    plan_path : str
        Path to HECRAS plan HDF5
    depth_threshold : float
        Wetted threshold in meters
    sample_fraction : float
        Fraction of wetted cells to sample (0.05-0.2 recommended)
    min_length : float
        Minimum centerline length in meters
    
    Returns
    -------
    shapely.LineString or None
    """
    import h5py
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA
    
    print("FAST CENTERLINE EXTRACTION")
    
    with h5py.File(plan_path, 'r') as hdf:
        # Load geometry
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:], dtype=np.float64)
        
        # Get depth at t=0
        depth_path = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
        depth_data = hdf[depth_path]
        depth = depth_data[0, :]  # First timestep
    
    # Find wetted cells
    wetted_mask = depth > depth_threshold
    wetted_coords = coords[wetted_mask]
    
    print(f"   Wetted cells: {len(wetted_coords):,} / {len(coords):,} ({100*len(wetted_coords)/len(coords):.1f}%)")
    
    if len(wetted_coords) < 100:
        print("   ERROR: Too few wetted cells for centerline extraction")
        return None
    
    # Sample wetted cells for speed
    n_sample = max(500, int(len(wetted_coords) * sample_fraction))
    n_sample = min(n_sample, len(wetted_coords))
    sample_idx = np.random.choice(len(wetted_coords), size=n_sample, replace=False)
    sample_coords = wetted_coords[sample_idx]
    
    print(f"   Using {n_sample:,} sampled points for centerline")
    
    # Use PCA to find principal flow direction
    pca = PCA(n_components=2)
    pca.fit(sample_coords)
    
    # First principal component = longitudinal axis
    longitudinal_axis = pca.components_[0]
    
    # Project all sample points onto longitudinal axis
    centered = sample_coords - pca.mean_
    projections = centered @ longitudinal_axis
    
    # Sort points by projection (upstream to downstream)
    sorted_idx = np.argsort(projections)
    sorted_coords = sample_coords[sorted_idx]
    
    # Bin points along longitudinal axis and take median laterally
    n_bins = min(200, len(sorted_coords) // 5)
    bins = np.linspace(projections.min(), projections.max(), n_bins)
    bin_idx = np.digitize(projections[sorted_idx], bins)
    
    centerline_points = []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() > 0:
            # Median position in this bin
            bin_coords = sorted_coords[mask]
            centerline_points.append(np.median(bin_coords, axis=0))
    
    centerline_points = np.array(centerline_points)
    
    print(f"   Centerline points after binning: {len(centerline_points)}")
    
    if len(centerline_points) < 2:
        print("   ERROR: Not enough centerline points")
        return None
    
    # Smooth the centerline
    if len(centerline_points) > 5:
        sigma = max(1, len(centerline_points) // 20)
        smoothed_x = gaussian_filter1d(centerline_points[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(centerline_points[:, 1], sigma=sigma)
        centerline_points = np.column_stack((smoothed_x, smoothed_y))
    
    centerline = LineString(centerline_points)
    
    print(f"   Centerline length: {centerline.length:.2f}m")
    
    if centerline.length < min_length:
        print(f"   WARNING: Centerline too short ({centerline.length:.2f}m < {min_length}m)")
        return None
    
    return centerline


def initialize_hecras_geometry(simulation, plan_path, depth_threshold=0.05, crs=None, 
                                target_cell_size=None, create_rasters=True):
    """Complete HECRAS geometry initialization workflow in correct order.
    
    This function orchestrates the proper initialization sequence for HECRAS mode:
    1. Load HECRAS plan geometry and build KDTree
    2. Extract centerline (FAST method using PCA on wetted cells)
    3. Optionally compute distance-to-bank if needed
    4. Optionally create regular grid rasters
    
    Parameters
    ----------
    simulation : simulation object
        The simulation instance to initialize
    plan_path : str
        Path to HECRAS plan HDF5 file
    depth_threshold : float
        Wetted/dry threshold in meters (default 0.05m)
    crs : CRS, optional
        Coordinate reference system
    target_cell_size : float, optional
        Target cell size for regular grid (auto-detected if None)
    create_rasters : bool
        Whether to create regular grid rasters (default True)
    
    Returns
    -------
    dict with:
        'centerline': shapely LineString
        'coords': HECRAS cell coordinates
        'transform': Affine transform (if rasters created)
    """
    import h5py
    from scipy.spatial import cKDTree
    
    print("="*80)
    print("HECRAS GEOMETRY INITIALIZATION")
    print("="*80)
    
    # Step 1: Load HECRAS geometry and build KDTree
    print("\n1. Loading HECRAS plan and building KDTree...")
    with h5py.File(plan_path, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:], dtype=np.float64)
    
    # Create and cache HECRASMap
    fields = ['Cell Hydraulic Depth', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y', 'Water Surface']
    if not hasattr(simulation, '_hecras_maps'):
        simulation._hecras_maps = {}
    key = (str(plan_path), tuple(fields))
    if key not in simulation._hecras_maps:
        simulation._hecras_maps[key] = HECRASMap(str(plan_path), field_names=fields)
    hecras_map = simulation._hecras_maps[key]
    print(f"   Loaded {len(coords):,} HECRAS cells")
    
    # Step 2: Fast centerline extraction
    print("\n2. Extracting centerline...")
    centerline = extract_centerline_fast_hecras(
        plan_path,
        depth_threshold=depth_threshold,
        sample_fraction=0.1,  # Sample 10% of wetted cells
        min_length=50
    )
    
    if centerline is None:
        print("   WARNING: Failed to extract centerline!")

    # Step 2b: Infer wetted perimeter (vectorize raster boundary)
    try:
        print("\n2b. Inferring wetted perimeter (vectorizing)...")
        wetted_info = infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=depth_threshold, timestep=0)
        perimeter_points = wetted_info.get('perimeter_points', None)
        perimeter_cells = wetted_info.get('perimeter_cells', None)
        median_spacing = wetted_info.get('median_spacing', None)
        print(f"   Perimeter points: {0 if perimeter_points is None else len(perimeter_points):,}")
    except Exception:
        perimeter_points = None
        perimeter_cells = None
        median_spacing = None
    
    # Step 3: Optionally create regular grid rasters
    transform = None
    if create_rasters:
        print("\n3. Creating regular grid rasters...")
        
        # Compute affine transform from HECRAS cell spacing
        transform = compute_affine_from_hecras(coords, target_cell_size=target_cell_size)
        
        # Determine raster dimensions from extents
        minx, miny = coords[:, 0].min(), coords[:, 1].min()
        maxx, maxy = coords[:, 0].max(), coords[:, 1].max()
        
        cell_size = abs(transform.a)  # Assuming square cells
        width = int(np.ceil((maxx - minx) / cell_size))
        height = int(np.ceil((maxy - miny) / cell_size))
        
        print(f"   Grid dimensions: {height} x {width} at {cell_size:.2f}m resolution")
        
        # Create x_coords, y_coords in HDF5
        ensure_hdf_coords_from_hecras(simulation, plan_path, target_shape=(height, width), target_transform=transform)
        
        # Map initial HECRAS fields to rasters
        print("\n4. Mapping HECRAS fields to rasters...")
        map_hecras_to_env_rasters(simulation, plan_path, field_names=fields, k=1)  # k=1 for speed
    
    return {
        'centerline': centerline,
        'coords': coords,
        'transform': transform,
        'perimeter_points': perimeter_points,
        'perimeter_cells': perimeter_cells,
        'median_spacing': median_spacing
    }


# # End HECRAS helpers


def get_arr(use_gpu=False):
    '''
    Method to get the appropriate array module.
    '''
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except ImportError:
            return np
    return np


def map_hecras_for_agents(simulation, agent_xy, plan_path, field_names=None, k=8):
    """Map `agent_xy` (N x 2) to one or more HECRAS fields.

    Returns:
      - If one field requested: (N,) array
      - If multiple fields: dict field_name -> (N,) array
    """
    # Create and cache HECRASMap
    if not hasattr(simulation, '_hecras_maps'):
        simulation._hecras_maps = {}
    if field_names is None:
        field_names = ['Cells Minimum Elevation']
    key = (str(plan_path), tuple(field_names))
    if key not in simulation._hecras_maps:
        simulation._hecras_maps[key] = HECRASMap(str(plan_path), field_names=field_names)
    m = simulation._hecras_maps[key]
    out = m.map_idw(agent_xy, k=k)
    # if single field requested, return array directly
    if len(out) == 1:
        return next(iter(out.values()))
    return out


# Ensure HDF x_coords/y_coords exist when using HECRAS so code expecting raster-style arrays works
def ensure_hdf_coords_from_hecras(simulation, plan_path, target_shape=None, target_transform=None):
    """Create `x_coords` and `y_coords` datasets in the simulation HDF5 file when missing.

    - `plan_path` : path to HECRAS HDF used to compute affine
    - `target_shape` : (height, width) to shape outputs; if None, compute from HECRAS coords and transform
    - `target_transform` : affine to map cols/rows -> x/y; if None derive with compute_affine_from_hecras
    """
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        return

    # load coords from HECRAS plan
    try:
        with h5py.File(str(plan_path), 'r') as ph:
            hecras_coords = ph['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
    except Exception:
        hecras_coords = None

    if hecras_coords is None:
        return

    # derive transform and raster shape if not provided
    if target_transform is None:
        target_transform = compute_affine_from_hecras(hecras_coords)
    if target_shape is None:
        # attempt to infer reasonable grid extent and size using median spacing
        aff = target_transform
        # find bounds from coords
        minx, miny = float(hecras_coords[:, 0].min()), float(hecras_coords[:, 1].min())
        maxx, maxy = float(hecras_coords[:, 0].max()), float(hecras_coords[:, 1].max())
        # compute cols/rows
        width = max(1, int(np.ceil((maxx - minx) / abs(aff.a))))
        height = max(1, int(np.ceil((maxy - miny) / abs(aff.e))))
        target_shape = (height, width)

    height, width = target_shape
    
    # Skip creation if dimensions are invalid (0,0) - means rasters haven't been imported yet
    if height == 0 or width == 0:
        return

    # create datasets if missing
    if 'x_coords' not in hdf:
        dset_x = hdf.create_dataset('x_coords', (height, width), dtype='float32')
    else:
        dset_x = hdf['x_coords']
    if 'y_coords' not in hdf:
        dset_y = hdf.create_dataset('y_coords', (height, width), dtype='float32')
    else:
        dset_y = hdf['y_coords']

    # populate once if empty (all zeros or nan)
    try:
        existing = np.asarray(dset_x[:])
        # Check if uninit: all non-finite OR all zeros
        needs_populate = not np.isfinite(existing).any() or np.allclose(existing, 0.0)
    except Exception:
        needs_populate = True

    if needs_populate:
        # build col and row indices then map to geo using target_transform
        cols = np.arange(width, dtype=np.float64)
        rows = np.arange(height, dtype=np.float64)
        col_grid, row_grid = np.meshgrid(cols, rows)
        # pixel centers to geo
        xs, ys = pixel_to_geo(target_transform, row_grid, col_grid)
        dset_x[:, :] = xs.astype('float32')
        dset_y[:, :] = ys.astype('float32')

    # store transforms on simulation for later geotransform usage
    simulation.depth_rast_transform = target_transform
    simulation.hdf_height = height
    simulation.hdf_width = width


def map_hecras_to_env_rasters(simulation, plan_path, field_names=None, k=1):
    """Map HECRAS nodal fields onto the full environment raster grid and write
    them into `simulation.hdf5['environment']` datasets. This is called each
    timestep in HECRAS mode so legacy raster readers see a time-varying grid.

    - field_names: list of HECRAS field names to map; if None, use simulation.hecras_fields
    - k: number of nearest neighbors for IDW (default=1 for speed; use k>1 for smoother interpolation)
    """
    if not hasattr(simulation, 'hdf5') or getattr(simulation, 'hdf5', None) is None:
        return False
    if field_names is None:
        field_names = getattr(simulation, 'hecras_fields', None)

    # ensure x/y coords exist
    try:
        ensure_hdf_coords_from_hecras(simulation, plan_path, target_transform=getattr(simulation, 'depth_rast_transform', None))
    except Exception:
        pass

    env = simulation.hdf5.require_group('environment')

    # build flattened grid coordinates (cache on simulation)
    if not hasattr(simulation, '_hecras_grid_xy') or simulation._hecras_grid_xy is None:
        if 'x_coords' in simulation.hdf5 and 'y_coords' in simulation.hdf5:
            xarr = np.asarray(simulation.hdf5['x_coords'])
            yarr = np.asarray(simulation.hdf5['y_coords'])
            h, w = xarr.shape
            XX = xarr.flatten()
            YY = yarr.flatten()
            simulation._hecras_grid_shape = (h, w)
            simulation._hecras_grid_xy = np.column_stack((XX, YY))
        else:
            # fallback: derive from HECRAS coords directly
            try:
                m = load_hecras_plan_cached(simulation, plan_path, field_names=[field_names[0]] if field_names else None)
                coords = m.coords
                aff = compute_affine_from_hecras(coords)
                # create small raster extent based on coords and median spacing
                cell = abs(aff.a)
                minx = float(coords[:, 0].min())
                maxx = float(coords[:, 0].max())
                miny = float(coords[:, 1].min())
                maxy = float(coords[:, 1].max())
                w = max(1, int(np.ceil((maxx - minx) / cell)))
                h = max(1, int(np.ceil((maxy - miny) / cell)))
                cols = np.arange(w)
                rows = np.arange(h)
                colg, rowg = np.meshgrid(cols, rows)
                xs, ys = pixel_to_geo(aff, rowg, colg)
                simulation._hecras_grid_shape = (h, w)
                simulation._hecras_grid_xy = np.column_stack((xs.flatten(), ys.flatten()))
            except Exception:
                return False

    grid_xy = simulation._hecras_grid_xy
    h, w = simulation._hecras_grid_shape

    # load HECRAS map and map all requested fields for the whole grid
    try:
        m = load_hecras_plan_cached(simulation, plan_path, field_names=field_names if field_names else None)
    except Exception:
        m = None

    # if map couldn't be loaded, bail
    if m is None:
        return False

    # perform IDW mapping on grid (may be large)
    try:
        mapped = m.map_idw(grid_xy, k=k)
    except Exception:
        # per-field fallback: try mapping each field separately
        mapped = {}
        for fname in field_names:
            try:
                vals = m.map_idw(grid_xy, k=k)[fname]
            except Exception:
                vals = np.full((grid_xy.shape[0],), np.nan)
            mapped[fname] = vals

    # write mapped arrays to environment datasets; use normalized short names
    for fname, vals in mapped.items():
        short = fname.lower().replace(' ', '_').replace('/', '_')
        # choose sensible environment dataset name mapping
        if 'velocity' in short and ('x' in short or 'velocity_x' in short or 'vel_x' in short):
            dname = 'vel_x'
        elif 'velocity' in short and ('y' in short or 'velocity_y' in short or 'vel_y' in short):
            dname = 'vel_y'
        elif 'depth' in short or 'hydraulic_depth' in short:
            dname = 'depth'
        elif 'water_surface' in short or 'wsel' in short:
            dname = 'wsel'
        else:
            dname = short

        arr = np.asarray(vals).reshape((h, w))
        if dname not in env:
            env.create_dataset(dname, data=arr.astype('f4'), shape=(h, w), dtype='f4', chunks=(min(256, h), min(256, w)))
        else:
            try:
                env[dname][:, :] = arr.astype('f4')
            except Exception:
                # replace dataset if incompatible
                del env[dname]
                env.create_dataset(dname, data=arr.astype('f4'), shape=(h, w), dtype='f4', chunks=(min(256, h), min(256, w)))

    # compute vel_mag and vel_dir if vel_x/vel_y present
    try:
        vx = env['vel_x'][:]
        vy = env['vel_y'][:]
        mag = np.sqrt(vx * vx + vy * vy)
        dir_ = np.arctan2(vy, vx)
        if 'vel_mag' not in env:
            env.create_dataset('vel_mag', data=mag.astype('f4'), shape=(h, w), dtype='f4', chunks=(min(256, h), min(256, w)))
        else:
            env['vel_mag'][:, :] = mag.astype('f4')
        if 'vel_dir' not in env:
            env.create_dataset('vel_dir', data=dir_.astype('f4'), shape=(h, w), dtype='f4', chunks=(min(256, h), min(256, w)))
        else:
            env['vel_dir'][:, :] = dir_.astype('f4')
    except Exception:
        pass

    # flush HDF to make sure subsequent reads see latest values (caller may disable flushes)
    try:
        safe_flush(simulation.hdf5)
    except Exception:
        pass

    return True


# def map_hecras_to_env_rasters(simulation, plan_path, raster_names, k=8):
#     """Map HECRAS nodal fields onto the HDF5 `environment` rasters and write them.

#     This flattens the HDF `x_coords`/`y_coords` arrays to a list of points, runs
#     IDW mapping via the cached HECRASMap, reshapes back to grid, and writes into
#     `simulation.hdf5['environment'][name]` for each requested raster name.
#     """
#     hdf = getattr(simulation, 'hdf5', None)
#     if hdf is None:
#         return
#     if 'x_coords' not in hdf or 'y_coords' not in hdf:
#         # try to create coords
#         ensure_hdf_coords_from_hecras(simulation, plan_path, target_shape=(getattr(simulation,'height',None) or 0, getattr(simulation,'width',None) or 0), target_transform=getattr(simulation,'depth_rast_transform', None))
#     if 'x_coords' not in hdf or 'y_coords' not in hdf:
#         return

#     xs = np.asarray(hdf['x_coords'][:], dtype=float)
#     ys = np.asarray(hdf['y_coords'][:], dtype=float)
#     # flatten
#     flat_x = xs.ravel()
#     flat_y = ys.ravel()
#     pts = np.column_stack((flat_x, flat_y))

#     # map each requested raster name to a candidate HECRAS field
#     candidate_map = {
#         'depth': 'Cell Hydraulic Depth',
#         'vel_x': 'Cell Velocity - Velocity X',
#         'vel_y': 'Cell Velocity - Velocity Y',
#         'vel_mag': 'Velocity Magnitude',
#         'vel_dir': 'Velocity Direction',
#         'wetted': 'Wetted'
#     }

#     # ensure environment group exists
#     env = hdf.require_group('environment')

#     # perform IDW mapping for flat grid points
#     for rn in raster_names:
#         field = candidate_map.get(rn, None)
#         if field is None:
#             # create empty dataset if missing
#             if rn not in env:
#                 env.create_dataset(rn, shape=xs.shape, dtype='f4', fillvalue=np.nan)
#             continue
#         try:
#             mapped = map_hecras_for_agents(simulation, pts, plan_path, field_names=[field], k=k)
#             mapped = np.asarray(mapped).reshape(xs.shape)
#         except Exception:
#             mapped = np.full(xs.shape, np.nan, dtype=float)
#         # write into environment dataset (create if missing)
#         if rn not in env:
#             env.create_dataset(rn, shape=xs.shape, dtype='f4', fillvalue=np.nan)
#         try:
#             env[rn][:] = mapped.astype('f4')
#         except Exception:
#             try:
#                 # fallback slower write
#                 env[rn][...] = mapped.astype('f4')
#             except Exception:
#                 pass
#     try:
#         # if hdf is a file object, flush; if it's a group, try to get parent file
#         if hasattr(hdf, 'flush'):
#             hdf.flush()
#         else:
#             # try to get filename and open file to flush
#             fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
#             if fname:
#                 try:
#                     with h5py.File(fname, 'r+') as hw:
#                         try:
#                             hw.flush()
#                         except Exception:
#                             pass
#                 except Exception:
#                     pass
#     except Exception:
#         pass

# # End HECRAS helpers


def compute_affine_from_hecras(coords, target_cell_size=None):
    """Compute a conservative Affine transform from HECRAS cell center coordinates.

    Strategy:
    - Compute nearest-neighbor distances for a random subset of points and take the median spacing.
    - Use that spacing as the `x`/`y` pixel size (square cells).
    - Use the min-x and max-y of coords as the origin (upper-left corner), adjusting by half-cell.

    Returns an `Affine` suitable for rasterizing / geo_to_pixel mapping.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    n = coords.shape[0]
    # sample up to 2000 points for spacing calc
    sample_n = min(2000, n)
    idx = np.random.choice(n, size=sample_n, replace=False)
    sample = coords[idx]

    # build KDTree and get 2nd NN distances (first is zero/self)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        # fallback to slower scipy.spatial.KDTree if cKDTree not available
        from scipy.spatial import KDTree as cKDTree
    tree = cKDTree(coords)
    dists, _ = tree.query(sample, k=2)
    # second column are nearest neighbor distances
    nn = dists[:, 1]
    # median spacing
    median_spacing = float(np.median(nn))
    if target_cell_size is not None:
        # prefer requested target_cell_size if provided but don't exceed median spacing
        cell = float(target_cell_size)
    else:
        cell = max(median_spacing, 1e-6)

    minx = float(coords[:, 0].min())
    maxy = float(coords[:, 1].max())
    # use minx, maxy as upper-left corner but shift by half-cell to center cells
    origin_x = minx - 0.5 * cell
    origin_y = maxy + 0.5 * cell

    try:
        from rasterio.transform import Affine as _Affine
        AffineLocal = _Affine
    except Exception:
        try:
            from affine import Affine as _Affine
            AffineLocal = _Affine
        except Exception:
            # last resort: build a simple stand-in
            class AffineLocal:
                def __init__(self, a, b, c, d, e, f):
                    self.a = a; self.b = b; self.c = c; self.d = d; self.e = e; self.f = f
                def __invert__(self):
                    raise RuntimeError('Affine inverse not available')
    return AffineLocal(cell, 0.0, origin_x, 0.0, -cell, origin_y)


def derive_centerline_from_distance_raster(distance_rast, transform=None, crs=None, footprint_size=5, min_length=50):
    """Derive centerline LineString(s) from a distance-to-edge raster.

    Returns the main LineString (longest) or None, and a list of all LineStrings.
    """
    from scipy.ndimage import maximum_filter
    from skimage.morphology import skeletonize
    from skimage.measure import label
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    if distance_rast is None or distance_rast.size == 0:
        return None, []

    # Find local maxima (ridge detection) - exact copy from visualization script
    local_max = maximum_filter(distance_rast, size=footprint_size)
    is_ridge = (distance_rast == local_max) & (distance_rast > 0.5)

    # Skeletonize
    skeleton = skeletonize(is_ridge)

    # Convert skeleton to LineString(s)
    labeled = label(skeleton, connectivity=2)

    centerlines = []
    for region_id in range(1, int(labeled.max()) + 1):
        region_mask = (labeled == region_id)
        ys, xs = np.where(region_mask)
        if len(xs) < 5:
            continue
        world_coords = []
        if transform is not None:
            for i in range(len(xs)):
                xw, yw = transform * (xs[i], ys[i])
                world_coords.append((xw, yw))
        else:
            for i in range(len(xs)):
                world_coords.append((float(xs[i]), float(ys[i])))
        if len(world_coords) >= 2:
            line = LineString(world_coords)
            centerlines.append(line)

    if not centerlines:
        return None, []

    merged = linemerge(centerlines)
    if isinstance(merged, LineString):
        main_centerline = merged
        all_lines = [merged]
    elif isinstance(merged, MultiLineString):
        all_lines = list(merged.geoms)
        main_centerline = max(all_lines, key=lambda g: g.length) if all_lines else None
    else:
        return None, centerlines

    if main_centerline is not None and main_centerline.length >= min_length:
        return main_centerline, all_lines
    return None, all_lines

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

font = {'family': 'serif','size': 6}
from matplotlib import rcParams
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'
rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'


def get_arr(use_gpu):
    '''
    Parameters
    ----------
    use_gpu : Boolean
        AI generated function that returns a CuPy array if true, otherwise it 
        returns a Numpy array.  

    Returns
    -------
    Array
        a CuPy or Numpy array.

    '''
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except:
            print("CuPy not found. Falling back to Numpy.")
            import numpy as np
            return np
    else:
        import numpy as np
        return np

# ... other utility functions ...

def geo_to_pixel(X, Y, transform):
    """
    Convert x, y coordinates to row, column indices in the raster grid.
    This function inverts the provided affine transform to convert geographic
    coordinates to pixel coordinates.

    Parameters:
    - X: array-like of x coordinates (longitude or projected x)
    - Y: array-like of y coordinates (latitude or projected y)
    - transform: affine transform of the raster

    Returns:
    - rows: array of row indices
    - cols: array of column indices
    """
    # Try to use vectorized affine math if transform exposes coefficients
    try:
        inv = get_inv_transform(getattr(transform, '__self__', None) or globals().get('sim', None), transform)
        # inv is an Affine; compute cols, rows via inv.c + inv.a*(x+0.5) + inv.b*(y+0.5)
        xs = np.asarray(X, dtype=float)
        ys = np.asarray(Y, dtype=float)
        # Affine multiplication for inverse: (col, row) = inv * (x, y)
        cols = inv.c + inv.a * (xs + 0.0) + inv.b * (ys + 0.0)
        rows = inv.f + inv.d * (xs + 0.0) + inv.e * (ys + 0.0)
        rows = np.rint(rows).astype(int)
        cols = np.rint(cols).astype(int)
        return rows, cols
    except Exception:
        # Fallback to per-point multiplication
        inv_transform = ~transform
        pixels = [inv_transform * (x, y) for x, y in zip(X, Y)]
        cols, rows = zip(*pixels)
        rows = np.round(rows).astype(int)
        cols = np.round(cols).astype(int)
        return rows, cols

def geo_to_pixel_from_inv(inv, X, Y):
    """Convert coordinates to pixel indices using precomputed inverse affine `inv`.

    `inv` is expected to have attributes a,b,c,d,e,f (an Affine object).
    """
    xs = np.asarray(X, dtype=float)
    ys = np.asarray(Y, dtype=float)
    cols = inv.c + inv.a * (xs + 0.0) + inv.b * (ys + 0.0)
    rows = inv.f + inv.d * (xs + 0.0) + inv.e * (ys + 0.0)
    return np.rint(rows).astype(int), np.rint(cols).astype(int)


def get_inv_transform(sim, transform):
    """Return cached inverse affine for `transform` on `sim`.

    Caches by id(transform) to avoid repeated Affine inversion costs.
    """
    try:
        key = id(transform)
    except Exception:
        return ~transform
    cache = getattr(sim, '_inv_transform_cache', None)
    if cache is None:
        cache = {}
        sim._inv_transform_cache = cache
    inv = cache.get(key)
    if inv is None:
        try:
            inv = ~transform
        except Exception:
            # best-effort: return direct invertible object
            return ~transform
        cache[key] = inv
    return inv


def safe_flush(hdf):
    """Safely flush an h5py file or group.

    Attempts to call `.flush()` on the object if present, otherwise tries
    to obtain a file object or filename and flush that file. Swallows
    all exceptions to avoid interrupting callers.
    """
    try:
        if hasattr(hdf, 'flush'):
            try:
                hdf.flush()
                return
            except Exception:
                pass
        # h5py Group has .file attribute referencing the File object
        fobj = getattr(hdf, 'file', None)
        if fobj is not None and hasattr(fobj, 'flush'):
            try:
                fobj.flush()
                return
            except Exception:
                pass
        # fallback: try to open by filename and flush
        fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
        if fname:
            try:
                with h5py.File(fname, 'r+') as hw:
                    try:
                        hw.flush()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass


# --- Performance helpers for simulation math ---
try:
    from numba import njit, prange
    import math
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _compute_drags_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
        n = fx.shape[0]
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                continue
            rvx = fx[i] - wx[i]
            rvy = fy[i] - wy[i]
            rel = math.sqrt(rvx * rvx + rvy * rvy)
            if rel < 1e-6:
                rel = 1e-6
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            # drag scalar prefactor
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = math.sqrt(dx * dx + dy * dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return drags


def _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    # vectorized numpy implementation mirroring the previous logic
    relative_velocities_x = fx - wx
    relative_velocities_y = fy - wy
    rel_norms = np.sqrt(relative_velocities_x ** 2 + relative_velocities_y ** 2)
    rel_norms_safe = np.maximum(rel_norms, 1e-6)
    unit_x = relative_velocities_x / rel_norms_safe
    unit_y = relative_velocities_y / rel_norms_safe
    relsq = rel_norms ** 2
    pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
    dx = pref * unit_x
    dy = pref * unit_y
    drags = np.stack((dx, dy), axis=1)
    # clip excessive drags for holding behavior
    drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
    mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
    if np.any(mask_excess):
        scales = 5.0 / drag_mags[mask_excess]
        drags[mask_excess, 0] *= scales
        drags[mask_excess, 1] *= scales
    # apply agent mask
    drags[~mask] = 0.0
    return drags


def compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    if _HAS_NUMBA:
        # ensure fixed dtypes and contiguous memory to avoid numba recompilation and hidden copies
        fx = np.ascontiguousarray(fx, dtype=np.float64)
        fy = np.ascontiguousarray(fy, dtype=np.float64)
        wx = np.ascontiguousarray(wx, dtype=np.float64)
        wy = np.ascontiguousarray(wy, dtype=np.float64)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        surface_areas = np.ascontiguousarray(surface_areas, dtype=np.float64)
        drag_coeffs = np.ascontiguousarray(drag_coeffs, dtype=np.float64)
        wave_drag = np.ascontiguousarray(wave_drag, dtype=np.float64)
        swim_behav = np.ascontiguousarray(swim_behav, dtype=np.int64)
        return _compute_drags_numba(fx, fy, wx, wy, mask, float(density), surface_areas, drag_coeffs, wave_drag, swim_behav)
    else:
        return _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)

# Numba wrapper for higher-level fatigue/behavior orchestration
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        # compute swim speeds
        for i in prange(n):
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            swim_speeds[i] = math.hypot(rx, ry)
        # compute bl/s
        bl_s = np.empty(n, dtype=np.float64)
        for i in prange(n):
            bl_s[i] = swim_speeds[i] / (1.0 if 0 else 1.0)
        # mask categories
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            prolonged[i] = (max_s_U < bl_s[i]) and (bl_s[i] <= max_p_U)
            sprint[i] = bl_s[i] > max_p_U
            sustained[i] = bl_s[i] <= max_s_U
        # write swim speeds into circular buffer last slot
        for i in prange(n):
            swim_speeds_buf[i, -1] = swim_speeds[i]
        return swim_speeds, bl_s, prolonged, sprint, sustained
else:
    def _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
        swim_speeds = np.sqrt((sog * np.cos(heading) - x_vel) ** 2 + (sog * np.sin(heading) - y_vel) ** 2)
        bl_s = swim_speeds / (1.0 if 0 else 1.0)
        prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
        sprint = bl_s > max_p_U
        sustained = bl_s <= max_s_U
        swim_speeds_buf[:, -1] = swim_speeds
        return swim_speeds, bl_s, prolonged, sprint, sustained


# Merged kernel: compute swim speeds, fatigue masks, and drags in one pass
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _merged_swim_drag_fatigue_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        bl_s = np.empty(n, dtype=np.float64)
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                swim_speeds[i] = 0.0
                bl_s[i] = 0.0
                prolonged[i] = False
                sprint[i] = False
                sustained[i] = False
                drags[i, 0] = 0.0
                drags[i, 1] = 0.0
                continue
            # fish velocity components
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            # relative to water
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            s = math.hypot(rx, ry)
            swim_speeds[i] = s
            # preserve existing bl_s semantics (previous implementation used a placeholder)
            bl_s[i] = s
            # fatigue masks
            prolonged[i] = (max_s_U[i] < bl_s[i]) and (bl_s[i] <= max_p_U[i])
            sprint[i] = bl_s[i] > max_p_U[i]
            sustained[i] = bl_s[i] <= max_s_U[i]
            # write into circular buffer last slot
            swim_speeds_buf[i, -1] = swim_speeds[i]

            # compute drag (same as _compute_drags_numba)
            rvx = fx - x_vel[i]
            rvy = fy - y_vel[i]
            rel = math.hypot(rvx, rvy)
            if rel < 1e-6:
                rel = 1e-6
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = math.hypot(dx, dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags
else:
    def _merged_swim_drag_fatigue_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
        # numpy fallback implementing the same logic in vectorized form
        fx = sog * np.cos(heading)
        fy = sog * np.sin(heading)
        rx = fx - x_vel
        ry = fy - y_vel
        swim_speeds = np.sqrt(rx * rx + ry * ry)
        bl_s = swim_speeds.copy()
        prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
        sprint = bl_s > max_p_U
        sustained = bl_s <= max_s_U
        # write into last slot
        swim_speeds_buf[:, -1] = swim_speeds
        rel = np.maximum(np.sqrt((fx - x_vel) ** 2 + (fy - y_vel) ** 2), 1e-6)
        unitx = (fx - x_vel) / rel
        unity = (fy - y_vel) / rel
        relsq = rel * rel
        pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
        dx = pref * unitx
        dy = pref * unity
        drags = np.stack((dx, dy), axis=1)
        mask_arr = np.asarray(mask, dtype=np.bool_)
        # clip excessive drags for holding behavior
        drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
        mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
        if np.any(mask_excess):
            scales = 5.0 / drag_mags[mask_excess]
            drags[mask_excess, 0] *= scales
            drags[mask_excess, 1] *= scales
        drags[~mask_arr] = 0.0
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags

# Wrap drag_fun to call compute_drags quickly
if _HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        # copy into out
        for i in prange(dr.shape[0]):
            out[i,0] = dr[i,0]
            out[i,1] = dr[i,1]
        return out
else:
    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        out[:,0] = dr[:,0]
        out[:,1] = dr[:,1]
        return out

# --- Additional Numba helpers for other hotspots ---
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _project_points_onto_line_numba(xs_line, ys_line, px, py):
        # compute segment vectors
        S = xs_line.size - 1
        seg_x0 = xs_line[:S]
        seg_y0 = ys_line[:S]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.empty(S, dtype=np.float64)
        for j in range(S):
            seg_len[j] = math.hypot(vx[j], vy[j])
        cumlen = np.empty(S + 1, dtype=np.float64)
        cumlen[0] = 0.0
        for j in range(S):
            cumlen[j + 1] = cumlen[j] + seg_len[j]

        M = px.size
        out = np.empty(M, dtype=np.float64)
        for i in prange(M):
            best_d2 = 1e308
            best_dist = 0.0
            xi = px[i]
            yi = py[i]
            for j in range(S):
                x0 = seg_x0[j]
                y0 = seg_y0[j]
                dx = vx[j]
                dy = vy[j]
                denom = dx * dx + dy * dy
                if denom == 0.0:
                    t = 0.0
                else:
                    t = ((xi - x0) * dx + (yi - y0) * dy) / denom
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                cx = x0 + t * dx
                cy = y0 + t * dy
                d2 = (xi - cx) * (xi - cx) + (yi - cy) * (yi - cy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_dist = cumlen[j] + t * seg_len[j]
            out[i] = best_dist
        return out
else:
    def _project_points_onto_line_numba(xs_line, ys_line, px, py):
        # fallback to numpy implementation
        seg_x0 = xs_line[:-1]
        seg_y0 = ys_line[:-1]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.hypot(vx, vy)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
        M = px.size
        px_e = px[:, None]
        py_e = py[:, None]
        x0_e = seg_x0[None, :]
        y0_e = seg_y0[None, :]
        vx_e = vx[None, :]
        vy_e = vy[None, :]
        wx = px_e - x0_e
        wy = py_e - y0_e
        denom = vx_e * vx_e + vy_e * vy_e
        denom = np.where(denom == 0, 1e-12, denom)
        t = (wx * vx_e + wy * vy_e) / denom
        t_clamped = np.clip(t, 0.0, 1.0)
        cx = x0_e + t_clamped * vx_e
        cy = y0_e + t_clamped * vy_e
        d2 = (px_e - cx) ** 2 + (py_e - cy) ** 2
        idx = np.argmin(d2, axis=1)
        chosen_t = t_clamped[np.arange(M), idx]
        chosen_seg = idx
        distances_along = cumlen[chosen_seg] + chosen_t * seg_len[chosen_seg]
        return distances_along

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _swim_speeds_numba(x_vel, y_vel, sog, heading):
        n = sog.size
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            out[i] = math.hypot(rx, ry)
        return out
else:
    def _swim_speeds_numba(x_vel, y_vel, sog, heading):
        fish_velocities_x = sog * np.cos(heading)
        fish_velocities_y = sog * np.sin(heading)
        relx = fish_velocities_x - x_vel
        rely = fish_velocities_y - y_vel
        return np.sqrt(relx * relx + rely * rely)

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _swim_speeds_numba_v2(x_vel, y_vel, sog, cos_h, sin_h, out):
        n = sog.size
        for i in prange(n):
            fx = sog[i] * cos_h[i]
            fy = sog[i] * sin_h[i]
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            out[i] = math.hypot(rx, ry)
        return out
else:
    def _swim_speeds_numba_v2(x_vel, y_vel, sog, cos_h, sin_h, out):
        fx = sog * cos_h
        fy = sog * sin_h
        relx = fx - x_vel
        rely = fy - y_vel
        out[:] = np.sqrt(relx * relx + rely * rely)
        return out

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _calc_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        n = battery.size
        # apply sustained recovery
        for i in prange(n):
            if mask_sustained[i]:
                battery[i] = battery[i] + per_rec[i]
        # non-sustained: scale battery by remaining ttf
        for i in prange(n):
            if not mask_sustained[i]:
                ttf0 = ttf[i] * battery[i]
                if ttf0 <= 0.0:
                    battery[i] = 0.0
                else:
                    ttf1 = ttf0 - dt
                    ratio = ttf1 / ttf0
                    if ratio < 0.0:
                        ratio = 0.0
                    battery[i] = battery[i] * ratio
        # clip
        for i in prange(n):
            if battery[i] < 0.0:
                battery[i] = 0.0
            elif battery[i] > 1.0:
                battery[i] = 1.0
        return battery
else:
    def _calc_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        # numpy fallback
        battery = battery.copy()
        battery[mask_sustained] += per_rec[mask_sustained]
        mask_non = ~mask_sustained
        ttf0 = ttf[mask_non] * battery[mask_non]
        ttf1 = ttf0 - dt
        # avoid division by zero
        safe = ttf0 != 0
        ratio = np.ones_like(ttf0)
        ratio[safe] = np.maximum(0.0, ttf1[safe] / ttf0[safe])
        battery[mask_non] = battery[mask_non] * ratio
        np.clip(battery, 0.0, 1.0, out=battery)
        return battery

# Single-pass optimized battery update kernel (merged / faster variant)
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        n = battery.size
        for i in prange(n):
            b = battery[i]
            if mask_sustained[i]:
                b = b + per_rec[i]
            else:
                t0 = ttf[i] * b
                if t0 <= 0.0:
                    b = 0.0
                else:
                    t1 = t0 - dt
                    ratio = t1 / t0
                    if ratio < 0.0:
                        ratio = 0.0
                    b = b * ratio
            # clip
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0
            battery[i] = b
        return battery
else:
    def _merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        # fallback to numpy implementation (single-pass)
        battery = battery.copy()
        for i in range(battery.size):
            if mask_sustained[i]:
                battery[i] = battery[i] + per_rec[i]
            else:
                t0 = ttf[i] * battery[i]
                if t0 <= 0.0:
                    battery[i] = 0.0
                else:
                    t1 = t0 - dt
                    ratio = t1 / t0
                    if ratio < 0.0:
                        ratio = 0.0
                    battery[i] = battery[i] * ratio
        np.clip(battery, 0.0, 1.0, out=battery)
        return battery

# --- Fatigue helpers: bout distance and time-to-fatigue ---
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _bout_distance_numba(prev_X, X, prev_Y, Y):
        n = prev_X.shape[0]
        dist = np.empty(n, dtype=np.float64)
        for i in prange(n):
            dx = prev_X[i] - X[i]
            dy = prev_Y[i] - Y[i]
            dist[i] = math.sqrt(dx * dx + dy * dy)
        return dist

    @njit(parallel=True, cache=True)
    def _time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
        n = swim_speeds.shape[0]
        ttf = np.empty(n, dtype=np.float64)
        for i in prange(n):
            ttf[i] = np.nan
            s = swim_speeds[i]
            if mask_prolonged[i]:
                ttf[i] = math.exp(a_p + s * b_p)
            if mask_sprint[i]:
                ttf[i] = math.exp(a_s + s * b_s)
        return ttf
else:
    def _bout_distance_numba(prev_X, X, prev_Y, Y):
        dx = prev_X - X
        dy = prev_Y - Y
        return np.sqrt(dx * dx + dy * dy)

    def _time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
        ttf = np.full_like(swim_speeds, np.nan, dtype=float)
        ttf = np.where(mask_prolonged, np.exp(a_p + swim_speeds * b_p), ttf)
        ttf = np.where(mask_sprint, np.exp(a_s + swim_speeds * b_s), ttf)
        return ttf

# Precompile numba functions at import time to avoid first-call JIT overhead
if _HAS_NUMBA:
    try:
        n = 8
        dummy = np.zeros(n, dtype=np.float64)
        _ = _compute_drags_numba(dummy, dummy, dummy, dummy, np.ones(n, dtype=np.bool_), 1.0, np.ones(n), np.ones(n), np.ones(n), np.zeros(n, dtype=np.int64))
        _ = _bout_distance_numba(dummy, dummy, dummy, dummy)
        _ = _time_to_fatigue_numba(dummy, np.ones(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), 0.0, 0.0, 0.0, 0.0)
        _ = _project_points_onto_line_numba(dummy, dummy, dummy, dummy)
        _ = _swim_speeds_numba(dummy, dummy, dummy, dummy)
        _ = _calc_battery_numba(dummy, dummy, dummy, np.ones(n, dtype=np.bool_), 0.1)
    except Exception:
        pass

    # helper to ensure compilation completes at import time
    def _numba_warmup(m=None):
        try:
            if m is None:
                m = max(64, n)
            else:
                m = int(m)
            d = np.zeros(m, dtype=np.float64)
            b = np.ones(m, dtype=np.bool_)
            bi = np.zeros(m, dtype=np.int64)
            _ = _compute_drags_numba(d, d, d, d, b, 1.0, np.ones(m, dtype=np.float64), np.ones(m, dtype=np.float64), np.ones(m, dtype=np.float64), bi)
            _ = _bout_distance_numba(d, d, d, d)
            _ = _time_to_fatigue_numba(d, b, np.zeros(m, dtype=np.bool_), 0.0, 0.0, 0.0, 0.0)
            _ = _project_points_onto_line_numba(d, d, d, d)
            _ = _swim_speeds_numba(d, d, d, d)
            _ = _calc_battery_numba(np.ones(m, dtype=np.float64), d, d, b, 0.1)
            _ = _swim_core_numba(d, d, d, d, d, d, np.zeros(m, dtype=np.bool_), np.zeros(m, dtype=np.bool_), b, 0.1)
        except Exception:
            pass

    # attempt warmup (may still spend time but at import not during timed loop)
    try:
        _numba_warmup()
    except Exception:
        pass

    def _numba_warmup_for_sim(sim):
        """Warm Numba kernels using arrays shaped to the given simulation instance.

        This calls `_numba_warmup` with a large `m` and then invokes a small set of
        kernels using arrays shaped exactly like `sim.num_agents` and `sim.swim_speeds`.
        """
        try:
            n = max(1024, int(getattr(sim, 'num_agents', 128)))
            _numba_warmup(m=n)
            # prepare exact-shape arrays
            na = int(getattr(sim, 'num_agents', n))
            max_ts = int(getattr(sim, 'swim_speeds', np.zeros((na,1))).shape[1])
            ones = np.ones(na, dtype=np.float64)
            zeros = np.zeros(na, dtype=np.float64)
            bmask = np.ones(na, dtype=np.bool_)
            bi = np.zeros(na, dtype=np.int64)
            # exact-shape warmups
            try:
                _compute_drags_numba(ones, ones, ones, ones, bmask, 1.0, ones, ones, ones, bi)
            except Exception:
                pass
            try:
                _swim_speeds_numba(ones, ones, ones, ones)
            except Exception:
                pass
            try:
                # swim_speeds buffer shaped (na, max_ts)
                buf = np.zeros((na, max_ts), dtype=np.float64)
                _assess_fatigue_core(ones, ones, ones, ones, ones, ones, ones, buf)
            except Exception:
                pass
        except Exception:
            pass


# --- Swim numeric core helpers ---
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
        n = fv0x.shape[0]
        dxdy = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i] or dead_mask[i]:
                continue
            vx = fv0x[i] + accx[i] * dt
            vy = fv0y[i] + accy[i] * dt
            if not tired_mask[i]:
                vx += pidx[i]
                vy += pidy[i]
            dxdy[i, 0] = vx * dt
            dxdy[i, 1] = vy * dt
        return dxdy
else:
    def _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
        vx = fv0x + accx * dt
        vy = fv0y + accy * dt
        # apply pid adjustment where not tired
        vx = np.where(~tired_mask, vx + pidx, vx)
        vy = np.where(~tired_mask, vy + pidy, vy)
        # zero dead or masked agents
        vx = np.where((~mask) | dead_mask, 0.0, vx)
        vy = np.where((~mask) | dead_mask, 0.0, vy)
        return np.stack((vx * dt, vy * dt), axis=1)

def pixel_to_geo(transform, rows, cols):
    """
    Convert row, column indices in the raster grid to x, y coordinates.

    Parameters:
    - transform: affine transform of the raster
    - rows: array-like or scalar of row indices
    - cols: array-like or scalar of column indices

    Returns:
    - xs: array of x coordinates
    - ys: array of y coordinates
    """
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)

    return xs, ys

# --- Safety wrappers to ensure stable Numba specializations ---
def _wrap_project_points_onto_line_numba(xs_line, ys_line, px, py):
    xs = np.ascontiguousarray(xs_line, dtype=np.float64)
    ys = np.ascontiguousarray(ys_line, dtype=np.float64)
    pxx = np.ascontiguousarray(px, dtype=np.float64)
    pyy = np.ascontiguousarray(py, dtype=np.float64)
    return _project_points_onto_line_numba(xs, ys, pxx, pyy)

def _wrap_drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out=None):
    fx_a = np.ascontiguousarray(fx, dtype=np.float64)
    fy_a = np.ascontiguousarray(fy, dtype=np.float64)
    wx_a = np.ascontiguousarray(wx, dtype=np.float64)
    wy_a = np.ascontiguousarray(wy, dtype=np.float64)
    mask_a = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_), dtype=np.bool_)
    sa = np.ascontiguousarray(surface_areas, dtype=np.float64)
    dc = np.ascontiguousarray(drag_coeffs, dtype=np.float64)
    wd = np.ascontiguousarray(wave_drag, dtype=np.float64)
    if out is None:
        out = np.zeros((fx_a.size, 2), dtype=np.float64)
    else:
        out = np.ascontiguousarray(out, dtype=np.float64)
    return _drag_fun_numba(fx_a, fy_a, wx_a, wy_a, mask_a, float(density), sa, dc, wd, np.ascontiguousarray(np.asarray(swim_behav, dtype=np.int64)), out)

def _wrap_merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
    batt = np.ascontiguousarray(battery, dtype=np.float64)
    perr = np.ascontiguousarray(per_rec, dtype=np.float64)
    ttf_a = np.ascontiguousarray(ttf, dtype=np.float64)
    mask_a = np.ascontiguousarray(np.asarray(mask_sustained, dtype=np.bool_), dtype=np.bool_)
    return _merged_battery_numba(batt, perr, ttf_a, mask_a, float(dt))

# Optional merged drag + battery kernel (single-pass). Not wired automatically; available for experiments.
if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery, per_rec, ttf, dt, update_battery):
        n = sog.size
        # outputs
        swim_speeds = np.empty(n, dtype=np.float64)
        bl_s = np.empty(n, dtype=np.float64)
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                swim_speeds[i] = 0.0
                bl_s[i] = 0.0
                prolonged[i] = False
                sprint[i] = False
                sustained[i] = False
                drags[i, 0] = 0.0
                drags[i, 1] = 0.0
                continue
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            relx = fx - x_vel[i]
            rely = fy - y_vel[i]
            rel = math.hypot(relx, rely)
            if rel < 1e-12:
                rel = 1e-12
            swim_speeds[i] = rel
            bl_s[i] = rel
            # masks (caller should compute thresholds externally if needed)
            # Here we leave it to the caller to interpret bl_s against thresholds
            prolonged[i] = False
            sprint[i] = False
            sustained[i] = False
            unitx = relx / rel
            unity = rely / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            # clip for holding behavior
            if swim_behav[i] == 3:
                mag = math.hypot(dx, dy)
                if mag > 5.0:
                    scale = 5.0 / mag
                    dx *= scale
                    dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
            # battery update if requested
            if update_battery:
                b = battery[i]
                # determine sustained by per_rec>0 as a proxy
                if per_rec is not None and per_rec.size == n and per_rec[i] > 0.0:
                    b = b + per_rec[i]
                else:
                    t0 = ttf[i] * b
                    if t0 <= 0.0:
                        b = 0.0
                    else:
                        t1 = t0 - dt
                        ratio = t1 / t0
                        if ratio < 0.0:
                            ratio = 0.0
                        b = b * ratio
                # clip
                if b < 0.0:
                    b = 0.0
                elif b > 1.0:
                    b = 1.0
                battery[i] = b
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags, battery
else:
    def _drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery, per_rec, ttf, dt, update_battery):
        # fallback numpy implementation
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        bl_s = np.empty(n, dtype=np.float64)
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        fx = sog * np.cos(heading)
        fy = sog * np.sin(heading)
        relx = fx - x_vel
        rely = fy - y_vel
        rel = np.sqrt(relx * relx + rely * rely)
        rel = np.where(rel == 0, 1e-12, rel)
        swim_speeds[:] = rel
        bl_s[:] = rel
        unitx = relx / rel
        unity = rely / rel
        relsq = rel * rel
        pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
        dx = pref * unitx
        dy = pref * unity
        drags = np.stack((dx, dy), axis=1)
        mask_arr = np.asarray(mask, dtype=np.bool_)
        drag_mags = np.sqrt(drags[:,0]**2 + drags[:,1]**2)
        mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
        if np.any(mask_excess):
            scales = 5.0 / drag_mags[mask_excess]
            drags[mask_excess,0] *= scales
            drags[mask_excess,1] *= scales
        drags[~mask_arr] = 0.0
        batt = battery.copy()
        if update_battery:
            for i in range(batt.size):
                if per_rec is not None and per_rec.size == batt.size and per_rec[i] > 0.0:
                    batt[i] = batt[i] + per_rec[i]
                else:
                    t0 = ttf[i] * batt[i]
                    if t0 <= 0.0:
                        batt[i] = 0.0
                    else:
                        t1 = t0 - dt
                        ratio = t1 / t0
                        if ratio < 0.0:
                            ratio = 0.0
                        batt[i] = batt[i] * ratio
        np.clip(batt, 0.0, 1.0, out=batt)
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags, batt


def compute_alongstream_raster(simulation, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist'):
    """Compute along-stream distance raster and write to the HDF.

    This function computes distance-to-outlet for traversable raster cells
    (defined by ``depth`` or ``wetted``) by building a sparse graph of
    8-neighbor connections and running Dijkstra from the outlet cell(s).

    Parameters
    ----------
    simulation : object
        Simulation instance exposing ``hdf5`` and raster transforms.
    outlet_xy : tuple or None
        (x, y) outlet coordinate. If ``None``, a downstream cell is chosen.
    depth_name, wetted_name : str
        Names of datasets under ``/environment`` to determine traversable cells.
    out_name : str
        Name of the dataset to write under ``/environment`` for results.

    Returns
    -------
    numpy.ndarray
        2D array (dtype ``float32``) of distances to outlet, same shape as rasters.
    """
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('simulation.hdf5 is required')

    env = hdf.get('environment')
    if env is None:
        raise RuntimeError('environment group missing in HDF')

    # read rasters
    if depth_name in env:
        depth = np.asarray(env[depth_name][:], dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0.0)
    elif wetted_name in env:
        wett = np.asarray(env[wetted_name][:])
        mask = (wett != 0)
    else:
        raise RuntimeError('Neither depth nor wetted raster found')

    # read transforms for pixel spacing
    try:
        t = getattr(simulation, 'depth_rast_transform', None)
        if t is None:
            # try generic transform
            t = getattr(simulation, 'vel_mag_rast_transform', None)
    except Exception:
        t = None
    if t is None:
        # fallback to unit pixels
        px = py = 1.0
    else:
        px = abs(t.a)
        py = abs(t.e)

    h, w = mask.shape

    # mapping from (r,c) to node id
    idx = -np.ones(mask.shape, dtype=np.int32)
    mask_flat = mask.ravel()
    node_ids = np.nonzero(mask_flat)[0]
    if node_ids.size == 0:
        # nothing to do
        arr = np.full(mask.shape, np.nan, dtype=np.float32)
        env.create_dataset(out_name, data=arr, dtype='f4')
        safe_flush(hdf)
        return arr

    idx_flat = -np.ones(h * w, dtype=np.int32)
    idx_flat[node_ids] = np.arange(node_ids.size, dtype=np.int32)
    idx = idx_flat.reshape(h, w)

    # neighbor offsets (8-neighbors)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    rows = []
    cols = []
    data = []
    # build sparse graph
    for r in range(h):
        for c in range(w):
            nid = idx[r, c]
            if nid < 0:
                continue
            for dr, dc in nbrs:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                nid2 = idx[rr, cc]
                if nid2 < 0:
                    continue
                # distance
                dist = np.hypot(dr * py, dc * px)
                rows.append(nid)
                cols.append(nid2)
                data.append(dist)

    n_nodes = node_ids.size
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # find outlet node(s)
    if outlet_xy is not None:
        ox, oy = outlet_xy
        try:
            orow, ocol = geo_to_pixel(simulation.depth_rast_transform, [oy], [ox])
            orow = int(orow[0]); ocol = int(ocol[0])
        except Exception:
            orow = None
        if orow is None or orow < 0 or orow >= h or ocol < 0 or ocol >= w or idx[orow, ocol] < 0:
            # fallback to nearest wetted cell by Euclidean
            flat_xy = np.column_stack((env['x_coords'][:].ravel(), env['y_coords'][:].ravel()))
            dists = np.hypot(flat_xy[:,0] - ox, flat_xy[:,1] - oy)
            cand = np.argmin(dists)
            if mask_flat[cand]:
                outlet_nodes = [idx_flat[cand]]
            else:
                # nearest wetted
                wett_inds = np.nonzero(mask_flat)[0]
                nearest = wett_inds[np.argmin(dists[wett_inds])]
                outlet_nodes = [idx_flat[nearest]]
        else:
            outlet_nodes = [int(idx[orow, ocol])]
    else:
        # default: pick wetted cell with minimal y coordinate (downstream-most)
        xcoords = env['x_coords'][:]
        ycoords = env['y_coords'][:]
        flat_y = ycoords.ravel()
        wett_inds = np.nonzero(mask_flat)[0]
        if wett_inds.size == 0:
            outlet_nodes = [0]
        else:
            out_ind = wett_inds[np.argmin(flat_y[wett_inds])]
            outlet_nodes = [int(idx_flat[out_ind])]

    # run Dijkstra from outlet(s) to all nodes
    dist_matrix = dijkstra(csgraph=graph, directed=False, indices=outlet_nodes)
    # dist_matrix shape: (len(outlet_nodes), n_nodes) or (n_nodes,) if one
    if dist_matrix.ndim == 2:
        dist = dist_matrix.min(axis=0)
    else:
        dist = dist_matrix

    out_arr = np.full(h * w, np.nan, dtype=np.float32)
    out_arr[node_ids] = dist.astype(np.float32)
    out_arr = out_arr.reshape(h, w)

    # write to HDF (handle read-only file by reopening in r+ mode)
    wrote = False
    try:
        if out_name in env:
            env[out_name][:] = out_arr
        else:
            env.create_dataset(out_name, data=out_arr, dtype='f4')
        try:
            safe_flush(hdf)
        except Exception:
            pass
        wrote = True
    except (TypeError, ValueError, RuntimeError) as e:
        # try reopening the underlying file in r+ mode if possible
        fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
        if fname:
            try:
                with h5py.File(fname, 'r+') as hw:
                    envw = hw.require_group('environment')
                    if out_name in envw:
                        envw[out_name][:] = out_arr
                    else:
                        envw.create_dataset(out_name, data=out_arr, dtype='f4')
                    try:
                        hw.flush()
                    except Exception:
                        pass
                    wrote = True
            except Exception:
                wrote = False
        if not wrote:
            # As a last resort, skip write and return the array
            pass
    return out_arr


def compute_coarsened_alongstream_raster(simulation, factor=4, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist'):
    """
    Compute an along-stream distance raster at a coarser resolution and upsample
    back to the simulation raster resolution. This reduces precompute cost for
    large models while providing a fast per-agent sampling raster.

    Parameters:
    - simulation: simulation instance with `hdf5` open and raster transforms.
    - factor: integer downsampling factor (e.g., 4 means coarse pixels are 4x4 blocks).
    - outlet_xy, depth_name, wetted_name: forwarded to the underlying compute.
    - out_name: dataset name to write (same as compute_alongstream_raster by default).

    Returns: upsampled 2D numpy array at original raster size (float32).
    """
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('simulation.hdf5 is required')
    env = hdf.get('environment')
    if env is None:
        raise RuntimeError('environment group missing in HDF')

    # read rasters
    if depth_name in env:
        depth = np.asarray(env[depth_name][:], dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0.0)
    elif wetted_name in env:
        wett = np.asarray(env[wetted_name][:])
        mask = (wett != 0)
    else:
        raise RuntimeError('Neither depth nor wetted raster found')

    h, w = mask.shape
    # compute coarse dims
    ch = max(1, h // factor)
    cw = max(1, w // factor)

    # shrink arrays by block-mean / any-true mask
    depth_coarse = np.full((ch, cw), np.nan, dtype=np.float32)
    mask_coarse = np.zeros((ch, cw), dtype=bool)
    for i in range(ch):
        for j in range(cw):
            r0 = i * factor
            c0 = j * factor
            block = mask[r0:r0 + factor, c0:c0 + factor]
            if np.any(block):
                mask_coarse[i, j] = True
                # average depth where present
                if depth_name in env:
                    db = depth[r0:r0 + factor, c0:c0 + factor]
                    vals = db[np.isfinite(db) & (db > 0.0)]
                    if vals.size:
                        depth_coarse[i, j] = float(np.mean(vals))
                    else:
                        depth_coarse[i, j] = np.nan
                else:
                    depth_coarse[i, j] = 1.0

    # Create a temporary in-memory simulation-like object for the coarse grid
    class _MiniSim:
        pass
    minisim = _MiniSim()
    # set a dummy transform that scales pixels accordingly
    try:
        t = getattr(simulation, 'depth_rast_transform', None)
        if t is None:
            t = getattr(simulation, 'vel_mag_rast_transform', None)
    except Exception:
        t = None
    if t is None:
        # unit transform
        from rasterio.transform import Affine as _Affine
        origin_x = 0.0
        origin_y = 0.0
        tcoarse = _Affine.scale(1, -1)
    else:
        # scale pixel sizes by factor and keep origin
        from rasterio.transform import Affine as _Affine
        tcoarse = _Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f)

    minisim.depth_rast_transform = tcoarse
    # create a small HDF-like group in memory: we'll write temporary datasets to disk via h5py
    # reuse existing HDF file path to create a transient group 'tmp_coarse' then call compute_alongstream_raster
    fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
    if not fname:
        # fallback: call compute_alongstream_raster directly on full grid
        return compute_alongstream_raster(simulation, outlet_xy=outlet_xy, depth_name=depth_name, wetted_name=wetted_name, out_name=out_name)

    import h5py
    tmp_name = 'tmp_coarse'
    with h5py.File(fname, 'r+') as hw:
        if tmp_name in hw:
            del hw[tmp_name]
        g = hw.create_group(tmp_name)
        # create coarse environment subgroup
        envc = g.create_group('environment')
        # write coarse rasters
        envc.create_dataset('depth', data=depth_coarse.astype('f4'))
        envc.create_dataset('wetted', data=mask_coarse.astype('i1'))
        # create x_coords/y_coords for coarse grid based on tcoarse
        cols = np.arange(cw, dtype=np.float32)
        rows = np.arange(ch, dtype=np.float32)
        colg, rowg = np.meshgrid(cols, rows)
        xs, ys = pixel_to_geo(tcoarse, rowg, colg)
        envc.create_dataset('x_coords', data=xs.astype('f4'))
        envc.create_dataset('y_coords', data=ys.astype('f4'))
        # attach this group to minisim by setting its hdf5 attr to a lightweight object
        minisim.hdf5 = hw[tmp_name]
        minisim.depth_rast_transform = tcoarse
        # run compute on the coarse group
        coarse_out = compute_alongstream_raster(minisim, outlet_xy=outlet_xy, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
        # upsample coarse_out back to original resolution by nearest neighbor repeat
        upsampled = np.repeat(np.repeat(coarse_out, factor, axis=0), factor, axis=1)
        upsampled = upsampled[:h, :w]
        # write upsampled raster to main environment via existing function (attempt re-open)
        try:
            env = hw.require_group('environment')
            if out_name in env:
                try:
                    env[out_name][:] = upsampled.astype('f4')
                except Exception:
                    del env[out_name]
                    env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
            else:
                env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
            hw.flush()
        except Exception:
            # if writing to same file is not desired, just return the upsampled array
            pass

    # clean up temp group
    try:
        with h5py.File(fname, 'r+') as hw2:
            if tmp_name in hw2:
                del hw2[tmp_name]
    except Exception:
        pass

    return upsampled.astype(np.float32)

def standardize_shape(arr, target_shape=(5, 5), fill_value=np.nan):
    if arr.shape != target_shape:
        # Create a new array with the target shape, filled with the fill value
        standardized_arr = np.full(target_shape, fill_value)
        # Copy data from the original array to the standardized array
        standardized_arr[:arr.shape[0], :arr.shape[1]] = arr
        return standardized_arr
    return arr

def calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0):
    num_agents = len(headings)

    # Convert headings to direction vectors (dx, dy)
    dx = np.cos(headings)[:, np.newaxis, np.newaxis]
    dy = np.sin(headings)[:, np.newaxis, np.newaxis]

    # Agent coordinates expanded to match the 5x5 grid
    agent_x_expanded = agent_x[:, np.newaxis, np.newaxis]
    agent_y_expanded = agent_y[:, np.newaxis, np.newaxis]

    # Calculate relative coordinates of each cell
    rel_x = x_coords - agent_x_expanded
    rel_y = y_coords - agent_y_expanded

    # Dot product to determine if cells are in front of the agent
    dot_product = dx * rel_x + dy * rel_y
    front_masks = (dot_product > 0).astype(int)

    # Set cells behind the agent to the user-defined value
    front_masks[dot_product <= 0] = behind_value

    return front_masks

def determine_slices_from_vectors(vectors, num_slices=4):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    normalized_angles = np.mod(angles, 2*np.pi)
    slice_width = 2*np.pi / num_slices

    slice_indices = (normalized_angles // slice_width).astype(int)
    return slice_indices

def determine_slices_from_headings(headings, num_slices=4):
    normalized_headings = np.mod(headings, 2*np.pi)
    slice_width = 2*np.pi / num_slices

    slice_indices = (normalized_headings // slice_width).astype(int)
    return slice_indices

def output_excel(records, model_dir, model_name):
    """
    Export the records of PID optimization errors and rankings to an excel file.
    
    Parameters:
    - records (dict): keys are generation iteration and values are the generation's
                      dataframe of errors and rankings.
    - model_dir: path to simulation output.
    - model_name (str): name of the model to help name the output excel file.
    
    """
    # export record results to excel via pandas
    print('\nexporting records to excel...')
    
    # Create an Excel writer object
    output_excel = os.path.join(model_dir,f'output_{model_name}.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        # iterate through the dictionary and write each dataframe to a sheet
        for generation_name, df in records.items():
            df.to_excel(writer,
                        sheet_name = 'gen' + str(generation_name),
                        index=False)
    
    print('records exported. check output excel file.')
    
def movie_maker(directory, model_name, crs, dt, depth_rast_transform):
    # connect to model
    model_directory = os.path.join(directory,'%s.h5'%(model_name))
    hdf5 = h5py.File(model_directory, 'r')
    X_arr = hdf5['agent_data/X'][:]
    Y_arr = hdf5['agent_data/Y'][:]
    
    # calculate important things, like the number of columns which should equal the number of timesteps
    shape = X_arr.shape

    # Number of columns is the second element of the 'shape' tuple
    num_columns = shape[1]

    # get depth raster
    depth_arr = hdf5['environment/depth'][:]
    depth = rasterio.MemoryFile()
    height = depth_arr.shape[0]
    width = depth_arr.shape[1]
    
    with depth.open(
        driver ='GTiff',
        height = depth_arr.shape[0],
        width = depth_arr.shape[1],
        count =1,
        dtype ='float32',
        crs = crs,
        transform = depth_rast_transform
    ) as dataset:
        dataset.write(depth_arr, 1)

        # define metadata for movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title= model_name, artist='Matplotlib',
                        comment='emergent model run %s'%(datetime.now()))
        writer = FFMpegWriter(fps = np.round(30/dt,0), metadata=metadata)

        #initialize plot
        fig, ax = plt.subplots(figsize = (10,5))

        background = ax.imshow(dataset.read(1),
                               origin = 'upper',
                               extent = [dataset.bounds[0],
                                          dataset.bounds[2],
                                          dataset.bounds[1],
                                          dataset.bounds[3]])

        agent_pts, = plt.plot([], [], marker = 'o', ms = 1, ls = '', color = 'red')

        plt.xlabel('Easting')
        plt.ylabel('Northing')

        # Update the frames for the movie
        with writer.saving(fig, 
                           os.path.join(directory,'%s.mp4'%(model_name)), 
                           dpi = 300):

            for i in range(int(num_columns)):


                # write frame
                agent_pts.set_data(X_arr[:, i],
                                   Y_arr[:, i])
                writer.grab_frame()
                    
                print ('Time Step %s complete'%(i))

    # clean up
    writer.finish()
    
def HECRAS (model_dir, HECRAS_dir, resolution, crs):
    """
    Import environment data from a HECRAS model and generate raster files.
    
    This method extracts data from a HECRAS model stored in HDF format and 
    interpolates the data to generate raster files for various environmental 
    parameters such as velocity, water surface elevation, and bathymetry.
    
    Parameters:
    - HECRAS_model (str): Path to the HECRAS model in HDF format.
    - resolution (float): Desired resolution for the interpolated rasters.
    
    Attributes set:
    
    Notes:
    - The method reads data from the HECRAS model, interpolates the data to the 
      desired resolution, and then writes the interpolated data to raster files.
    - The generated raster files are saved in the model directory with names corresponding 
      to the environmental parameter they represent (e.g., 'elev.tif', 'wsel.tif').
    - The method uses LinearNDInterpolator for interpolation and rasterio for raster 
      generation and manipulation.
    """
    # Initialization Part 1: Connect to HECRAS model and import environment
    hdf = h5py.File(HECRAS_dir+'.hdf','r')
    
    # Extract Data from HECRAS HDF model
    print ("Extracting Model Geometry and Results")
    
    pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
    vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
    wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1]
    elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
    
    hdf.close()
    
    # create list of xy tuples
    geom = list(tuple(zip(pts[:,0],pts[:,1])))
    
    # create a dataframe with geom column and observations
    df = pd.DataFrame.from_dict({'index':np.arange(0,len(pts),1),
                                 'geom_tup':geom,
                                 'vel_x':vel_x,
                                 'vel_y':vel_y,
                                 'wsel':wsel,
                                 'elev':elev})
    
    # add a geometry column
    df['geometry'] = df.geom_tup.apply(Point)
    
    # convert into a geodataframe
    gdf = gpd.GeoDataFrame(df,crs = crs)
    
    print ("Create multidimensional interpolator functions for velocity, wsel, elev")
    
    vel_x_interp = LinearNDInterpolator(pts,gdf.vel_x)
    vel_y_interp = LinearNDInterpolator(pts,gdf.vel_y)
    wsel_interp = LinearNDInterpolator(pts,gdf.wsel)
    elev_interp = LinearNDInterpolator(pts,gdf.elev)
    
    # first identify extents of image
    xmin = np.min(pts[:,0])
    xmax = np.max(pts[:,0])
    ymin = np.min(pts[:,1])
    ymax = np.max(pts[:,1])
    
    # interpoate velocity, wsel, and elevation at new xy's
    ## TODO ISHA TO CHECK IF RASTER OUTPUTS LOOK DIFFERENT AT 0.5m vs 1m
    xint = np.arange(xmin,xmax,resolution)
    yint = np.arange(ymax,ymin,resolution * -1.)
    xnew, ynew = np.meshgrid(xint,yint, sparse = True)
    
    print ("Interpolate Velocity East")
    vel_x_new = vel_x_interp(xnew, ynew)
    print ("Interpolate Velocity North")
    vel_y_new = vel_y_interp(xnew, ynew)
    print ("Interpolate WSEL")
    wsel_new = wsel_interp(xnew, ynew)
    print ("Interpolate bathymetry")
    elev_new = elev_interp(xnew, ynew)
    
    # create a depth raster
    depth = wsel_new - elev_new
    
    # calculate velocity magnitude
    vel_mag = np.sqrt((np.power(vel_x_new,2)+np.power(vel_y_new,2)))
    
    # calculate velocity direction in radians
    vel_dir = np.arctan2(vel_y_new,vel_x_new)
    
    print ("Exporting Rasters")

    # create raster properties
    driver = 'GTiff'
    width = elev_new.shape[1]
    height = elev_new.shape[0]
    count = 1
    crs = crs
    #transform = Affine.translation(xnew[0][0] - 0.5, ynew[0][0] - 0.5) * Affine.scale(1,-1)
    transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution)\
        * Affine.scale(resolution,-1 * resolution)

    # write elev raster
    with rasterio.open(os.path.join(model_dir,'elev.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as elev_rast:
        elev_rast.write(elev_new,1)
    
    # write wsel raster
    with rasterio.open(os.path.join(model_dir,'wsel.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as wsel_rast:
        wsel_rast.write(wsel_new,1)
                    
    # write depth raster
    with rasterio.open(os.path.join(model_dir,'depth.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as depth_rast:
        depth_rast.write(depth,1)
        
    # write velocity dir raster
    with rasterio.open(os.path.join(model_dir,'vel_dir.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_dir_rast:
        vel_dir_rast.write(vel_dir,1)
                    
    # write velocity .mag raster
    with rasterio.open(os.path.join(model_dir,'vel_mag.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_mag_rast:
        vel_mag_rast.write(vel_mag,1)
                
    # write velocity x raster
    with rasterio.open(os.path.join(model_dir,'vel_x.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_x_rast:
        vel_x_rast.write(vel_x_new,1)
                    
    # write velocity y raster
    with rasterio.open(os.path.join(model_dir,'vel_y.tif'),
                       mode = 'w',
                       driver = driver,
                       width = width,
                       height = height,
                       count = count,
                       dtype = 'float64',
                       crs = crs,
                       transform = transform) as vel_y_rast:
        vel_y_rast.write(vel_y_new,1)


class PID_controller:
    def __init__(self, n_agents, k_p = 0., k_i = 0., k_d = 0., tau_d = 1):
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        self.integral = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.previous_error = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.derivative_filtered = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        # Attempt to initialize PID plane parameters; fall back to safe defaults
        try:
            self.interp_PID()
        except Exception:
            # Default: P = 1.0 constant, I = 0, D = 0
            self.P_params = np.array([0.0, 0.0, 1.0])
            self.I_params = np.array([0.0, 0.0, 0.0])
            self.D_params = np.array([0.0, 0.0, 0.0])

    def update(self, error, dt, status):
        # create a mask - if this fish is fatigued, this doesn't matter
        mask = np.where(status == 3,True,False)
        
        self.integral = np.where(~mask, self.integral + error, self.integral)
        derivative = error - self.previous_error
        self.previous_error = error
    
        p_term = self.k_p[:, np.newaxis] * error
        i_term = self.k_i[:, np.newaxis] * self.integral
        d_term = self.k_d[:, np.newaxis] * derivative
        
        # # calculate unsaturated output
        # unsaturated_output = p_term + i_term + d_term

        # # Apply limits
        # max_limit = [[20.,20.]]
        # min_limit = [[-20.,-20.]]
        
        # actual_output = np.clip(unsaturated_output, min_limit, max_limit)
        
        # # Back-calculation if necessary
        # if np.any(actual_output != unsaturated_output):
        #     excess = unsaturated_output - actual_output
        #     excess_mask = np.where(excess != 0., True, False)
        #     integral_adjustment = np.where(~mask,
        #                                    excess / self.k_i[:, np.newaxis],
        #                                    [[0., 0.]])
            
        #     print ('initial integral \n %s'%(self.integral))
            
        #     self.integral = np.where(~excess_mask,
        #                              self.integral + integral_adjustment,
        #                              [[0., 0.]])
            
        #     i_term = self.k_i[:, np.newaxis] * self.integral
        #     print ('i term: \n %s'%(i_term))
        #     print ('new integral \n %s'%(self.integral))
            
        
        # # Apply low-pass filter to derivative
        # self.derivative_filtered += (dt / (self.tau_d + dt)) * (derivative - self.derivative_filtered)
    
        # # Update for next iteration
        # self.previous_measurement = error
    
        # # Calculate D term with filtered derivative
        # d_term = -self.k_d[:, np.newaxis] * self.derivative_filtered
    
        return np.where(~mask,p_term + i_term + d_term,0.0) #+ i_term + d_term
        
    def interp_PID(self):
        '''
        Parameters
        ----------
        data_ws : file directory.

        Returns
        -------
        tuple consisting of (P,I,D).
        '''
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/pid_optimize_Nushagak.csv')
        # get data
        df = pd.read_csv(data_dir)
        
        # get data arrays
        length = df.loc[:, 'fish_length'].values
        velocity = df.loc[:, 'avg_water_velocity'].values
        P = df.loc[:, 'p'].values
        I = df.loc[:, 'i'].values
        D = df.loc[:, 'd'].values
        
        # Plane model function
        def plane_model(coords, a, b, c):
            length, velocity = coords
            return a * length + b * velocity + c
        
        # fit plane for P, I, and D values
        self.P_params, _ = curve_fit(plane_model, (length, velocity), P)
        
        self.I_params, _ = curve_fit(plane_model, (length, velocity), I)
        
        self.D_params, _ = curve_fit(plane_model, (length, velocity), D)
    
    def PID_func(self, velocity, length):
        '''
        
        '''
        # Ensure parameters exist
        a_P, b_P, c_P = self.P_params
        a_I, b_I, c_I = self.I_params
        a_D, b_D, c_D = self.D_params

        # Coerce inputs to numpy arrays and broadcast to the same shape
        vel = np.asarray(velocity)
        leng = np.asarray(length)

        # If scalars, expand to 1-D arrays
        if vel.ndim == 0:
            vel = np.full(1, float(vel))
        if leng.ndim == 0:
            leng = np.full(1, float(leng))

        # Broadcast to common shape
        try:
            vel_b, leng_b = np.broadcast_arrays(vel, leng)
        except ValueError:
            # Fallback: flatten and match lengths if possible
            vel_b = vel.ravel()
            leng_b = np.broadcast_to(leng.ravel(), vel_b.shape)

        P = a_P * leng_b + b_P * vel_b + c_P
        I = a_I * leng_b + b_I * vel_b + c_I
        D = a_D * leng_b + b_D * vel_b + c_D

        # Return 1-D arrays
        return np.asarray(P).ravel(), np.asarray(I).ravel(), np.asarray(D).ravel()
    
class PID_optimization():
    '''
    Python class object for solving a genetic algorithm to optimize PID controller values. 
    '''
    def __init__(self,
                 pop_size,
                 generations,
                 min_p_value,
                 max_p_value,
                 min_i_value,
                 max_i_value,
                 min_d_value,
                 max_d_value):
        """
        Initializes an individual's genetic traits.
    
        """
        self.num_genes = 3
        self.min_p_value = min_p_value
        self.max_p_value = max_p_value
        self.min_i_value = min_i_value
        self.max_i_value = max_i_value
        self.min_d_value = min_d_value
        self.max_d_value = max_d_value
        
        # population size, number of individuals to create
        self.pop_size = pop_size
        
        # number of generations to run the alogrithm for
        self.generations = generations
        
        ## for non-uniform range across p/i/d values
        self.p_component = np.random.uniform(self.min_p_value, self.max_p_value, size=1)
        self.i_component = np.random.uniform(self.min_i_value, self.max_i_value, size=1)
        self.d_component = np.random.uniform(self.min_d_value, self.max_d_value, size=1)
        self.genes = np.concatenate((self.p_component, self.i_component, self.d_component), axis=None)
        
        self.cross_ratio = 0.9 # percent of offspring that are crossover vs mutation
        self.mutation_count = 0 # dummy value, will be overwritten
        self.p = {}
        self.i = {}
        self.d = {}
        self.errors = {}
        self.velocities = {}
        self.batteries = {}
        

    def fitness(self):
        '''
        Overview

        This fitness function is designed to evaluate a population of individuals 
        based on three key criteria: error magnitude, array length, and battery life. 
        The function ranks each individual by combining these criteria into a single 
        score, with the goal of minimizing error magnitude, maximizing array length, 
        and maximizing battery life.
        
        Attributes
        
            pop_size (int): The number of individuals in the population. Each 
            individual's performance is evaluated against the set criteria.
            errors (dict): A dictionary where keys are individual identifiers and 
            values are arrays representing the error magnitude for each timestep.
            p, i, d (arrays): Parameters associated with each individual, potentially 
            relevant to the context of the evaluation (e.g., PID controller parameters).
            velocities (array): An array containing the average velocities for each 
            individual, which might be relevant for certain analyses.
            batteries (array): An array containing the battery life values for each 
            individual. Higher values indicate better performance.
        
        Returns
        
            error_df (DataFrame): A pandas DataFrame containing the following 
            columns for each individual:
                individual: The identifier for the individual.
                p, i, d: The PID controller parameters or other relevant parameters 
                for the individual.
                magnitude: The sum of squared errors, representing the error magnitude. 
                Lower values are better.
                array_length: The length of the error array, indicative of the operational 
                duration. Higher values are better.
                avg_velocity: The average velocity for the individual. Included for 
                contextual information.
                battery: The battery life of the individual. Higher values are better.
                arr_len_score: Normalized score based on array_length. Higher scores are better.
                mag_score: Normalized score based on magnitude. Higher scores are better (inverted).
                battery_score: Normalized score based on battery. Higher scores are better.
                rank: The final ranking score, calculated by combining arr_len_score, mag_score, 
                and battery_score according to their respective weights.
        
        Methodology
        
            Data Preparation: The function iterates through each individual in 
            the population, calculating the magnitude of errors and extracting 
            other relevant parameters. It then appends this information to error_df.
        
            Normalization: Each criterion (array length, magnitude, and battery) 
            is normalized to a [0, 1] scale. For array length and battery, higher 
            values result in higher scores. For magnitude, the normalization is 
            inverted so that lower values result in higher scores.
        
            Weighting and Preference Matrix: The criteria are weighted according 
            to their perceived importance to the overall fitness. A pairwise 
            preference matrix is constructed based on these weighted scores, 
            comparing each individual against every other individual.
        
            Ranking: The final rank for each individual is determined by summing 
            up their preferences in the preference matrix. The DataFrame is then 
            sorted by these ranks in descending order, with higher ranks indicating 
            better overall fitness according to the defined criteria.
        
        Customization
        
            The weights assigned to each criterion (array_len_weight, 
                                                    magnitude_weight, 
                                                    battery_weight) can be adjusted 
            to reflect their relative importance in the specific context of use. The 
            default weights are set based on a balanced assumption but should be 
            tailored to the specific requirements of the evaluation.
            Additional criteria can be incorporated into the evaluation by extending 
            the DataFrame to include new columns, normalizing these new criteria,
            and adjusting the preference matrix calculation to account for these 
            criteria.
        
        Usage
        
        To use this function, instantiate the class with the relevant data 
        (errors, parameters, velocities, and batteries) and call the fitness method. 
        The method returns a ranked DataFrame, which can be used to select the 
        top-performing individuals for further analysis or operations.
                
                
                
        '''
        error_df = pd.DataFrame(columns=['individual', 
                                         'p', 
                                         'i', 
                                         'd', 
                                         'magnitude',
                                         'array_length',
                                         'avg_velocity',
                                         'battery',
                                         'arr_len_score',
                                         'mag_score',
                                         'battery_score',
                                         'rank'])

        for i in range(self.pop_size):
            filtered_array = self.errors[i][:-1]
            magnitude = np.nansum(np.power(filtered_array, 2))

            row_data = {
                'individual': i,
                'p': self.p[i],
                'i': self.i[i],
                'd': self.d[i],
                'magnitude': magnitude,
                'array_length': len(filtered_array),
                'avg_velocity': np.nanmean(self.velocities[i]),
                'battery': self.batteries[i]  # Assuming you have battery data in self.batteries
            }

            error_df = error_df.append(row_data, ignore_index=True)

        # Normalize the criteria
        error_df['arr_len_score'] = (error_df['array_length'] - error_df['array_length'].min()) / (error_df['array_length'].max() - error_df['array_length'].min())
        error_df['mag_score'] = (error_df['magnitude'].max() - error_df['magnitude']) / (error_df['magnitude'].max() - error_df['magnitude'].min())
        error_df['battery_score'] = (error_df['battery'] - error_df['battery'].min()) / (error_df['battery'].max() - error_df['battery'].min())

        error_df.set_index('individual', inplace=True)

        # Update weights to include battery
        array_len_weight = 0.35
        magnitude_weight = 0.40
        battery_weight = 1 - array_len_weight - magnitude_weight

        n = len(error_df)
        preference_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    preference_matrix[i, j] = (array_len_weight * (error_df.at[i, 'arr_len_score'] > error_df.at[j, 'arr_len_score'])) + \
                                              (magnitude_weight * (error_df.at[i, 'mag_score'] > error_df.at[j, 'mag_score'])) + \
                                              (battery_weight * (error_df.at[i, 'battery_score'] > error_df.at[j, 'battery_score']))

        final_scores = np.sum(preference_matrix, axis=1)
        error_df['rank'] = final_scores
        error_df.reset_index(drop=False, inplace=True)
        error_df.sort_values(by='rank', ascending=False, inplace=True)

        return error_df
    
    def selection(self, error_df):
        """
        Selects the highest performing indivduals to become parents, based on
        solution rank. Assigns a number of offspring to each parent pair based
        on a beta probability distribution function. Fitter parents produce more
        offspring.
        
        Parameters:
        - error_df (dataframe): a ranked dataframe of indidvidual solutions.
                                output of the self.fitness() function.
        
        Attributes set:
        - pop_size (int): number of indidivduals in population. useful for defining
                          the number of offspring to ensure population doesn't balloon.
        - cross_ratio (float): controls the ratio of crossover offspring vs mutation offspring
                          
        Returns: list of dataframes. each dataframe contained paired parents with
                 assigned number of offspring
        
        """
        # selects the top 80% of individuals to be parents
        index_80_percent = int(0.8 * len(error_df))
        parents = error_df.iloc[:index_80_percent]
        
        # create a list of dataframes -> pairs of parents by fitness
        pairs_parents = []
        for i in np.arange(0, len(parents), 2):
            pairs_parents.append(parents[i:(i + 2)])
        
        # shape parameters for the beta distribution -> have more fit parents produce more offspring
        # https://en.wikipedia.org/wiki/Beta_distribution#/media/File:Beta_distribution_pdf.svg
        a = 1
        b = 3
        
        # calculate PDF values of the beta distribution based on the length of the list
        beta_values = beta.pdf(np.linspace(0, 0.5, len(pairs_parents)), a, b)
        
        # scale values to number of offspring desired
        offspring = self.cross_ratio * self.pop_size # generate XX% of offspring as crossover
        scaled_values = offspring * beta_values / sum(beta_values)
        scaled_values = np.round(scaled_values).astype(int)
        
        # assign beta values (as offspring weight) to appropriate parent pair
        for i, df in enumerate(pairs_parents):
            df['offspring_weight'] = scaled_values[i]  # Assign array value to the column
        
        return pairs_parents
    
    def crossover(self, pairs_parents):
        """
        Generate new genes for offspring based on existing parent genes. Number of offspring
        per parent pair is dictated by 'offspring_weight' as set in selection function.
        
        Parameters:
        - pairs_parents (list): list of dataframes. each dataframe contained paired
                                parents with assigned number of offspring
                                
        Returns: list of lists, each list contains random p,i,d values between parent values
                                
        """
        offspring = []

        for i in pairs_parents:
            parent1 = i[:1]
            parent2 = i[1:]
            num_offspring = parent1.iloc[0]['offspring_weight'].astype(int)
            
            for j in range(num_offspring):
                p = random.uniform(parent1.iloc[0]['p'], parent2.iloc[0]['p'])
                i = random.uniform(parent1.iloc[0]['i'], parent2.iloc[0]['i'])
                d = random.uniform(parent1.iloc[0]['d'], parent2.iloc[0]['d'])
                offspring.append([p,i,d])
        
        # set a number of mutations to generate
        # this ensures the correct number of offspring are generated
        self.mutation_count = self.pop_size - len(offspring)
        
        return offspring

    def mutation(self, error_df):
        """
        Generate new genes for offspring independent of parent genes. Uses the min/max
        gene values set in the first generation population.
        
        Attributes set:
        - mutation_count (int): number of mutation individuals to create. defined by the crossover
                                function, this ensures that the offspring total are the same as the
                                previous population so it doesn't change.
        - min_gene_value: minimum for gene value. same as defined in initial population
        - max_gene_value: maximum for gene value. same as defined in initial population
        - num_genes: number of genes to create. should always be 3 for pid controller
                                
        Returns: list of lists, each list contains random p,i,d values between min/max gene values.
                 this list will be combined with the crossover offspring to produce the full
                 population of the next generation.
        
        """
        population = []

        for i in range(self.mutation_count):
            # individual = [random.uniform(self.min_gene_value, self.max_gene_value) for _ in range(self.num_genes)]
            P = np.abs(error_df.iloc[i]['p'] + np.random.uniform(-4.0,4.0,1)[0])
            I = np.abs(error_df.iloc[i]['i'] + np.random.uniform(-0.1,0.1,1)[0])
            D = np.abs(error_df.iloc[i]['d'] + np.random.uniform(-1.0,1.0,1)[0])
            
            individual = np.concatenate((P, I, D), axis=None)
            
            population.append(individual)
   
        return population

    def population_create(self):
        """
        Generate the population of individuals.
        
        Attributes set:
        - genes
        - pop_size
        - num_genes
        - min_gene_value
        - max_gene_value
                                
        Returns: array of population p/i/d values, one set for each individual.
        
        """      
        population = []

        for _ in range(self.pop_size):
        # create a new instance of the solution class for each individual
            individual = PID_optimization(self.pop_size,
                                          self.generations,
                                          self.min_p_value,
                                          self.max_p_value,
                                          self.min_i_value,
                                          self.max_i_value,
                                          self.min_d_value,
                                          self.max_d_value)
            population.append(individual.genes)

        return population
    
    def run(self, population, sockeye, model_dir, crs, basin, water_temp, pid_tuning_start, fish_length, ts, n, dt):
        """
        Run the genetic algorithm.
        
        Parameters:
        - population (array): collection of solutions (population of individuals)
        - sockeye: sockeye model
        - model_dir (str): Directory where the model data will be stored.
        - crs (str): Coordinate reference system for the model.
        - basin (str): Name or identifier of the basin.
        - water_temp (float): Water temperature in degrees Celsius.
        - pid_tuning_start (tuple): A tuple of two values (x, y) defining the point where agents start.
        - ts (int, optional): Number of timesteps for the simulation. Defaults to 100.
        - n (int, optional): Number of agents in the simulation. Defaults to 100.
        - dt (float): The duration of each time step.
        
        Attributes:
        - generations
        - pop_size
        - p
        - i
        - d
        - errors
        - velocities
        
        Returns:
        - records (dict): dictionary holding each generation's errors and rankings. 
                          Generation number is used as the dictionary key. Each key's value
                          is the dataframe of PID values and ranking metrics.
        """
        records = {}
        
        for generation in range(self.generations):
            
            # keep track of the timesteps before error (length of error array),
            # also used to calc magnitude of errors
            pop_error_array = []
            
            prev_error_sum = np.zeros(1)

            #for i in range(len(self.population)):
            for i in range(self.pop_size):
            
                print(f'\nrunning individual {i+1} of generation {generation+1}, {generation+1}, {generation+1}, {generation+1}, {generation+1}...')
                
                # useful to have these in pid_solution
                self.p[i] = population[i][0]
                self.i[i] = population[i][1]
                self.d[i] = population[i][2]
                
                print(f'P: {self.p[i]:0.3f}, I: {self.i[i]:0.3f}, D: {self.d[i]:0.3f}')
                
                # set up the simulation
                sim = sockeye.simulation(model_dir,
                                         'solution',
                                         crs,
                                         basin,
                                         water_temp,
                                         pid_tuning_start,
                                         fish_length,
                                         ts,
                                         n,
                                         use_gpu = False,
                                         pid_tuning = True)
                
                # run the model and append the error array

                try:
                    sim.run('solution',
                            n = ts,
                            dt = dt,
                            k_p = self.p[i], # k_p
                            k_i = self.i[i], # k_i
                            k_d = self.d[i], # k_d
                            )
                    
                except:
                    print(f'failed --> P: {self.p[i]:0.3f}, I: {self.i[i]:0.3f}, D: {self.d[i]:0.3f}\n')
                    pop_error_array.append(sim.error_array)
                    self.errors[i] = sim.error_array
                    self.velocities[i] = np.sqrt(np.power(sim.vel_x_array,2) + np.power(sim.vel_y_array,2))
                    self.batteries[i] = sim.battery[-1]
                    sim.close()

                    continue

            # run the fitness function -> output is a df
            error_df = self.fitness()
            # print(f'Generation {generation+1}: {error_df.head()}')
            
            # update logging dictionary
            records[generation] = error_df

            # selection -> output is list of paired parents dfs
            selected_parents = PID_optimization.selection(self, error_df)

            # crossover -> output is list of crossover pid values
            cross_offspring = PID_optimization.crossover(self, selected_parents)

            # mutation -> output is list of muation pid values
            mutated_offspring = PID_optimization.mutation(self, error_df)
            # combine crossover and mutation offspring to get next generation
            population = cross_offspring + mutated_offspring
            
            print(f'completed generation {generation+1}.... ')
            
            if np.all(error_df.magnitude.values == 0):
                return records
                        
        return records    
      
class simulation():
    '''Python class object that implements an Agent Based Model of adult upstream
    migrating sockeye salmon through a modeled riffle cascade complex.  Rather
    than traditional OOP architecture, this version of the software employs a 
    Structure of Arrays data management philosophy - in other words - we processin
    on a GPU now'''
    
    def __init__(self, 
                 model_dir, 
                 model_name, 
                 crs, 
                 basin, 
                 water_temp, 
                 start_polygon,
                 centerline,
                 env_files=None,
                 fish_length = None,
                 num_timesteps = 100, 
                 num_agents = 100,
                 use_gpu = False,
                 pid_tuning = False,
                 hecras_plan_path=None,
                 hecras_fields=None,
                 hecras_k=8,
                 use_hecras=False,
                 hecras_write_rasters=False,
                 defer_hdf=False,
                 defer_log_dir=None,
                 defer_log_fmt='npz'):
        """
         Initialize the simulation environment.
         
         Parameters:
         - model_dir (str): Directory where the model data will be stored.
         - model_name (str): Name of the model.
         - crs (str): Coordinate reference system for the model.
         - basin (str): Name or identifier of the basin.
         - water_temp (float): Water temperature in degrees Celsius.
         - starting_box (tuple): A tuple of four values (xmin, xmax, ymin, ymax) defining the bounding box 
                                 where agents start.
         - env_files (dict): A dictionary of file names that make up the agent based model
                             environment.  Must inlcude keys: wsel (water surface elevation), 
                                                             depth, 
                                                             elev (elevation),
                                                             x_vel (water velocity x direction),
                                                             y_vel (water velocity y direction),
                                                             vel_dir (water direction in radians),
                                                             vel_mag (water velocity magnitude)
         - num_timesteps (int, optional): Number of timesteps for the simulation. Defaults to 100.
         - num_agents (int, optional): Number of agents in the simulation. Defaults to 100.
         - use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
         
         Attributes:
         - arr (module): Module used for array operations (either numpy or cupy).
         - model_dir, model_name, crs, basin, etc.: Various parameters and states for the simulation.
         - X, Y, prev_X, prev_Y: Arrays representing the X and Y coordinates of agents.
         - drag, thrust, Hz, etc.: Arrays representing various movement parameters of agents.
         - kcal: Array representing the kilocalorie counter for each agent.
         - hdf5 (h5py.File): HDF5 file object for storing simulation data.
         
         Notes:
         - The method initializes various agent properties and states.
         - It creates a new HDF5 file for the project and writes initial data to it.
         - Agent properties that do not change with time are written to the HDF5 file.
        """        
        self.arr = get_arr(use_gpu)
        # deferred HDF logging — when True, per-timestep agent data is written
        # to fast binary .npz logs in `defer_log_dir` via LogWriter and converted later.
        self.defer_hdf = defer_hdf
        self.defer_log_dir = defer_log_dir
        self.defer_log_fmt = defer_log_fmt
        if self.defer_hdf:
            try:
                log_dir = self.defer_log_dir or os.path.join(self.model_dir, 'logs', 'deferred')
                if getattr(self, 'defer_log_fmt', 'npz') == 'memmap':
                    from emergent.io.log_writer_memmap import MemmapLogWriter
                    # we don't yet have num_timesteps and num_agents until after init; create placeholder
                    # actual memmaps will be created after agent arrays are allocated; store config for later
                    self._memmap_config = {'out_dir': log_dir}
                    self._log_writer = None
                    self._memmap_writer = None
                else:
                    from emergent.io.log_writer import LogWriter
                    self._log_writer = LogWriter(log_dir)
                    self._memmap_writer = None
            except Exception:
                self._log_writer = None
                self._memmap_writer = None

        # cache for inverse Affine transforms to avoid repeated `~transform` calls
        self._inv_transform_cache = {}
        # HDF5 buffering parameters
        self.flush_interval = 50  # timesteps between HDF5 flushes (configurable)
        self._hdf5_buffers = {}
        self._buffer_pos = 0

        # NOTE: Numba helpers are precompiled at module import time when available.
        
        # If we are tuning the PID controller, special settings used
        if pid_tuning:
            self.pid_tuning = pid_tuning
            self.vel_x_array = np.array([])
            self.vel_y_array = np.array([])
            
        else:
            self.pid_tuning = False
        
        # model directory and model name
        self.model_dir = model_dir
        self.model_name = model_name
        
        # Save start polygon path for episode resets
        self.start_polygon_path = start_polygon
        self.db = os.path.join(self.model_dir,'%s.h5'%(self.model_name))
                
        # coordinate reference system for the model
        self.crs = crs
        self.basin = basin
        
        # model parameters
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.water_temp = water_temp
        
        # initialize agent properties and internal states
        self.sim_sex()
        self.sim_length(fish_length)
        self.sim_weight()
        self.sim_body_depth()
        # Locate recovery.csv robustly: try module-relative, repo data locations, or search upwards
        from pathlib import Path
        file_candidates = []
        module_dir = Path(__file__).resolve().parent
        # module relative
        file_candidates.append(module_dir.joinpath('..', 'data', 'recovery.csv'))
        # repo-level data paths (walk up parents)
        for p in module_dir.parents[:6]:
            file_candidates.append(p.joinpath('data', 'salmon_abm', 'recovery.csv'))
            file_candidates.append(p.joinpath('data', 'recovery.csv'))
        # also check known workspace location (OneDrive path)
        file_candidates.append(Path(r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\recovery.csv"))

        data_dir = None
        for candidate in file_candidates:
            if candidate.exists():
                data_dir = str(candidate)
                break

        if data_dir is None:
            raise FileNotFoundError('Could not locate recovery.csv in repo; looked at: %s' % ','.join([str(p) for p in file_candidates]))

        recover = pd.read_csv(data_dir)
        recover['Seconds'] = recover.Minutes * 60.
        self.recovery = CubicSpline(recover.Seconds, recover.Recovery, extrapolate = True,)
        del recover
        self.swim_behav = self.arr.repeat(1, num_agents)               # 1 = migratory , 2 = refugia, 3 = station holding
        self.swim_mode = self.arr.repeat(1, num_agents)      # 1 = sustained, 2 = prolonged, 3 = sprint
        self.battery = self.arr.repeat(1.0, num_agents)
        self.recover_stopwatch = self.arr.repeat(0.0, num_agents)
        self.ttfr = self.arr.repeat(0.0, num_agents)
        self.time_out_of_water = self.arr.repeat(0.0, num_agents)
        self.time_of_abandon = self.arr.repeat(0.0, num_agents)
        self.time_since_abandon = self.arr.repeat(0.0, num_agents)
        # Death / drying configuration
        # If True, agents on dry ground are marked dead immediately
        self.death_on_dry_immediate = True
        # Time (timesteps) allowed out of water before death if immediate flag is False
        self.time_out_of_water_threshold = 5
        self.dead = np.zeros(num_agents)
        
        # create initial positions
        gdf = gpd.read_file(start_polygon)

        # Get the geometry of the shapefile
        geometry = gdf.geometry.unary_union
        
        minx, miny, maxx, maxy = geometry.bounds
        X = []
        Y = []

        while len(X) <= self.num_agents:
            random_points = np.random.uniform([minx, miny], [maxx, maxy], size=(self.num_agents*2, 2))
            for x, y in random_points:
                pnt = Point(x, y)
                if geometry.contains(pnt):
                    X.append(x)
                    Y.append(y)

        self.X = np.array(X)[:self.num_agents]
        self.Y = np.array(Y)[:self.num_agents]
            
        self.prev_X = self.X
        self.prev_Y = self.Y
        
        # create short term memory for eddy escpement 
        max_timesteps = 600  # Maximum number of timesteps to track

        self.swim_speeds = np.full((num_agents, max_timesteps), np.nan)
        self.past_centerline_meas = np.full((num_agents, max_timesteps), np.nan)
        self.current_centerline_meas = np.zeros_like(self.X)

        self.in_eddy = np.zeros_like(self.X)
        self.time_since_eddy_escape = np.zeros_like(self.X)
        self.max_eddy_escape_seconds = 45.
        
        # Time to Fatigue values for Sockeye digitized from Bret 1964
        #TODO - we need to scale these numbers by size, way too big for tiny fish
        adult_slope_adjustment = 0.2 # 0.5 or 0.1
        adult_intercept_adjustment = 1.5 # 1.5 or 2.1
        prolonged_swim_speed_adjustment = 2.1
        self.max_s_U = 2.77      # maximum sustained swim speed in bl/s
        self.max_p_U = 4.43 + prolonged_swim_speed_adjustment  # maximum prolonged swim speed
        self.a_p = 8.643 + adult_intercept_adjustment   # prolonged intercept
        self.b_p = -2.0894 * adult_slope_adjustment  # prolonged slope
        self.a_s = 0.1746  + adult_intercept_adjustment    # sprint intercept
        self.b_s = -0.1806 * adult_slope_adjustment   # sprint slope
        
        # initialize movement parameters
        self.drag = self.arr.zeros(num_agents)           # computed theoretical drag

        # If memmap deferred logging is requested, pre-create the memmap writer now
        if getattr(self, 'defer_hdf', False) and getattr(self, 'defer_log_fmt', 'npz') == 'memmap':
            try:
                # variables mirrored from timestep_flush buf_vals
                vars_to_log = ['X','Y','Z','prev_X','prev_Y','heading','sog','ideal_sog','swim_speed',
                               'battery','swim_behav','swim_mode','recover_stopwatch','ttfr','time_out_of_water',
                               'drag','thrust','Hz','bout_no','dist_per_bout','bout_dur','kcal']
                var_shapes = {k: (self.num_agents, self.num_timesteps) for k in vars_to_log}
                from emergent.io.log_writer_memmap import MemmapLogWriter
                out_dir = (self.defer_log_dir or os.path.join(self.model_dir, 'logs', 'deferred'))
                self._memmap_writer = MemmapLogWriter(out_dir, var_shapes, dtype=np.float32)
                # also make a small internal mapping so flush uses the same variables
                self._memmap_vars = vars_to_log
            except Exception:
                self._memmap_writer = None
                self._memmap_vars = []
        self.thrust = self.arr.zeros(num_agents)         # computed theoretical thrust Lighthill 
        self.Hz = self.arr.zeros(num_agents)             # tail beats per second
        self.bout_no = self.arr.zeros(num_agents)        # bout number - new bout whenever fish recovers
        self.dist_per_bout = self.arr.zeros(num_agents)  # running counter of the distance travelled per bout
        self.bout_dur = self.arr.zeros(num_agents)       # running bout timer 
        self.time_of_jump = self.arr.zeros(num_agents)   # time since last jump - can't happen every timestep
        
        # initialize odometer
        self.kcal = self.arr.zeros(num_agents)           #kilo calorie counter
        # cache for per-timestep precomputed pixel indices
        self._pixel_index_cache = {}
    
        # create a project database and write initial arrays to HDF
        self.hdf5 = h5py.File(self.db, 'w')
        self.initialize_hdf5()
        # Build an in-memory cache of environment rasters to avoid repeated h5py reads
        self._env_cache = {}
        try:
            env = self.hdf5.get('environment')
            if env is not None:
                for name in ('depth','wetted','vel_x','vel_y','vel_mag','vel_dir','x_coords','y_coords','along_stream_dist'):
                    try:
                        if name in env:
                            self._env_cache[name] = np.asarray(env[name][:])
                    except Exception:
                        self._env_cache[name] = None
        except Exception:
            self._env_cache = {}

        # Ensure Numba kernels are warmed with representative sizes to avoid JIT stalls in timed loops
        try:
            if _HAS_NUMBA:
                # warmup with representative size to move JIT compile cost out of timed loop
                warm_n = max(1024, int(getattr(self, 'num_agents', 128)))
                _numba_warmup(m=warm_n)
        except Exception:
            pass

        # perform an exact-shape warmup using this simulation's sizes
        try:
            if _HAS_NUMBA:
                _numba_warmup_for_sim(self)
        except Exception:
            pass

        # Optionally compute along-stream raster on init if user provided
        # an external environment HDF path via env_files special key 'hecras_hdf'
        # or if the simulation was given a hecras_plan_path. The user requested
        # to use the .p05 file in data; if present, compute a coarsened raster.
        try:
            # default config (can be overridden by setting these attrs before init)
            factor = getattr(self, 'alongstream_factor', 4)
            create_on_init = getattr(self, 'create_alongstream_on_init', True)
        except Exception:
            factor = 4
            create_on_init = True

        if create_on_init:
            # try to find a matching .p05 in workspace data if a path wasn't provided
            try:
                # user-specified override in env_files under key 'hecras_hdf'
                hecras_hdf = None
                if isinstance(env_files, dict):
                    hecras_hdf = env_files.get('hecras_hdf')
                if not hecras_hdf:
                    # default path the user indicated
                    hecras_dir = os.path.join(self.model_dir, 'data', 'salmon_abm', '20240506')
                    hecras_dir = os.path.normpath(hecras_dir)
                    candidate = os.path.join(hecras_dir, 'Nuyakuk_Production_.p05.hdf')
                    if os.path.exists(candidate):
                        hecras_hdf = candidate
                # if we found an HDF, set simulation.hdf5 to it for compute, otherwise compute on current file
                if hecras_hdf and os.path.exists(hecras_hdf):
                    # If the file is a HECRAS plan HDF without 'environment', use mapping helpers
                    try:
                        # ensure x_coords/y_coords and environment rasters via mapping helper
                        try:
                            # attempt to map commonly used rasters from HECRAS onto our HDF grid
                            map_hecras_to_env_rasters(self, hecras_hdf, raster_names=['depth','wetted','vel_x','vel_y'])
                        except Exception:
                            # fallback: try loading HECRAS plan into cache (KDTree) and then map
                            try:
                                load_hecras_plan_cached(self, hecras_hdf)
                                map_hecras_to_env_rasters(self, hecras_hdf, raster_names=['depth','wetted','vel_x','vel_y'])
                            except Exception:
                                pass

                        # compute coarsened alongstream raster using created env rasters
                        try:
                            compute_coarsened_alongstream_raster(self, factor=factor, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
                        except Exception:
                            try:
                                compute_alongstream_raster(self, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
                            except Exception:
                                pass
                    except Exception:
                        # if reading external HDF fails, try computing on current hdf5
                        try:
                            compute_coarsened_alongstream_raster(self, factor=factor)
                        except Exception:
                            pass
                else:
                    # compute on in-project HDF if environment rasters were loaded
                    try:
                        compute_coarsened_alongstream_raster(self, factor=factor)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Initialize agent properties using the restored methods from backup
        self.sim_sex()
        self.sim_length(fish_length)
        self.sim_weight()
        self.sim_body_depth()
            
        # write agent properties that do not change with time
        self.hdf5["agent_data/sex"][:] = self.sex
        self.hdf5["agent_data/length"][:] = self.length
        self.hdf5["agent_data/ucrit"][:] = self.ucrit
        self.hdf5["agent_data/weight"][:] = self.weight
        self.hdf5["agent_data/body_depth"][:] = self.body_depth
        self.hdf5["agent_data/too_shallow"][:] = self.too_shallow
        self.hdf5["agent_data/opt_wat_depth"][:] = self.opt_wat_depth
        
        # import environment only when file paths are present and exist (robust to missing files)
        def _maybe_import(key, surface_type):
            if env_files is None:
                return
            fp = env_files.get(key)
            if not fp:
                return
            # Use absolute path if provided, otherwise join with model_dir
            path = fp if os.path.isabs(fp) else os.path.join(model_dir, fp)
            if os.path.exists(path):
                print(f'Importing {key} from {path} as {surface_type}')
                self.enviro_import(path, surface_type)
                print(f'Successfully imported {key}')
            else:
                print(f'Raster file not found: {path}')

        # If a HECRAS plan is provided, prefer it and skip raster imports
        self.hecras_plan_path = hecras_plan_path
        self.hecras_fields = hecras_fields if hecras_fields is not None else ['Cells Minimum Elevation']
        self.hecras_k = hecras_k
        self.use_hecras = bool(use_hecras)
        self.hecras_write_rasters = bool(hecras_write_rasters)
        if not self.hecras_plan_path:
            _maybe_import('x_vel', 'velocity x')
            _maybe_import('y_vel', 'velocity y')
            _maybe_import('depth', 'depth')
            _maybe_import('wsel', 'wsel')
            _maybe_import('elev', 'elevation')
            _maybe_import('vel_dir', 'velocity direction')
            _maybe_import('vel_mag', 'velocity magnitude')
            _maybe_import('wetted', 'wetted')
        else:
            # HECRAS MODE: Initialize geometry with proper workflow
            print("\n" + "="*80)
            print("INITIALIZING HECRAS MODE")
            print("="*80)
            
            try:
                # Call comprehensive HECRAS initialization
                hecras_info = initialize_hecras_geometry(
                    simulation=self,
                    plan_path=self.hecras_plan_path,
                    depth_threshold=0.05,
                    crs=self.crs,
                    target_cell_size=None,  # Auto-detect
                    create_rasters=self.hecras_write_rasters
                )
                
                # Store results on simulation
                self._hecras_geometry_info = hecras_info
                
                # If centerline was derived, use it
                if hecras_info['centerline'] is not None:
                    self.centerline = hecras_info['centerline']
                    centerline_derived = True
                    print(f"Using HECRAS-derived centerline ({self.centerline.length:.2f}m)")
                
                # Set depth_rast_transform if rasters were created
                if hecras_info['transform'] is not None:
                    self.depth_rast_transform = hecras_info['transform']
                    coords = hecras_info['coords']
                    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                    print(f"HECRAS extent: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
                
            except Exception as e:
                print(f"ERROR: HECRAS initialization failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to loading KDTree only
                try:
                    load_hecras_plan_cached(self, self.hecras_plan_path, field_names=self.hecras_fields)
                except Exception:
                    self.hecras_plan_path = None
                    
        self.hdf5.flush()
        # Initialize in-memory buffers for per-timestep datasets to reduce small writes
        buf_len = min(self.flush_interval, max(1, self.num_timesteps))
        buffer_shape = (self.num_agents, buf_len)
        dset_names = [
            "X","Y","Z","prev_X","prev_Y","heading","sog","ideal_sog","swim_speed",
            "battery","swim_behav","swim_mode","recover_stopwatch","ttfr","time_out_of_water",
            "drag","thrust","Hz","bout_no","dist_per_bout","bout_dur","time_of_jump","kcal"
        ]
        for name in dset_names:
            self._hdf5_buffers[name] = np.zeros(buffer_shape, dtype='float32')
        self._buffer_pos = 0
        
        # import centerline shapefile (or derive from HECRAS if requested)
        centerline_derived = False
        if self.use_hecras and self.hecras_plan_path:
            # If user didn't provide a centerline_path file, derive a crude centerline
            if centerline is None or not os.path.exists(centerline):
                try:
                    # Prefer extracting a centerline from rasters if available (wetted or distance_to)
                    # Prefer rasters already present in the simulation HDF (`self.hdf5['environment']`)
                    env = None
                    try:
                        env = self.hdf5.get('environment') if self.hdf5 is not None else None
                    except Exception:
                        env = None

                    distance_to = None
                    wetted = None
                    x_coords = None
                    y_coords = None
                    if env is not None:
                        try:
                            if 'distance_to' in env:
                                distance_to = np.array(env['distance_to'])
                            if 'wetted' in env:
                                wetted = np.array(env['wetted'])
                            if 'x_coords' in env:
                                x_coords = np.array(env['x_coords'])
                            if 'y_coords' in env:
                                y_coords = np.array(env['y_coords'])
                        except Exception:
                            distance_to = None
                            wetted = None

                    # If not present in sim HDF, try to open the HECRAS plan HDF as a fallback
                    if distance_to is None and self.hecras_plan_path:
                        try:
                            with h5py.File(self.hecras_plan_path, 'r') as hdf:
                                env2 = hdf.get('environment') if 'environment' in hdf else None
                                if env2 is not None and 'distance_to' in env2:
                                    distance_to = np.array(env2['distance_to'])
                                if env2 is not None and 'wetted' in env2:
                                    wetted = np.array(env2['wetted'])
                                if env2 is not None and 'x_coords' in env2:
                                    x_coords = x_coords if x_coords is not None else np.array(env2['x_coords'])
                                if env2 is not None and 'y_coords' in env2:
                                    y_coords = y_coords if y_coords is not None else np.array(env2['y_coords'])
                        except Exception:
                            pass

                    # If distance_to not present, try to compute from wetted raster
                    distance_to_rast = None
                    if distance_to is not None and distance_to.size > 0:
                        distance_to_rast = distance_to
                    elif wetted is not None and wetted.size > 0:
                        try:
                            mask = (wetted != -9999) & (wetted > 0)
                            pix = getattr(self, 'depth_rast_transform', None)
                            pixel_width = pix.a if pix is not None else 1.0
                            distance_to_rast = distance_transform_edt(mask) * pixel_width
                        except Exception:
                            distance_to_rast = None

                    if distance_to_rast is None or distance_to_rast.size == 0:
                        # cannot extract from rasters; fall back to file-based import if provided
                        if centerline is not None and os.path.exists(centerline):
                            self.centerline = self.centerline_import(centerline)
                            centerline_derived = True
                        else:
                            raise RuntimeError('Could not derive centerline from HECRAS rasters and no centerline provided')
                    else:
                        # Use the standardized helper to extract centerlines and prefer a known Affine transform
                        try:
                            # prefer an existing raster transform stored on simulation
                            target_affine = getattr(self, 'depth_rast_transform', None)

                            # if that isn't available, try to derive an Affine from available x_coords/y_coords
                            if target_affine is None:
                                try:
                                    # if we have full 2D x_coords/y_coords from the HDF environment use them
                                    if x_coords is not None and y_coords is not None and getattr(x_coords, 'ndim', 0) == 2 and getattr(y_coords, 'ndim', 0) == 2:
                                        coords = np.column_stack((x_coords.flatten(), y_coords.flatten()))
                                        target_affine = compute_affine_from_hecras(coords)
                                    else:
                                        # try to read cell center coords directly from the HECRAS plan HDF
                                        try:
                                            with h5py.File(self.hecras_plan_path, 'r') as ph:
                                                hecras_coords = ph['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
                                            target_affine = compute_affine_from_hecras(hecras_coords)
                                        except Exception:
                                            target_affine = None
                                except Exception:
                                    target_affine = None

                            main_centerline, all_lines = derive_centerline_from_distance_raster(distance_to_rast, transform=target_affine, footprint_size=5, min_length=50)
                            if main_centerline is not None and main_centerline.length > 50:
                                self.centerline = self.centerline_import(main_centerline)
                                print('Derived centerline from HECRAS rasters (skeletonized)')
                                centerline_derived = True
                            else:
                                # fallback to provided centerline file if available
                                if centerline is not None and os.path.exists(centerline):
                                    self.centerline = self.centerline_import(centerline)
                                    centerline_derived = True
                                else:
                                    raise RuntimeError('Extracted centerline is invalid or too short')
                        except Exception as e:
                            raise
                except Exception as e:
                    print(f'Warning: could not derive centerline from HECRAS: {e}')
                    # fall back to file-based import only if a valid file was provided
                    if centerline is not None and os.path.exists(centerline):
                        self.centerline = self.centerline_import(centerline)
                        centerline_derived = True
                    else:
                        raise RuntimeError(f'Failed to derive centerline from HECRAS and no valid centerline file provided: {e}')
            else:
                self.centerline = self.centerline_import(centerline)
                centerline_derived = True
        
        if not centerline_derived:
            self.centerline = self.centerline_import(centerline)
        # boundary_surface
        self.boundary_surface()

        # initialize mental maps
        self.avoid_cell_size = 10.
        self.initialize_mental_map()
        self.refugia_cell_size = 1.
        self.initialize_refugia_map()
        
        # initialize heading
        self.initial_heading()
        
        # initialize swim speed
        self.initial_swim_speed() 
        
        # Compute vel_mag from x_vel and y_vel (required by timestep)
        if hasattr(self, 'x_vel') and hasattr(self, 'y_vel'):
            self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
        else:
            self.vel_mag = np.zeros(self.num_agents)
        
        # If using HECRAS mode, initialize agent environment attributes
        if self.use_hecras and self.hecras_plan_path:
            try:
                with h5py.File(self.hecras_plan_path, 'r') as hdf:
                    pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
                    
                    # Set depth_rast_transform from HECRAS extent (required by vel_cue)
                    if pts is not None:
                        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                        cell_size = 10.0  # Approximate cell size
                        self.depth_rast_transform = from_origin(x_min, y_max, cell_size, cell_size)
                    
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
                
                if pts is not None and node_fields:
                    # Initialize agent arrays
                    if not hasattr(self, 'depth'):
                        self.depth = np.zeros(self.num_agents, dtype=float)
                    if not hasattr(self, 'x_vel'):
                        self.x_vel = np.zeros(self.num_agents, dtype=float)
                    if not hasattr(self, 'y_vel'):
                        self.y_vel = np.zeros(self.num_agents, dtype=float)
                    if not hasattr(self, 'wet'):
                        self.wet = np.ones(self.num_agents, dtype=float)
                    if not hasattr(self, 'distance_to'):
                        self.distance_to = np.zeros(self.num_agents, dtype=float)
                    
                    # Enable HECRAS mapping
                    self.enable_hecras(pts, node_fields, k=self.hecras_k)
                    
                    # Sample initial values
                    if 'depth' in node_fields:
                        self.depth = self.apply_hecras_mapping(node_fields['depth'])
                    if 'vel_x' in node_fields:
                        self.x_vel = self.apply_hecras_mapping(node_fields['vel_x'])
                    if 'vel_y' in node_fields:
                        self.y_vel = self.apply_hecras_mapping(node_fields['vel_y'])
                        self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
                    
            except Exception as e:
                print(f"Warning: Failed to initialize HECRAS environment: {e}")
        
        # error array
        self.error_array = np.array([])
        
        # initialize cumulative time
        self.cumulative_time = 0.
        
        # RL: Initialize behavioral weights (will be loaded/trained)
        self.behavioral_weights = BehavioralWeights()
        self._initial_positions = None  # Store initial agent positions for spatial reset

    # =============================================================================
    # RL-Specific Methods for Behavioral Learning
    # =============================================================================
    
    def apply_behavioral_weights(self, weights):
        """Apply learned behavioral weights to simulation.
        
        This updates the force weights used in behavioral cues (cohesion, alignment,
        separation, border avoidance, etc.) without changing spatial state.
        
        Args:
            weights: BehavioralWeights object with learned parameters
        """
        self.behavioral_weights = weights
        try:
            wdict = weights.to_dict()
            keys = ', '.join(sorted(wdict.keys()))
            print(f"Applied behavioral weights (keys): {keys}")
        except Exception:
            try:
                print('Applied behavioral weights (some fields may be unavailable)')
            except Exception:
                pass
        # Numba warm-up: call compiled loops with tiny inputs to trigger JIT at initialization
        try:
            if 'numba' in globals():
                # create tiny dummy arrays
                dummy_pos = np.zeros((2, 2), dtype=np.float64)
                dummy_head = np.zeros(2, dtype=np.float64)
                dummy_data = np.array([0, 1], dtype=np.int32)
                dummy_offsets = np.array([0, 2], dtype=np.int32)
                dummy_coh = np.zeros(2, dtype=np.float64)
                dummy_align = np.zeros(2, dtype=np.float64)
                try:
                    _compute_schooling_loop(dummy_pos, dummy_head, dummy_data, dummy_offsets, 1.0, 1.0, dummy_coh, dummy_align, 2)
                except Exception:
                    pass
                dummy_drag = np.zeros(2, dtype=np.float64)
                try:
                    _compute_drafting_loop(dummy_pos, dummy_head, dummy_data, dummy_offsets, 0.5, 0.1, 0.2, dummy_drag, 2)
                except Exception:
                    pass
        except Exception:
            pass
    
    def reset_spatial_state(self, reset_positions=False):
        """Reset all spatial/ephemeral state for a new simulation episode.
        
        This preserves learned behavioral weights but resets:
        - Agent positions (back to starting area or randomized)
        - Velocities and headings (randomized)
        - Energy/battery levels
        - Memory buffers (eddy escape, past positions)
        - Dead/alive status
        
        Args:
            reset_positions: If True, randomize positions within start polygon.
                           If False, restore to saved initial positions.
        
        Behavioral weights (cohesion, separation, rheotaxis, etc.) are NOT reset.
        """
        # Save initial positions if not already saved
        if self._initial_positions is None or reset_positions:
            if reset_positions and hasattr(self, 'start_polygon_path'):
                # Re-randomize positions
                import geopandas as gpd
                from shapely.geometry import Point
                start_poly = gpd.read_file(self.start_polygon_path)
                geometry = start_poly.geometry.iloc[0]
                minx, miny, maxx, maxy = geometry.bounds
                
                X, Y = [], []
                attempts = 0
                max_attempts = self.num_agents * 10
                while len(X) < self.num_agents and attempts < max_attempts:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    pnt = Point(x, y)
                    if geometry.contains(pnt):
                        X.append(x)
                        Y.append(y)
                    attempts += 1
                
                if len(X) < self.num_agents:
                    raise ValueError(f"Could only find {len(X)} valid positions in start polygon")
                
                self.X = np.array(X)[:self.num_agents]
                self.Y = np.array(Y)[:self.num_agents]
            
            self._initial_positions = {
                'X': self.X.copy(),
                'Y': self.Y.copy(),
                'Z': self.Z.copy() if hasattr(self, 'Z') else None
            }
        else:
            # Reset positions to saved initial state
            self.X = self._initial_positions['X'].copy()
        # Randomize headings and small SOG variation so agents do not all point into flow
        try:
            # Add small random heading perturbation uniform [-pi, pi]
            self.heading = np.random.uniform(-np.pi, np.pi, size=self.num_agents)
        except Exception:
            pass
        try:
            # Slightly randomize SOG around current ideal_sog (±10%)
            if hasattr(self, 'ideal_sog'):
                base = np.asarray(self.ideal_sog)
                if base.size == 1:
                    base = np.full(self.num_agents, float(base))
                # choose jitter fraction depending on 'strong' flag
                frac = self.strong_initial_sog_jitter_fraction if getattr(self, 'strong_initial_sog_jitter', False) else self.initial_sog_jitter_fraction
                low = 1.0 - float(frac)
                high = 1.0 + float(frac)
                self.sog = base * np.random.uniform(low, high, size=self.num_agents)
                self.ideal_sog = self.sog.copy()
        except Exception:
            pass

        # Add a small random initial velocity jitter so agents don't all move identically
        try:
            jitter_scale = getattr(self, 'initial_velocity_jitter', 0.05)
            vel_jitter = np.random.normal(scale=float(jitter_scale), size=(self.num_agents, 2))
            if not hasattr(self, 'x_vel') or not hasattr(self, 'y_vel'):
                # initialize velocity arrays if missing
                self.x_vel = np.zeros(self.num_agents)
                self.y_vel = np.zeros(self.num_agents)
            self.x_vel = self.x_vel + vel_jitter[:, 0]
            self.y_vel = self.y_vel + vel_jitter[:, 1]
        except Exception:
            pass

        # Optionally randomize behavioral weights per agent (small gaussian perturbation)
        try:
            if hasattr(self, 'behavioral_weights') and getattr(self, 'randomize_weights_on_reset', True):
                bw = self.behavioral_weights
                # Perturb numeric fields only
                for k, v in bw.to_dict().items():
                    if isinstance(v, (int, float)):
                        # Add small gaussian noise (std = 5% of value or 0.05 absolute)
                        std = max(abs(v) * 0.05, 0.05)
                        try:
                            newv = float(v) + np.random.normal(0, std)
                            setattr(bw, k, newv)
                        except Exception:
                            pass
                # apply mutated weights
                try:
                    self.apply_behavioral_weights(bw)
                except Exception:
                    pass
        except Exception:
            pass
            self.Y = self._initial_positions['Y'].copy()
            if self._initial_positions['Z'] is not None:
                self.Z = self._initial_positions['Z'].copy()
        
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()
        
        # Reset velocities and heading (randomize based on flow)
        self.initial_heading()
        self.initial_swim_speed()
        
        # Reset energy/battery
        if hasattr(self, 'battery'):
            self.battery = np.ones(self.num_agents) * 100.0  # Full battery
        
        # Reset swim behavior state
        if hasattr(self, 'swim_behav'):
            self.swim_behav = np.zeros(self.num_agents)
        if hasattr(self, 'swim_mode'):
            self.swim_mode = np.ones(self.num_agents)  # Default mode
        if hasattr(self, 'recover_stopwatch'):
            self.recover_stopwatch = np.zeros(self.num_agents)
        
        # Reset memory buffers
        max_timesteps = getattr(self, 'swim_speeds', np.array([])).shape[1] if hasattr(self, 'swim_speeds') else 600
        self.swim_speeds = np.full((self.num_agents, max_timesteps), np.nan)
        self.past_centerline_meas = np.full((self.num_agents, max_timesteps), np.nan)
        self.current_centerline_meas = np.zeros_like(self.X)
        self.in_eddy = np.zeros_like(self.X)
        self.time_since_eddy_escape = np.zeros_like(self.X)
        
        # Reset dead/alive status
        if hasattr(self, 'dead'):
            self.dead = np.zeros(self.num_agents)
        if hasattr(self, 'time_out_of_water'):
            self.time_out_of_water = np.zeros(self.num_agents)
        
        # Reset odometer
        if hasattr(self, 'kcal'):
            self.kcal = np.zeros(self.num_agents)
        if hasattr(self, 'dist_per_bout'):
            self.dist_per_bout = np.zeros(self.num_agents)
        if hasattr(self, 'bout_dur'):
            self.bout_dur = np.zeros(self.num_agents)
        if hasattr(self, 'bout_no'):
            self.bout_no = np.zeros(self.num_agents)
        
        # Clear pixel index cache (spatial)
        self._pixel_index_cache = {}
        # Clear other caches that may hold large arrays or objects
        try:
            if hasattr(self, '_kdtree_cache'):
                try:
                    del self._kdtree_cache
                except Exception:
                    self._kdtree_cache = {}
        except Exception:
            pass
        try:
            if hasattr(self, '_hecras_map'):
                try:
                    del self._hecras_map
                except Exception:
                    self._hecras_map = None
        except Exception:
            pass
        # Drop large historical buffers if present to reclaim memory
        try:
            if hasattr(self, 'swim_speeds'):
                self.swim_speeds = np.full((self.num_agents, 1), np.nan)
            if hasattr(self, 'past_centerline_meas'):
                self.past_centerline_meas = np.full((self.num_agents, 1), np.nan)
        except Exception:
            pass
        # Force garbage collection to free memory immediately
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        
        # Re-sample environmental conditions at reset positions
        if self.use_hecras and hasattr(self, 'hecras_node_fields'):
            self.update_hecras_mapping_for_current_positions()
            if 'depth' in self.hecras_node_fields:
                self.depth = self.apply_hecras_mapping(self.hecras_node_fields['depth'])
                self.wet = np.where(self.depth > self.too_shallow / 2.0, 1.0, 0.0)
            if 'vel_x' in self.hecras_node_fields:
                self.x_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_x'])
            if 'vel_y' in self.hecras_node_fields:
                self.y_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_y'])
                self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
            if 'distance_to' in self.hecras_node_fields:
                self.distance_to = self.apply_hecras_mapping(self.hecras_node_fields['distance_to'])
        
        print("Spatial state reset complete. Behavioral weights preserved.")
    
    def save_behavioral_weights(self, filepath):
        """Save current behavioral weights to file."""
        self.behavioral_weights.save(filepath)
    
    def load_behavioral_weights(self, filepath):
        """Load pre-trained behavioral weights from file."""
        self.behavioral_weights.load(filepath)
        self.apply_behavioral_weights(self.behavioral_weights)

    def sim_sex(self):
        """
        Simulate the sex distribution of agents based on the basin.
    
        Notes:
        - The method sets the `sex` attribute of the class, which is an array representing the sex of each agent.
        - Currently, the method only has data for the "Nushagak River" basin. For this basin, the sex distribution 
          is determined based on given probabilities for male (0) and female (1).
        - If the basin is not "Nushagak River", uses default 50/50 distribution.
    
        Attributes set:
        - sex (array): Array of size `num_agents` with values 0 (male) or 1 (female) representing the sex of each agent.
        """
        if self.basin == "Nushagak River":
            self.sex = self.arr.random.choice([0,1], size = self.num_agents, p = [0.503,0.497])
        else:
            # Default 50/50 sex distribution for other basins
            self.sex = self.arr.random.choice([0,1], size = self.num_agents, p = [0.5,0.5])
            
    def sim_length(self, fish_length = None):
        """
        Simulate the length distribution of agents based on the basin and their sex.
    
        Notes:
        - The method sets the `length` attribute of the class, which is an array representing the length of each agent in mm.
        - Currently, the method only has data for the "Nushagak River" basin. For this basin, the length distribution 
          is determined based on given lognormal distributions for male and female agents.
        - The method also sets other attributes that are functions of the agent's length.
    
        Attributes set:
        - length (array): Array of size `num_agents` representing the length of each agent in mm.
        - sog (array): Speed over ground for each agent, assumed to be 1 body length per second.
        - ideal_sog (array): Ideal speed over ground for each agent, set to be the same as `sog`.
        - swim_speed (array): Initial swim speed for each agent, set to be the same as `length/1000`.
        - ucrit (array): Critical swimming speed for each agent, set to be `sog * 7`.
    
        Raises:
        - ValueError: If the `sex` attribute is not recognized.
        """
        # length in mm
        if self.pid_tuning == True:
            self.length = np.repeat(fish_length,self.num_agents) # testing
        
        else:
            if self.basin == "Nushagak River":
                self.length=np.where(self.sex==0,
                         self.arr.random.lognormal(mean = 6.426,sigma = 0.072,size = self.num_agents),
                         self.arr.random.lognormal(mean = 6.349,sigma = 0.067,size = self.num_agents))
            else:
                # Default length distribution for other basins
                self.length = self.arr.random.lognormal(mean = 6.39, sigma = 0.07, size = self.num_agents)
        
        # we can also set these arrays that contain parameters that are a function of length
        # ensure self.length exists (fallback to 475 mm if not previously set)
        if not hasattr(self, 'length') or self.length is None:
            self.length = np.repeat(475., self.num_agents)
        self.length = np.where(self.length < 475., 475., self.length)
        self.sog = self.length/1000. #* 0.8  # sog = speed over ground - assume fish maintain 1 body length per second
        self.ideal_sog = self.length/1000. # self.sog
        self.opt_sog = self.length/1000. #* 0.8
        self.school_sog = self.length/1000.
       # self.swim_speed = self.length/1000.        # set initial swim speed
        self.ucrit = self.sog * 1.6    # TODO - what is the ucrit for sockeye?
        
    def sim_weight(self):
        '''function simulates a fish weight out of the user provided basin and 
        sex of fish'''
        # using a W = a * L^b relationship given in fishbase - weight in kg
        self.weight = (0.0155 * (self.length/10.0)**3)/1000.
        
    def sim_body_depth(self):
        '''function simulates a fish body depth out of the user provided basin and 
        sex of fish'''
        # body depth is in cm
        if self.basin == "Nushagak River":
            self.body_depth=np.where(self.sex==0,
                        self.arr.exp(-1.938 + np.log(self.length) * 1.084 + 0.0435) / 10.,
                        self.arr.exp(-1.938 + np.log(self.length) * 1.084) / 10.)
        else:
            # Default body depth calculation for other basins (sex-averaged)
            self.body_depth = self.arr.exp(-1.938 + np.log(self.length) * 1.084 + 0.02175) / 10.
                
        # ensure body_depth exists
        if not hasattr(self, 'body_depth') or self.body_depth is None:
            self.body_depth = np.repeat(1.0, self.num_agents)
        self.too_shallow = self.body_depth /100. / 2. # m
        self.opt_wat_depth = self.body_depth /100 * 3.0 + self.too_shallow
        
    def initialize_hdf5(self):
        '''Initialize an HDF5 database for a simulation'''
        # Create groups for organization (optional)
        agent_data = self.hdf5.create_group("agent_data")
        
        # Create datasets for agent properties that are static
        agent_data.create_dataset("sex", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("length", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("ucrit", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("weight", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("body_depth", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("too_shallow", (self.num_agents,), dtype='f4')
        agent_data.create_dataset("opt_wat_depth", (self.num_agents,), dtype='f4')
      
        # Create datasets for agent properties that change with time
        # Use chunking optimized for column (per-timestep) writes: chunks=(num_agents,1)
        chunk_shape = (self.num_agents, 1)
        agent_data.create_dataset("X", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Y", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Z", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("prev_X", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("prev_Y", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("heading", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("sog", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("ideal_sog", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_speed", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("battery", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_behav", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_mode", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("recover_stopwatch", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("ttfr", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("time_out_of_water", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("drag", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("thrust", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Hz", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("bout_no", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("dist_per_bout", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("bout_dur", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("time_of_jump", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("kcal", (self.num_agents, self.num_timesteps), dtype='f4', chunks=chunk_shape)
        
        # Set attributes (metadata) if needed
        self.hdf5.attrs['simulation_name'] = "%s Sockeye Movement Simulation"%(self.basin)
        self.hdf5.attrs['num_agents'] = self.num_agents 
        self.hdf5.attrs['num_timesteps'] = self.num_timesteps
        self.hdf5.attrs['basin'] = self.basin
        self.hdf5.attrs['crs'] = self.crs
        
        self.hdf5.flush()
        
    def timestep_flush(self, timestep):
        if self.pid_tuning == False:
            '''function writes to the open hdf5 file '''
            # write into in-memory buffers and flush periodically
            buf_vals = {
                'X': self.X.astype('float32'),
                'Y': self.Y.astype('float32'),
                'Z': self.z.astype('float32'),
                'prev_X': self.prev_X.astype('float32'),
                'prev_Y': self.prev_Y.astype('float32'),
                'heading': self.heading.astype('float32'),
                'sog': self.sog.astype('float32'),
                'ideal_sog': self.ideal_sog.astype('float32'),
                'swim_speed': self.swim_speed.astype('float32'),
                'battery': self.battery.astype('float32'),
                'swim_behav': self.swim_behav.astype('float32'),
                'swim_mode': self.swim_mode.astype('float32'),
                'recover_stopwatch': self.recover_stopwatch.astype('float32'),
                'ttfr': self.ttfr.astype('float32'),
                'time_out_of_water': self.time_out_of_water.astype('float32'),
                'drag': np.linalg.norm(self.drag, axis=-1).astype('float32'),
                'thrust': np.linalg.norm(self.thrust, axis=-1).astype('float32'),
                'Hz': self.Hz.astype('float32'),
                'bout_no': self.bout_no.astype('float32'),
                'dist_per_bout': self.dist_per_bout.astype('float32'),
                'bout_dur': self.bout_dur.astype('float32'),
                'kcal': self.kcal.astype('float32')
            }

            pos = self._buffer_pos
            for k, v in buf_vals.items():
                if k in self._hdf5_buffers:
                    self._hdf5_buffers[k][:, pos] = v

            self._buffer_pos += 1
            if self._buffer_pos >= self.flush_interval or timestep == (self.num_timesteps - 1):
                write_len = self._buffer_pos
                t_end = timestep
                t_start = t_end - write_len + 1
                # If deferring HDF writes, write buffered agent arrays to the fast LogWriter
                if getattr(self, 'defer_hdf', False):
                    # If memmap writer requested, initialize it on first flush
                    if getattr(self, 'defer_log_fmt', 'npz') == 'memmap' and getattr(self, '_memmap_writer', None) is None:
                        try:
                            var_shapes = {k: (self.num_agents, self.num_timesteps) for k in self._hdf5_buffers.keys()}
                            from emergent.io.log_writer_memmap import MemmapLogWriter
                            out_dir = self._memmap_config.get('out_dir', os.path.join(self.model_dir, 'logs', 'deferred'))
                            self._memmap_writer = MemmapLogWriter(out_dir, var_shapes, dtype=np.float32)
                        except Exception:
                            self._memmap_writer = None

                    # write buffered timesteps using memmap writer if available
                    if getattr(self, '_memmap_writer', None) is not None:
                        # Build a dict of 2D arrays shaped (num_agents, write_len)
                        arrays_2d = {}
                        for k in self._hdf5_buffers.keys():
                            try:
                                arrays_2d[k] = self._hdf5_buffers[k][:, :write_len].astype('f4')
                            except Exception:
                                arrays_2d[k] = np.zeros((self.num_agents, write_len), dtype='f4')
                        try:
                            self._memmap_writer.append_block(t_start, arrays_2d)
                        except Exception:
                            # fallback to per-step append on failure
                            for offset in range(write_len):
                                t_idx = t_start + offset
                                arrays = {k: self._hdf5_buffers[k][:, offset].astype('f4') for k in self._hdf5_buffers.keys()}
                                try:
                                    self._memmap_writer.append(t_idx, arrays)
                                except Exception:
                                    pass
                    elif getattr(self, '_log_writer', None) is not None:
                        # fallback to existing npz writer
                        for offset in range(write_len):
                            t_idx = t_start + offset
                            arrays = {k: self._hdf5_buffers[k][:, offset].astype('f4') for k in self._hdf5_buffers.keys()}
                            try:
                                self._log_writer.append(t_idx, arrays)
                            except Exception:
                                pass
                else:
                    for k, buf in self._hdf5_buffers.items():
                        ds_name = f'agent_data/{k}'
                        if ds_name in self.hdf5:
                            try:
                                self.hdf5[ds_name][:, t_start:t_end+1] = buf[:, :write_len]
                            except Exception:
                                for offset in range(write_len):
                                    self.hdf5[ds_name][:, t_start + offset] = buf[:, offset]
                # reset buffer
                for k in list(self._hdf5_buffers.keys()):
                    self._hdf5_buffers[k][:] = 0
                self._buffer_pos = 0
                # Flush mental-map accumulators to HDF5 in a batched manner
                try:
                    mem_grp = self.hdf5['memory']
                    for aid in range(self.num_agents):
                        ds = mem_grp.get(str(aid))
                        if ds is None:
                            continue
                        # Logical OR the accumulator into the stored dataset
                        acc = self.mental_map_accumulator[aid]
                        if np.any(acc):
                            # read existing, OR, and write back in one shot
                            existing = ds[:, :]
                            np.maximum(existing, acc, out=existing)
                            ds[:, :] = existing
                            # clear accumulator for this agent
                            acc.fill(0)
                    # Flush refugia accumulators if present
                    try:
                        refugia_grp = self.hdf5['refugia']
                        for aid in range(self.num_agents):
                            acc_r = getattr(self, 'refugia_accumulator', None)
                            if acc_r is None:
                                break
                            acc_arr = acc_r[aid]
                            if not np.any(acc_arr):
                                continue
                            dsr = refugia_grp.get(str(aid))
                            if dsr is None:
                                continue
                            # read existing and overwrite where accumulator non-zero
                            existing_r = dsr[:, :]
                            mask = acc_arr != 0
                            existing_r[mask] = acc_arr[mask]
                            dsr[:, :] = existing_r
                            acc_arr.fill(0)
                    except Exception:
                        pass
                except Exception:
                    # If memory group doesn't exist or write fails, skip silently
                    pass
                self.hdf5.flush()

    def enviro_import(self, data_dir, surface_type):
        """
        Imports environmental raster data and writes it to an HDF5 file.
    
        Parameters:
        - data_dir (str): Path to the raster file to be imported.
        - surface_type (str): Type of the surface data being imported. 
                              Acceptable values include 'velocity x', 'velocity y', 'depth', 
                              'wsel', 'elevation', 'velocity direction', 'velocity magnitude'.
    
        Attributes set:
        - vel_x_rast_transform, vel_y_rast_transform, depth_rast_transform, etc. (Affine): 
          Transformation matrix for the respective raster data.
        - width (int): Width of the raster data.
        - height (int): Height of the raster data.
    
        Notes:
        - The method creates a group named "environment" in the HDF5 file to organize the raster datasets.
        - The raster data is read using rasterio and written to the HDF5 file under the "environment" group.
        - The transformation matrix for each raster type is stored as an attribute of the Simulation class.
        - The method sets the width and height attributes of the Simulation class based on the raster dimensions.
    
        Raises:
        - ValueError: If the provided surface_type is not recognized.
        """
        
        # Helper to create or overwrite HDF5 dataset
        def _create_or_replace_dataset(group, name, shape, dtype='f4', chunks=None):
            if name in group:
                del group[name]
            return group.create_dataset(name, shape, dtype=dtype, chunks=chunks)
       

        
        # Guard: if data_dir is falsy or file does not exist, skip
        if not data_dir or not os.path.exists(data_dir):
            return
        # get raster properties
        try:
            src = rasterio.open(data_dir)
        except Exception:
            return
        num_bands = src.count
        width = src.width
        height = src.height
        dtype = np.float32
        transform = src.transform
        self.no_data_value = src.nodatavals[0]
        
        # Create groups for organization (optional)
        if 'environment' not in self.hdf5:
            env_data = self.hdf5.create_group("environment")
        else:
            env_data = self.hdf5['environment']
        
        # Always set width/height from current raster
        self.width = width
        self.height = height

        # Create x_coords and y_coords if they don't exist (needed for all behavioral cues)
        if 'x_coords' not in self.hdf5:
            print(f"Creating x_coords and y_coords with dimensions: height={height}, width={width}, rows={src.shape[0]}, cols={src.shape[1]}")
            # Get the dimensions of the raster
            rows, cols = src.shape
        
            # Define chunk size (you can adjust this based on your memory constraints)
            chunk_size = 1024  # Example chunk size
        
            # Set up HDF5 file and datasets
            dset_x = self.hdf5.create_dataset('x_coords', (height, width), dtype='float32')
            dset_y = self.hdf5.create_dataset('y_coords', (height, width), dtype='float32')
            print(f"Created datasets with shape: {dset_x.shape}")
        
            # Process and write in chunks
            for i in range(0, rows, chunk_size):
                row_chunk = slice(i, min(i + chunk_size, rows))
                row_indices, col_indices = np.meshgrid(np.arange(row_chunk.start, row_chunk.stop), np.arange(cols), indexing='ij')
    
                # Apply the affine transformation
                x_coords, y_coords = transform * (col_indices, row_indices)
    
                # Write the chunk to the HDF5 datasets
                dset_x[row_chunk, :] = x_coords
                dset_y[row_chunk, :] = y_coords
            
            # Flush to ensure data is written
            self.hdf5.flush()
            print(f"Successfully created x_coords and y_coords; flushed to disk. Shape: {dset_x.shape}, first value: {dset_x[0,0]}")
        else:
            print(f"x_coords already exists with shape: {self.hdf5['x_coords'].shape}")

        shape = (num_bands, height, width)
        #shape = (num_bands, width, height)
        
        if surface_type == 'wetted':
            # set transform as parameter of simulation
            self.wetted_transform = transform
                 
            # get data 
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "wetted", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/wetted'][:, :] = arr
            
        elif surface_type == 'velocity x':
            # set transform as parameter of simulation
            self.vel_x_rast_transform = transform
            
            # get data 
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "vel_x", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/vel_x'][:, :] = arr

        elif surface_type == 'velocity y':
            # set transform as parameter of simulation            
            self.vel_y_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "vel_y", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/vel_y'][:, :] = arr
            
        elif surface_type == 'depth':
            # set transform as parameter of simulation            
            self.depth_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)
           
            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "depth", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/depth'][:, :] =arr
            
        elif surface_type == 'wsel':
            # set transform as parameter of simulation            
            self.wsel_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "wsel", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/wsel'][:, :] = src.read(1)
            
        elif surface_type == 'elevation':
            # set transform as parameter of simulation                        
            self.elev_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "elevation", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/elevation'][:, :] = arr
                
        elif surface_type == 'velocity direction':          
            # set transform as parameter of simulation                        
            self.vel_dir_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)

            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "vel_dir", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/vel_dir'][:, :] = src.read(1) 
                
        elif surface_type == 'velocity magnitude': 
            # set transform as parameter of simulation                        
            self.vel_mag_rast_transform = transform
            
            # get data and desribe
            arr = src.read(1)
            
            # create an hdf5 array and write to it
            _create_or_replace_dataset(env_data, "vel_mag", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
            self.hdf5['environment/vel_mag'][:, :] = arr
            
        self.width = width
        self.height = height
        self.hdf5.flush()
        src.close()


    def centerline_import(self, shapefile):
        # Load the shapefile with the centerline, or accept a shapely geometry
        if shapefile is None:
            self.centerline = None
            # Keep legacy attribute name in sync
            # legacy attribute removed; use `centerline` only
            return None

        # If a shapely geometry was passed in directly, use it
        if isinstance(shapefile, BaseGeometry):
            self.centerline = shapefile
            # Centerline attribute set by initialize_hecras_geometry or boundary_surface
            # legacy attribute removed; use `centerline` only
            return self.centerline

        # Otherwise assume a filepath and read with geopandas
        line_gdf = gpd.read_file(shapefile)
        if len(line_gdf) == 0:
            self.centerline = None
            return None
        self.centerline = line_gdf.geometry.iloc[0]
        # Centerline attribute used throughout for upstream progress tracking
        return self.centerline
        
    def compute_linear_positions(self, line):
        # Vectorized projection of points onto a polyline (line can be a shapely LineString)
        # Convert line coordinates to numpy arrays
        coords = np.asarray(line.coords)
        xs_line = coords[:, 0]
        ys_line = coords[:, 1]

        # Agent points
        px = self.X.flatten()
        py = self.Y.flatten()

        # Helper: project points to cumulative distance along polyline
        def _project_points_onto_line_numpy(xs_line, ys_line, px, py):
            # segments
            seg_x0 = xs_line[:-1]
            seg_y0 = ys_line[:-1]
            seg_x1 = xs_line[1:]
            seg_y1 = ys_line[1:]
            vx = seg_x1 - seg_x0
            vy = seg_y1 - seg_y0
            seg_len = np.hypot(vx, vy)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])

            # For each point, compute projection onto each segment, pick the closest
            # vectorized over segments using broadcasting where feasible
            # px shape (M,), seg_x0 shape (S,) -> we compute per-point per-segment
            M = px.size
            S = seg_x0.size

            # Expand arrays
            px_e = px[:, None]
            py_e = py[:, None]
            x0_e = seg_x0[None, :]
            y0_e = seg_y0[None, :]
            vx_e = vx[None, :]
            vy_e = vy[None, :]
            seg_len_e = seg_len[None, :]

            wx = px_e - x0_e
            wy = py_e - y0_e
            # projection factor along each segment (t) = (w.v) / (v.v)
            denom = vx_e * vx_e + vy_e * vy_e
            # avoid div0
            denom = np.where(denom == 0, 1e-12, denom)
            t = (wx * vx_e + wy * vy_e) / denom
            t_clamped = np.clip(t, 0.0, 1.0)

            # Closest point coords
            cx = x0_e + t_clamped * vx_e
            cy = y0_e + t_clamped * vy_e

            # distances squared from point to projected point
            d2 = (px_e - cx) ** 2 + (py_e - cy) ** 2

            # choose segment with minimum distance
            idx = np.argmin(d2, axis=1)
            # gather t and cumulative length per point
            chosen_t = t_clamped[np.arange(M), idx]
            chosen_seg = idx
            distances_along = cumlen[chosen_seg] + chosen_t * seg_len[chosen_seg]
            return distances_along

        # If an along-stream raster exists, prefer sampling that raster for speed/scale
        try:
            env = self.hdf5.get('environment')
            if env is not None and 'along_stream_dist' in env:
                # sample raster at agent positions (vectorized)
                # prefer in-memory cache if available
                xs = self._env_cache.get('x_coords') if getattr(self, '_env_cache', None) is not None else None
                ys = self._env_cache.get('y_coords') if getattr(self, '_env_cache', None) is not None else None
                rast = self._env_cache.get('along_stream_dist') if getattr(self, '_env_cache', None) is not None else None
                if xs is None or ys is None or rast is None:
                    xs = env['x_coords'][:]
                    ys = env['y_coords'][:]
                    rast = env['along_stream_dist'][:]
                # convert agent XY to pixel indices using inverse transform if available
                try:
                    inv = get_inv_transform(self, getattr(self, 'depth_rast_transform', None))
                    rows, cols = geo_to_pixel_from_inv(inv, self.X.flatten(), self.Y.flatten())
                except Exception:
                    # fallback to brute nearest neighbor in coords arrays
                    flat_xy = np.column_stack((xs.ravel(), ys.ravel()))
                    pts = np.column_stack((self.X.flatten(), self.Y.flatten()))
                    from scipy.spatial import cKDTree
                    tree = cKDTree(flat_xy)
                    dists, inds = tree.query(pts, k=1)
                    rows = (inds // xs.shape[1]).astype(int)
                    cols = (inds % xs.shape[1]).astype(int)
                # clamp
                rows = np.clip(rows, 0, rast.shape[0]-1)
                cols = np.clip(cols, 0, rast.shape[1]-1)
                vals = rast[rows, cols]
                # reshape to agent grid
                return vals.reshape(self.X.shape)
        except Exception:
            pass

        # prefer numba helper when available
        if _HAS_NUMBA:
            try:
                dists = _wrap_project_points_onto_line_numba(xs_line, ys_line, px, py)
            except Exception:
                dists = _project_points_onto_line_numpy(xs_line, ys_line, px, py)
        else:
            dists = _project_points_onto_line_numpy(xs_line, ys_line, px, py)
        return dists.reshape(self.X.shape)

    def boundary_surface(self):
        """Compute distance-to-boundary surface.
        
        For HECRAS mode: Uses distance_to_bank from wetted perimeter
        For raster mode: Uses distance transform on wetted raster
        """
        # Check if we're in HECRAS mode and already have distance_to_bank
        if self.use_hecras and hasattr(self, '_hecras_geometry_info'):
            print("Using HECRAS distance-to-bank (already computed)")
            # distance_to_bank was already written to HDF5 by initialize_hecras_geometry
            return
        
        # Fallback: Raster-based mode
        # Guard: require the 'environment' group and 'wetted' dataset
        try:
            env = self.hdf5.get('environment')
            if env is None or 'wetted' not in env:
                print("Warning: No wetted raster found, skipping boundary_surface")
                return
            raster = env['wetted'][:]
            # Additional guard: ensure raster has valid shape
            if raster.size == 0 or raster.shape[0] == 0 or raster.shape[1] == 0:
                return
        except Exception:
            return

        # pixel width fallback
        try:
            pixel_width = self.depth_rast_transform.a
        except Exception:
            pixel_width = 1.0

        # Compute the Euclidean distance transform. This computes the distance to the nearest zero (background) for all non-zero (foreground) pixels.
        dist_to_bound = distance_transform_edt(raster != -9999) * pixel_width
        
        # Additional guard: ensure distance transform succeeded
        if dist_to_bound.size == 0 or dist_to_bound.shape[0] == 0 or dist_to_bound.shape[1] == 0:
            return
        
        # Create or access 'environment' group
        if 'environment' not in self.hdf5:
            env_data = self.hdf5.create_group('environment')
        else:
            env_data = self.hdf5['environment']

        # determine shape for distance dataset
        if hasattr(self, 'height') and hasattr(self, 'width'):
            h = int(self.height)
            w = int(self.width)
        else:
            # try to infer from x_coords/y_coords if present
            try:
                xs = env_data['x_coords'][:]
                h, w = xs.shape
                self.height = h
                self.width = w
            except Exception:
                # cannot determine shape; skip writing
                return

        # Create 'distance_to' dataset and write data
        if 'distance_to' in env_data:
            del env_data['distance_to']
        env_data.create_dataset('distance_to', (h, w), dtype='float32')
        env_data['distance_to'][:, :] = dist_to_bound

        safe_flush(self.hdf5)

    def initialize_mental_map(self):
        """
        Initializes the mental map for each agent.
        
        The mental map is a 3D array where each row corresponds to an agent, 
        and each agent's row is a 2D raster of shape (self.width, self.height).
        The values in the raster can represent various things, such as visited locations, 
        perceived obstacles, etc. Initially, all values are set to zero.
        
        Attributes:
            map (ndarray): A 3D array representing the mental maps of all agents.
                           Shape: (self.num_agents, self.width, self.height)
        """

        # Ensure a 'memory' group exists and create compact per-agent avoid maps.
        mem_data = self.hdf5.require_group('memory')

        # Handle missing width/height by using conservative defaults
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
            avoid_height = 256
            avoid_width = 256
        else:
            avoid_height = int(np.round(self.height / self.avoid_cell_size)) + 1
            avoid_width = int(np.round(self.width / self.avoid_cell_size)) + 1

        # Create per-agent datasets (if missing) using a compact dtype and
        # row-friendly chunking so our grouped-row reads/writes are efficient.
        for i in range(self.num_agents):
            name = str(i)
            if name not in mem_data:
                mem_data.create_dataset(name, shape=(avoid_height, avoid_width),
                                        dtype='i2', chunks=(1, min(avoid_width, 4096)))

        # Transform mapping agent coords -> mental map cell indices
        if hasattr(self, 'depth_rast_transform') and self.depth_rast_transform is not None:
            base_transform = self.depth_rast_transform
        else:
            # Fallback: derive transform from HECRAS coords when available
            base_transform = None
            try:
                if hasattr(self, '_hecras_maps') and len(self._hecras_maps) > 0:
                    m = next(iter(self._hecras_maps.values()))
                    coords = m.coords
                    base_transform = compute_affine_from_hecras(coords, target_cell_size=self.avoid_cell_size)
                    # set width/height approximate if missing using coords bbox
                    if not hasattr(self, 'width') or not hasattr(self, 'height'):
                        xrange = float(coords[:, 0].max() - coords[:, 0].min())
                        yrange = float(coords[:, 1].max() - coords[:, 1].min())
                        self.width = int(np.ceil(xrange / self.avoid_cell_size)) + 1
                        self.height = int(np.ceil(yrange / self.avoid_cell_size)) + 1
            except Exception:
                base_transform = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.avoid_cell_size)

        self.mental_map_transform = compute_affine_from_hecras(base_transform * 0.0 if False else np.array([[base_transform.c, base_transform.f]]), target_cell_size=self.avoid_cell_size) if False else base_transform

        # In-memory accumulator to batch per-timestep updates; dtype u1 is enough
        # since we only flag visited cells (0/1). Shape: (num_agents, H, W)
        self.mental_map_accumulator = np.zeros((self.num_agents, avoid_height, avoid_width), dtype='u1')

        # How frequently (timesteps) to flush the accumulator to HDF5.
        # Default kept at 50 unless user sets `mental_map_flush_interval` earlier.
        self.mental_map_flush_interval = getattr(self, 'mental_map_flush_interval', 50)

        self.hdf5.flush()
        
    def initialize_refugia_map(self):
        """
        Initializes the velocity map for each agent.
        
        The velocity map is a 3D array where each row corresponds to an agent, 
        and each agent's row is a 2D raster of shape (self.width, self.height).
        The values in the raster represent veloicty refugia.
        
        Attributes:
            map (ndarray): A 3D array representing the mental maps of all agents.
                           Shape: (self.num_agents, self.width, self.height)
        """

        # Ensure refugia group exists and create per-agent datasets
        mem_grp = self.hdf5.require_group('refugia')
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
            refugia_height = 256
            refugia_width = 256
        else:
            refugia_height = int(np.round(self.height / self.refugia_cell_size)) + 1
            refugia_width = int(np.round(self.width / self.refugia_cell_size)) + 1

        for i in range(self.num_agents):
            name = str(i)
            if name not in mem_grp:
                # create dataset with default fill (do not write large zero block)
                mem_grp.create_dataset(name, shape=(refugia_height, refugia_width), dtype='f4',
                                       chunks=(1, min(refugia_width, 4096)), fillvalue=0.0)

        # Apply the scaling transform for refugia (fallback to HECRAS-derived transform)
        if hasattr(self, 'depth_rast_transform') and self.depth_rast_transform is not None:
            base_t = self.depth_rast_transform
        else:
            try:
                if hasattr(self, '_hecras_maps') and len(self._hecras_maps) > 0:
                    m = next(iter(self._hecras_maps.values()))
                    coords = m.coords
                    base_t = compute_affine_from_hecras(coords, target_cell_size=self.refugia_cell_size)
                else:
                    base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.refugia_cell_size)
            except Exception:
                base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=self.refugia_cell_size)

        self.refugia_map_transform = base_t

        # In-memory refugia accumulator (stores float values per agent)
        self.refugia_accumulator = np.zeros((self.num_agents, refugia_height, refugia_width), dtype='f4')

        self.hdf5.flush()
 
    def sample_environment(self, transform, raster_name):
        """
        Sample the raster values at the given x, y coordinates using an open HDF5 file.

        Parameters:
        - x_coords: array of x coordinates
        - y_coords: array of y coordinates
        - raster_name: name of the raster dataset in the HDF5 file

        Returns:
        - values: array of sampled raster values
        """
        # Get the row, col indices for the coordinates
        rows, cols = geo_to_pixel(self.X, self.Y, transform)

        # Prefer in-memory cache if present to avoid h5py reads
        cache = getattr(self, '_env_cache', None)
        if cache is not None and raster_name in cache and cache[raster_name] is not None:
            data = cache[raster_name]
            # Ensure that the indices are within the bounds of the raster data
            rows = np.clip(np.round(rows).astype(int), 0, data.shape[0] - 1)
            cols = np.clip(np.round(cols).astype(int), 0, data.shape[1] - 1)

            # If agents are clustered in a small bbox, read a single contiguous block and slice
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            if (rmax - rmin + 1) * (cmax - cmin + 1) <= max(4 * self.num_agents, 256):
                block = data[rmin:rmax+1, cmin:cmax+1]
                vals = block[rows - rmin, cols - cmin]
                return np.asarray(vals).flatten()

            # fallback to grouped-row efficient indexing on the in-memory array
            vals = data[rows, cols]
            return np.asarray(vals).flatten()

        # Use the already open HDF5 file object to read the specified raster dataset (no flush)
        env = self.hdf5['environment']
        raster_dataset = env[raster_name] if raster_name in env else self.hdf5['environment/%s' % (raster_name)]

        # Ensure that the indices are within the bounds of the raster data
        rows = np.clip(np.round(rows).astype(int), 0, raster_dataset.shape[0] - 1)
        cols = np.clip(np.round(cols).astype(int), 0, raster_dataset.shape[1] - 1)

        # If agents are clustered in a small bbox, read a single contiguous block and slice
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        if (rmax - rmin + 1) * (cmax - cmin + 1) <= max(4 * self.num_agents, 256):
            block = raster_dataset[rmin:rmax+1, cmin:cmax+1]
            vals = block[rows - rmin, cols - cmin]
            return np.asarray(vals).flatten()

        # fallback to grouped-row efficient indexing
        values = self._h5_advanced_index(raster_dataset, rows, cols)
        return np.asarray(values).flatten()

    def batch_sample_environment(self, transforms, raster_names):
        """Sample multiple rasters in a single pass and return dict of name->values.

        transforms: list of Affine transforms matching raster_names.
        raster_names: list of dataset names in HDF5 'environment' group.
        """
        # If running in HECRAS mode, map HECRAS values for agents each timestep
        if getattr(self, 'use_hecras', False) and getattr(self, 'hecras_plan_path', None):
            # ensure we have x_coords/y_coords in HDF for legacy consumers
            try:
                target_shape = None
                target_transform = getattr(self, 'depth_rast_transform', None)
                if hasattr(self, 'height') and hasattr(self, 'width'):
                    target_shape = (self.height, self.width)
                ensure_hdf_coords_from_hecras(self, self.hecras_plan_path, target_shape=target_shape, target_transform=target_transform)
            except Exception:
                pass

            # Optionally update full environment rasters from HECRAS each timestep
            # (controlled by `self.hecras_write_rasters`, default False for speed).
            if getattr(self, 'hecras_write_rasters', False):
                try:
                    map_hecras_to_env_rasters(self, self.hecras_plan_path, field_names=getattr(self, 'hecras_fields', None), k=getattr(self, 'hecras_k', 8))
                except Exception:
                    pass

            # Map HECRAS values directly to agents for requested raster names
            agent_xy = np.column_stack((self.X, self.Y))

            results = {}
            # candidate HECRAS field name choices per raster_name
            candidates = {
                'depth': ['Cell Hydraulic Depth', 'Water Surface', 'Cells Minimum Elevation', 'Cell Hydraulic Depth'],
                'vel_x': ['Cell Velocity - Velocity X', 'Velocity X', 'Vel X', 'Cell Velocity X'],
                'vel_y': ['Cell Velocity - Velocity Y', 'Velocity Y', 'Vel Y', 'Cell Velocity Y'],
                'vel_mag': ['Velocity Magnitude', 'Velocity Speed', 'Cell Velocity Magnitude', 'Cell Velocity - Velocity Magnitude'],
                'wetted': ['Wetted', 'Wetted Area', 'WettedIndicator'],
                'distance_to': ['Distance', 'Distance To']
            }
            k = getattr(self, 'hecras_k', 8)
            for name in raster_names:
                mapped = None
                if name in candidates:
                    for cand in candidates[name]:
                        try:
                            vals = map_hecras_for_agents(self, agent_xy, self.hecras_plan_path, field_names=[cand], k=k)
                            # map_hecras_for_agents returns array for single field
                            if vals is not None:
                                mapped = np.asarray(vals).flatten()
                                break
                        except Exception:
                            mapped = None
                            continue
                # fallback: fill with NaN
                if mapped is None:
                    mapped = np.full(self.num_agents, np.nan, dtype=float)
                results[name] = mapped
            return results

        # If the first transform is None, rasters are unavailable (HECRAS-only mode)
        if transforms is None or transforms[0] is None:
            # return NaN arrays for each requested raster name
            results = {name: np.full(self.num_agents, np.nan, dtype=float) for name in raster_names}
            return results

        # compute pixel indices once using the first transform (assumes same grid)
        try:
            # prefer using a cached inverse affine to avoid repeated inversion
            inv = get_inv_transform(self, transforms[0])
            rows, cols = geo_to_pixel_from_inv(inv, self.X, self.Y)
        except Exception:
            rows, cols = geo_to_pixel(self.X, self.Y, transforms[0])
        rows = np.clip(rows, 0, self.height - 1).astype(int)
        cols = np.clip(cols, 0, self.width - 1).astype(int)

        # prepare results container
        results = {}

        # Prefer in-memory cache if present to avoid repeated h5 reads
        cache = getattr(self, '_env_cache', None)
        if cache is not None:
            out = {}
            for name in raster_names:
                arr = cache.get(name)
                if arr is None:
                    out[name] = np.full(self.num_agents, np.nan, dtype=float)
                    continue
                # sample directly from cached array
                r_idx = np.clip(np.round(rows).astype(int), 0, arr.shape[0]-1)
                c_idx = np.clip(np.round(cols).astype(int), 0, arr.shape[1]-1)
                out[name] = arr[r_idx, c_idx].flatten()
            return out

        # If the HDF5 'environment' group does not exist (HECRAS-only), return NaNs
        if 'environment' not in self.hdf5:
            return {name: np.full(self.num_agents, np.nan, dtype=float) for name in raster_names}

        # compute bbox
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        # Heuristic: if bbox area is small, use multi-raster window read
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        if bbox_area <= max(4 * self.num_agents, 256):
            # use cache-backed multi-raster reader
            patches = self._read_env_window_multi(rmin, rmax, cmin, cmax, raster_names)
            # Stack patches into a single array of shape (n_rasters, h, w)
            stacked = np.stack([patches[name] for name in raster_names], axis=0)
            # compute local indices into the window
            r_idx = (rows - rmin).astype(int)
            c_idx = (cols - cmin).astype(int)
            # gather values for all rasters and agents in one indexed pass -> shape (n_rasters, num_agents)
            vals_all = stacked[:, r_idx, c_idx]
            # assign back to results dict
            for i, name in enumerate(raster_names):
                results[name] = np.asarray(vals_all[i]).flatten()
            return results

        # fallback: grouped row reads
        env = self.hdf5['environment']
        h5idx = self._h5_advanced_index
        for name in raster_names:
            dset = env[name]
            results[name] = np.asarray(h5idx(dset, rows, cols)).flatten()
        return results

    def _read_env_window_multi(self, rmin, rmax, cmin, cmax, raster_names):
        """Read multiple rasters in a single HDF5 window and return a dict name->2D array.

        Uses an LRU cache keyed by (rmin,rmax,cmin,cmax,tuple(raster_names)) to avoid
        repeating identical reads across timesteps.
        """
        # Build cache on first use
        if not hasattr(self, '_env_window_cache'):
            from collections import OrderedDict

            class _LRUCache:
                def __init__(self, maxsize=64):
                    self.maxsize = maxsize
                    self.od = OrderedDict()

                def get(self, key):
                    try:
                        val = self.od.pop(key)
                        self.od[key] = val
                        return val
                    except KeyError:
                        return None

                def put(self, key, val):
                    if key in self.od:
                        self.od.pop(key)
                    self.od[key] = val
                    if len(self.od) > self.maxsize:
                        self.od.popitem(last=False)

            self._env_window_cache = _LRUCache(maxsize=128)

        key = (int(rmin), int(rmax), int(cmin), int(cmax), tuple(raster_names))
        cached = self._env_window_cache.get(key)
        if cached is not None:
            return cached

        # Prefer to serve from in-memory env cache if present
        cache = getattr(self, '_env_cache', None)
        out = {}
        if cache is not None:
            can_serve = True
            for name in raster_names:
                arr = cache.get(name)
                if arr is None:
                    can_serve = False
                    break
            if can_serve:
                for name in raster_names:
                    arr = cache[name]
                    out[name] = arr[rmin:rmax+1, cmin:cmax+1]
                self._env_window_cache.put(key, out)
                return out

        env = self.hdf5['environment']
        out = {}
        for name in raster_names:
            out[name] = env[name][rmin:rmax+1, cmin:cmax+1]

        self._env_window_cache.put(key, out)
        return out

    def _h5_advanced_index(self, dset, rows, cols):
        """Efficiently gather values from an HDF5 2D dataset given row,col arrays.

        h5py does not support arbitrary advanced indexing when index arrays are
        unordered. This helper groups positions by row, reads contiguous slices
        for each unique row, and assembles results in the original order. This
        avoids loading the full dataset into memory and minimizes I/O.
        """
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        # Bound check
        rows = np.clip(rows, 0, dset.shape[0] - 1)
        cols = np.clip(cols, 0, dset.shape[1] - 1)

        # Group indices by row
        order = np.arange(len(rows))
        # Lexsort by rows then cols to allow contiguous reads
        sorter = np.lexsort((cols, rows))
        rows_s = rows[sorter]
        cols_s = cols[sorter]
        order_s = order[sorter]

        result_s = np.empty_like(rows_s, dtype=dset.dtype)

        # iterate contiguous runs of same row
        start = 0
        n = len(rows_s)
        while start < n:
            r = rows_s[start]
            end = start + 1
            while end < n and rows_s[end] == r:
                end += 1

            # we have indices for row r in cols_s[start:end]
            c_inds = cols_s[start:end]
            # read the row once
            row_data = dset[r, :]
            result_s[start:end] = row_data[c_inds]
            start = end

        # restore original order
        result = np.empty_like(result_s)
        result[order_s] = result_s
        return result

    def build_hecras_mapping(self, hecras_nodes, abm_points, k=3, eps=1e-8):
        """Precompute KDTree mapping from irregular HECRAS nodes to ABM points.

        hecras_nodes: (N,2) array of node coordinates (x,y)
        abm_points: (M,2) array of target ABM coordinates (x,y) to interpolate
        k: number of nearest HECRAS nodes to use for interpolation

        Stores in self.hecras_map a dict with:
         - 'indices': (M,k) int array of nearest node indices
         - 'weights': (M,k) float array of weights (sum to 1)
         - 'tree': cKDTree of HECRAS nodes (cached for fast re-queries)
         - 'nodes': HECRAS node coordinates
         - 'k': number of neighbors
         - 'eps': epsilon for inverse distance weighting

        Weighting uses inverse-distance with small eps to avoid div0.
        """
        hecras_nodes = np.asarray(hecras_nodes, dtype=float)
        abm_points = np.asarray(abm_points, dtype=float)

        # Build KDTree once and cache it
        if not hasattr(self, 'hecras_map') or self.hecras_map.get('tree') is None:
            tree = cKDTree(hecras_nodes)
        else:
            # Reuse existing tree if nodes haven't changed
            tree = self.hecras_map['tree']
            
        # Query tree for current agent positions
        try:
            dists, inds = tree.query(abm_points, k=k, n_jobs=-1)
        except TypeError:
            dists, inds = tree.query(abm_points, k=k)
        # ensure shapes (M,k)
        if k == 1:
            dists = dists[:, None]
            inds = inds[:, None]

        # inverse-distance weights
        inv = 1.0 / (dists + eps)
        w = inv / np.sum(inv, axis=1)[:, None]

        self.hecras_map = {
            'indices': inds.astype(np.int32),
            'weights': w.astype(np.float32),
            'nodes': hecras_nodes,
            'tree': tree,  # Cache the tree for reuse
            'k': k,
            'eps': eps
        }

        return self.hecras_map
    
    def update_hecras_mapping_for_current_positions(self):
        """Fast update of HECRAS mapping for current agent positions.
        
        Uses cached KDTree to avoid rebuilding. Call this after agents move.
        """
        if not hasattr(self, 'hecras_map') or 'tree' not in self.hecras_map:
            raise RuntimeError('HECRAS mapping not initialized. Call build_hecras_mapping first.')
        
        abm_points = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        tree = self.hecras_map['tree']
        k = self.hecras_map.get('k', 3)
        eps = self.hecras_map.get('eps', 1e-8)
        
        # Query tree for current positions
        try:
            dists, inds = tree.query(abm_points, k=k, n_jobs=-1)
        except TypeError:
            dists, inds = tree.query(abm_points, k=k)
            
        if k == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        
        # Recompute weights
        inv = 1.0 / (dists + eps)
        w = inv / np.sum(inv, axis=1)[:, None]
        
        # Update indices and weights in place
        self.hecras_map['indices'] = inds.astype(np.int32)
        self.hecras_map['weights'] = w.astype(np.float32)

    def apply_hecras_mapping(self, hecras_field):
        """Apply precomputed HECRAS mapping to a HECRAS nodal field.

        hecras_field: (N,) or (N,...) array of nodal values. If multiple
        attributes per node, shape should be (N, attrs) and result is (M, attrs).

        Returns interpolated values at ABM points as (M,) or (M, attrs).
        """
        if not hasattr(self, 'hecras_map'):
            raise RuntimeError('HECRAS map not built. Call build_hecras_mapping first.')

        inds = self.hecras_map['indices']
        w = self.hecras_map['weights']

        hecras_field = np.asarray(hecras_field)
        # if field is 1D
        if hecras_field.ndim == 1:
            sampled = np.sum(hecras_field[inds] * w, axis=1)
            return sampled
        else:
            # hecras_field shape: (N, attrs)
            sampled = np.einsum('mk,mk->m', hecras_field[inds], w)
            return sampled
    
    def initial_swim_speed(self):
        """
        Calculates the initial swim speed required for each fish to overcome
        current water velocities and maintain their ideal speed over ground (SOG).
    
        Attributes:
        x_vel, y_vel (array): Water velocity components in m/s for each fish.
        ideal_sog (array): Ideal speed over ground in m/s for each fish.
        heading (array): Heading in radians for each fish.
        swim_speed (array): Calculated swim speed for each fish to maintain ideal SOG.
    
        Notes:
        - The function assumes that the input attributes are structured as arrays of
          values, with each index across the arrays corresponding to a different fish.
        - The function updates the swim_speed attribute for each fish based on the
          calculated swim speed necessary to maintain the ideal SOG against water currents.
        """

        # Attempt to sample raster velocities; fall back to zeros if transforms are missing
        try:
            vals = self.batch_sample_environment([self.vel_x_rast_transform, self.vel_y_rast_transform], ['vel_x', 'vel_y'])
            self.x_vel = vals.get('vel_x', np.zeros(self.num_agents))
            self.y_vel = vals.get('vel_y', np.zeros(self.num_agents))
        except Exception:
            self.x_vel = np.zeros(self.num_agents)
            self.y_vel = np.zeros(self.num_agents)
        
        # Vector components of water velocity for each fish
        water_velocities = np.sqrt(self.x_vel**2 + self.y_vel**2)
    
        # Vector components of ideal velocity for each fish
        ideal_velocities = np.stack((self.ideal_sog * np.cos(self.heading),
                                     self.ideal_sog * np.sin(self.heading)), axis=-1)
    
        # Calculate swim speed for each fish
        # Subtracting the scalar water velocity from the vector ideal velocity
        # requires broadcasting the water_velocities array to match the shape of ideal_velocities
        self.swim_speed = np.linalg.norm(ideal_velocities - np.array([self.x_vel,self.y_vel]).T, axis = -1)
                

    def initial_heading (self):
        """
        Calculate the initial heading for each agent based on the velocity direction raster.
    
        This function performs the following steps:
        - Converts the geographic coordinates of each agent to pixel coordinates.
        - Samples the environment to get the velocity direction at each agent's location.
        - Adjusts the heading based on the flow direction, ensuring it is within the range [0, 2π).
        - Calculates the maximum practical speed over ground (SOG) for each agent based on their heading and SOG.
    
        Attributes updated:
        - self.heading: The heading for each agent in radians.
        - self.max_practical_sog: The maximum practical speed over ground for each agent as a 2D vector (m/s).
        """
        # get the x, y position of the agent (use cache if available)
        try:
            if 'vel' in self._pixel_index_cache:
                row, col = self._pixel_index_cache['vel']
            else:
                row, col = geo_to_pixel(self.X, self.Y, self.vel_dir_rast_transform)

            # get the initial heading values from vel_dir raster
            values = self.batch_sample_environment([self.vel_dir_rast_transform], ['vel_dir'])['vel_dir']

            # set direction using raster-derived values (degrees -> radians adjustment)
            self.heading = self.arr.where(values < 0,
                                           (self.arr.radians(360) + values) - self.arr.radians(180),
                                           values - self.arr.radians(180))
        except Exception:
            # Fallback: compute heading from vector velocity components (x_vel, y_vel)
            try:
                vals_deg = np.degrees(np.arctan2(self.y_vel, self.x_vel))
            except Exception:
                vals_deg = np.zeros(self.num_agents, dtype=float)

            # Convert degrees to radians and shift by 180 degrees (consistent with raster logic)
            self.heading = np.deg2rad(vals_deg) - np.pi

        # set initial max practical speed over ground as well
        self.max_practical_sog = self.arr.array([self.sog * self.arr.cos(self.heading), 
                                                          self.sog * self.arr.sin(self.heading)]) #meters/sec       

    def update_mental_map(self, current_timestep):
        """
        Update the mental map for each agent in the HDF5 dataset.
    
        This function performs the following steps:
        - Converts the geographic coordinates (X and Y) of each agent to pixel coordinates.
        - Updates the mental map for each agent in the HDF5 dataset at the corresponding pixel location with the current timestep.
    
        Parameters:
        - current_timestep: The current timestep to record in the mental map.
    
        The mental map is stored in an HDF5 dataset with shape (num_agents, width, height), where each 'slice' corresponds to an agent's mental map.
        """
        # Vectorized: compute mental-map cell indices for all agents and set
        # the corresponding cells in the in-memory accumulator. We do not
        # perform HDF5 writes here to avoid many small write calls.
        if 'mental_map' in self._pixel_index_cache:
            rows, cols = self._pixel_index_cache['mental_map']
        else:
            rows, cols = geo_to_pixel(self.X, self.Y, self.mental_map_transform)

        # Round/clip to integer cell indices
        rows = np.clip(np.round(rows).astype(int), 0, self.mental_map_accumulator.shape[1] - 1)
        cols = np.clip(np.round(cols).astype(int), 0, self.mental_map_accumulator.shape[2] - 1)

        agents = np.arange(self.num_agents, dtype=int)
        self.mental_map_accumulator[agents, rows, cols] = 1

        # Do not flush here; flushing happens in `timestep_flush` to batch writes.

    def update_refugia_map(self, current_velocity):
        """
        Update the mental map for each agent in the HDF5 dataset.
    
        This function performs the following steps:
        - Converts the geographic coordinates (X and Y) of each agent to pixel coordinates.
        - Updates the mental map for each agent in the HDF5 dataset at the corresponding pixel location with the current timestep.
    
        Parameters:
        - current_timestep: The current timestep to record in the mental map.
    
        The mental map is stored in an HDF5 dataset with shape (num_agents, width, height), where each 'slice' corresponds to an agent's mental map.
        """
        # Convert geographic coordinates to refugia-map pixel coordinates
        # use the refugia map transform if available, otherwise fall back
        transform = getattr(self, 'refugia_map_transform', self.mental_map_transform)
        # use cached indices for refugia/mental map transforms if available
        key = 'refugia' if hasattr(self, 'refugia_map_transform') else 'mental_map'
        if key in self._pixel_index_cache:
            rows, cols = self._pixel_index_cache[key]
        else:
            rows, cols = geo_to_pixel(self.X, self.Y, transform)

        # Determine refugia dataset shape (use first agent dataset as reference)
        try:
            sample_ds = next(iter(self.hdf5['refugia'].values()))
            max_row = sample_ds.shape[0] - 1
            max_col = sample_ds.shape[1] - 1
        except Exception:
            max_row = int(np.round(self.height / self.refugia_cell_size))
            max_col = int(np.round(self.width / self.refugia_cell_size))

        rows = np.clip(np.round(rows).astype(int), 0, max_row)
        cols = np.clip(np.round(cols).astype(int), 0, max_col)

        # Populate in-memory refugia accumulator for each agent (vectorized)
        agents = np.arange(self.num_agents, dtype=int)
        # Ensure current_velocity is an array of floats
        cv = np.asarray(current_velocity, dtype=float)
        # Mask invalid velocities (NaN/inf) to avoid assignment errors
        valid = np.isfinite(cv)
        if np.any(valid):
            ai = agents[valid]
            ri = rows[valid].astype(int)
            ci = cols[valid].astype(int)
            try:
                self.refugia_accumulator[ai, ri, ci] = cv[valid]
            except Exception:
                # Fallback to safe per-agent assignment if vectorized write fails
                for i in ai:
                    r = int(rows[i])
                    c = int(cols[i])
                    try:
                        self.refugia_accumulator[i, r, c] = float(cv[i])
                    except Exception:
                        continue

        # Flushing to HDF5 happens in `timestep_flush` in batch.

    def environment(self):
        """
        Updates environmental parameters for each agent and identifies neighbors within a defined buffer.
    
        This function creates a GeoDataFrame from the agents' positions and sets the coordinate reference system (CRS).
        It then samples environmental data such as depth, x-velocity, and y-velocity at each agent's position.
        The function also tracks the time each agent spends in shallow water (out of water) and identifies neighboring agents within a specified buffer.
    
        Attributes updated:
            self.depth (np.ndarray): Depth at each agent's position.
            self.x_vel (np.ndarray): X-component of velocity at each agent's position.
            self.y_vel (np.ndarray): Y-component of velocity at each agent's position.
            self.time_out_of_water (np.ndarray): Incremented time that each agent spends in water too shallow for swimming.
    
        Returns:
            agents_within_buffers_dict (dict): A dictionary where each key is an agent index and the value is a list of indices of other agents within that agent's buffer.
            closest_agent_dict (dict): A dictionary where each key is an agent index and the value is the index of the closest agent within that agent's buffer.
    
        Example:
            # Assuming `self` is an instance with appropriate attributes:
            self.environment()
            # After execution, `self.depth`, `self.x_vel`, `self.y_vel`, and `self.time_out_of_water` are updated.
            # `agents_within_buffers_dict` and `closest_agent_dict` are available for further processing.
        """
        # (The rest of your function code follows here...)        
        # Convert positions into a simple Nx2 array — avoid GeoDataFrame/CRS overhead
        # and batch-sample environment rasters
        names = ['depth', 'vel_x', 'vel_y', 'vel_mag', 'wetted', 'distance_to']
        # Build transforms defensively — some may be missing when running HECRAS-only
        transforms = []
        transforms.append(getattr(self, 'depth_rast_transform', None))
        transforms.append(getattr(self, 'vel_x_rast_transform', None))
        transforms.append(getattr(self, 'vel_y_rast_transform', None))
        transforms.append(getattr(self, 'vel_mag_rast_transform', None))
        transforms.append(getattr(self, 'wetted_transform', None))
        transforms.append(getattr(self, 'depth_rast_transform', None))

        sampled = self.batch_sample_environment(transforms, names)
        self.depth = sampled['depth']
        self.x_vel = sampled['vel_x']
        self.y_vel = sampled['vel_y']
        self.vel_mag = sampled['vel_mag']
        self.wet = sampled['wetted']
        self.distance_to = sampled['distance_to']
        self.current_centerline_meas = self.compute_linear_positions(self.centerline)

        # Ensure neighbor lists and closest_agent are present — some code paths
        # assume these are set before behavior arbitration. Compute fallbacks
        # here if they do not exist.
        try:
            if not hasattr(self, 'closest_agent') or getattr(self, 'closest_agent', None) is None:
                clean_x = self.X.flatten()[~np.isnan(self.X.flatten())]
                clean_y = self.Y.flatten()[~np.isnan(self.Y.flatten())]
                if clean_x.size > 0:
                    positions = np.vstack([clean_x, clean_y]).T
                    tree = cKDTree(positions)
                    distances, indices = tree.query(positions, k=2)
                    nearest_neighbors = np.where(distances[:, 1] != np.inf, indices[:, 1], np.nan)
                    self.closest_agent = nearest_neighbors
                    nearest_neighbor_distances = np.where(distances[:, 1] != np.inf, distances[:, 1], np.nan)
                    self.nearest_neighbor_distance = nearest_neighbor_distances
                else:
                    self.closest_agent = np.full(self.num_agents, np.nan)
                    self.nearest_neighbor_distance = np.full(self.num_agents, np.nan)
        except Exception:
            # If neighbor computation fails, provide a safe fallback
            self.closest_agent = np.full(self.num_agents, np.nan)

        # Optional: override or augment raster-sampled fields with HECRAS plan mapping
        # To enable, set `self.hecras_plan_path` (string path) and `self.hecras_fields` (list of field names)
        if hasattr(self, 'hecras_plan_path') and self.hecras_plan_path:
            # default fields if not provided
            field_names = getattr(self, 'hecras_fields', ['Cells Minimum Elevation'])
            k = getattr(self, 'hecras_k', 8)
            # build N x 2 agent array
            agent_xy = np.column_stack((self.X, self.Y))
            out = map_hecras_for_agents(self, agent_xy, self.hecras_plan_path, field_names=field_names, k=k)
            # `out` is either array (single field) or dict of arrays
            if isinstance(out, dict):
                for fname, arr in out.items():
                    # store HECRAS-mapped fields under hecras_<name>
                    attr = 'hecras_' + fname.replace(' ', '_').replace('/', '_').lower()
                    setattr(self, attr, arr)
                    # if use_hecras is True, override common attributes
                    if self.use_hecras:
                        lname = fname.lower()
                        if 'elev' in lname or 'minimum elevation' in lname:
                            # prefer wsel - elev for depth if both present later
                            self.depth = arr
                        elif 'water surface' in lname or 'wsel' in lname:
                            self.wsel = arr
                        elif 'velocity x' in lname or 'vel_x' in lname or 'cell velocity - velocity x' in lname.lower():
                            self.x_vel = arr
                        elif 'velocity y' in lname or 'vel_y' in lname or 'cell velocity - velocity y' in lname.lower():
                            self.y_vel = arr
            else:
                # single-field array -> store as hecras_<fieldname> and optionally override depth
                setattr(self, 'hecras_' + field_names[0].replace(' ', '_').lower(), out)
                if self.use_hecras:
                    self.depth = out

        # Use HECRAS mapping only when explicitly enabled by the user via
        # `enable_hecras(...)`. Raster sampling remains the default fallback.
        if getattr(self, 'use_hecras', False):
            try:
                if hasattr(self, 'hecras_node_fields') and hasattr(self, 'hecras_map'):
                    if 'depth' in self.hecras_node_fields:
                        vals = self.apply_hecras_mapping(self.hecras_node_fields['depth'])
                        self.depth = np.asarray(vals).flatten()
                    if 'vel_x' in self.hecras_node_fields:
                        vals = self.apply_hecras_mapping(self.hecras_node_fields['vel_x'])
                        self.x_vel = np.asarray(vals).flatten()
                    if 'vel_y' in self.hecras_node_fields:
                        vals = self.apply_hecras_mapping(self.hecras_node_fields['vel_y'])
                        self.y_vel = np.asarray(vals).flatten()
            except Exception:
                # If mapping fails for any reason, silently fall back to raster values
                self.use_hecras = False

    def precompute_pixel_indices(self):
        """
        Precompute and cache row/col indices for commonly used raster transforms
        for the current agent positions. This avoids calling geo_to_pixel multiple
        times per timestep across different cue functions.
        """
        X = self.X
        Y = self.Y
        cache = {}
        mapping = {
            'depth': getattr(self, 'depth_rast_transform', None),
            'vel': getattr(self, 'vel_mag_rast_transform', None),
            'vel_dir': getattr(self, 'vel_dir_rast_transform', None),
            'refugia': getattr(self, 'refugia_map_transform', None),
            'mental_map': getattr(self, 'mental_map_transform', None)
        }
        for key, transform in mapping.items():
            if transform is None:
                cache[key] = (np.full_like(X, -1, dtype=int), np.full_like(Y, -1, dtype=int))
            else:
                try:
                    inv = get_inv_transform(self, transform)
                    rows, cols = geo_to_pixel_from_inv(inv, X, Y)
                except Exception:
                    rows, cols = geo_to_pixel(X, Y, transform)
                cache[key] = (rows.astype(np.int32), cols.astype(np.int32))

        self._pixel_index_cache = cache

    def enable_hecras(self, hecras_nodes, hecras_node_fields, k=3):
        """Enable HECRAS mapping for the simulation.

        - hecras_nodes: (N,2) array of HECRAS node coordinates
        - hecras_node_fields: dict of nodal arrays e.g. {'depth':..., 'vel_x':..., 'vel_y':...}
        - k: number of nearest nodes to use for inverse-distance interpolation
        """
        # build mapping for the current agent positions
        agent_points = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        self.build_hecras_mapping(hecras_nodes, agent_points, k=k)
        self.hecras_node_fields = hecras_node_fields
        self.use_hecras = True
        self.hecras_mapping_enabled = True  # Flag for behavioral cues to use per-agent velocities
        
        # Sample initial HECRAS values at agent positions
        if 'depth' in hecras_node_fields:
            self.depth = self.apply_hecras_mapping(hecras_node_fields['depth'])
            # Compute initial wetted status from depth
            self.wet = np.where(self.depth > self.too_shallow / 2.0, 1.0, 0.0)
        if 'vel_x' in hecras_node_fields:
            self.x_vel = self.apply_hecras_mapping(hecras_node_fields['vel_x'])
        if 'vel_y' in hecras_node_fields:
            self.y_vel = self.apply_hecras_mapping(hecras_node_fields['vel_y'])
            self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
        
    
        # Avoid divide by zero by setting zero velocities to a small number
        # self.x_vel[self.x_vel == 0.0] = 0.0001
        # self.y_vel[self.y_vel == 0.0] = 0.0001
    
        # keep track of the amount of time a fish spends out of water
        self.time_out_of_water = np.where(self.depth < self.too_shallow, 
                                          self.time_out_of_water + 1, 
                                          0.0)

        # Determine death due to drying:
        if getattr(self, 'death_on_dry_immediate', False):
            # Immediate death when fish are on dry ground (wet != 1)
            self.dead = np.where(self.wet != 1., 1, self.dead)
        else:
            # Use a short timeout threshold (configurable)
            thresh = int(getattr(self, 'time_out_of_water_threshold', 5))
            self.dead = np.where(np.logical_or(self.time_out_of_water > thresh,
                                                self.wet != 1.), 
                                  1,
                                  self.dead)

        # Force random headings after HECRAS mapping so agents don't all face upstream
        try:
            self.heading = np.random.uniform(-np.pi, np.pi, size=self.num_agents)
            # small additional SOG perturbation
            if hasattr(self, 'ideal_sog'):
                base = np.asarray(self.ideal_sog)
                if base.size == 1:
                    base = np.full(self.num_agents, float(base))
                frac = getattr(self, 'initial_sog_jitter_fraction', 0.1)
                self.sog = base * np.random.uniform(1.0 - frac, 1.0 + frac, size=self.num_agents)
                self.ideal_sog = self.sog.copy()
            # small velocity jitter
            try:
                jitter_scale = getattr(self, 'initial_velocity_jitter', 0.05)
                vel_jitter = np.random.normal(scale=float(jitter_scale), size=(self.num_agents, 2))
                if not hasattr(self, 'x_vel') or not hasattr(self, 'y_vel'):
                    self.x_vel = np.zeros(self.num_agents)
                    self.y_vel = np.zeros(self.num_agents)
                self.x_vel += vel_jitter[:, 0]
                self.y_vel += vel_jitter[:, 1]
            except Exception:
                pass
        except Exception:
            pass
                
        # self.dead = np.where(self.wet != 1., 
        #                      1,
        #                      self.dead)
        
        # if np.any(self.dead):
        #     print ('why did they die?')
        #     print ('wet status: %s'%(self.wet))
        #     # sys.exit()
            
            
        
        # For dead fish, zero out positions and velocity
        self.x_vel = np.where(self.dead,np.zeros_like(self.x_vel), self.x_vel)
        self.y_vel = np.where(self.dead,np.zeros_like(self.y_vel), self.y_vel)
        self.X = np.where(self.dead,np.zeros_like(self.X), self.X)
        self.Y = np.where(self.dead,np.zeros_like(self.Y), self.Y)

        
        
        clean_x = self.X.flatten()[~np.isnan(self.X.flatten())]
        clean_y = self.Y.flatten()[~np.isnan(self.Y.flatten())]
        
        positions = np.vstack([clean_x,clean_y]).T
        
        # Creating a KDTree for efficient spatial queries
        try:
           tree = cKDTree(positions)
        except ValueError:
            print ('something wrong with positions - is an agent off the map?')
            print ('XY: %s'%(positions))
            print ('wetted: %s'%(self.wet))
            print ('dead: %s' %(self.dead))
            sys.exit()
        
        # Radius for nearest neighbors search
        #TODO changed from 2 to xx
        radius = 6.
        
        # Find agents within the specified radius for each agent
        agents_within_radius = tree.query_ball_tree(tree, r=radius)
        
        # Batch query for the two nearest neighbors (including self)
        distances, indices = tree.query(positions, k=2)
        
        # Exclude self from results and handle no neighbors case
        nearest_neighbors = np.where(distances[:, 1] != np.inf, indices[:, 1], np.nan)

        # Extract the distance to the closest agent, excluding self
        nearest_neighbor_distances = np.where(distances[:, 1] != np.inf, distances[:, 1], np.nan)

        # Now `agents_within_buffers_dict` is a dictionary where each key is an agent index
        self.agents_within_buffers = agents_within_radius
        self.closest_agent = nearest_neighbors
        self.nearest_neighbor_distance = nearest_neighbor_distances
            
    def odometer(self, t, dt):
        """
        Updates the running counter of the amount of kCal consumed by each fish during a simulation timestep.
    
        Parameters:
        t (float): The current time in the simulation.
    
        Attributes:
        water_temp (array): Water temperature experienced by each fish.
        weight (array): Weight of each fish.
        wave_drag (array): Wave drag acting on each fish.
        swim_speed (array): Swimming speed of each fish.
        ucrit (array): Critical swimming speed for each fish.
        kcal (array): Cumulative kilocalories burned by each fish.
    
        Notes:
        - The function assumes that the input attributes are structured as arrays of
          values, with each index across the arrays corresponding to a different fish.
        - The function updates the kcal attribute for each fish based on their oxygen
          consumption converted to calories using metabolic equations from Brett (1964)
          and Brett and Glass (1973).
        """
    
        # Convert kg → grams for metabolic rate equations
        weight_g = self.weight * 1000.0
    
        # Standard Metabolic Rate (SMR) – Brett & Glass 1973
        sr_o2_rate = self.arr.where(
            self.water_temp <= 5.3,
            self.arr.exp(0.0565 * self.arr.power(self.arr.log(weight_g), 0.9141)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.1498 * self.arr.power(self.arr.log(weight_g), 0.8465)),
                self.arr.exp(0.1987 * self.arr.power(self.arr.log(weight_g), 0.8844))
            )
        )
    
        # Active Metabolic Rate (AMR) – Brett & Glass 1973
        ar_o2_rate = self.arr.where(
            self.water_temp <= 5.3,
            self.arr.exp(0.4667 * self.arr.power(self.arr.log(weight_g), 0.9989)),
            self.arr.where(
                self.water_temp <= 15,
                self.arr.exp(0.9513 * self.arr.power(self.arr.log(weight_g), 0.9632)),
                self.arr.exp(0.8237 * self.arr.power(self.arr.log(weight_g), 0.9947))
            )
        )
    
        # Interpolated swim cost – Hughes 2004 (linear interpolation)
        # Optional: use exponent b=2 for high effort swimming
        b = 1.0
        frac_speed = self.swim_speed / self.ucrit
        frac_speed = self.arr.clip(frac_speed, 0.0, 1.0)
        swim_cost = sr_o2_rate + (ar_o2_rate - sr_o2_rate) * self.arr.power(frac_speed, b)
    
        # Convert metabolic rate from mg O2/kg/hr → kcal
        hours = dt / 3600.0
        mg_O2 = swim_cost * hours * self.weight  # self.weight in kg
        kcal  = mg_O2 * (3.36 / 1000.0)          # kcal burned this timestep
    
        if np.any(kcal < 0):
            print("🚨 Warning: Negative kcal detected!")
    
        # Update cumulative odometer
        self.kcal += kcal
    
        # # Calculate active and standard metabolic rate using equations from Brett and Glass (1973)
        # # O2_rate in units of mg O2/hr
        # sr_o2_rate = self.arr.where(
        #     self.water_temp <= 5.3,
        #     self.arr.exp(0.0565 * np.power(self.arr.log(self.weight), 0.9141)),
        #     self.arr.where(
        #         self.water_temp <= 15,
        #         self.arr.exp(0.1498 * self.arr.power(self.arr.log(self.weight), 0.8465)),
        #         self.arr.exp(0.1987 * self.arr.power(self.arr.log(self.weight), 0.8844))
        #     )
        # )
    
        # ar_o2_rate = self.arr.where(
        #     self.water_temp <= 5.3,
        #     self.arr.exp(0.4667 * self.arr.power(self.arr.log(self.weight), 0.9989)),
        #     self.arr.where(
        #         self.water_temp <= 15,
        #         self.arr.exp(0.9513 * self.arr.power(self.arr.log(self.weight), 0.9632)),
        #         self.arr.exp(0.8237 * self.arr.power(self.arr.log(self.weight), 0.9947))
        #     )
        # )
    
        # # Calculate total metabolic rate
        # if self.num_agents > 1:
        #     swim_cost = sr_o2_rate + self.wave_drag * (
        #         self.arr.exp(np.log(sr_o2_rate) + self.swim_speed * (
        #             (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
        #         ) - sr_o2_rate)
        #     )
        # else:
        #     swim_cost = sr_o2_rate + self.wave_drag.flatten() * (
        #         self.arr.exp(np.log(sr_o2_rate) + np.linalg.norm(self.swim_speed.flatten(), axis = -1) * (
        #             (self.arr.log(ar_o2_rate) - self.arr.log(sr_o2_rate)) / self.ucrit
        #         ) - sr_o2_rate)
        #     )
        # (odometer logic continues)
        # # swim cost is expressed in mg O2 _kg _hr.  convert to mg O2 _ kg
        # hours = dt * (1./3600.)
        # per_capita_swim_cost = swim_cost * hours
        # mg_O2 = per_capita_swim_cost * self.weight
        # # Brett (1973) used a mean oxycalorific equivalent of 3.36 cal/ mg O2 (RQ = 0.8) 
        # kcal = mg_O2 * (3.36 / 1000)
        
        # if np.any(kcal < 0):
        #     print ('why kcal negative')
        # # Update kilocalories burned
        # self.kcal += kcal
        
    class movement():
        
        def __init__(self, simulation_object):
            self.simulation = simulation_object
            
        def find_z(self):
            """
            Calculate the z-coordinate for an agent based on its depth and body depth.
        
            This function determines the z-coordinate (vertical position) of an agent.
            If the depth at the agent's position is less than one-third of its body depth,
            the z-coordinate is set to the sum of the depth and a predefined shallow water threshold.
            Otherwise, it is set to one-third of the agent's body depth.
        
            Attributes updated:
                self.z (array-like): The calculated z-coordinate for the agent.
        
            Note:
                The function uses `self.arr.where` for vectorized conditional operations,
                which implies that `self.depth`, `self.body_depth`, and `self.too_shallow` should be array-like
                and support broadcasting if they are not scalar values.
            """
            self.simulation.z = np.where(
                self.simulation.depth < self.simulation.body_depth * 3 / 100.,
                self.simulation.depth + self.simulation.too_shallow,
                self.simulation.body_depth * 3 / 100.)
            
            # make sure 
            self.simulation.z = np.where(self.simulation.z < 0,0,self.simulation.z)
        
        def thrust_fun(self, mask, t, dt, fish_velocities = None):
            """
            Calculates the thrust for a collection of agents based on Lighthill's elongated-body theory of fish propulsion.
            
            This method uses piecewise linear interpolation for the amplitude, wave, and trailing edge
            as functions of body length and swimming speed. It is designed to work with array operations
            and can utilize either NumPy or CuPy for calculations to support execution on both CPUs and GPUs.
            
            The method assumes a freshwater environment with a density of 1.0 kg/m^3. The thrust calculation
            uses the agents' lengths, velocities, ideal speeds over ground (SOG), headings, and tail-beat frequencies
            to compute thrust vectors for each agent.
            
            Attributes
            ----------
            length : array_like
                The lengths of the agents in meters.
            x_vel : array_like
                The x-components of the water velocity vectors for the agents in m/s.
            y_vel : array_like
                The y-components of the water velocity vectors for the agents in m/s.
            ideal_sog : array_like
                The ideal speeds over ground for the agents in m/s.
            heading : array_like
                The headings of the agents in radians.
            Hz : array_like
                The tail-beat frequencies of the agents in Hz.
            
            Returns
            -------
            thrust : ndarray
                An array of thrust vectors for each agent, where each vector is a 2D vector
                representing the thrust in the x and y directions in N/m.
            
            Notes
            -----
            The function assumes that the input arrays are of equal length, with each index
            corresponding to a different agent. The thrust calculation is vectorized to handle
            multiple agents simultaneously.
            
            The piecewise linear interpolation is used for the amplitude, wave, and trailing
            edge based on the provided data points. This approach simplifies the computation and
            is suitable for scenarios where a small amount of error is acceptable.
            
            Examples
            --------
            Assuming `agents` is an instance of the simulation class with all necessary properties set as arrays:
            
            >>> thrust = agents.thrust_fun()
            
            The `thrust` array will contain the thrust vectors for each agent after the function call.
            """

            # Constants
            rho = 1.0  # density of freshwater
            theta = 32.  # theta that produces cos(theta) = 0.85
            length_cm = self.simulation.length / 1000 * 100.
            
            # Calculate swim speed
            water_vel = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_dir = np.arctan2(fish_y_vel,fish_x_vel)
                    fish_mag = np.hypot(fish_x_vel, fish_y_vel)

                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)),
                                               axis=-1)
                        
            ideal_swim_speed = np.linalg.norm(fish_velocities - water_vel, axis=-1)

            swim_speed_cms = ideal_swim_speed * 100.
        
            # Cache splines on the simulation object to avoid recreating every timestep
            if not hasattr(self.simulation, '_thrust_splines'):
                length_dat = np.array([5., 10., 15., 20., 25., 30., 40., 50., 60.])
                speed_dat = np.array([37.4, 58., 75.1, 90.1, 104., 116., 140., 161., 181.])
                amp_dat = np.array([1.06, 2.01, 3., 4.02, 4.91, 5.64, 6.78, 7.67, 8.4])
                wave_dat = np.array([53.4361, 82.863, 107.2632, 131.7, 148.125, 166.278, 199.5652, 230.0044, 258.3])
                edge_dat = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.])
                self.simulation._thrust_splines = (
                    UnivariateSpline(length_dat, amp_dat, k=2, ext=0),
                    UnivariateSpline(speed_dat, wave_dat, k=1, ext=0),
                    UnivariateSpline(length_dat, edge_dat, k=1, ext=0)
                )

            A_spline, V_spline, B_spline = self.simulation._thrust_splines
            A = A_spline(length_cm)
            V = V_spline(swim_speed_cms)
            B = B_spline(length_cm)
        
            # Calculate thrust
            m = (np.pi * rho * B**2) / 4.
            W = (self.simulation.Hz * A * np.pi) / 1.414
            w = W * (1 - swim_speed_cms / V)
        
            # Thrust calculation
            thrust_erg_s = m * W * w * swim_speed_cms - (m * w**2 * swim_speed_cms) / (2. * np.cos(np.radians(theta)))
            thrust_Nm = thrust_erg_s / 10000000.
            thrust_N = thrust_Nm / (self.simulation.length / 1000.)
        
            # Convert thrust to vector
            thrust_x = np.where(mask, thrust_N * np.cos(self.simulation.heading), 0.0)
            thrust_y = np.where(mask, thrust_N * np.sin(self.simulation.heading), 0.0)
            thrust = np.stack((thrust_x, thrust_y), axis=1)
                
            self.simulation.thrust = thrust
        
        def frequency(self, mask, t, dt, fish_velocities = None):
            ''' Calculate tailbeat frequencies for a collection of agents in a vectorized manner.
            
                This method computes tailbeat frequencies based on Lighthill's elongated-body theory,
                considering each agent's length, velocity, and drag. It then adjusts these frequencies
                using a vectorized PID controller to better match the desired speed over ground.
            
                Parameters
                ----------
                mask : array_like
                    A boolean array indicating which agents to include in the calculation.
                pid_controller : VectorizedPIDController
                    An instance of the VectorizedPIDController class for adjusting Hz values.
            
                Returns
                -------
                Hzs : ndarray
                    An array of adjusted tailbeat frequencies for each agent, in Hz.
            
                Notes
                -----
                The function assumes that all input arrays are of equal length, corresponding
                to different agents. It uses vectorized operations for efficiency and is
                compatible with a structure-of-arrays approach.
            
                The PID controller adjusts frequencies based on the error between the actual
                and desired speeds, improving the model's realism and accuracy.
            
                    # ... function implementation ...'''

            # Constants
            rho = 1.0  # density of freshwater
            theta = 32.  # theta for cos(theta) = 0.85
        
            # Convert lengths from meters to centimeters
            lengths_cm = self.simulation.length / 10
        
            # Calculate swim speed in cm/s
            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            alternate = True
            
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_velocities = np.stack((fish_x_vel,fish_y_vel)).T
                
                alternate = False
            
            swim_speeds_cms = np.linalg.norm(fish_velocities - water_velocities, axis=-1) * 100 + 0.00001

            # sockeye parameters (Webb 1975, Table 20) units in CM!!! 
            length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
            speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
            amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
            wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
            edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
            
            # Cache splines (shared with thrust_fun)
            if not hasattr(self.simulation, '_thrust_splines'):
                length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
                speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.])
                amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4])
                wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
                edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.])
                self.simulation._thrust_splines = (
                    UnivariateSpline(length_dat, amp_dat, k=2, ext=0),
                    UnivariateSpline(speed_dat, wave_dat, k=1, ext=0),
                    UnivariateSpline(length_dat, edge_dat, k=1, ext=0)
                )

            A_spline, V_spline, B_spline = self.simulation._thrust_splines
            A = A_spline(lengths_cm)
            V = V_spline(swim_speeds_cms)
            B = B_spline(lengths_cm)
            
            # get the ideal drag - aka drag if fish is moving how we want it to
            if alternate == True:
                ideal_drag = self.ideal_drag_fun(fish_velocities = fish_velocities)
            else:
                ideal_drag = self.ideal_drag_fun()
                
            # Convert drag to erg/s
            drags_erg_s = np.where(mask,np.linalg.norm(ideal_drag, axis = -1) * self.simulation.length/1000 * 10000000,0)
            
            #TODO min_Hz should be the minimum tailbeat required to match the maximum sustained swim speed 
            # self.max_s_U = 2.77 bl/s
            min_Hz = np.interp(self.simulation.length, [450, 7.5], [690, 2.])
        
            # Solve for Hz
            Hz = np.where(self.simulation.swim_behav == 3, min_Hz,
                          np.sqrt(drags_erg_s * V**2 * np.cos(np.radians(theta))/\
                                  (A**2 * B**2 * swim_speeds_cms * np.pi**3 * rho * \
                                  (swim_speeds_cms - V) * \
                                  (-0.062518880701972 * swim_speeds_cms - \
                                  0.125037761403944 * V * np.cos(np.radians(theta)) + \
                                   0.062518880701972 * V)
                                   )
                                  )
                          )
            # Set Hz to 0 for stuck fish if the attribute exists
            if hasattr(self.simulation, 'is_stuck'):
                Hz = np.where(self.simulation.is_stuck, 0, Hz)
            
            self.simulation.prev_Hz = self.simulation.Hz   
            self.simulation.Hz = np.where(self.simulation.Hz > 20, 20, Hz)
             
        def kin_visc(self, temp):
            """
            Calculates the kinematic viscosity of water at a given temperature using
            interpolation from a predefined dataset.
        
            Parameters
            ----------
            temp : float
                The temperature of the water in degrees Celsius for which the kinematic
                viscosity is to be calculated.
        
            Returns
            -------
            float
                The kinematic viscosity of water at the specified temperature in m^2/s.
        
            Notes
            -----
            The function uses a dataset of kinematic viscosity values at various
            temperatures sourced from the Engineering Toolbox. It employs linear
            interpolation to estimate the kinematic viscosity at the input temperature.
        
            Examples
            --------
            >>> kin_viscosity = kin_visc(20)
            >>> print(f"The kinematic viscosity at 20°C is {kin_viscosity} m^2/s")
        
            This will output the kinematic viscosity at 20 degrees Celsius.
            """
            # Dataset for kinematic viscosity (m^2/s) at various temperatures (°C)
            kin_temp = np.array([0.01, 10., 20., 25., 30., 40., 50., 60., 70., 80.,
                                 90., 100., 110., 120., 140., 160., 180., 200.,
                                 220., 240., 260., 280., 300., 320., 340., 360.])
        
            kin_visc = np.array([0.00000179180, 0.00000130650, 0.00000100350,
                                 0.00000089270, 0.00000080070, 0.00000065790,
                                 0.00000055310, 0.00000047400, 0.00000041270,
                                 0.00000036430, 0.00000032550, 0.00000029380,
                                 0.00000026770, 0.00000024600, 0.00000021230,
                                 0.00000018780, 0.00000016950, 0.00000015560,
                                 0.00000014490, 0.00000013650, 0.00000012990,
                                 0.00000012470, 0.00000012060, 0.00000011740,
                                 0.00000011520, 0.00000011430])
        
            # Interpolate kinematic viscosity based on the temperature
            f_kinvisc = np.interp(temp, kin_temp, kin_visc)
        
            return f_kinvisc

        def wat_dens(self, temp):
            """
            Calculates the density of water at a given temperature using interpolation
            from a predefined dataset.
        
            Parameters
            ----------
            temp : float
                The temperature of the water in degrees Celsius for which the density
                is to be calculated.
        
            Returns
            -------
            float
                The density of water at the specified temperature in g/cm^3.
        
            Notes
            -----
            The function uses a dataset of water density values at various temperatures
            sourced from reliable references. It employs linear interpolation to estimate
            the water density at the input temperature.
        
            Examples
            --------
            >>> water_density = wat_dens(20)
            >>> print(f"The density of water at 20°C is {water_density} g/cm^3")
        
            This will output the density of water at 20 degrees Celsius.
            """
            # Dataset for water density (g/cm^3) at various temperatures (°C)
            dens_temp = np.array([0.1, 1., 4., 10., 15., 20., 25., 30., 35., 40.,
                                  45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
                                  95., 100., 110., 120., 140., 160., 180., 200.,
                                  220., 240., 260., 280., 300., 320., 340., 360.,
                                  373.946])
        
            density = np.array([0.9998495, 0.9999017, 0.9999749, 0.9997, 0.9991026,
                                0.9982067, 0.997047, 0.9956488, 0.9940326, 0.9922152,
                                0.99021, 0.98804, 0.98569, 0.9832, 0.98055, 0.97776,
                                0.97484, 0.97179, 0.96861, 0.96531, 0.96189, 0.95835,
                                0.95095, 0.94311, 0.92613, 0.90745, 0.887, 0.86466,
                                0.84022, 0.81337, 0.78363, 0.75028, 0.71214, 0.66709,
                                0.61067, 0.52759, 0.322])
        
            # Interpolate water density based on the temperature
            f_density = np.interp(temp, dens_temp, density)
        
            return f_density

        def calc_Reynolds(self, visc, water_vel):
            """
            Calculates the Reynolds number for each fish in an array based on their lengths,
            the kinematic viscosity of the water, and the velocity of the water.
        
            The Reynolds number is a dimensionless quantity that predicts flow patterns in
            fluid flow situations. It is the ratio of inertial forces to viscous forces and
            is used to determine whether a flow will be laminar or turbulent.
        
            This function is designed to work with libraries that support NumPy-like array
            operations, such as NumPy or CuPy, allowing for efficient computation on either
            CPUs or GPUs.
        
            Parameters
            ----------
            visc : float
                The kinematic viscosity of the water in m^2/s.
            water_vel : float
                The velocity of the water in m/s.
        
            Returns
            -------
            array_like
                An array of Reynolds numbers, one for each fish.
        
            Examples
            --------
            >>> lengths = self.arr.array([200, 250, 300])
            >>> visc = 1e-6
            >>> water_vel = 0.5
            >>> reynolds_numbers = calc_Reynolds(lengths, visc, water_vel)
            >>> print(f"The Reynolds numbers are {reynolds_numbers}")
        
            This will output the Reynolds numbers for fish of lengths 200 mm, 250 mm, and 300 mm
            in water with a velocity of 0.5 m/s and a kinematic viscosity of 1e-6 m^2/s.
            """
            # Convert length from millimeters to meters
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = water_vel * length_m / visc
        
            return reynolds_numbers
     
        def calc_surface_area(self):
            """
            Calculates the surface area of each fish in an array based on their lengths.
            
            The surface area is determined using a power-law relationship, which is a common
            empirical model in biological studies to relate the size of an organism to some
            physiological or ecological property, in this case, the surface area.
        
            This function is designed to work with libraries that support NumPy-like array
            operations, such as NumPy or CuPy, allowing for efficient computation on either
            CPUs or GPUs.
        
            Atrributes
            ----------
            length : array_like
                An array of fish lengths in millimeters.
        
            Returns
            -------
            array_like
                An array of surface areas, one for each fish.
        
            Notes
            -----
            The power-law relationship used here is given by the formula:
            SA = 10 ** (a + b * log10(length))
            where `a` and `b` are empirically derived constants.
        
            Examples
            --------
            >>> lengths = self.arr.array([200, 250, 300])
            >>> surface_areas = calc_surface_area(lengths)
            >>> print(f"The surface areas are {surface_areas}")
        
            This will output the surface areas for fish of lengths 200 mm, 250 mm, and 300 mm
            using the power-law relationship with constants a = -0.143 and b = 1.881.
            """
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
        
            # Calculate the surface area for each fish
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length))
        
            return surface_areas

        def drag_coeff(self, reynolds):
            """
            Calculates the drag coefficient for each value in an array of Reynolds numbers.
            
            The relationship between drag coefficient and Reynolds number is modeled using
            a logarithmic fit. This function is designed to work with libraries that support
            NumPy-like array operations, such as NumPy or CuPy, allowing for efficient
            computation on either CPUs or GPUs.
        
            Parameters
            ----------
            reynolds : array_like
                An array of Reynolds numbers.
            arr : module, optional
                The array library to use for calculations (default is NumPy).
        
            Returns
            -------
            array_like
                An array of drag coefficients corresponding to the input Reynolds numbers.
        
            Examples
            --------
            >>> reynolds_numbers = arr.array([2.5e4, 5.0e4, 7.4e4])
            >>> drag_coeffs = drag_coeff(reynolds_numbers)
            >>> print(f"The drag coefficients are {drag_coeffs}")
            """
            # Coefficients from the dataframe, converted to arrays for vectorized operations
            reynolds_data = np.array([2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5])
            drag_data = np.array([0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
        
            # Fit the logarithmic model to the data
            drag_coefficients = np.interp(reynolds, reynolds_data, drag_data)
        
            return drag_coefficients

        def drag_fun(self, mask, t, dt, fish_velocities = None):
            """
            Calculate the drag force on a sockeye salmon swimming upstream.
        
            This function computes the drag force experienced by a sockeye salmon as it
            swims against the current. It takes into account the fish's velocity, the
            water velocity, and the water temperature to determine the kinematic
            viscosity and density of the water. The drag force is calculated using the
            drag equation from fluid dynamics, which incorporates the Reynolds number,
            the surface area of the fish, and the drag coefficient.
        
            Attributes:
                sog (array): Speed over ground of the fish in m/s.
                heading (array): Heading of the fish in radians.
                x_vel, y_vel (array): Water velocity components in m/s.
                water_temp (array): Water temperature in degrees Celsius.
                length (array): Length of the fish in meters.
                wave_drag (array): Additional drag factor due to wave-making.
        
            Returns:
                ndarray: An array of drag force vectors for each fish, where each vector
                is a 2D vector representing the drag force in the x and y directions in N.
        
            Notes:
                - The function assumes that the input arrays are structured as arrays of
                  values, with each index across the arrays corresponding to a different
                  fish.
                - The drag force is computed in a vectorized manner, allowing for
                  efficient calculations over multiple fish simultaneously.
                - The function uses np.stack and np.newaxis to ensure proper alignment
                  and broadcasting of array operations.
                - The drag is calculated in SI units (N).
        
            Examples:
                >>> # Assuming all properties are set in the class
                >>> drags = self.drag_fun()
                >>> print(drags)
                # Output: array of drag force vectors for each fish
            """
            tired_mask = np.where(self.simulation.swim_behav == 3,True,False)

            # Calculate fish velocities
            if fish_velocities is None:
                if t == 0:
                    fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                                self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
                else:
                    fish_x_vel = (self.simulation.X - self.simulation.prev_X)/dt
                    fish_y_vel = (self.simulation.Y - self.simulation.prev_Y)/dt
                    fish_velocities = np.stack((fish_x_vel,fish_y_vel)).T

            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            
            
            water_velocities = np.where(tired_mask[:,np.newaxis], 
                                        water_velocities * 0.2,
                                        water_velocities * 1.)
        
            # Ensure non-zero fish velocity for calculation (avoid division by zero)
            fish_speeds = np.linalg.norm(fish_velocities, axis=-1)
            zero_mask = fish_speeds == 0.0
            if np.any(zero_mask):
                fish_speeds = fish_speeds.copy()
                fish_speeds[zero_mask] = 1e-4
                fish_velocities = fish_velocities.copy()
                fish_velocities[zero_mask] = 1e-4
        
            # Calculate kinematic viscosity and density based on water temperature
            viscosity = self.kin_visc(self.simulation.water_temp)
            density = self.wat_dens(self.simulation.water_temp)

            # Calculate Reynolds numbers
            #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
        
            # Calculate surface areas
            
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
        
            # Calculate the surface area for each fish
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))
        
            # Calculate drag coefficients
            drag_coeffs = self.drag_coeff(reynolds_numbers)
        
            # delegate bulk computation to optimized helper
            fx = fish_velocities[:, 0]
            fy = fish_velocities[:, 1]
            wx = water_velocities[:, 0]
            wy = water_velocities[:, 1]
            n = fx.size
            out = np.zeros((n,2), dtype=np.float64)
            try:
                if _HAS_NUMBA:
                    out = _wrap_drag_fun_numba(fx, fy, wx, wy, mask, float(density), surface_areas, drag_coeffs, self.simulation.wave_drag, self.simulation.swim_behav, out)
                else:
                    out = _wrap_drag_fun_numba(fx, fy, wx, wy, mask, float(density), surface_areas, drag_coeffs, self.simulation.wave_drag, self.simulation.swim_behav, out)
            except Exception:
                # fallback
                out = compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, self.simulation.wave_drag, self.simulation.swim_behav)
            self.simulation.drag = out

        def ideal_drag_fun(self, fish_velocities = None):
            """
            Calculate the ideal drag force on multiple sockeye salmon swimming upstream.
            
            This function computes the ideal drag force for each fish based on its length,
            water velocity, fish velocity, and water temperature. The drag force is computed
            using the drag equation from fluid dynamics, incorporating the Reynolds number,
            surface area, and drag coefficient.
            
            Attributes:
                x_vel, y_vel (array): Water velocity components in m/s for each fish.
                ideal_sog (array): Ideal speed over ground in m/s for each fish.
                heading (array): Heading in radians for each fish.
                water_temp (array): Water temperature in degrees Celsius for each fish.
                length (array): Length of each fish in meters.
                swim_behav (array): Swimming behavior for each fish.
                max_s_U (array): Maximum sustainable swimming speed in m/s for each fish.
                wave_drag (array): Additional drag factor due to wave-making for each fish.
            
            Returns:
                ndarray: An array of ideal drag force vectors for each fish, where each vector
                is a 2D vector representing the drag force in the x and y directions in N.
            
            Notes:
                - The function assumes that the input arrays are structured as arrays of
                  values, with each index across the arrays corresponding to a different
                  fish.
                - The drag force is computed in a vectorized manner, allowing for
                  efficient calculations over multiple fish simultaneously.
                - The function adjusts the fish velocity if it exceeds the maximum
                  sustainable speed based on the fish's behavior.
            """
            # Vector components of water velocity and speed over ground for each fish
            water_velocities = np.stack((self.simulation.x_vel, self.simulation.y_vel), axis=-1)
            
            if fish_velocities is None:
                fish_velocities = np.stack((self.simulation.ideal_sog * np.cos(self.simulation.heading),
                                            self.simulation.ideal_sog * np.sin(self.simulation.heading)), axis=-1)
        
            # calculate ideal swim speed  
            ideal_swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
           
            # make sure fish isn't swimming faster than it should
            refugia_mask = (self.simulation.swim_behav == 2) & (ideal_swim_speeds > self.simulation.max_s_U)
            holding_mask = (self.simulation.swim_behav == 3) & (ideal_swim_speeds > self.simulation.max_s_U)
            too_fast = refugia_mask + holding_mask
            
            fish_velocities = np.where(too_fast[:,np.newaxis],
                                       (self.simulation.max_s_U / ideal_swim_speeds[:,np.newaxis]) * fish_velocities,
                                       fish_velocities)
        
            # Calculate the maximum practical speed over ground
            self.simulation.max_practical_sog = fish_velocities
            
            if self.simulation.num_agents > 1:
                self.simulation.max_practical_sog[np.linalg.norm(self.simulation.max_practical_sog, axis=1) == 0.0] = [0.0001, 0.0001]
            else:
                pass

            # Kinematic viscosity and density based on water temperature for each fish
            viscosity = self.kin_visc(self.simulation.water_temp)
            density = self.wat_dens(self.simulation.water_temp)
        
            # Reynolds numbers for each fish
            #reynolds_numbers = self.calc_Reynolds(self.length, viscosities, np.linalg.norm(water_velocities, axis=1))
            length_m = self.simulation.length / 1000.
        
            # Calculate the Reynolds number for each fish
            reynolds_numbers = np.linalg.norm(water_velocities, axis = -1) * length_m / viscosity
            
            # Surface areas for each fish
            # Constants for the power-law relationship
            a = -0.143
            b = 1.881
            #surface_areas = self.calc_surface_area(self.length)
            surface_areas = 10 ** (a + b * np.log10(self.simulation.length / 1000. * 100.))
        
            # Drag coefficients for each fish
            drag_coeffs = self.drag_coeff(reynolds_numbers)
        
            # Calculate ideal drag forces
            relative_velocities = self.simulation.max_practical_sog - water_velocities
            relative_speeds_squared = np.linalg.norm(relative_velocities, axis=-1)**2
            unit_max_practical_sog = self.simulation.max_practical_sog / np.linalg.norm(self.simulation.max_practical_sog, axis=1)[:, np.newaxis]
        
            # Ideal drag calculation
            ideal_drags = -0.5 * (density * 1000) * \
                (surface_areas[:,np.newaxis] / 100**2) * drag_coeffs[:,np.newaxis] \
                    * relative_speeds_squared[:, np.newaxis] * unit_max_practical_sog \
                          * self.simulation.wave_drag[:, np.newaxis]
        
            return ideal_drags
                

        def swim(self, t, dt, pid_controller, mask):
            """
            Method propels each fish agent forward by calculating its new speed over ground 
            (sog) and updating its position.
        
            Parameters:
            - dt: The time step over which to update the fish's position.
        
            The function performs the following steps for each fish agent:
            1. Calculates the initial velocity of the fish based on its speed over ground 
            (sog) and heading.
            2. Computes the surge by adding the thrust and drag forces, rounding them to 
            two decimal places.
            3. Calculates the acceleration by dividing the surge by the fish's weight and 
            rounding to two decimal places.
            4. Applies a dampening factor to the acceleration to simulate the effect of 
            water resistance.
            5. Updates the fish's velocity by adding the dampened acceleration to the 
            initial velocity.
            6. Updates the fish's speed over ground (sog) based on the new velocity.
            7. Prepares to update the fish's position in the main simulation loop 
            (not implemented here).
        
            Note: The position update is not performed within this function. The 
            'prevPosX' and 'prevPosY' attributes are set to 'self.posX' and 'self.posY' 
            to prepare for the position update, which should be handled in the main 
            simulation loop where this method is called.
            """
            tired_mask = np.where(self.simulation.swim_behav == 3,True,False)
            
            # Step 1: Calculate fish velocity in vector form for each fish
            if t == 0:
                fish_vel_0_x = np.where(mask, self.simulation.sog * np.cos(self.simulation.heading),0) 
                fish_vel_0_y = np.where(mask, self.simulation.sog * np.sin(self.simulation.heading),0)  
                fish_vel_0 = np.stack((fish_vel_0_x, fish_vel_0_y)).T
            else:
                fish_vel_0_x = (self.simulation.X - self.simulation.prev_X)/dt
                fish_vel_0_y = (self.simulation.Y - self.simulation.prev_Y)/dt
                fish_vel_0 = np.stack((fish_vel_0_x,fish_vel_0_y)).T
            
            
            
            ideal_vel_x = np.where(mask, self.simulation.ideal_sog * np.cos(self.simulation.heading),0) 
            ideal_vel_y = np.where(mask, self.simulation.ideal_sog * np.sin(self.simulation.heading),0)  
            
            ideal_vel = np.stack((ideal_vel_x, ideal_vel_y)).T
            
            # Step 2: Calculate surge for each fish
            surge_ini = self.simulation.thrust + self.simulation.drag
            
            # Step 3: Calculate acceleration for each fish
            acc_ini = np.round(surge_ini / self.simulation.weight[:,np.newaxis], 2)  
            
            # Step 4: Update velocity for each fish
            fish_vel_1_ini = fish_vel_0 + acc_ini * dt
                
            # Step 5: Thrust feedback PID controller 
            error = np.where(mask[:,np.newaxis], 
                             np.round(ideal_vel - fish_vel_1_ini,12),
                             0.)
            
            #error = np.where(self.simulation.is_stuck[:,np.newaxis],np.zeros_like(error),error)
            
            self.simulation.error = error
            self.simulation.dead = np.where(np.isnan(error[:, 0]),1,self.simulation.dead)
            
                
            if self.simulation.pid_tuning == True:
                self.simulation.error_array = np.append(self.simulation.error_array, error[0])
                self.simulation.vel_x_array = np.append(self.simulation.vel_x_array, self.simulation.x_vel)
                self.simulation.vel_y_array = np.append(self.simulation.vel_y_array, self.simulation.y_vel)
                
                curr_vel = np.round(np.sqrt(np.power(self.simulation.x_vel,2) + np.power(self.simulation.y_vel,2)),2)
                
                print (f'error: {error}')
                print (f'current velocity: {curr_vel}')
                print (f'Hz: {self.Hz}')
                print (f'thrust: {np.round(self.thrust,2)}')
                print (f'drag: {np.round(self.drag,2)}')
                print (f'sog: {np.round(self.sog,4)}')

                if np.any(np.isnan(error)):
                    print('nan in error')
                    sys.exit()
                    
            else:
                k_p, k_i, k_d = pid_controller.PID_func(np.sqrt(np.power(self.simulation.x_vel,2) + np.power(self.simulation.y_vel,2)),
                                                        self.simulation.length)
                # Ensure k_p/k_i/k_d are numpy arrays with shape (n_agents,)
                k_p = np.asarray(k_p)
                k_i = np.asarray(k_i)
                k_d = np.asarray(k_d)
                # If scalars, expand to per-agent arrays
                if k_p.ndim == 0:
                    k_p = np.full(self.simulation.num_agents, float(k_p))
                if k_i.ndim == 0:
                    k_i = np.full(self.simulation.num_agents, float(k_i))
                if k_d.ndim == 0:
                    k_d = np.full(self.simulation.num_agents, float(k_d))
                pid_controller.k_p = k_p
                pid_controller.k_i = k_i
                pid_controller.k_d = k_d
                
            # Adjust Hzs using the PID controller (vectorized)
            pid_adjustment = pid_controller.update(error, dt, None)
            self.simulation.integral = pid_controller.integral
            self.simulation.pid_adjustment = pid_adjustment
            
            # Step 6: add adjustment to original velocity computation via fast helper
            fv0x = fish_vel_0[:, 0]
            fv0y = fish_vel_0[:, 1]
            accx = acc_ini[:, 0]
            accy = acc_ini[:, 1]
            pidx = pid_adjustment[:, 0]
            pidy = pid_adjustment[:, 1]
            dead_mask = self.simulation.dead == 1
            try:
                fv0x_a = np.ascontiguousarray(fv0x, dtype=np.float64)
                fv0y_a = np.ascontiguousarray(fv0y, dtype=np.float64)
                accx_a = np.ascontiguousarray(accx, dtype=np.float64)
                accy_a = np.ascontiguousarray(accy, dtype=np.float64)
                pidx_a = np.ascontiguousarray(pidx, dtype=np.float64)
                pidy_a = np.ascontiguousarray(pidy, dtype=np.float64)
                tired_a = np.ascontiguousarray(tired_mask, dtype=np.bool_)
                dead_a = np.ascontiguousarray(dead_mask, dtype=np.bool_)
                mask_a = np.ascontiguousarray(mask, dtype=np.bool_)
                dxdy = _swim_core_numba(fv0x_a, fv0y_a, accx_a, accy_a, pidx_a, pidy_a, tired_a, dead_a, mask_a, float(dt))
            except Exception:
                dxdy = _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)
                
            # if np.any(np.linalg.norm(fish_vel_1,axis = -1) > 2* self.ideal_sog):
            #     print ('fuck - why')
            return dxdy
                            
        def jump(self, t, g, mask):
            """
            Simulates each fish jumping using a ballistic trajectory.
        
            Parameters:
            t (float): The current time in the simulation.
            g (float): The acceleration due to gravity.
        
            Attributes:
            time_of_jump (array): The time each fish jumps.
            ucrit (array): Critical swimming speed for each fish.
            sog (array): Speed over ground for each fish.
            heading (array): Heading for each fish.
            y_vel (array): Y-component of water velocity.
            x_vel (array): X-component of water velocity.
            pos_x (array): X-coordinate of the current position of each fish.
            pos_y (array): Y-coordinate of the current position of each fish.
        
            Notes:
            - The function assumes that the input attributes are structured as arrays of
              values, with each index across the arrays corresponding to a different fish.
            - The function updates the position and speed over ground (sog) for each fish
              based on their jump.
            """

            # Reset jump time for each fish
            
            self.simulation.time_of_jump = np.where(mask,t,self.simulation.time_of_jump)
        
            # Get jump angle for each fish
            jump_angles = np.where(mask,np.random.choice([np.radians(45), np.radians(60)], size=self.simulation.ucrit.shape),0)
        
            # Calculate time airborne for each fish
            time_airborne = np.where(mask,(2 * self.simulation.ucrit * np.sin(jump_angles)) / g, 0)
        
            # Calculate displacement for each fish
            displacement = self.simulation.ucrit * time_airborne * np.cos(jump_angles)
            
            # Calculate the new position for each fish
            # should we make this the water direction?  calculate unit vector, multiply by displacement, and add to current position
            
            dx = np.where(mask, displacement * np.cos(self.simulation.heading), 0.0)
            dy = np.where(mask, displacement * np.sin(self.simulation.heading), 0.0)
            
            dxdy = np.stack((dx,dy)).T
            
            # Debug at t=0
            if t == 0:
                print(f'JUMP DEBUG t=0: mask[:3]={mask[:3]}, displacement[:3]={displacement[:3]}')
                print(f'JUMP DEBUG t=0: heading[:3]={self.simulation.heading[:3]}, dx[:3]={dx[:3]}, dy[:3]={dy[:3]}')
                print(f'JUMP DEBUG t=0: dxdy[:3]={dxdy[:3]}')
            
            if np.any(dxdy > 3):
                print ('check jump parameters')
           
            return dxdy        
        
            
    class behavior():
        
        def __init__(self, dt, simulation_object):
            self.dt = dt
            self.simulation = simulation_object
            # convenience copies used throughout behavior methods
            try:
                self.num_agents = int(getattr(simulation_object, 'num_agents', 0))
            except Exception:
                self.num_agents = 0
            self.X = getattr(simulation_object, 'X', None)
            self.Y = getattr(simulation_object, 'Y', None)
            self.sim_time = 0

        def _batch_read_env_patches(self, dataset_name, row_mins, row_maxs, col_mins, col_maxs, target_shape=None):
            """
            Read a single global window from an HDF5 2D dataset that covers all per-agent
            requested patches, then slice out each agent's patch from that global window.

            Returns an array of shape (num_agents, patch_h, patch_w).
            """
            num_agents = len(row_mins)

            ds = self.simulation.hdf5[dataset_name]
            height, width = ds.shape

            # ensure integer arrays
            row_mins = np.asarray(row_mins, dtype=int)
            row_maxs = np.asarray(row_maxs, dtype=int)
            col_mins = np.asarray(col_mins, dtype=int)
            col_maxs = np.asarray(col_maxs, dtype=int)

            global_rmin = max(0, np.min(row_mins))
            global_rmax = min(height, np.max(row_maxs))
            global_cmin = max(0, np.min(col_mins))
            global_cmax = min(width, np.max(col_maxs))

            if global_rmin >= global_rmax or global_cmin >= global_cmax:
                # nothing to read -> return zeros
                ph = 0 if target_shape is None else target_shape[0]
                pw = 0 if target_shape is None else target_shape[1]
                return np.zeros((num_agents, ph, pw), dtype=ds.dtype)

            global_block = ds[global_rmin:global_rmax, global_cmin:global_cmax]

            patches = []
            for r0, r1, c0, c1 in zip(row_mins, row_maxs, col_mins, col_maxs):
                lo_r = r0 - global_rmin
                hi_r = r1 - global_rmin
                lo_c = c0 - global_cmin
                hi_c = c1 - global_cmin
                patch = global_block[lo_r:hi_r, lo_c:hi_c]
                if target_shape is not None:
                    patch = standardize_shape(patch, target_shape=target_shape)
                patches.append(patch)

            return np.stack(patches)
            
        def already_been_here(self, weight, t):
            """
            Calculate repulsive forces based on agents' historical locations within a specified time frame,
            simulating a tendency to avoid areas recently visited.
            
            In HECRAS mode: uses in-memory history tracking without HDF5 I/O.
            In raster mode: uses HDF5 mental map dataset.
        
            This function retrieves the current X and Y positions of agents, converts these positions to
            row and column indices in the mental map's raster grid, and then accesses the relevant sections
            of the mental map from an HDF5 file. It computes the repulsive force exerted by each cell in the
            mental map based on the time since the agent's last visit, applying a time-dependent weighting factor.
        
            Parameters
            ----------
            weight : float
                The strength of the repulsive force.
            t : int
                The current time step in the simulation.
        
            Returns
            -------
            numpy.ndarray
                An array containing the sum of the repulsive forces in the X and Y directions for each agent.
        
            Notes
            -----
            - The method assumes that the HDF5 dataset supports numpy-style advanced indexing.
            - The mental map dataset within the HDF5 file is expected to be named in the format 'memory/{agent_idx}'.
            - The forces are normalized to unit vectors to ensure that the direction of the force is independent of the distance.
            - The method uses a buffer zone, currently set to a 4-cell radius, to limit the computation to a manageable area around each agent.
            - The time-dependent weighting factor (`multiplier`) is applied to cells visited within a specified time range, with a repulsive force applied to these cells.
            """
            # HECRAS mode - use in-memory history instead of HDF5
            if getattr(self.simulation, 'use_hecras', False):
                # For HECRAS mode, track history in memory to avoid HDF5 I/O
                # Initialize history buffer if it doesn't exist
                if not hasattr(self.simulation, '_position_history'):
                    # Store last N positions per agent
                    history_length = 100
                    self.simulation._position_history = {
                        'x': np.zeros((self.simulation.num_agents, history_length)),
                        'y': np.zeros((self.simulation.num_agents, history_length)),
                        't': np.zeros((self.simulation.num_agents, history_length)),
                        'idx': np.zeros(self.simulation.num_agents, dtype=int)  # circular buffer index
                    }
                
                hist = self.simulation._position_history
                repulsive_forces = np.zeros((self.simulation.num_agents, 2), dtype=float)
                
                # Update history for current timestep
                for i in range(self.simulation.num_agents):
                    idx = hist['idx'][i]
                    hist['x'][i, idx] = self.simulation.X[i]
                    hist['y'][i, idx] = self.simulation.Y[i]
                    hist['t'][i, idx] = t
                    hist['idx'][i] = (idx + 1) % hist['x'].shape[1]  # circular buffer
                
                # Calculate repulsive forces from recent history
                for i in range(self.simulation.num_agents):
                    # Get valid history entries (non-zero times)
                    valid_mask = hist['t'][i] > 0
                    if not np.any(valid_mask):
                        continue
                    
                    hist_x = hist['x'][i, valid_mask]
                    hist_y = hist['y'][i, valid_mask]
                    hist_t = hist['t'][i, valid_mask]
                    
                    # Time since visit
                    t_since = t - hist_t
                    
                    # Apply time-dependent weight (same logic as raster mode)
                    multiplier = np.where((t_since > 10) & (t_since < 7200), 
                                         1 - (t_since - 5) / 7195, 0)
                    
                    # Calculate repulsive force from each historical position
                    delta_x = self.simulation.X[i] - hist_x
                    delta_y = self.simulation.Y[i] - hist_y
                    magnitudes = np.sqrt(delta_x**2 + delta_y**2) + 1e-6
                    
                    # Unit vectors weighted by time and distance
                    force_x = np.sum(weight * (delta_x / magnitudes**2) * multiplier)
                    force_y = np.sum(weight * (delta_y / magnitudes**2) * multiplier)
                    
                    repulsive_forces[i, 0] = force_x
                    repulsive_forces[i, 1] = force_y
                
                return repulsive_forces
            
            # Original raster-based HDF5 implementation
            # Step 1: Get the x, y position of the agents
            x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)
        
            # Step 2: Convert these positions to mental map's pixel indices (use cache if available)
            if 'mental_map' in self.simulation._pixel_index_cache:
                mental_map_rows, mental_map_cols = self.simulation._pixel_index_cache['mental_map']
            else:
                mental_map_rows, mental_map_cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        
            # Define buffer zone around current positions
            buff = 10
            row_min = np.clip(mental_map_rows - buff, 0, None)
            row_max = np.clip(mental_map_rows + buff + 1, None,
                              self.simulation.hdf5['memory/0'].shape[0])
            col_min = np.clip(mental_map_cols - buff, 0, None)
            col_max = np.clip(mental_map_cols + buff + 1, None,
                              self.simulation.hdf5['memory/0'].shape[1])
        
            # Using list comprehension to access the relevant sections from the mental map and calculate forces
            repulsive_forces_per_agent = np.array([
                self._calculate_repulsive_force(agent_idx, rmin, rmax, cmin, cmax, weight, t)
                for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
            ])
            # if np.any(np.linalg.norm(repulsive_forces_per_agent, axis = -1) != 0.):
            #     print ('check')
            return repulsive_forces_per_agent

        def _calculate_repulsive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight, t):
            """
            A helper function to calculate the repulsive force for a single agent based on a section of the mental map.
        
            This function computes the repulsive force exerted on an agent by previously visited cells within
            a specified buffer zone around the agent's current position. It considers the time since each cell was last visited,
            applying a conditional weight based on this time to modulate the repulsive force.
        
            Parameters
            ----------
            agent_idx : int
                The index of the agent for whom the repulsive force is being calculated.
            row_min : int
                The minimum row index of the buffer zone in the mental map's raster grid.
            row_max : int
                The maximum row index of the buffer zone in the mental map's raster grid.
            col_min : int
                The minimum column index of the buffer zone in the mental map's raster grid.
            col_max : int
                The maximum column index of the buffer zone in the mental map's raster grid.
            weight : float
                The strength of the repulsive force.
            t : int
                The current time step in the simulation.
        
            Returns
            -------
            numpy.ndarray
                An array containing the repulsive force in the X and Y directions exerted on the specified agent.
        
            Notes
            -----
            - This function is designed to be called within a list comprehension in the `already_been_here` function.
            - It accesses a specific section of the mental map for the given agent, determined by the provided row and column bounds.
            - The force calculation considers the Euclidean distance from each cell to the agent's current position, normalizing the force to unit vectors.
            """
            # Access the relevant section from the mental map
            mmap_section = self.simulation.hdf5['memory/%s' % agent_idx][row_min:row_max, col_min:col_max]
        
            # Calculate time since last visit and apply conditional weight
            t_since = mmap_section - t
            multiplier = np.where((t_since > 10) & (t_since < 7200), 1 - (t_since - 5) / (7195), 0)
        
            # Relative positions and magnitudes
            delta_x = self.simulation.X[agent_idx] - np.arange(col_min, col_max)
            delta_y = self.simulation.Y[agent_idx] - np.arange(row_min, row_max)[:, np.newaxis]
            magnitudes = np.sqrt(delta_x**2 + delta_y**2)
            magnitudes[magnitudes == 0] = 0.000001  # Avoid division by zero
        
            # Unit vectors and repulsive force
            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes
            x_force = ((weight * unit_vector_x) / magnitudes) * multiplier
            y_force = ((weight * unit_vector_y) / magnitudes) * multiplier
        
            # Sum forces for this agent
            total_x_force = np.nansum(x_force)
            total_y_force = np.nansum(y_force)
        
            return np.array([total_x_force, total_y_force])    
        
        def find_nearest_refuge(self, weight):
            """
            Calculate the attractive force towards the nearest refuge cell for agents.
            
            In HECRAS mode: identifies low-velocity areas as refugia.
            In raster mode: uses pre-computed refugia map.
        
            Parameters:
            - weight: float, the weight of the attraction force.
        
            Returns:
            - np.array: Array containing the x and y components of the attraction force for the agents.
            """
            # HECRAS mode - find low velocity areas as refuge (vectorized)
            if getattr(self.simulation, 'use_hecras', False):
                attractive_forces = np.zeros((self.simulation.num_agents, 2), dtype=float)
                
                # Define low velocity threshold
                vel_mag = np.sqrt(self.simulation.x_vel**2 + self.simulation.y_vel**2)
                low_vel_threshold = 0.3 * np.mean(vel_mag)
                
                # For HECRAS, use KDTree to find nearest low-velocity nodes (vectorized)
                if hasattr(self.simulation, 'hecras_map') and 'nodes' in self.simulation.hecras_map:
                    hecras_nodes = self.simulation.hecras_map['nodes']
                    vel_x_field = self.simulation.hecras_node_fields.get('vel_x', np.zeros(len(hecras_nodes)))
                    vel_y_field = self.simulation.hecras_node_fields.get('vel_y', np.zeros(len(hecras_nodes)))
                    hecras_vel_mag = np.sqrt(vel_x_field**2 + vel_y_field**2)
                    
                    # Identify low-velocity refuge nodes
                    refuge_mask = hecras_vel_mag < low_vel_threshold
                    if np.any(refuge_mask):
                        refuge_nodes = hecras_nodes[refuge_mask]
                        
                        # Build KDTree for refuge nodes (cached if possible)
                        if not hasattr(self.simulation, '_refuge_tree') or self.simulation._refuge_tree is None:
                            from scipy.spatial import cKDTree
                            self.simulation._refuge_tree = cKDTree(refuge_nodes)
                            self.simulation._refuge_nodes = refuge_nodes
                        
                        # Query nearest refuge for all agents at once (vectorized)
                        agent_positions = np.column_stack([self.simulation.X, self.simulation.Y])
                        dists, idxs = self.simulation._refuge_tree.query(agent_positions, k=1)
                        
                        # Calculate direction vectors to nearest refuge
                        nearest_refuge = self.simulation._refuge_nodes[idxs]
                        dx = nearest_refuge[:, 0] - self.simulation.X
                        dy = nearest_refuge[:, 1] - self.simulation.Y
                        dist_to_refuge = np.sqrt(dx**2 + dy**2) + 1e-6
                        
                        # Only apply force if refuge is within search radius
                        search_radius = 50.0
                        within_range = dists < search_radius
                        
                        # Attractive force magnitude (vectorized)
                        force_mag = np.where(within_range, weight / dist_to_refuge, 0.0)
                        attractive_forces[:, 0] = force_mag * (dx / dist_to_refuge)
                        attractive_forces[:, 1] = force_mag * (dy / dist_to_refuge)
                
                return attractive_forces  # Return (num_agents, 2) to match raster mode
            
            # Original raster-based implementation
            # Step 1: Get the x, y position of the agents
            x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)
        
            # Step 2: Convert these positions to mental map's pixel indices
            if 'refugia' in self.simulation._pixel_index_cache:
                refugia_map_rows, refugia_map_cols = self.simulation._pixel_index_cache['refugia']
            else:
                refugia_map_rows, refugia_map_cols = geo_to_pixel(x, y, self.simulation.refugia_map_transform)
        
            # Define buffer zone around current positions
            buff = 50
        
            row_min = np.clip(refugia_map_rows - buff, 0, None)
            row_max = np.clip(refugia_map_rows + buff + 1, None, self.simulation.hdf5['refugia/0'].shape[0])
            col_min = np.clip(refugia_map_cols - buff, 0, None)
            col_max = np.clip(refugia_map_cols + buff + 1, None, self.simulation.hdf5['refugia/0'].shape[1])
        
            # Using list comprehension to access the relevant sections from the mental map and calculate forces
            attractive_forces_per_agent = np.array([
                self._calculate_attractive_force(agent_idx, rmin, rmax, cmin, cmax, weight)
                for agent_idx, rmin, rmax, cmin, cmax in zip(np.arange(self.simulation.num_agents), row_min, row_max, col_min, col_max)
            ])
            
            return attractive_forces_per_agent
            
        def _calculate_attractive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight):
            # Access the relevant section from the mental map
            refugia_section = self.simulation.hdf5['refugia/%s'% agent_idx][row_min:row_max, col_min:col_max]
        
            # Create a binary mask for the refuge cells
            refuge_mask = (refugia_section == 1)
        
            if np.any(refuge_mask):
                # Compute the distance transform
                distances = distance_transform_edt(~refuge_mask)
            
                # Find the coordinates of the nearest refuge cell
                nearest_refuge_coords = np.unravel_index(np.argmin(distances), distances.shape)
            
                # Convert pixel coordinates to geographic coordinates
                ref_xy = pixel_to_geo(self.simulation.refugia_map_transform,
                                      nearest_refuge_coords[0],
                                      nearest_refuge_coords[1])
            
                # Calculate the attraction force
                delta_x = ref_xy[0] - self.simulation.X
                delta_y = ref_xy[1] - self.simulation.Y
            
                magnitudes = np.sqrt(delta_x**2 + delta_y**2)
                magnitudes[magnitudes == 0] = 0.000001  # Avoid division by zero
            
                # Unit vectors and attractive force
                unit_vector_x = delta_x / magnitudes
                unit_vector_y = delta_y / magnitudes
                x_force = (weight * unit_vector_x)
                y_force = (weight * unit_vector_y)
            
                # Sum forces for this agent
                attract_x = np.nansum(x_force)
                attract_y = np.nansum(y_force)
            
                return np.array([attract_x, attract_y])
            
            else:
                return np.array([0,0])
        
        def vel_cue(self, weight):
            """
            Calculate the velocity cue for each agent based on the surrounding water velocity.
        
            This function determines the direction with the lowest water velocity within a specified
            buffer around each agent. The buffer size is determined by the agent's swim mode:
            if in 'refugia' mode, the buffer is 15 body lengths; otherwise, it is 5 body lengths.
            The function then computes a velocity cue that points in the direction of the lowest
            water velocity within this buffer.
        
            Parameters:
            - weight (float): A weighting factor applied to the velocity cue.
        
            Returns:
            - velocity_min (ndarray): An array of velocity cues for each agent, where each cue
              is a vector pointing in the direction of the lowest water velocity within the buffer.
              The magnitude of the cue is scaled by the given weight and the agent's body length.
        
            Notes:
            - The function assumes that the HDF5 dataset 'environment/vel_mag' is accessible and
              supports numpy-style advanced indexing.
            - The velocity cue is calculated as a unit vector in the direction of the lowest velocity,
              scaled by the weight and normalized by the square of 5 body lengths in meters.
            """
            # If using HECRAS, return zero cue (no spatial velocity map available)
            if hasattr(self.simulation, 'hecras_mapping_enabled') and self.simulation.hecras_mapping_enabled:
                return np.zeros((2, self.simulation.num_agents), dtype=float)
            
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length#.get() if isinstance(self.length, cp.ndarray) else self.length
            
            # calculate buffer size based on swim mode, if we are in refugia mode buffer is 15 body lengths else 5
            #buff = np.where(self.swim_mode == 2, 15 * length_numpy, 5 * length_numpy)
            
            # for array operations, buffers are represented as a slice (# rows and columns)
            buff = 2
            
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
            
            # find the row and column in the direction raster (use cache if available)
            if 'depth' in self.simulation._pixel_index_cache:
                rows, cols = self.simulation._pixel_index_cache['depth']
            else:
                rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
            
            # Access the velocity dataset from the HDF5 file by slicing and dicing
            
            # get slices 
            xmin = cols - buff
            xmax = cols + buff + 1
            ymin = rows - buff
            ymax = rows + buff + 1
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)

            # Prepare per-agent bounding boxes and read patches in one HDF5 read
            row_mins = ymin
            row_maxs = ymax
            col_mins = xmin
            col_maxs = xmax

            vel3d = self._batch_read_env_patches('environment/vel_mag', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2*buff+1,2*buff+1))
            x_coords = self._batch_read_env_patches('x_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2*buff+1,2*buff+1))
            y_coords = self._batch_read_env_patches('y_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2*buff+1,2*buff+1))
            
            vel3d_multiplier = calculate_front_masks(self.simulation.heading.flatten(), 
                                                     x_coords, 
                                                     y_coords, 
                                                     np.nan_to_num(self.simulation.X.flatten()), 
                                                     np.nan_to_num(self.simulation.Y.flatten()), 
                                                     behind_value = 999.9)
                
            vel3d = vel3d * vel3d_multiplier
                
            num_agents, rows, cols = vel3d.shape
            
            # Reshape the 3D array into a 2D array where each row represents an agent
            vel3d = vel3d.reshape(num_agents, rows * cols)
            
            # Find the index of the minimum value in each row (agent)
            flat_indices = np.argmin(vel3d, axis=1)
            
            # Convert flat indices to row and column indices
            min_row_indices = flat_indices // cols
            min_col_indices = flat_indices % cols
                
            # Convert the index back to geographical coordinates
            min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        min_row_indices + ymin, 
                                        min_col_indices + xmin)
            min_x = min_x
            min_y = min_y
            
            # delta_x = self.X - min_x
            # delta_y = self.Y - min_y
            delta_x = min_x - self.simulation.X
            delta_y = min_y - self.simulation.Y
            delta_x_sq = np.power(delta_x,2)
            delta_y_sq = np.power(delta_y,2)
            dist = np.sqrt(delta_x_sq + delta_y_sq)

            # Initialize an array to hold the velocity cues for each agent
            velocity_min = np.zeros((self.simulation.num_agents, 2), dtype=float)

            # attract_x = (weight * delta_x/dist) / np.power(buff,2)
            # attract_y = (weight * delta_y/dist) / np.power(buff,2)
            attract_x = weight * delta_x/dist
            attract_y = weight * delta_y/dist
            return np.array([attract_x,attract_y])

        def rheo_cue(self, weight, downstream = False):
            """
            Calculate the rheotactic heading command for each agent.
        
            This function computes a heading command based on the water velocity direction at the
            agent's current position. The heading is adjusted to face upstream by subtracting 180
            degrees from the sampled velocity direction. The resulting vector is scaled by the
            given weight and normalized by the square of twice the agent's body length in meters.
        
            Parameters:
            - weight (float): A weighting factor applied to the rheotactic cue.
        
            Returns:
            - rheotaxis (ndarray): An array of rheotactic cues for each agent, where each cue
              is a vector pointing upstream. The magnitude of the cue is scaled by the given
              weight and the agent's body length.
        
            Notes:
            - The function assumes that the method `sample_environment` is available and can
              sample the 'vel_dir' from the environment given a transformation matrix.
            - The function converts the velocity direction from degrees to radians and adjusts
              it to point upstream.
            - If `self.length` is a CuPy array, it is converted to a NumPy array for computation.
            """
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length#.get() if isinstance(self.length, cp.ndarray) else self.length
        
            # If using HECRAS node-based mapping, use the per-agent velocities directly
            if hasattr(self.simulation, 'hecras_mapping_enabled') and self.simulation.hecras_mapping_enabled:
                x_vel = self.simulation.x_vel
                y_vel = self.simulation.y_vel
            else:
                # Batch sample x and y velocity to avoid repeated HDF reads
                vals = self.simulation.batch_sample_environment([self.simulation.vel_dir_rast_transform,
                                                                self.simulation.vel_dir_rast_transform],
                                                               ['vel_x', 'vel_y'])
                x_vel = vals['vel_x']
                y_vel = vals['vel_y']
            
            # Ensure velocities are finite and non-NaN
            x_vel = np.nan_to_num(x_vel, nan=0.0, posinf=0.0, neginf=0.0)
            y_vel = np.nan_to_num(y_vel, nan=0.0, posinf=0.0, neginf=0.0)
                
            if downstream == False:
                x_vel = x_vel * -1
                y_vel = y_vel * -1
            
            # Calculate the unit vector in the upstream direction
            v = np.column_stack([x_vel, y_vel])  
            v_norm = np.linalg.norm(v, axis = -1)[:,np.newaxis]
            # Avoid division by zero - use larger epsilon for stability
            v_norm = np.where(v_norm < 1e-6, 1e-6, v_norm)
            v_hat = v / v_norm
            
            # Calculate the rheotactic cue
            rheotaxis = weight* v_hat

            return rheotaxis
        
        def border_cue(self, weight, t):
            """
            Calculate the border cue for each agent based on the surrounding distance 
            from the boundary.
            
            Works in both raster mode (uses distance_to raster) and HECRAS mode (simplified).
            """
            
            # HECRAS mode - use per-agent `sim.distance_to` (vectorized, fast)
            if getattr(self.simulation, 'use_hecras', False):
                # Get per-agent distance to dry land (meters). If unavailable, fallback to zeros.
                current_distances = getattr(self.simulation, 'distance_to', None)
                if current_distances is None:
                    return np.zeros((2, self.simulation.num_agents), dtype=float)

                # Determine per-agent body length (m) and threshold (m)
                length_arr = np.asarray(getattr(self.simulation, 'length', 0.0))
                if length_arr.size == 1:
                    body_length_m = float(length_arr) / 1000.0
                    threshold_m = max(2.0 * body_length_m, 1.0)
                    too_close = current_distances <= threshold_m
                else:
                    body_length_m = length_arr / 1000.0
                    threshold_m = np.maximum(2.0 * body_length_m, 1.0)
                    too_close = current_distances <= threshold_m

                if not np.any(too_close):
                    return np.zeros((2, self.simulation.num_agents), dtype=float)

                # Find direction to channel center by sampling nearby nodes with higher distance_to values
                if hasattr(self.simulation, 'hecras_map') and 'nodes' in self.simulation.hecras_map:
                    hecras_nodes = self.simulation.hecras_map['nodes']

                    # Get distance_to values at all HECRAS nodes
                    if hasattr(self.simulation, 'hecras_node_fields') and 'distance_to' in self.simulation.hecras_node_fields:
                        node_distances = self.simulation.hecras_node_fields['distance_to']

                        from scipy.spatial import cKDTree
                        if not hasattr(self.simulation, '_hecras_tree_cached'):
                            self.simulation._hecras_tree_cached = cKDTree(hecras_nodes)

                        agent_positions = np.column_stack([self.simulation.X, self.simulation.Y])
                        # Choose k neighbors (at most number of nodes)
                        k = min(20, len(hecras_nodes))
                        query_positions = agent_positions[too_close]
                        dists, idxs = self.simulation._hecras_tree_cached.query(query_positions, k=k)

                        center_direction_x = np.zeros(self.simulation.num_agents)
                        center_direction_y = np.zeros(self.simulation.num_agents)

                        too_close_indices = np.where(too_close)[0]
                        max_force = 5.0
                        for i, agent_idx in enumerate(too_close_indices):
                            neigh_idx = idxs[i]
                            neigh_idx = np.atleast_1d(neigh_idx)
                            neighbor_distances = node_distances[neigh_idx]
                            best_local = np.argmax(neighbor_distances)
                            best_node_idx = neigh_idx[best_local]
                            best_node = hecras_nodes[best_node_idx]

                            # Direction toward center
                            dx = best_node[0] - self.simulation.X[agent_idx]
                            dy = best_node[1] - self.simulation.Y[agent_idx]
                            dist_to_center = np.hypot(dx, dy) + 1e-6

                            # Compute raw force and cap it to avoid runaway/oscillation
                            raw_force = weight / dist_to_center
                            force_mag = np.clip(raw_force, -max_force, max_force)
                            center_direction_x[agent_idx] = force_mag * (dx / dist_to_center)
                            center_direction_y[agent_idx] = force_mag * (dy / dist_to_center)

                        # Debug: print summary and a small sample of forces
                        try:
                            n_close = int(np.sum(too_close))
                            sample_n = min(6, n_close)
                            sample_ids = too_close_indices[:sample_n]
                            sample_fx = center_direction_x[sample_ids].tolist()
                            sample_fy = center_direction_y[sample_ids].tolist()
                            if np.ndim(threshold_m) > 0:
                                thr_repr = (float(np.min(threshold_m)), float(np.max(threshold_m)))
                            else:
                                thr_repr = float(threshold_m)
                            print(f"border_cue: {n_close} agents too close, threshold={thr_repr}; sample forces (x,y): {list(zip(sample_fx, sample_fy))}")
                        except Exception:
                            pass

                        return np.vstack([center_direction_x, center_direction_y])

                # Fallback if no HECRAS node data available
                return np.zeros((2, self.simulation.num_agents), dtype=float)
            
            # Original raster-based implementation
            # Convert self.length to a NumPy array if it's a CuPy array
            length_numpy = self.simulation.length  # .get() if isinstance(self.length, cp.ndarray) else self.length
            
            # For simplicity, let's use a fixed buffer size. You can adjust this as needed.
            buff = 2  # This could be dynamic based on your requirements
        
            # get the x, y position of the agent 
            x, y = (np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y))
            
            if 'depth' in self.simulation._pixel_index_cache:
                rows, cols = self.simulation._pixel_index_cache['depth']
            else:
                rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
            
            # get slices 
            xmin = cols - buff
            xmax = cols + buff + 1
            ymin = rows - buff
            ymax = rows + buff + 1
        
            # Ensure indices are within valid range
            xmin = np.clip(xmin, 0, self.simulation.hdf5['environment/distance_to'].shape[1] - 1)
            xmax = np.clip(xmax, 0, self.simulation.hdf5['environment/distance_to'].shape[1])
            ymin = np.clip(ymin, 0, self.simulation.hdf5['environment/distance_to'].shape[0] - 1)
            ymax = np.clip(ymax, 0, self.simulation.hdf5['environment/distance_to'].shape[0])
        
            row_mins = ymin
            row_maxs = ymax
            col_mins = xmin
            col_maxs = xmax

            x_coords = self._batch_read_env_patches('x_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1))
            y_coords = self._batch_read_env_patches('y_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1))

            front_multiplier = calculate_front_masks(self.simulation.heading,
                                                     x_coords,
                                                     y_coords,
                                                     self.simulation.X, 
                                                     self.simulation.Y)
            # get distance to border raster per agent
            dist3d = self._batch_read_env_patches('environment/distance_to', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1)) * front_multiplier
            
            num_agents, rows, cols = dist3d.shape
            
            # Reshape the 3D array into a 2D array where each row represents an agent
            dist3d = dist3d.reshape(num_agents, rows * cols) 
            
            # Find the index of the maximum value in each row (agent)
            flat_indices = np.argmax(dist3d, axis=1)
            
            # Convert flat indices to row and column indices
            max_row_indices = flat_indices // cols
            max_col_indices = flat_indices % cols
            
            # Convert the index back to geographical coordinates
            max_x, max_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        max_row_indices + ymin, 
                                        max_col_indices + xmin)
            
            delta_x = np.zeros(self.simulation.X.shape)
            delta_y = np.zeros(self.simulation.Y.shape)
            
            # delta_x = self.simulation.X - max_x
            # delta_y = self.simulation.Y - max_y
            
            # rather than being repelled here, we are actually attracted to the furthest point from the border
            delta_x = max_x - self.simulation.X 
            delta_y = max_y - self.simulation.Y 
            
            dist = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            
            # check if fish is in the center of the channel?
            # Calculate the current distance to the border for each agent
            current_distances = self.simulation.sample_environment(self.simulation.depth_rast_transform, 'distance_to')
            self.simulation.current_distances = current_distances
        
            # Identify agents that are too close to the border 
            too_close = np.where(current_distances <= 1 * (self.simulation.length/1000.),1,0)# self.simulation.length / 1000.) #| \
                #(self.simulation.in_eddy == 1)
                
            # if np.any(too_close == 1):
            #     print ('boundary force needed')
                
            too_close = np.where(self.simulation.in_eddy == 1,1,too_close)
            
            # calculate repulsive force
            repulse_x = np.where(too_close, 
                                 weight * delta_x / dist,
                                 np.zeros_like(delta_x))
            repulse_y = np.where(too_close,
                                 weight * delta_y / dist,
                                 np.zeros_like(delta_y))

            return np.array([repulse_x, repulse_y])


        def shallow_cue(self, weight):
            """
            Calculate the repulsive force vectors from shallow water areas within a 
            specified buffer around each agent using a vectorized approach.
        
            Works in both raster mode (reads from HDF5) and HECRAS mode (uses current depth values).
        
            Parameters:
            - weight (float): The weighting factor to scale the repulsive force.
        
            Returns:
            - np.ndarray: A 2D array where each row corresponds to an agent and 
            contains the sum of the repulsive forces in the X and Y directions.
            """
            
            # Simple depth-based repulsion for HECRAS mode
            # If agent is in shallow water, create repulsive force pointing away from shallow areas
            if getattr(self.simulation, 'use_hecras', False):
                repulsive_forces = np.zeros((self.simulation.num_agents, 2), dtype=float)
                min_depth = self.simulation.too_shallow
                
                # For agents in shallow water, calculate repulsive force
                shallow_mask = self.simulation.depth < min_depth
                if np.any(shallow_mask):
                    # Create force pointing in direction of deeper water
                    # Use velocity gradient as proxy for depth gradient (water flows from high to low)
                    # Perpendicular to flow is often toward shore (shallow)
                    vel_x = getattr(self.simulation, 'x_vel', np.zeros_like(self.simulation.X))
                    vel_y = getattr(self.simulation, 'y_vel', np.zeros_like(self.simulation.Y))
                    vel_mag = np.sqrt(vel_x**2 + vel_y**2) + 1e-6
                    
                    # Normalized velocity direction (parallel to flow)
                    flow_dir_x = vel_x / vel_mag
                    flow_dir_y = vel_y / vel_mag
                    
                    # Force magnitude scaled by how shallow
                    depth_deficit = np.maximum(0, min_depth - self.simulation.depth)
                    force_mag = weight * depth_deficit / (min_depth + 1e-6)
                    
                    # Apply force in flow direction (toward deeper water) for shallow agents
                    repulsive_forces[shallow_mask, 0] = force_mag[shallow_mask] * flow_dir_x[shallow_mask]
                    repulsive_forces[shallow_mask, 1] = force_mag[shallow_mask] * flow_dir_y[shallow_mask]
                
                return repulsive_forces
            
            # Original raster-based implementation
            buff = 2 #* self.length / 1000.  # 2 meters
        
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
        
            if 'depth' in self.simulation._pixel_index_cache:
                rows, cols = self.simulation._pixel_index_cache['depth']
            else:
                rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        
            # calculate array slice bounds for each agent
            xmin = cols - buff
            xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
            ymin = rows - buff
            ymax = rows + buff + 1  # +1 for the same reason
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)
        
            # Initialize an array to hold the repulsive forces for each agent
            repulsive_forces = np.zeros((self.simulation.num_agents,2), dtype=float)
            
            #min_depth = (self.simulation.body_depth * 1.1) / 100.# Use advanced indexing to create a boolean mask for the slices
            min_depth = self.simulation.too_shallow
            
            row_mins = ymin
            row_maxs = ymax
            col_mins = xmin
            col_maxs = xmax

            depths = self._batch_read_env_patches('environment/depth', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1))
            x_coords = self._batch_read_env_patches('x_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1))
            y_coords = self._batch_read_env_patches('y_coords', row_mins, row_maxs, col_mins, col_maxs, target_shape=(2 * buff + 1,2 * buff + 1))

            front_multiplier = calculate_front_masks(self.simulation.heading,
                                                     x_coords,
                                                     y_coords,
                                                     self.simulation.X, 
                                                     self.simulation.Y)

            # create a multiplier
            depth_multiplier = np.where(depths < min_depth[:,np.newaxis,np.newaxis], 1, 0)

            # Calculate the difference vectors
            # delta_x = x_coords - self.simulation.X[:,np.newaxis,np.newaxis]
            # delta_y = y_coords - self.simulation.Y[:,np.newaxis,np.newaxis]
            
            delta_x =  self.simulation.X[:,np.newaxis,np.newaxis] - x_coords
            delta_y =  self.simulation.Y[:,np.newaxis,np.newaxis] - y_coords
            
            # Calculate the magnitude of each vector
            magnitudes = np.sqrt(np.power(delta_x,2) + np.power(delta_y,2))
            
            #TODO rater than norm of every delta_x, we just grab the mean delta_x, delta_y?

            # Avoid division by zero
            magnitudes = np.where(magnitudes == 0, 0.000001, magnitudes)
            
            # Normalize each vector to get the unit direction vectors
            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes
        
            # Calculate repulsive force in X and Y directions for this agent
            x_force = ((weight * unit_vector_x) / magnitudes) * depth_multiplier * front_multiplier
            y_force = ((weight * unit_vector_y) / magnitudes) * depth_multiplier * front_multiplier
        
            # Sum the forces for this agent
            if self.simulation.num_agents > 1:
                total_x_force = np.nansum(x_force, axis = (1, 2))
                total_y_force = np.nansum(y_force, axis = (1, 2))
            else:
                total_x_force = np.nansum(x_force)
                total_y_force = np.nansum(y_force)
        
            repulsive_forces =  np.array([total_x_force, total_y_force]).T
            
            return repulsive_forces

        def wave_drag_multiplier(self):
            """
            Calculate the wave drag multiplier based on the body depth of the fish 
            submerged and data from Hughes 2004.
        
            This function reads a CSV file containing digitized data from Hughes 2004 
            Figure 3, which relates the body depth of fish submerged to the wave drag 
            multiplier. It sorts this data, fits a univariate spline to it, and then 
            uses this fitted function to calculate the wave drag multiplier for the 
            current instance based on its submerged body depth.
        
            The wave drag multiplier is used to adjust the drag force experienced by 
            the fish due to waves, based on how much of the fish's body is submerged. 
            A multiplier of 1 indicates no additional drag (fully submerged), while 
            values less than 1 indicate increased drag due to the fish's body interacting 
            with the water's surface.
        
            The function updates the instance's `wave_drag` attribute with the 
            calculated wave drag multipliers.
        
            Notes:
            - The CSV file should be located at '../data/wave_drag_huges_2004_fig3.csv'.
            - The CSV file is expected to have columns 'body_depths_submerged' and 'wave_drag_multiplier'.
            - The spline fit is of degree 3 and extends with a constant value (ext=0) outside the range of the data.
            - The function assumes that the body depth of the fish (`self.body_depth`) is provided in centimeters.
            - The `self.z` attribute represents the depth at which the fish is currently swimming.
        
            Returns:
            - None: The function updates the `self.wave_drag` attribute in place.
            """

            # get data
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '../data/wave_drag_huges_2004_fig3.csv')
            try:
                hughes = pd.read_csv(data_dir)
                hughes.sort_values(by='body_depths_submerged', ascending=True, inplace=True)
                wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged,
                                                 hughes.wave_drag_multiplier,
                                                 k=3, ext=0)
                body_depths = self.simulation.z / (self.simulation.body_depth / 100.)
                self.simulation.wave_drag = np.where(body_depths >= 3, 1, wave_drag_fun(body_depths))
            except Exception:
                # Fallback: if CSV not available, assume no additional wave drag
                self.simulation.wave_drag = np.ones_like(self.simulation.z)
           
        def wave_drag_cue(self, weight):
            """
            Calculate the direction to the optimal depth cell for each agent to minimize wave drag.
        
            This function computes the direction vectors pointing towards the depth that is closest to each agent's optimal water depth. The optimal water depth is the depth at which the agent experiences the least wave drag. The function uses a buffer around each agent to search for the optimal depth within that area.
        
            Parameters:
            - weight (float): A weighting factor applied to the direction vectors.
        
            Returns:
            - weighted_direction_vectors (ndarray): An array of weighted direction vectors for each agent. Each vector points towards the cell with the depth closest to the agent's optimal water depth.
        
            Notes:
            - The function assumes that the HDF5 dataset 'environment/depth' is accessible and contains the depth raster data.
            - The buffer size is set to 2 meters around each agent's position.
            - The function uses the `geo_to_pixel` method to convert geographic coordinates to pixel indices in the raster.
            - The direction vectors are normalized to unit vectors and then scaled by the given weight.
            - If the agent is already at the optimal depth, the direction vector is set to zero, indicating no movement is necessary.
            - The function iterates over each agent to calculate their respective direction vectors.
            """
            
            # If using HECRAS, return zero cue (no spatial depth map available)
            if hasattr(self.simulation, 'hecras_mapping_enabled') and self.simulation.hecras_mapping_enabled:
                return np.zeros((2, self.simulation.num_agents), dtype=float)
        
            # identify buffer
            buff = 2.  # 2 meters
            
            # get the x, y position of the agent 
            x, y = (self.simulation.X, self.simulation.Y)
        
            if 'depth' in self.simulation._pixel_index_cache:
                rows, cols = self.simulation._pixel_index_cache['depth']
            else:
                rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        
            # calculate array slice bounds for each agent
            xmin = cols - buff
            xmax = cols + buff + 1  # +1 because slicing is exclusive on the upper bound
            ymin = rows - buff
            ymax = rows + buff + 1  # +1 for the same reason
            
            xmin = xmin.astype(np.int32)
            xmax = xmax.astype(np.int32)
            ymin = ymin.astype(np.int32)
            ymax = ymax.astype(np.int32)
            
            # Initialize an array to hold the direction vectors for each agent
            direction_vectors = np.zeros((len(x), 2), dtype=float)
            
            row_mins = ymin
            row_maxs = ymax
            col_mins = xmin
            col_maxs = xmax

            dep3D = self._batch_read_env_patches('environment/depth', row_mins, row_maxs, col_mins, col_maxs)
            x_coords = self._batch_read_env_patches('x_coords', row_mins, row_maxs, col_mins, col_maxs)
            y_coords = self._batch_read_env_patches('y_coords', row_mins, row_maxs, col_mins, col_maxs)

            # Safety check: if patches are empty (agents out of bounds), return zero vectors
            if dep3D.size == 0 or dep3D.shape[1] == 0 or dep3D.shape[2] == 0:
                return np.zeros((2, len(x)), dtype=float)

            dep3D_multiplier = calculate_front_masks(self.simulation.heading.flatten(), 
                                                     x_coords, 
                                                     y_coords, 
                                                     self.simulation.X.flatten(), 
                                                     self.simulation.Y.flatten(), 
                                                     behind_value = 99999.9)
            dep3D = dep3D * dep3D_multiplier

            num_agents, rows, cols = dep3D.shape
     
            # Reshape the 3D array into a 2D array where each row represents an agent
            reshaped_dep3D = dep3D.reshape(num_agents, rows * cols)
            
            # Find the cell with the depth closest to the agent's optimal depth
            optimal_depth_diff = np.abs(reshaped_dep3D - self.simulation.opt_wat_depth[:,np.newaxis])

            # Find the index of the minimum value in each row (agent)
            flat_indices = np.argmin(optimal_depth_diff, axis=1)
            
            # Convert flat indices to row and column indices
            min_row_indices = flat_indices // cols
            min_col_indices = flat_indices % cols

            # Convert the index back to geographical coordinates
            min_x, min_y = pixel_to_geo(self.simulation.vel_mag_rast_transform, 
                                        min_row_indices + ymin, 
                                        min_col_indices + xmin)
            
            # delta_x = self.X - min_x
            # delta_y = self.Y - min_y
            delta_x = min_x - self.simulation.X
            delta_y = min_y - self.simulation.Y
            delta_x_sq = np.power(delta_x,2)
            delta_y_sq = np.power(delta_y,2)
            dist = np.sqrt(delta_x_sq + delta_y_sq)

            # Initialize an array to hold the velocity cues for each agent
            velocity_min = np.zeros((self.simulation.num_agents, 2), dtype=float)

            attract_x = weight * delta_x/dist
            attract_y = weight * delta_y/dist
            
            return np.array([attract_x,attract_y])
        
        def cohesion_cue(self, weight, consider_front_only=False):
            """
            Calculate the attractive force towards the average position (cohesion) of the school for each agent.
        
            Parameters:
            - weight (float): The weighting factor to scale the attractive force.
            - consider_front_only (bool): If True, consider only the fish in front of each agent.
        
            Returns:
            - np.ndarray: An array of attractive force vectors towards the average position of the school for each agent.
        
            Notes:
            - The function assumes that `self.simulation.X` and `self.simulation.Y` are arrays 
              containing the x and y coordinates of all agents.
            """
            num_agents = self.simulation.num_agents
        
            # Flatten the list of neighbor indices and create a corresponding array of agent indices
            neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
            agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
            
            # Aggregate X and Y coordinates of all neighbors
            x_neighbors = self.simulation.X[neighbor_indices]
            y_neighbors = self.simulation.Y[neighbor_indices]
            
            # Calculate vectors from agents to their neighbors
            vectors_to_neighbors_x = x_neighbors - self.simulation.X[agent_indices]
            vectors_to_neighbors_y = y_neighbors - self.simulation.Y[agent_indices]
        
            if consider_front_only:
                # Calculate agent velocity vectors
                agent_velocities_x = self.simulation.x_vel[agent_indices]
                agent_velocities_y = self.simulation.y_vel[agent_indices]
                
                # Calculate dot products
                dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
        
                # Filter out neighbors that are behind the agent
                valid_neighbors_mask = dot_products > 0
            else:
                # Consider all neighbors
                valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)
        
            # Filter valid neighbor indices and their corresponding agent indices
            valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
            valid_agent_indices = agent_indices[valid_neighbors_mask]
        
            # Calculate Cohesion vectors
            
            # Calculate centroid for cohesion
            center_x = np.zeros(num_agents)
            center_y = np.zeros(num_agents)
            np.add.at(center_x, valid_agent_indices, x_neighbors[valid_neighbors_mask])
            np.add.at(center_y, valid_agent_indices, y_neighbors[valid_neighbors_mask])
            counts = np.bincount(valid_agent_indices, minlength=num_agents)
            center_x /= counts + (counts == 0)  # Avoid division by zero
            center_y /= counts + (counts == 0)  # Avoid division by zero
            
            # Calculate vectors to average position (centroid)
            vectors_to_center_x = center_x - self.simulation.X
            vectors_to_center_y = center_y - self.simulation.Y
        
            # Calculate distances to average position (centroid)
            distances_to_center = np.sqrt(vectors_to_center_x**2 + vectors_to_center_y**2)
        
            # Normalize vectors (add a small epsilon to distances to avoid division by zero)
            epsilon = 1e-10
            v_hat_center_x = np.divide(vectors_to_center_x, distances_to_center + epsilon, out=np.zeros_like(self.simulation.x_vel), where=distances_to_center+epsilon != 0)
            v_hat_center_y = np.divide(vectors_to_center_y, distances_to_center + epsilon, out=np.zeros_like(self.simulation.y_vel), where=distances_to_center+epsilon != 0)
        
            # Calculate attractive forces
            cohesion_array = np.zeros((num_agents, 2))
            cohesion_array[:, 0] = weight * v_hat_center_x
            cohesion_array[:, 1] = weight * v_hat_center_y
            
    
            return np.nan_to_num(cohesion_array)
        
        def alignment_cue(self, weight, consider_front_only=False, use_sog=True, sog_weight=1.0):
            """
            Calculate the attractive force towards the average heading (alignment) of the school for each agent.
        
            Parameters:
            - weight (float): The weighting factor to scale the attractive force.
            - consider_front_only (bool): If True, consider only the fish in front of each agent.
        
            Returns:
            - np.ndarray: An array of attractive force vectors towards the average heading of the school for each agent.
        
            Notes:
            - The function assumes that `self.simulation.heading` is an array 
              containing the heading angles of all agents.
            - The function assumes that `self.simulation.x_vel` and `self.simulation.y_vel` are arrays 
              containing the x and y components of velocity for all agents.
            """
            num_agents = self.simulation.num_agents
        
            # Flatten the list of neighbor indices and create a corresponding array of agent indices
            neighbor_indices = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
            agent_indices = np.repeat(np.arange(num_agents), [len(neighbors) for neighbors in self.simulation.agents_within_buffers]).astype(np.int32)
            
            # Aggregate headings of all neighbors
            headings_neighbors = self.simulation.heading[neighbor_indices]
            
            # Calculate vectors from agents to their neighbors
            vectors_to_neighbors_x = self.simulation.X[neighbor_indices] - self.simulation.X[agent_indices]
            vectors_to_neighbors_y = self.simulation.Y[neighbor_indices] - self.simulation.Y[agent_indices]
        
            if consider_front_only:
                # Calculate agent velocity vectors
                agent_velocities_x = self.simulation.x_vel[agent_indices]
                agent_velocities_y = self.simulation.y_vel[agent_indices]
                
                # Calculate dot products
                dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
        
                # Filter out neighbors that are behind the agent
                valid_neighbors_mask = dot_products > 0
            else:
                # Consider all neighbors
                valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)
        
            # Filter valid neighbor indices and their corresponding agent indices
            valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
            valid_agent_indices = agent_indices[valid_neighbors_mask]
        
            # Calculate average headings for valid neighbors
            avg_heading = np.zeros(num_agents)
            np.add.at(avg_heading, valid_agent_indices, headings_neighbors[valid_neighbors_mask])
            counts = np.bincount(valid_agent_indices, minlength=num_agents)
            avg_heading /= counts + (counts == 0)  # Avoid division by zero
            no_school = np.where(avg_heading == 0., 0., 1.)
        
            # Calculate unit vectors for average headings
            avg_heading_x = np.cos(avg_heading)
            avg_heading_y = np.sin(avg_heading)

            # Default alignment vector: point fish velocity toward average heading unit vector
            vectors_to_heading_x = avg_heading_x - self.simulation.x_vel
            vectors_to_heading_y = avg_heading_y - self.simulation.y_vel

            # Calculate distances to average headings
            distances = np.sqrt(vectors_to_heading_x**2 + vectors_to_heading_y**2)

            # Normalize vectors (add a small epsilon to distances to avoid division by zero)
            epsilon = 1e-10
            v_hat_align_x = np.divide(vectors_to_heading_x, distances + epsilon, out=np.zeros_like(self.simulation.x_vel), where=distances+epsilon != 0)
            v_hat_align_y = np.divide(vectors_to_heading_y, distances + epsilon, out=np.zeros_like(self.simulation.y_vel), where=distances+epsilon != 0)

            # Base alignment force (heading alignment)
            alignment_array = np.zeros((num_agents, 2))
            alignment_array[:, 0] = weight * v_hat_align_x * no_school
            alignment_array[:, 1] = weight * v_hat_align_y * no_school

            # Optional: augment alignment with SOG/velocity similarity
            if use_sog:
                try:
                    # For each agent, compute mean neighbor SOG and desired velocity vector
                    # Build adjacency by reusing agents_within_buffers (list of neighbor index arrays)
                    neighbor_sogs = np.empty(num_agents, dtype=float)
                    for ag in range(num_agents):
                        neigh = self.simulation.agents_within_buffers[ag]
                        if len(neigh) > 0:
                            neighbor_sogs[ag] = np.mean(self.simulation.sog[neigh])
                        else:
                            neighbor_sogs[ag] = self.simulation.sog[ag]
                    # Ensure minima
                    neighbor_sogs = np.where(neighbor_sogs < 0.5 * self.simulation.length / 1000,
                                              0.5 * self.simulation.length / 1000,
                                              neighbor_sogs)
                    # Desired neighbor velocity vectors (unit directions times neighbor_sogs)
                    desired_vx = neighbor_sogs * np.cos(avg_heading)
                    desired_vy = neighbor_sogs * np.sin(avg_heading)

                    # Velocity difference (desired - current water-relative velocity)
                    vel_diff_x = desired_vx - self.simulation.x_vel
                    vel_diff_y = desired_vy - self.simulation.y_vel
                    vel_diff_norm = np.sqrt(vel_diff_x**2 + vel_diff_y**2)
                    v_hat_vel_x = np.divide(vel_diff_x, vel_diff_norm + epsilon, out=np.zeros_like(vel_diff_x), where=vel_diff_norm+epsilon != 0)
                    v_hat_vel_y = np.divide(vel_diff_y, vel_diff_norm + epsilon, out=np.zeros_like(vel_diff_y), where=vel_diff_norm+epsilon != 0)

                    # Combine heading alignment and SOG alignment weighted by sog_weight
                    alignment_array[:, 0] = (1.0 - sog_weight) * alignment_array[:, 0] + sog_weight * weight * v_hat_vel_x * no_school
                    alignment_array[:, 1] = (1.0 - sog_weight) * alignment_array[:, 1] + sog_weight * weight * v_hat_vel_y * no_school
                except Exception:
                    # On any error, fall back to heading-only alignment
                    pass
            
            # Calculate a new ideal speed based on the mean speed of those fish around
            sogs = np.empty(num_agents, dtype=float)
            for ag in range(num_agents):
                neigh = self.simulation.agents_within_buffers[ag]
                if len(neigh) > 0:
                    sogs[ag] = np.mean(self.simulation.sog[neigh])
                else:
                    sogs[ag] = self.simulation.sog[ag]
            #sogs = np.array([np.mean(self.simulation.opt_sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])
            #sogs = np.array([np.min(self.simulation.opt_sog[neighbor_indices[np.where(agent_indices == agent)]]) for agent in np.arange(num_agents)])

            # make sure sogs don't get too low
            sogs = np.where(sogs < 0.5 * self.simulation.length / 1000,
                            0.5 * self.simulation.length / 1000,
                            sogs)
            
            self.simulation.school_sog = sogs
        
            return np.nan_to_num(alignment_array)

        def collision_cue(self, weight):
            """
            Generates an array of repulsive force vectors for each agent to avoid collisions,
            based on the positions of their nearest neighbors. This function leverages
            vectorization for efficient computation across multiple agents.
        
            Parameters
            ----------
            weight : float
                The weighting factor that scales the magnitude of the repulsive forces.
            closest_agent_dict : dict
                A dictionary mapping each agent's index to the index of its closest neighbor.
        
            Returns
            -------
            collision_cue_array : ndarray
                An array where each element is a 2D vector representing the repulsive force
                exerted on an agent by its closest neighbor. The force is directed away from
                the neighbor and is scaled by the weight and the inverse square of the distance
                between the agents.
        
            Notes
            -----
            The function internally calls `collision_repulsion`, which computes the repulsive
            force for an individual agent. The `np.vectorize` decorator is used to apply this
            function across all agents, resulting in an array of repulsive forces. The
            `excluded` parameter in the vectorization process is set to exclude the last three
            parameters from vectorization, as they are common to all function calls.
        
            Example
            -------
            # Assuming an instance of the class is created and initialized as `agent_model`
            # and closest_agent_dict is already computed:
            repulsive_forces = agent_model.collision_cue(weight=0.5,
                                                         closest_agent_dict=closest_agents)
            """
            
            # Filter out invalid indices (where nearest_neighbors is nan)
            valid_indices = ~np.isnan(self.simulation.closest_agent)
            
            # Initialize arrays for closest X and Y positions
            closest_X = np.full_like(self.simulation.X, np.nan)
            closest_Y = np.full_like(self.simulation.Y, np.nan)
            
            # Extract the closest X and Y positions using the valid indices
            closest_X[valid_indices] = self.simulation.X[self.simulation.closest_agent[valid_indices].astype(int)]
            closest_Y[valid_indices] = self.simulation.Y[self.simulation.closest_agent[valid_indices].astype(int)]
            
            # calculate vector pointing from neighbor to self
            self_2_closest = np.column_stack((closest_X.flatten() - self.simulation.X.flatten(), 
                                              closest_Y.flatten() - self.simulation.Y.flatten()))
            closest_2_self = np.column_stack((self.simulation.X.flatten() - closest_X.flatten(), 
                                              self.simulation.Y.flatten() - closest_Y.flatten()))
            
            coll_slice = determine_slices_from_vectors(closest_2_self, num_slices = 8)
            head_slice = determine_slices_from_headings(self.simulation.heading, num_slices = 8)
            
            # Handling np.nan values
            # If either component of a vector is np.nan, you might want to treat the whole vector as invalid
            invalid_vectors = np.isnan(closest_2_self).any(axis=1)
            closest_2_self[invalid_vectors] = [np.nan, np.nan]
            closest_2_self = np.nan_to_num(closest_2_self)
            
            # Replace zeros and NaNs in distances to avoid division errors
            # This step assumes that a zero distance implies the agent is its own closest neighbor, 
            # which might result in a zero vector or a scenario you'll want to handle separately.
            safe_distances = np.where(self.simulation.nearest_neighbor_distance > 0, 
                                      self.simulation.nearest_neighbor_distance, 
                                      np.nan)
            
            safe_distances_mm = safe_distances * 1000
            
            # Calculate unit vector components
            v_hat_x = np.divide(closest_2_self[:,0], safe_distances, 
                                out=np.zeros_like(closest_2_self[:,0]), where=safe_distances!=0)
            v_hat_y = np.divide(closest_2_self[:,1], safe_distances, 
                                out=np.zeros_like(closest_2_self[:,1]), where=safe_distances!=0)
                            
            # Calculate collision cue components
            collision_cue_x = np.divide(weight * v_hat_x, safe_distances**2, 
                                        out=np.zeros_like(v_hat_x), where=safe_distances!=0) #* same_quad_multiplier
            collision_cue_y = np.divide(weight * v_hat_y, safe_distances**2, 
                                        out=np.zeros_like(v_hat_y), where=safe_distances!=0) #* same_quad_multiplier
            
            # Optional: Combine the components into a single array
            collision_cue_mm = np.column_stack((collision_cue_x, collision_cue_y))
            #collision_cue = collision_cue_mm / 1000.
            
            np.nan_to_num(collision_cue_mm, copy = False)
            
            return collision_cue_mm     
        
        def is_in_eddy(self,t):
            """
            Assess whether each agent is in an eddy based on several conditions,
            including displacement and behavioral states. This function updates the
            `in_eddy` attribute of the class, which is a Boolean array where each element
            indicates whether the corresponding agent is considered to be in an eddy.
            
            Parameters:
            - t (int): The current timestep. This parameter is not currently used in the
                       function but can be included for future extensions or conditional
                       checks based on time.
            
            Notes:
            - The function initializes all agents as not being in an eddy.
            - Displacement is calculated between the first and the last recorded positions
              in `self.past_x` and `self.past_y`, provided these positions are not NaN.
            - Agents are determined to be in an eddy if:
                1. Their displacement is less than 30 units.
                2. Their `swim_behav` attribute is equal to 1.
                3. Their `swim_mode` attribute is equal to 1.
                4. Their `current_distances` are less than or equal to 6 times the ratio
                   of their `length` attribute to 1000.
            - This function relies on vectorized operations for efficient computation and
              assumes that `self.past_x`, `self.past_y`, `self.swim_behav`, `self.swim_mode`,
              `self.current_distances`, and `self.length` are numpy arrays of appropriate
              dimensions and have been initialized correctly.
            
            Returns:
            - None: The function updates the `self.in_eddy` attribute in place and does not
                    return any value.
            
            Raises:
            - This function does not explicitly raise errors but will fail if the input arrays
              are not correctly formatted or initialized.
            
            Example of usage:
            - Assuming an instance `simulation` of a class where `is_in_eddy` is defined,
              you might call it at a timestep `t` as follows:
                simulation.is_in_eddy(t=100)
            """
            
            # Compute linear positions based on available data
            if getattr(self.simulation, 'centerline', None) is not None:
                # Use centerline profile shapefile if available
                linear_positions = self.simulation.compute_linear_positions(self.simulation.centerline)
            else:
                # For HECRAS mode without centerline profile, compute cumulative distance traveled
                # along the migration path. For upstream migration, this is the integral of movement
                # against the local flow direction. For downstream, it's movement with the flow.
                
                # Initialize on first call
                if not hasattr(self.simulation, 'cumulative_migration_distance'):
                    self.simulation.cumulative_migration_distance = np.zeros(self.simulation.num_agents, dtype=float)
                    self.simulation.eddy_prev_X = self.simulation.X.copy()
                    self.simulation.eddy_prev_Y = self.simulation.Y.copy()
                
                # Compute displacement since last timestep
                dx = self.simulation.X.flatten() - self.simulation.eddy_prev_X.flatten()
                dy = self.simulation.Y.flatten() - self.simulation.eddy_prev_Y.flatten()
                
                # Get local velocity at each agent position
                if hasattr(self.simulation, 'x_vel') and hasattr(self.simulation, 'y_vel'):
                    vx = self.simulation.x_vel.flatten()
                    vy = self.simulation.y_vel.flatten()
                    
                    # For upstream migration: progress = movement against flow
                    # Project displacement onto local upstream direction (-velocity)
                    # progress = dx*(-vx) + dy*(-vy) normalized by velocity magnitude
                    vel_mag = np.sqrt(vx**2 + vy**2)
                    vel_mag = np.where(vel_mag == 0, 1e-12, vel_mag)  # Avoid divide by zero
                    
                    # Dot product of displacement with upstream direction (normalized)
                    migration_upstream = True  # TODO: add parameter
                    if migration_upstream:
                        progress_increment = -(dx * vx + dy * vy) / vel_mag
                    else:
                        progress_increment = (dx * vx + dy * vy) / vel_mag
                else:
                    # No velocity, use straight-line distance (assumes migration is eastward)
                    progress_increment = dx
                
                # Accumulate progress
                self.simulation.cumulative_migration_distance += progress_increment
                
                # Update previous positions for eddy tracking
                self.simulation.eddy_prev_X = self.simulation.X.copy()
                self.simulation.eddy_prev_Y = self.simulation.Y.copy()
                
                linear_positions = self.simulation.cumulative_migration_distance
            
            self.current_centerline_meas = linear_positions
            # Shift data to the left
            self.simulation.past_centerline_meas[:, :-1] = self.simulation.past_centerline_meas[:, 1:]
            self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]

        
            # Insert new position data at the last column
            self.simulation.past_centerline_meas[:, -1] = linear_positions
            self.simulation.swim_speeds[:, -1] = self.simulation.sog
                      
            # Check for valid entries in both the first and last columns
            valid_entries = ~np.isnan(self.simulation.swim_speeds[:, 0]) & ~np.isnan(self.simulation.swim_speeds[:, -1])
            
            # initialize and calculate average speeds
            avg_speeds = np.full(self.simulation.swim_speeds.shape[0], np.nan)
            avg_speeds[valid_entries] = np.max(self.simulation.swim_speeds[valid_entries], axis = -1)
            
            # total displacements
            total_displacement = np.full(self.simulation.past_centerline_meas.shape[0], np.nan)
            total_displacement[valid_entries] = self.simulation.past_centerline_meas[valid_entries,-1] - self.simulation.past_centerline_meas[valid_entries,0]
            
            # calculate the change in centerline measure, length of memory, and expected displacement given avg velocity
            delta = self.simulation.past_centerline_meas[valid_entries,0] - self.simulation.past_centerline_meas[valid_entries,-1]
            dt = self.simulation.past_centerline_meas.shape[1]
            expected_displacement = avg_speeds * dt
            
            # calculate change in centerline position (only if we have at least 2 timesteps)
            if self.simulation.past_centerline_meas.shape[1] >= 2:
                centerline_dir = self.simulation.past_centerline_meas[:,-2] - self.simulation.past_centerline_meas[:,-1]
            else:
                centerline_dir = np.zeros(self.simulation.num_agents)

            # Check if agents have moved less than expected, if they are moving backwards, and if they are sustained swimming mode
            if delta.shape == total_displacement.shape and t >= 1800.:
                # stuck_conditions = (expected_displacement >= 2* total_displacement) & \
                #     (self.simulation.swim_mode == 1) & (np.sign(delta) > 0) 
                    
                stuck_conditions = (expected_displacement >= 5. * np.abs(total_displacement)) \
                    & (self.simulation.swim_behav == 1)
            else:
                stuck_conditions = np.zeros_like(self.simulation.X)
            
            not_in_eddy_anymore = self.simulation.time_since_eddy_escape >= self.simulation.max_eddy_escape_seconds
            # Set a specific value (9999) for past positions and swim speeds where not in eddy anymore
            self.simulation.swim_speeds[not_in_eddy_anymore, :] = np.nan
            self.simulation.past_centerline_meas[not_in_eddy_anymore, :] = np.nan
            self.simulation.time_since_eddy_escape[not_in_eddy_anymore] = 0.0
                                 
            # Update in_eddy status based on conditions
            already_in_eddy = self.simulation.in_eddy == True
            self.simulation.in_eddy = np.where(np.logical_or(stuck_conditions,already_in_eddy), True, False) 
            self.simulation.in_eddy[not_in_eddy_anymore] = False
            self.simulation.time_since_eddy_escape[self.simulation.in_eddy == True] += 1
            
            if np.any(self.simulation.in_eddy):
                print ('check eddy escape calcs')
            
        def arbitrate(self,t):

            """
            Arbitrates between different behavioral cues to determine a new heading for each agent.
            This method considers the agent's behavioral mode and prioritizes different sensory inputs
            to calculate the most appropriate heading.
        
            Parameters
            ----------
            t : int
                The current time step in the simulation, used for cues that depend on historical data.
        
            Returns
            -------
            None
                This method updates the agent's heading in place.
        
            Notes
            -----
            - The method calculates various steering cues with predefined weights, such as rheotaxis,
              shallow water preference, wave drag, low-speed areas, historical avoidance, schooling behavior,
              and collision avoidance.
            - These cues are then organized into two dictionaries: `order_dict` which maintains the order
              of behavioral cues based on their importance, and `cue_dict` which holds all the steering cues.
            - The `f4ck_allocate` method is vectorized and called for each agent to allocate a limited
              amount of 'effort' to these cues, resulting in a new heading based on the agent's current
              swimming behavior.
            - The agent's heading is updated in place, reflecting the new direction based on the arbitration
              of cues.
        
            Example
            -------
            # Assuming an instance of the class is created and initialized as `agent_model`:
            agent_model.arbitrate(t=current_time_step)
            # The agent's heading is updated based on the arbitration of behavioral cues.
            """            
            # calculate behavioral cues
            if self.simulation.pid_tuning == True:
                rheotaxis = self.rheo_cue(50000)

            else:
                # Get behavioral weights (use learned values if available, otherwise defaults)
                bw = getattr(self.simulation, 'behavioral_weights', BehavioralWeights())
                
                # calculate attractive forces (using learned weights)
                rheotaxis = self.rheo_cue(bw.rheotaxis_weight * 10000)      # Default: 25000
                # Support SOG-aware alignment: BehavioralWeights may expose 'use_sog' and 'sog_weight'
                use_sog = getattr(bw, 'use_sog', True)
                sog_weight = getattr(bw, 'sog_weight', 0.5)
                alignment = self.alignment_cue(bw.alignment_weight * 20500, use_sog=use_sog, sog_weight=sog_weight) # Default: 20500
                cohesion = self.cohesion_cue(bw.cohesion_weight * 11000)    # Default: 11000
                low_speed = self.vel_cue(1500)                               # 1500 (keep fixed)
                wave_drag = self.wave_drag_cue(0)                            # 0 (keep fixed)
                refugia = self.find_nearest_refuge(50000)                    # 50000 (keep fixed)
                
                # calculate high priority repulsive forces (using learned weights)
                border = self.border_cue(bw.border_cue_weight, t)           # Default: 50000
                shallow = self.shallow_cue(100000)                           # 100000 (keep fixed - safety critical)
                avoid = self.already_been_here(25000, t)                     # 25000 (keep fixed)
                collision = self.collision_cue(bw.collision_weight * 10000) # Default: 50000
            
            # Create dictionary that has order of behavioral cues
            order_dict = {0: 'shallow',
                          1: 'border',
                          2: 'avoid',
                          3: 'collision', 
                          4: 'alignment', 
                          5: 'cohesion',
                          6: 'low_speed',
                          7: 'rheotaxis',  
                          8: 'wave_drag'}
            
            # Create dictionary that holds all steering cues
            cue_dict = {'rheotaxis': rheotaxis, 
                        'shallow': shallow, 
                        'border': border.T,
                        'wave_drag': wave_drag.T, 
                        'low_speed': low_speed.T, 
                        'avoid': avoid, 
                        'alignment': alignment,
                        'cohesion': cohesion,
                        'collision': collision,
                        'refugia': refugia}
            
            low_bat_cue_dict = {0:'shallow',
                                1:'border',
                                2:'refugia'}
            
            self.is_in_eddy(t)
            
            # Arbitrate between different behaviors
            # how many f4cks does this fish have?
            tolerance = 50000

            # Vectorized accumulation: stack ordered cues and add them while the
            # accumulated magnitude per-agent is below tolerance. This avoids
            # creating many large temporaries with np.where and removes Python
            # level loops over agents.
            num_agents = self.num_agents
            # Build ordered list of migratory cues (exclude refugia)
            migratory_cues = []
            for i in order_dict.keys():
                cue = order_dict[i]
                if cue != 'refugia':
                    migratory_cues.append(cue_dict[cue])
            if len(migratory_cues) > 0:
                # shape to (num_cues, num_agents, 2)
                migr_cues = np.stack(migratory_cues, axis=0)
                # transpose to (num_agents, num_cues, 2)
                migr_cues = np.transpose(migr_cues, (1, 0, 2))
            else:
                migr_cues = np.zeros((num_agents, 0, 2))

            vec_sum_migratory = np.zeros((num_agents, 2), dtype=float)
            # iterate over cues (small loop over number of cues)
            for j in range(migr_cues.shape[1]):
                add = migr_cues[:, j, :]
                mask = np.linalg.norm(vec_sum_migratory, axis=1) < tolerance
                if not np.any(mask):
                    break
                vec_sum_migratory[mask] += add[mask]

            # low-battery cues (small set)
            low_cues = [low_bat_cue_dict[i] for i in sorted(low_bat_cue_dict.keys())]
            low_stack = [cue_dict[c] for c in low_cues]
            if len(low_stack) > 0:
                low_cues_arr = np.stack(low_stack, axis=0)
                low_cues_arr = np.transpose(low_cues_arr, (1, 0, 2))
            else:
                low_cues_arr = np.zeros((num_agents, 0, 2))

            vec_sum_tired = np.zeros((num_agents, 2), dtype=float)
            for j in range(low_cues_arr.shape[1]):
                add = low_cues_arr[:, j, :]
                mask = np.linalg.norm(vec_sum_tired, axis=1) < tolerance
                if not np.any(mask):
                    break
                vec_sum_tired[mask] += add[mask]

            # now creating a heading vector for each fish - which is complicated because they are in different behavioral modes 
            head_vec = np.zeros((num_agents, 2), dtype=float)

            # assign by behavior masks (vectorized indexing)
            migr_mask = (self.simulation.swim_behav == 1)
            tired_mask = (self.simulation.swim_behav == 2) | (self.simulation.swim_behav == 3)
            eddy_mask = (self.simulation.in_eddy == 1)

            if np.any(migr_mask):
                head_vec[migr_mask] = vec_sum_migratory[migr_mask]
            if np.any(tired_mask):
                head_vec[tired_mask] = vec_sum_tired[tired_mask]
            if np.any(eddy_mask):
                head_vec[eddy_mask] = (cue_dict['border'] + cue_dict['shallow'])[eddy_mask]
            
            if len(head_vec.shape) == 2:
                new_heading = np.arctan2(head_vec[:, 1], head_vec[:, 0])
                # If head_vec is zero (no cues), keep current heading unchanged
                zero_vec_mask = np.linalg.norm(head_vec, axis=1) < 1e-9
                # Only update heading where we have non-zero cues
                final_heading = np.where(zero_vec_mask, self.simulation.heading, new_heading)
                return final_heading
            else:
                new_heading = np.arctan2(head_vec[:, 0, 1], head_vec[:, 0, 0])
                zero_vec_mask = np.linalg.norm(head_vec[:, 0, :], axis=1) < 1e-9
                final_heading = np.where(zero_vec_mask, self.simulation.heading, new_heading)
                return final_heading        

    class fatigue():
        '''
        A class dedicated to managing the fatigue and related physiological 
        parameters of a simulated fish population based on dynamic interactions 
        with their environment.
    
        Attributes:
            t (float): The current time in the simulation.
            dt (float): The time step increment for the simulation.
            simulation (object): An instance of another class handling specific 
            simulation details such as fish velocities, positions, and other metrics.
        '''
        
        def __init__ (self, t, dt, simulation_object):
            '''
            Initializes the fatigue class with time, timestep and a reference to 
            the simulation object.
            
            Parameters:
                t (float): The current time in the simulation.
                dt (float): The time step increment for the simulation.
                simulation_object (object): An instance of the class that handles 
                the environmental and biological simulation details.
            '''
            
            self.t = t
            self.dt = dt
            self.simulation = simulation_object
            
        def swim_speeds(self):
            '''
            Calculates the swim speeds for each fish by considering the difference 
            between the fish's velocity and the water velocity.
            
            Returns:
                numpy.ndarray: Array of swim speeds for each fish.
            '''
            # Vector components of water velocity and speed over ground for each fish
            water_velocities = np.column_stack((self.simulation.x_vel, 
                                                self.simulation.y_vel))
            fish_velocities = np.column_stack((self.simulation.sog * np.cos(self.simulation.heading),
                                               self.simulation.sog * np.sin(self.simulation.heading)))
        
            # Calculate swim speeds for each fish (relative to water)
            if _HAS_NUMBA:
                try:
                    # Use merged kernel to compute swim speeds and drags when possible
                    x_vel = np.ascontiguousarray(self.simulation.x_vel, dtype=np.float64)
                    y_vel = np.ascontiguousarray(self.simulation.y_vel, dtype=np.float64)
                    sog = np.ascontiguousarray(self.simulation.sog, dtype=np.float64)
                    heading = np.ascontiguousarray(self.simulation.heading, dtype=np.float64)
                    mask_arr = np.ascontiguousarray(np.ones(self.simulation.num_agents, dtype=np.bool_), dtype=np.bool_)
                    surf = np.ascontiguousarray(self.simulation.calc_surface_area(self.simulation.length), dtype=np.float64)
                    dragc = np.ascontiguousarray(self.simulation.drag_coeff(np.hypot(self.simulation.x_vel, self.simulation.y_vel) * (self.simulation.length/1000.) / np.maximum(self.kin_visc(self.simulation.water_temp), 1e-12)), dtype=np.float64)
                    wave = np.ascontiguousarray(self.simulation.wave_drag, dtype=np.float64)
                    # call merged kernel but do not update battery here
                    ss, bl_s_tmp, prolonged_tmp, sprint_tmp, sustained_tmp, drags_tmp, _ = _drag_and_battery_numba(
                        sog, heading, x_vel, y_vel, mask_arr,
                        float(self.wat_dens(self.simulation.water_temp).mean() if hasattr(self.simulation, 'water_temp') else 1.0),
                        surf, dragc, wave, self.simulation.swim_behav,
                        np.ascontiguousarray(self.simulation.battery, dtype=np.float64),
                        np.zeros(self.simulation.num_agents, dtype=np.float64),
                        np.zeros(self.simulation.num_agents, dtype=np.float64),
                        float(self.dt), False
                    )
                    swim_speeds = ss
                    # store drags into simulation state for later use
                    try:
                        self.simulation.drag = drags_tmp
                    except Exception:
                        self.simulation.drag = np.ascontiguousarray(drags_tmp)
                except Exception:
                    swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
            else:
                swim_speeds = np.linalg.norm(fish_velocities - water_velocities, axis=-1)
            # Shift the circular buffer and insert the new swim speeds
            self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]
            self.simulation.swim_speeds[:, -1] = swim_speeds
            return swim_speeds
        
        def bl_s(self, swim_speeds):
            '''
            Calculates the number of body lengths per second a fish swims.
            
            Parameters:
                swim_speeds (numpy.ndarray): The swim speeds of each fish as an array.
            
            Returns:
                numpy.ndarray: The number of body lengths per second for each fish.
            '''
            # calculate body lenghts per second
            bl_s = swim_speeds / (self.simulation.length/1000.)
            
            return bl_s

        def bout_distance(self):
            '''
            Updates the total distance traveled by each fish in a bout, and 
            increments the duration of the bout.
            
            Side effects:
                Modifies instance attributes related to the distance traveled 
                per bout and bout duration.
            '''
            # Calculate distances travelled and update bout odometer and duration
            try:
                prev_X = np.ascontiguousarray(self.simulation.prev_X, dtype=np.float64)
                X = np.ascontiguousarray(self.simulation.X, dtype=np.float64)
                prev_Y = np.ascontiguousarray(self.simulation.prev_Y, dtype=np.float64)
                Y = np.ascontiguousarray(self.simulation.Y, dtype=np.float64)
                dist_travelled = _bout_distance_numba(prev_X, X, prev_Y, Y)
            except Exception:
                dist_travelled = _bout_distance_numba(self.simulation.prev_X, self.simulation.X, self.simulation.prev_Y, self.simulation.Y)
            self.simulation.dist_per_bout += dist_travelled

            self.simulation.bout_dur += self.dt
            
        def time_to_fatigue(self, swim_speeds, mask_dict, method = 'CastroSantos'):
            '''
            Calculates the time to fatigue for each fish based on their current 
            swimming speeds and the selected fatigue model.
            
            Parameters:
                swim_speeds (numpy.ndarray): Array of current swim speeds for each fish.
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
                method (str): The fatigue model to apply. Default is 'CastroSantos'.
            
            Returns:
                numpy.ndarray: Array of time to fatigue for each fish.
            '''
            # Initialize time to fatigue (ttf) array
            ttf = np.full_like(swim_speeds, np.nan)
            
            if method == 'CastroSantos':
                a_p = self.simulation.a_p
                b_p = self.simulation.b_p
                a_s = self.simulation.a_s
                b_s = self.simulation.b_s
                lengths = self.simulation.length
                
                # Implement T Castro Santos (2005) via optimized helper
                try:
                    ss = np.ascontiguousarray(swim_speeds, dtype=np.float64)
                    m_pro = np.asarray(mask_dict['prolonged'], dtype=np.bool_)
                    m_sprint = np.asarray(mask_dict['sprint'], dtype=np.bool_)
                    ttf = _time_to_fatigue_numba(ss, m_pro, m_sprint, float(a_p), float(b_p), float(a_s), float(b_s))
                except Exception:
                    ttf = _time_to_fatigue_numba(swim_speeds, mask_dict['prolonged'].astype(bool), mask_dict['sprint'].astype(bool), a_p, b_p, a_s, b_s)
                return ttf
                
            elif method == 'Katapodis_Gervais':
                
                genus = 'Oncorhyncus'
                                                                                            
                # Regression parameters extracted from the document, indexed by species or group
                regression_params = {
                    'Oncorhyncus':{'K':3.5825,'b':-0.2621}
                }
                
                if genus in regression_params:
                    k = 6.3234 #regression_params[genus]['K'] # at upper 95% CI  
                    b = regression_params[genus]['b']
                else:
                    raise ValueError ("Species not found. Please use a species from the provided list or check the spelling.")
                    sys.exit()
                    
                # Calculate time to fatigue using the regression equation
                ttf = np.zeros(self.num_agents)
                ttf[~mask_dict['sustained']] = \
                    (swim_speeds[~mask_dict['sustained']]/k)** (1/b)
                
                return ttf
            
            else:
                raise ValueError ('emergent does not recognize %s the method passed'%(method))
                sys.exit()   
                
        def set_swim_mode(self, mask_dict):
            '''
            Sets the swim mode for each fish based on the provided behavior masks.
            
            Parameters:
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
            '''
            # Set swimming modes based on swim speeds
            mask_prolonged = mask_dict['prolonged']
            mask_sprint = mask_dict['sprint']
            
            # set swim mode
            self.simulation.swim_mode = np.where(mask_prolonged, 
                                                 2, 
                                                 self.simulation.swim_mode)
            self.simulation.swim_mode = np.where(mask_sprint,
                                                 3, 
                                                 self.simulation.swim_mode)
            self.simulation.swim_mode = np.where(~(mask_prolonged | mask_sprint), 
                                                 1, 
                                                 self.simulation.swim_mode)
            
        def recovery(self):
            '''
            Calculates the recovery percentage for each fish at the beginning 
            and end of the time step.
            
            Returns:
                numpy.ndarray: Array of recovery percentages for each fish.
            '''
            # Calculate recovery at the beginning and end of the time step
            rec0 = self.simulation.recovery(self.simulation.recover_stopwatch) / 100.
            rec0[rec0 < 0.0] = 0.0
            rec1 = self.simulation.recovery(self.simulation.recover_stopwatch + self.dt) / 100.
            rec1[rec1 > 1.0] = 1.0
            rec1[rec1 < 0.] = 0.0
            per_rec = rec1 - rec0
            
            # Recovery for fish that are station holding
            mask_station_holding = self.simulation.swim_behav == 3
            #self.bout_no[mask_station_holding] += 1.
            self.simulation.bout_dur[mask_station_holding] = 0.0
            self.simulation.dist_per_bout[mask_station_holding] = 0.0
            self.simulation.battery[mask_station_holding] += per_rec[mask_station_holding]
            self.simulation.recover_stopwatch[mask_station_holding] += self.dt
            
            return per_rec
        
        def calc_battery(self, per_rec, ttf, mask_dict):
            '''
            Updates the battery levels for each fish based on their swimming mode
            and recovery.
            
            Parameters:
                dt (float): The time step of the simulation.
                per_rec (numpy.ndarray): Array of percentages representing recovery 
                for each fish.
                ttf (numpy.ndarray): Array of time to fatigue values for each fish.
                mask_dict (dict): Dictionary of boolean arrays categorizing fish 
                swimming behaviors.
            '''
            
            # get fish that are swimming at a sustained level
            mask_sustained = mask_dict['sustained']
            if mask_sustained.ndim == 2:
                mask_sustained = mask_sustained.squeeze()
            # ensure shapes
            battery = self.simulation.battery
            per_rec_arr = per_rec
            ttf_arr = ttf
            # use numba kernel when available
            if _HAS_NUMBA:
                try:
                    batt = np.ascontiguousarray(battery, dtype=np.float64)
                    per_rec_a = np.ascontiguousarray(per_rec_arr, dtype=np.float64)
                    ttf_a = np.ascontiguousarray(ttf_arr, dtype=np.float64)
                    mask_sust = np.asarray(mask_sustained, dtype=np.bool_)
                    new_batt = _wrap_merged_battery_numba(batt, per_rec_a, ttf_a, mask_sust, float(self.dt))
                    self.simulation.battery = new_batt
                except Exception:
                    # fallback to original numpy behavior
                    battery[mask_sustained] += per_rec_arr[mask_sustained]
                    mask_non_sustained = ~mask_sustained
                    ttf0 = ttf_arr[mask_non_sustained] * battery[mask_non_sustained]
                    ttf1 = ttf0 - self.dt
                    safe = ttf0 != 0
                    battery[mask_non_sustained] *= np.where(safe, np.maximum(0.0, ttf1 / ttf0), 0.0)
                    self.simulation.battery = np.clip(battery, 0, 1)
            else:
                battery[mask_sustained] += per_rec_arr[mask_sustained]
                mask_non_sustained = ~mask_sustained
                ttf0 = ttf_arr[mask_non_sustained] * battery[mask_non_sustained]
                ttf1 = ttf0 - self.dt
                safe = ttf0 != 0
                battery[mask_non_sustained] *= np.where(safe, np.maximum(0.0, ttf1 / ttf0), 0.0)
                self.simulation.battery = np.clip(battery, 0, 1)
            
        def set_swim_behavior(self, battery_state_dict):
            '''
            Adjusts the swim behavior of the fish based on their current battery state.
            
            Parameters:
                battery_state_dict (dict): Dictionary defining categories of battery states.
            '''
            # Set swimming behavior based on battery level
            mask_low_battery = battery_state_dict['low'] 
            mask_mid_battery = battery_state_dict['mid'] 
            mask_high_battery = battery_state_dict['high'] 
        
            self.simulation.swim_behav = np.where(mask_low_battery, 
                                                  3, 
                                                  self.simulation.swim_behav)
            
            self.simulation.swim_behav = np.where(mask_mid_battery, 
                                                  2,
                                                  self.simulation.swim_behav)
            
            self.simulation.swim_behav = np.where(mask_high_battery,
                                                  1,
                                                  self.simulation.swim_behav)
            
        def set_ideal_sog(self, mask_dict, battery_state_dict):
            '''
            Calculates the optimal swim speeds for each fish based on vector flow speeds and mode.
            
            Parameters:
                flow_velocities (numpy.ndarray): The flow speeds experienced by each fish (n x 2 array).
            
            Returns:
                numpy.ndarray: Optimal swim speeds for each fish (n x 2 array).
            '''
            # Set swimming behavior based on battery level
            mask_low_battery = battery_state_dict['low'] 
            mask_mid_battery = battery_state_dict['mid'] 
            mask_high_battery = battery_state_dict['high'] 
            
            self.simulation.ideal_sog[mask_high_battery] = np.where(self.simulation.battery[mask_high_battery] == 1., 
                                                                    self.simulation.school_sog[mask_high_battery], 
                                                                    np.round((self.simulation.opt_sog[mask_high_battery] * \
                                                                             self.simulation.battery[mask_high_battery])/2, 2)
                                                                    )

            # Set ideal speed over ground based on battery level
            self.simulation.ideal_sog[mask_low_battery] = 0.0
            self.simulation.ideal_sog[mask_mid_battery] = 0.1
            
            # if np.any(mask_dict['sprint']):
            #     print ('fuck')

        def ready_to_move(self):
            '''
            Determines which fish are ready to resume movement based on their recovery state.
            
            Side effects:
                Modifies simulation attributes related to the movement readiness of the fish.
            '''
            # Fish ready to start moving again after recovery
            mask_ready_to_move = self.simulation.battery >= 0.85
            self.simulation.recover_stopwatch[mask_ready_to_move] = 0.0
            self.simulation.swim_behav[mask_ready_to_move] = 1
            self.simulation.swim_mode[mask_ready_to_move] = 1
            
        def PID_checks(self):
            '''
            Performs PID (Proportional-Integral-Derivative) checks if PID tuning
            is active in the simulation, primarily for debugging and optimization purposes.
            '''
            if self.simulation.pid_tuning == True:
                print(f'battery: {np.round(self.battery,4)}')
                print(f'swim behavior: {self.swim_behav[0]}')
                print(f'swim mode: {self.swim_mode[0]}')

                if np.any(self.simulation.swim_behav == 3):
                    print('error no longer counts, fatigued')
                    sys.exit()
                    
        def assess_fatigue(self):
            '''
            Comprehensive method to assess fatigue based on swim speeds, 
            calculates distances traveled, updates time to fatigue, recovery,
            and adjusts battery states accordingly.
            '''            
            # Use compiled core to compute swim speeds and masks where possible
            try:
                sog_a = np.ascontiguousarray(self.simulation.sog, dtype=np.float64)
                heading_a = np.ascontiguousarray(self.simulation.heading, dtype=np.float64)
                xv = np.ascontiguousarray(self.simulation.x_vel, dtype=np.float64)
                yv = np.ascontiguousarray(self.simulation.y_vel, dtype=np.float64)
                maxs = np.ascontiguousarray(self.simulation.max_s_U, dtype=np.float64)
                maxp = np.ascontiguousarray(self.simulation.max_p_U, dtype=np.float64)
                batt = np.ascontiguousarray(self.simulation.battery, dtype=np.float64)
                swim_buf = np.ascontiguousarray(self.simulation.swim_speeds, dtype=np.float64)
                # use merged kernel to compute swim speeds, fatigue masks, and drags
                mask_arr = np.ascontiguousarray(self.simulation.alive_mask if hasattr(self.simulation, 'alive_mask') else np.ones(self.simulation.num_agents, dtype=np.bool_), dtype=np.bool_)
                surf = np.ascontiguousarray(self.simulation.calc_surface_area(self.simulation.length), dtype=np.float64)
                dragc = np.ascontiguousarray(self.simulation.drag_coeff(np.hypot(self.simulation.x_vel, self.simulation.y_vel) * (self.simulation.length/1000.) / np.maximum(self.kin_visc(self.simulation.water_temp), 1e-12)), dtype=np.float64)
                wave = np.ascontiguousarray(self.simulation.wave_drag, dtype=np.float64)
                # precompute trig to avoid repeated cos/sin per-element
                cos_h = np.ascontiguousarray(np.cos(heading_a), dtype=np.float64)
                sin_h = np.ascontiguousarray(np.sin(heading_a), dtype=np.float64)
                # reuse an output drags array to avoid extra allocations
                drags_out = np.zeros((self.simulation.num_agents, 2), dtype=np.float64)
                swim_speeds, bl_s, prolonged, sprint, sustained = _merged_swim_drag_fatigue_numba(sog_a, cos_h, sin_h, xv, yv, mask_arr, float(self.wat_dens(self.simulation.water_temp).mean() if hasattr(self.simulation, 'water_temp') else 1.0), surf, dragc, wave, self.simulation.swim_behav, maxs, maxp, batt, swim_buf)
                mask_dict = {'prolonged': prolonged, 'sprint': sprint, 'sustained': sustained}
                # set drag into simulation state using the preallocated buffer
                try:
                    self.simulation.drag = drags_out
                except Exception:
                    self.simulation.drag = np.ascontiguousarray(drags_out)
            except Exception:
                swim_speeds = self.swim_speeds()
                bl_s = self.bl_s(swim_speeds)
                mask_dict = dict()
                mask_dict['prolonged'] = np.where((self.simulation.max_s_U < bl_s) & (bl_s <= self.simulation.max_p_U), True, False)
                mask_dict['sprint'] = np.where(bl_s > self.simulation.max_p_U, True, False)
                mask_dict['sustained'] = bl_s <= self.simulation.max_s_U

            # calculate how far this fish has travelled this bout
            self.bout_distance()

            # assess time to fatigue
            ttf = self.time_to_fatigue(bl_s, mask_dict)

            # set swim mode
            self.set_swim_mode(mask_dict)

            # assess recovery
            per_rec = self.recovery()

            # check battery: try single-pass merged drag+battery kernel for fewer Python<->Numba boundaries
            try:
                # prepare contiguous arrays
                sog_a = np.ascontiguousarray(self.simulation.sog, dtype=np.float64)
                heading_a = np.ascontiguousarray(self.simulation.heading, dtype=np.float64)
                xv_a = np.ascontiguousarray(self.simulation.x_vel, dtype=np.float64)
                yv_a = np.ascontiguousarray(self.simulation.y_vel, dtype=np.float64)
                mask_a = np.ascontiguousarray(self.simulation.alive_mask if hasattr(self.simulation, 'alive_mask') else np.ones(self.simulation.num_agents, dtype=np.bool_), dtype=np.bool_)
                surf_a = np.ascontiguousarray(self.simulation.calc_surface_area(self.simulation.length), dtype=np.float64)
                dragc_a = np.ascontiguousarray(self.simulation.drag_coeff(np.hypot(self.simulation.x_vel, self.simulation.y_vel) * (self.simulation.length/1000.) / np.maximum(self.kin_visc(self.simulation.water_temp), 1e-12)), dtype=np.float64)
                wave_a = np.ascontiguousarray(self.simulation.wave_drag, dtype=np.float64)
                batt_a = np.ascontiguousarray(self.simulation.battery, dtype=np.float64)
                per_rec_a = np.ascontiguousarray(per_rec, dtype=np.float64)
                ttf_a = np.ascontiguousarray(ttf, dtype=np.float64)
                swim_behav_a = np.ascontiguousarray(self.simulation.swim_behav, dtype=np.int64)
                # call single-pass kernel: returns drags, updated battery
                new_drags, new_batt = _drag_and_battery_numba(sog_a, heading_a, xv_a, yv_a, mask_a, float(self.wat_dens(self.simulation.water_temp).mean() if hasattr(self.simulation, 'water_temp') else 1.0), surf_a, dragc_a, wave_a, swim_behav_a, batt_a, per_rec_a, ttf_a, float(self.dt), True)
                # write back
                try:
                    self.simulation.drag = new_drags
                    self.simulation.battery = new_batt
                except Exception:
                    self.simulation.drag = np.ascontiguousarray(new_drags)
                    self.simulation.battery = np.ascontiguousarray(new_batt)
            except Exception:
                # fallback: previous behavior
                self.calc_battery(per_rec, ttf,  mask_dict)
            
            # set battery masks
            battery_dict = dict()
            battery_dict['low'] = self.simulation.battery <= 0.1
            battery_dict['mid'] = (self.simulation.battery > 0.1) & (self.simulation.battery <= 0.3)
            battery_dict['high'] = self.simulation.battery > 0.3
            
            # set swim behavior
            self.set_swim_behavior(battery_dict)
            
            # calculate ideal speed over ground
            #self.simulation.ideal_sog = self.set_ideal_sog(mask_dict, battery_dict)
            self.set_ideal_sog(mask_dict, battery_dict)
            
            # are fatigued fish ready to move?
            self.ready_to_move()
            
            # perform PID checks if we are optimizing controller
            self.PID_checks()
            
    def timestep(self, t, dt, g, pid_controller):
        """
        Simulates a single time step for all fish in the simulation."""
    
        # Instantiate inner classes for this timestep
        movement = self.movement(self)
        behavior = self.behavior(dt, self)
        fatigue = self.fatigue(t, dt, self)

        # Precompute pixel indices for this timestep to avoid repeated geo_to_pixel calls
        # Skip in HECRAS-only mode (no rasters loaded)
        if not getattr(self, 'use_hecras', False) or not getattr(self, 'hecras_mapping_enabled', False):
            try:
                self.precompute_pixel_indices()
            except Exception:
                # If precompute fails, do not block timestep (fallback to on-demand geo_to_pixel)
                pass
        
        # Update refugia map - skip in HECRAS-only mode (requires HDF5 file I/O)
        if not getattr(self, 'use_hecras', False) or not getattr(self, 'hecras_mapping_enabled', False):
            self.update_refugia_map(self.vel_mag)

        # Optimize vertical position
        movement.find_z()
        
        # Get wave drag multiplier
        behavior.wave_drag_multiplier()
        
        # Assess fatigue
        fatigue.assess_fatigue()
        
        # Calculate the ratio of ideal speed over ground to the magnitude of water velocity
        sog_to_water_vel_ratio = self.sog / np.hypot(self.x_vel, self.y_vel)
        
        # Calculate the sign of the heading and the water flow direction
        heading_sign = np.sign(self.heading)
        water_flow_direction_sign = np.sign(np.arctan2(self.y_vel, self.x_vel))
        
        # Calculate the time since the last jump
        time_since_jump = t - self.time_of_jump
        
        # Create a boolean mask for the fish that should jump
        should_jump = (sog_to_water_vel_ratio <= 0.10) & \
            (time_since_jump > 60) & \
                (self.battery >= 0.25)
        
        # Apply the jump or swim functions based on the condition
        # For each fish that should jump
        dxdy_jump = movement.jump(t=t, g = g, mask=should_jump)
        
        # For each fish that should swim
        movement.drag_fun(mask=~should_jump, t = t, dt = dt)
        movement.frequency(mask=~should_jump, t = t, dt = dt)
        movement.thrust_fun(mask=~should_jump, t = t, dt = dt)
        # optional profiling
        if getattr(self, 'profile', False):
            t_start = time.time()
            stage_times = {}
        if getattr(self, 'profile', False):
            s = time.time()
        dxdy_swim = movement.swim(t, dt, pid_controller = pid_controller, mask=~should_jump)
        if getattr(self, 'profile', False):
            stage_times['movement_swim'] = time.time() - s
        
        # Arbitrate amongst behavioral cues
        tolerance = 0.1  # A small tolerance level to account for floating-point arithmetic issues
        if getattr(self, 'profile', False):
            s = time.time()
        self.heading = behavior.arbitrate(t)
        if getattr(self, 'profile', False):
            stage_times['behavior_arbitrate'] = time.time() - s

        # Store old positions for sog calculation
        old_X = self.X.copy()
        old_Y = self.Y.copy()
            
        # Update positions
        self.X = self.X + dxdy_swim[:,0] + dxdy_jump[:,0]
        self.Y = self.Y + dxdy_swim[:,1] + dxdy_jump[:,1]

        # Simple billiard-ball collision response: if two agents are closer than 0.25*body_length, push them apart
        try:
            # Convert lengths (stored in mm) to meters for threshold
            if hasattr(self, 'length'):
                # length may be scalar or per-agent
                lengths_m = (np.asarray(self.length) / 1000.0)
                if lengths_m.size == 1:
                    lengths_m = np.full(self.num_agents, float(lengths_m))
            else:
                # fallback average length 0.45 m
                lengths_m = np.full(self.num_agents, 0.45)

            touch_thresh = 0.25 * lengths_m  # meters

            # Build pairwise KDTree for alive agents only to be efficient
            from scipy.spatial import cKDTree
            positions = np.column_stack([self.X, self.Y])
            tree = cKDTree(positions)
            # query pairs within max threshold
            max_thresh = np.max(touch_thresh)
            pairs = tree.query_pairs(r=max_thresh)

            if pairs:
                # For each pair, compute required separation and apply half displacement to each agent
                disp = np.zeros_like(positions)
                for i, j in pairs:
                    dx = self.X[j] - self.X[i]
                    dy = self.Y[j] - self.Y[i]
                    dist = np.hypot(dx, dy)
                    if dist <= 1e-6:
                        # If exactly overlapping, nudge randomly
                        nx, ny = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                        dist = np.hypot(nx, ny)
                        nx /= dist
                        ny /= dist
                    else:
                        nx, ny = dx / dist, dy / dist

                    # threshold for this pair = 0.25 * average body length
                    pair_thresh = 0.25 * (lengths_m[i] + lengths_m[j]) * 0.5
                    if dist < pair_thresh:
                        # separation amount required
                        sep = pair_thresh - dist
                        # apply half separation to each, opposite directions
                        disp_i = -0.5 * sep * np.array([nx, ny])
                        disp_j = 0.5 * sep * np.array([nx, ny])
                        disp[i] += disp_i
                        disp[j] += disp_j

                # Apply the displacement to positions and small velocity impulse
                if np.any(disp != 0):
                    self.X += disp[:, 0]
                    self.Y += disp[:, 1]
                    # small velocity impulse away from collision
                    self.x_vel += disp[:, 0] / max(1e-6, dt)
                    self.y_vel += disp[:, 1] / max(1e-6, dt)
        except Exception:
            pass
        
        # Re-sample HECRAS environmental data at new positions
        # Update every N frames for performance (hydraulics change slowly)
        hecras_update_interval = getattr(self, 'hecras_update_interval', 5)
        should_update_hecras = (int(t) % hecras_update_interval == 0)
        
        if getattr(self, 'use_hecras', False) and hasattr(self, 'hecras_node_fields') and should_update_hecras:
            try:
                # Fast update using cached KDTree (no rebuild)
                if getattr(self, 'profile', False):
                    s = time.time()
                self.update_hecras_mapping_for_current_positions()
                if getattr(self, 'profile', False):
                    stage_times['hecras_update_kdtree'] = time.time() - s
                
                # Re-sample depth, velocities
                if 'depth' in self.hecras_node_fields:
                    self.depth = self.apply_hecras_mapping(self.hecras_node_fields['depth'])
                    # Compute wetted status from depth (dry if depth <= too_shallow/2)
                    self.wet = np.where(self.depth > self.too_shallow / 2.0, 1.0, 0.0)
                if 'vel_x' in self.hecras_node_fields:
                    self.x_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_x'])
                if 'vel_y' in self.hecras_node_fields:
                    self.y_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_y'])
                    self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
                if 'distance_to' in self.hecras_node_fields:
                    self.distance_to = self.apply_hecras_mapping(self.hecras_node_fields['distance_to'])
                
                # Constrain agents to wetted areas - revert to previous position if now on land
                on_land = self.wet < 0.5
                if np.any(on_land):
                    # Revert to old position (border cue will push them away next timestep)
                    self.X = np.where(on_land, old_X, self.X)
                    self.Y = np.where(on_land, old_Y, self.Y)
                    
                    # Re-sample for corrected positions
                    self.update_hecras_mapping_for_current_positions()
                    if 'depth' in self.hecras_node_fields:
                        self.depth = self.apply_hecras_mapping(self.hecras_node_fields['depth'])
                        self.wet = np.where(self.depth > self.too_shallow / 2.0, 1.0, 0.0)
                    if 'vel_x' in self.hecras_node_fields:
                        self.x_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_x'])
                    if 'vel_y' in self.hecras_node_fields:
                        self.y_vel = self.apply_hecras_mapping(self.hecras_node_fields['vel_y'])
                        self.vel_mag = np.sqrt(self.x_vel**2 + self.y_vel**2)
                    if 'distance_to' in self.hecras_node_fields:
                        self.distance_to = self.apply_hecras_mapping(self.hecras_node_fields['distance_to'])
            except Exception as e:
                # Log error - this shouldn't fail silently
                print(f'ERROR resampling HECRAS data: {e}')
                import traceback
                traceback.print_exc()
                # Revert to old positions on error
                self.X = old_X
                self.Y = old_Y
                # record hecras resample time if profiling and no exception
        if getattr(self, 'profile', False):
            # finalize timing and write a log line
            try:
                stage_times['total_timestep'] = time.time() - t_start
            except Exception:
                stage_times['total_timestep'] = 0.0
            out_dir = os.path.join('outputs', 'rl_training')
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            log_path = os.path.join(out_dir, 'sim_profile.log')
            try:
                with open(log_path, 'a') as f:
                    f.write(f"t={t}, ")
                    f.write(', '.join([f"{k}={v:.6f}" for k, v in stage_times.items()]))
                    f.write('\n')
            except Exception:
                pass
        
        # Calculate sog from displacement
        self.sog = np.where(should_jump,
                            self.ideal_sog,
                            np.sqrt(np.power(self.X - old_X,2)+ np.power(self.Y - old_Y,2)) / dt)
        
        # Now update prev positions for next timestep
        self.prev_X = old_X
        self.prev_Y = old_Y
        
        if np.any(np.isnan(self.X)):
            print ('fish off map - why?')
            print ('dxdy swim: %s'%(dxdy_swim))
            print ('dxdy jump: %s'%(dxdy_jump))
            sys.exit()
            self.dead = np.where(np.isnan(self.X),1,self.dead)
    
        # Calculate mileage
        self.odometer(t=t, dt = dt)  
        
        # Log the timestep data - skip if using HECRAS mode for performance
        if not getattr(self, 'use_hecras', False):
            self.timestep_flush(t)
        
        # accumulate time
        self.cumulative_time = self.cumulative_time + dt
          
    def run(self, model_name, n, dt, video=False, k_p=None, k_i=None, k_d=None, interactive=False, g=9.81):
        """
        The simulation uses raster data for depth and agent positions to visualize the movement of agents in the environment.
        When `video` is True this writes an mp4 movie; when `interactive` is True this shows
        an on-screen real-time animation (no file output). If both are False the simulation
        runs headless and only prints progress.
        - model_name: A string representing the name of the model, used for titling the output movie.
        - n: An integer representing the number of time steps to simulate.
        - dt: Timestep duration in seconds
        - g: Gravitational acceleration (m/s^2), default 9.81

                # interactive (real-time) mode: show on-screen animation without saving
                if interactive:
                    plt.ion()
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # display depth raster as background
                    try:
                        depth_arr = self.hdf5['environment/depth'][:]
                        depth = rasterio.MemoryFile()
                        with depth.open(
                            driver='GTiff',
                            height=depth_arr.shape[0],
                            width=depth_arr.shape[1],
                            count=1,
                            dtype='float32',
                            crs=self.crs,
                            transform=self.depth_rast_transform
                        ) as dataset:
                            img = dataset.read(1)
                            background = ax.imshow(img,
                                                   origin='upper',
                                                   extent=[dataset.bounds[0],
                                                           dataset.bounds[2],
                                                           dataset.bounds[1],
                                                           dataset.bounds[3]],
                                                   cmap='viridis')
                    except Exception:
                        background = None

                    agent_pts, = ax.plot(self.X, self.Y, marker='o', ms=2, ls='', color='red')
                    ax.set_xlabel('Easting')
                    ax.set_ylabel('Northing')
                    fig.canvas.draw()

                    for i in range(int(n)):
                        self.timestep(i, dt, g, pid_controller)
                        agent_pts.set_data(self.X, self.Y)
                        fig.canvas.flush_events()
                        # small pause to allow GUI event loop to update; cap pause to a reasonable minimum
                        try:
                            plt.pause(max(0.001, float(dt)))
                        except Exception:
                            time.sleep(max(0.001, float(dt)))
                        print('Time Step %s complete' % (i))

                    plt.ioff()
                    try:
                        self.hdf5.flush()
                        self.hdf5.close()
                    except Exception:
                        pass

                else:
                    # iterate over timesteps 
                    for i in range(int(n)):
                        self.timestep(i, dt, g, pid_controller)
                        print ('Time Step %s complete'%(i))
        3. Initializes the plot for the simulation visualization.
        4. Iterates over the specified number of time steps, updating the agents and capturing each frame.
        5. Cleans up resources and finalizes the movie file.

        The simulation uses raster data for depth and agent positions to visualize the movement of agents in the environment. The output is a movie file that shows the progression of the simulation over time.



        Note:
        - The function assumes that the HDF5 dataset, coordinate reference system (CRS), and depth raster transformation are already set as attributes of the class instance.
        - The function prints the completion of each time step to the console.
        - The movie is saved in the directory specified by `self.model_dir`.

        Returns:
        None. The result of the function is the creation of a movie file visualizing the simulation.
        """
        t0 = time.time()
        if self.pid_tuning == False:
            if video == True:
                # get depth raster
                depth_arr = self.hdf5['environment/depth'][:]
                depth = rasterio.MemoryFile()
                height = depth_arr.shape[0]
                width = depth_arr.shape[1]
                
                with depth.open(
                    driver ='GTiff',
                    height = depth_arr.shape[0],
                    width = depth_arr.shape[1],
                    count =1,
                    dtype ='float32',
                    crs = self.crs,
                    transform = self.depth_rast_transform
                ) as dataset:
                    dataset.write(depth_arr, 1)
        
                    # define metadata for movie
                    FFMpegWriter = manimation.writers['ffmpeg']
                    metadata = dict(title= model_name, artist='Matplotlib',
                                    comment='emergent model run %s'%(datetime.now()))
                    writer = FFMpegWriter(fps = np.round(30/dt,0), metadata=metadata)
        
                    #initialize plot
                    fig, ax = plt.subplots(figsize = (10,5))
        
                    background = ax.imshow(dataset.read(1),
                                           origin = 'upper',
                                           extent = [dataset.bounds[0],
                                                      dataset.bounds[2],
                                                      dataset.bounds[1],
                                                      dataset.bounds[3]])
        
                    agent_pts, = plt.plot([], [], marker = 'o', ms = 1, ls = '', color = 'red')
        
                    plt.xlabel('Easting')
                    plt.ylabel('Northing')
        
                    # Update the frames for the movie
                    with writer.saving(fig, 
                                       os.path.join(self.model_dir,'%s.mp4'%(model_name)), 
                                       dpi = 300):
                        # set up PID controller 
                        #TODO make PID controller a function of length and water velocity
                        pid_controller = PID_controller(self.num_agents,
                                                        k_p, 
                                                        k_i, 
                                                        k_d)
                        
                        pid_controller.interp_PID()
                        for i in range(int(n)):
                            self.timestep(i, dt, g, pid_controller)
        
                            # write frame
                            agent_pts.set_data(self.X,
                                               self.Y)
                            writer.grab_frame()
                                
                            print ('Time Step %s complete'%(i))
    
                # clean up
                writer.finish()
                self.hdf5.flush()
                self.hdf5.close()
                depth.close()     
                t1 = time.time() 
            
            else:
                #TODO make PID controller a function of length and water velocity
                pid_controller = PID_controller(self.num_agents,
                                                k_p, 
                                                k_i, 
                                                k_d)
                
                pid_controller.interp_PID()
                
                # iterate over timesteps 
                for i in range(int(n)):
                    self.timestep(i, dt, g, pid_controller)
                    print ('Time Step %s complete'%(i))
                    
                # close and cleanup
                self.hdf5.flush()
                self.hdf5.close()
                t1 = time.time() 
                    
        else:
            pid_controller = PID_controller(self.num_agents,
                                            k_p, 
                                            k_i, 
                                            k_d)
            for i in range(n):
                self.timestep(i, dt, g, pid_controller)
                
                print ('Time Step %s %s %s %s %s %s complete'%(i,i,i,i,i,i))
                
                if i == range(n)[-1]:
                    self.hdf5.close()
                    sys.exit()
            
        print ('ABM took %s to compile'%(t1-t0))

    def close(self):
        self.hdf5.flush()
        self.hdf5.close()


class summary:
    '''The power of an agent based model lies in its ability to produce emergent
    behavior of interest to managers.  novel self organized patterns that 
    only happen once are a consequence, predictable self organized patterns  
    are powerful.  Each Emergent simulation should be run no less than 30 times. 
    This summary class object is designed to iterate over a parent directory, 
    extract data from child directories, and compile statistics.  The parent 
    directory describes a single scenario (for sockeye these are discharge)
    while each child directory is an individual iteration.  
    
    The class object iterates over child directories, extracts and manipulates data,
    calculate basic descriptive statistics, manages information, and utilizes 
    Poisson kriging to produce a surface that depicts the average number of agents
    per cell per second.  High use corridors should be visible in the surface.
    These corridors are akin to the desire paths we see snaking through college
    campuses and urban parks the world over.
    
    '''
    def __init__(self, parent_directory, tif_path):
        # set the model directory path
        self.parent_directory = parent_directory
        
        # where are the background tiffs stored?
        self.tif_path = tif_path
        
        #set input WS as parent_directory for compatibility with methods
        self.inputWS = parent_directory
        
        
        # get h5 files associated with this model
        self.h5_files = self.find_h5_files()
        
        # create empty thigs to hold agent data
        self.ts = gpd.GeoDataFrame(columns = ['agent','timestep','X','Y','kcal','Hz','filename','geometry'])
        self.morphometrics = pd.DataFrame()
        self.success_rates = {}

    def load_tiff(self, crs):
        # Define the desired CRS
        desired_crs = CRS.from_epsg(crs)

        # Open the TIFF file with rasterio
        with rasterio.open(self.tif_path) as tiff_dataset:
            # Calculate the transformation parameters for reprojecting
            transform, width, height = calculate_default_transform(
                tiff_dataset.crs, desired_crs, tiff_dataset.width, tiff_dataset.height,
                *tiff_dataset.bounds)
            
            cell_size = 2.
            # Calculate the new transform for 10x10 meter resolution
            new_transform = from_origin(transform.c, transform.f, cell_size, cell_size)
            
            # Calculate new width and height
            new_width = int((tiff_dataset.bounds.right - tiff_dataset.bounds.left) / cell_size)
            new_height = int((tiff_dataset.bounds.top - tiff_dataset.bounds.bottom) / cell_size)
            
            self.transform = new_transform
            self.width = new_width
            self.height = new_height

            # Reproject the TIFF image to the desired CRS
            image_data, _ = reproject(
                source=tiff_dataset.read(1),
                src_crs=tiff_dataset.crs,
                src_transform=tiff_dataset.transform,
                dst_crs=desired_crs,
                resampling=rasterio.enums.Resampling.bilinear)

            # Update the extent based on the reprojected data
            tiff_extent = rasterio.transform.array_bounds(height, width, transform)

        return image_data, tiff_extent
   
    # Find the .h5 files
    def find_h5_files(self):
        
        # create empty holders for all of the h5 files and child directories
        h5_files=[]
        child_dirs = []
        
        # first iterate over the parent diretory to find the children (iterations)
        for item in os.listdir(self.parent_directory):
            # create a full path object
            full_path = os.path.join(self.parent_directory, item)
            
            if full_path.endswith('.h5'):
                h5_files.append(full_path)
            
            # if full path is a directory and not a file - we found a child
            if os.path.isdir(full_path):
                child_dirs.append(full_path)
        
        # iterate over child directories and find the h5 files
        for child_dir in child_dirs:
            for filename in os.listdir(child_dir):
                if filename.endswith('.h5'):
                    h5_files.append(os.path.join(child_dir,filename))
        
        # we found our files
        return h5_files
        
    # Collect, rearrange, and manage data
    def get_data(self, h5_files):

        # Iterate through each HDF5 file in the specified directory and get data
        for filename in h5_files:

            with h5py.File(filename, 'r') as hdf:
                cell_center_x = pd.DataFrame(hdf['x_coords'][:])
                cell_center_x['row'] = np.arange(len(cell_center_x))
                cell_center_y = pd.DataFrame(hdf['y_coords'][:])
                cell_center_y['row'] = np.arange(len(cell_center_y))

                melted_center_x = pd.melt(cell_center_x, id_vars = ['row'], var_name = 'column', value_name = 'X')
                melted_center_y = pd.melt(cell_center_y, id_vars = ['row'], var_name = 'column', value_name = 'Y')
                melted_center = pd.merge(melted_center_x, melted_center_y, on = ['row','column'])
                self.melted_center = melted_center
                self.x_coords = hdf['x_coords'][:]
                self.y_coords = hdf['y_coords'][:]
                              
                if 'agent_data' in hdf:
                    # timestep data
                    X = pd.DataFrame(hdf['agent_data/X'][:])
                    X['agent'] = np.arange(X.shape[0])
                    Y = pd.DataFrame(hdf['agent_data/Y'][:])
                    Y['agent'] = np.arange(Y.shape[0])
                    Hz = pd.DataFrame(hdf['agent_data/Hz'][:])
                    Hz['agent'] = np.arange(Hz.shape[0])
                    kcal = pd.DataFrame(hdf['agent_data/kcal'][:])
                    kcal['agent'] = np.arange(kcal.shape[0])  
                    
                    # agent specific 
                    length = pd.DataFrame(hdf['agent_data/length'][:])
                    length['agent'] = np.arange(len(length))
                    length.rename(mapper = {0:'length'}, axis = 'columns', inplace = True)
                    
                    weight = pd.DataFrame(hdf['agent_data/weight'][:])
                    weight['agent'] = np.arange(len(weight))
                    weight.rename(mapper = {0:'weight'}, axis = 'columns', inplace = True)

                    body_depth = pd.DataFrame(hdf['agent_data/body_depth'][:])
                    body_depth['agent'] = np.arange(len(body_depth))
                    body_depth.rename(mapper = {0:'body_depth'}, axis = 'columns', inplace = True)

                    # melt time series data
                    melted_X = pd.melt(X, id_vars=['agent'], var_name='timestep', value_name='X')
                    melted_Y = pd.melt(Y, id_vars=['agent'], var_name='timestep', value_name='Y')
                    melted_kcal = pd.melt(kcal, id_vars=['agent'], var_name='timestep', value_name='kcal')
                    melted_Hz = pd.melt(Hz, id_vars=['agent'], var_name='timestep', value_name='Hz')
                    
                    # make one dataframe 
                    ts = pd.merge(melted_X, melted_Y, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_kcal, on = ['agent','timestep'])
                    ts = pd.merge(ts, melted_Hz, on = ['agent','timestep'])
                    ts['filename'] = filename
                    
                    print ('Data Imported ')
                    # turn ts into a geodataframe and find the fish that passed
                    geometry = [Point(xy) for xy in zip(ts['X'], ts['Y'])]
                    geo_ts = gpd.GeoDataFrame(ts, geometry=geometry)            
                    
                    # make one morphometric dataframe
                    morphometrics = pd.merge(length, weight, on = ['agent'])
                    morphometrics = pd.merge(morphometrics, body_depth, on = ['agent'])
                    
                    # add to summary data
                    self.ts = pd.concat([self.ts,geo_ts], ignore_index = True)
                    self.morphometrics = pd.concat([self.morphometrics,morphometrics], 
                                                  ignore_index = True) 
                    
                    print ('File %s imported'%(filename))
                    
                    
                    
    # Collect histograms of agent lengths
    def plot_lengths(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            pdf_filename = f"{base_name}_Lengths_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(output_folder, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        lengths = file['/agent_data/length'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            lengths_by_sex = lengths[sex_mask]
                            lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                            if lengths_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(lengths_by_sex, [75, 25])
                                    bin_width = 2 * (q75 - q25) * len(lengths_by_sex) ** (-1 / 3)

                                    if bin_width <= 0 or np.isnan(bin_width):
                                        bin_width = (max(lengths_by_sex) - min(lengths_by_sex)) / 10

                                    bins = max(1, round((max(lengths_by_sex) - min(lengths_by_sex)) / bin_width))
                                    ax.hist(lengths_by_sex, bins=bins, alpha=0.7, color='blue' if sex == 0 else 'pink')
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    continue

                                ax.set_title(f'{base_name} - {sex_label} Agent Lengths')
                                ax.set_xlabel('Length (mm)')
                                ax.set_ylabel('Frequency')
                                plt.tight_layout()
                                pdf.savefig(fig)
                                plt.close()
                            else:
                                print(f"No length values found for {sex_label}.")

    def length_statistics(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            stats_file_name = f"{base_name}_length_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(h5_file, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    lengths = file['/agent_data/length'][:]
                    sexes = file['/agent_data/sex'][:]

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        lengths_by_sex = lengths[sex_mask]
                        lengths_by_sex = lengths_by_sex[~np.isnan(lengths_by_sex)]

                        if lengths_by_sex.size > 1:
                            mean_length = np.mean(lengths_by_sex)
                            median_length = np.median(lengths_by_sex)
                            std_dev_length = np.std(lengths_by_sex, ddof=1)
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Length: {mean_length:.2f}\n")
                            output_file.write(f"  Median Length: {median_length:.2f}\n")
                            output_file.write(f"  Standard Deviation of Length: {std_dev_length:.2f}\n\n")
                        elif lengths_by_sex.size == 1:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Length: {lengths_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid length values found for {sex_label}.\n\n")

    def plot_weights(self):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_directory = os.path.dirname(h5_file)
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            pdf_filename = f"{base_name}_Weights_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(base_directory, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(h5_file, 'r') as file:
                    if 'agent_data' in file:
                        weights = file['/agent_data/weight'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            weights_by_sex = weights[sex_mask]
                            weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]

                            if weights_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(weights_by_sex, [75, 25])
                                    iqr = q75 - q25
                                    if iqr > 0:
                                        bin_width = 2 * iqr * len(weights_by_sex) ** (-1 / 3)
                                        bins = max(1, round((max(weights_by_sex) - min(weights_by_sex)) / bin_width))
                                    else:
                                        bins = 10

                                    ax.hist(weights_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                    ax.set_title(f'{base_name} - {sex_label} Agent Weights')
                                    ax.set_xlabel('Weight')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    pdf.savefig(fig)
                                    plt.close()
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    plt.close(fig)
                            else:
                                print(f"No weight values found for {sex_label} in {base_name}.")

    def weight_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_weight_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    weights = file['/agent_data/weight'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        weights_by_sex = weights[sex_mask]
                        weights_by_sex = weights_by_sex[~np.isnan(weights_by_sex)]  # Filter out NaN values

                        if weights_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                            mean_weight = np.mean(weights_by_sex)
                            median_weight = np.median(weights_by_sex)
                            std_dev_weight = np.std(weights_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Weight: {mean_weight:.2f}\n")
                            output_file.write(f"  Median Weight: {median_weight:.2f}\n")
                            output_file.write(f"  Standard Deviation of Weight: {std_dev_weight:.2f}\n\n")
                        elif weights_by_sex.size == 1:
                            # Handle single value case
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Weight: {weights_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid weight values found for {sex_label}.\n\n")

    def plot_body_depths(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            pdf_filename = f"{base_name}_Body_Depths_By_Sex_Comparison.pdf"
            pdf_filepath = os.path.join(output_folder, pdf_filename)

            with PdfPages(pdf_filepath) as pdf:
                with h5py.File(hdf_path, 'r') as file:
                    if 'agent_data' in file:
                        body_depths = file['/agent_data/body_depth'][:]
                        sexes = file['/agent_data/sex'][:]

                        for sex in np.unique(sexes):
                            sex_label = 'Male' if sex == 0 else 'Female'
                            sex_mask = sexes == sex
                            body_depths_by_sex = body_depths[sex_mask]
                            body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]

                            if body_depths_by_sex.size > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                try:
                                    q75, q25 = np.percentile(body_depths_by_sex, [75, 25])
                                    iqr = q75 - q25
                                    if iqr > 0:
                                        bin_width = 2 * iqr * len(body_depths_by_sex) ** (-1 / 3)
                                    else:
                                        bin_width = (max(body_depths_by_sex) - min(body_depths_by_sex)) / max(10, len(body_depths_by_sex))  # Avoid zero division

                                    bins = max(1, round((max(body_depths_by_sex) - min(body_depths_by_sex)) / bin_width))
                                    ax.hist(body_depths_by_sex, bins=bins, edgecolor='black', color='blue' if sex == 0 else 'pink')
                                    ax.set_title(f'{base_name} - {sex_label} Body Depths')
                                    ax.set_xlabel('Body Depth')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    pdf.savefig(fig)
                                    plt.close()
                                except Exception as e:
                                    print(f"Error in calculating histogram for {sex_label}: {e}")
                                    plt.close(fig)
                            else:
                                print(f"No body depth values found for {sex_label} in {base_name}.")

    def body_depth_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_body_depth_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)

            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    body_depths = file['/agent_data/body_depth'][:]
                    sexes = file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes

                    for sex in np.unique(sexes):
                        sex_mask = sexes == sex
                        body_depths_by_sex = body_depths[sex_mask]
                        body_depths_by_sex = body_depths_by_sex[~np.isnan(body_depths_by_sex)]  # Filter out NaN values

                        if body_depths_by_sex.size > 1:  # Ensure there's more than one value for statistical calculations
                            mean_body_depth = np.mean(body_depths_by_sex)
                            median_body_depth = np.median(body_depths_by_sex)
                            std_dev_body_depth = np.std(body_depths_by_sex, ddof=1)  # ddof=1 for sample standard deviation
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label}:\n")
                            output_file.write(f"  Average (Mean) Body Depth: {mean_body_depth:.2f}\n")
                            output_file.write(f"  Median Body Depth: {median_body_depth:.2f}\n")
                            output_file.write(f"  Standard Deviation of Body Depth: {std_dev_body_depth:.2f}\n\n")
                        elif body_depths_by_sex.size == 1:
                            # Handle single value case
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"Statistics for {sex_label} (only one data point):\n")
                            output_file.write(f"  Body Depth: {body_depths_by_sex[0]:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid body depth values found for {sex_label}.\n\n")
    def kcal_statistics(self):
        h5_files = self.h5_files
        for hdf_path in h5_files:
            base_name = os.path.splitext(os.path.basename(hdf_path))[0]
            output_folder = os.path.dirname(hdf_path)
            stats_file_name = f"{base_name}_kcal_statistics_by_sex.txt"
            stats_file_path = os.path.join(output_folder, stats_file_name)
    
            with h5py.File(hdf_path, 'r') as file, open(stats_file_path, 'w') as output_file:
                if 'agent_data' in file:
                    kcals = file['/agent_data/kcal'][:]  # Get kcal data for all agents and timesteps
                    sexes = file['/agent_data/sex'][:]    # Get sex data for all agents
    
                    # Sum kcal data across all timesteps for each agent
                    total_kcals_per_agent = np.nansum(kcals, axis=1)  # Sum along the timestep axis, ignoring NaNs
    
                    # Perform statistics on the total kcal data across all agents
                    mean_kcal = np.mean(total_kcals_per_agent)
                    median_kcal = np.median(total_kcals_per_agent)
                    std_dev_kcal = np.std(total_kcals_per_agent, ddof=1)  # Use ddof=1 for sample standard deviation
                    min_kcal = np.min(total_kcals_per_agent)
                    max_kcal = np.max(total_kcals_per_agent)
    
                    for i, (total_kcal, sex) in enumerate(zip(total_kcals_per_agent, sexes)):
                        if not np.isnan(total_kcal):  # Check if the total kcal is a valid number
                            sex_label = 'Male' if sex == 0 else 'Female'
    
                            output_file.write(f"Agent {i + 1} ({sex_label}):\n")
                            output_file.write(f"  Total Kcal: {total_kcal:.2f}\n")
                            output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                            output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                            output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                            output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                            output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                        else:
                            sex_label = 'Male' if sex == 0 else 'Female'
                            output_file.write(f"No valid kcal values found for Agent {i + 1} ({sex_label}).\n\n")
                            
    def kcal_statistics_directory(self):
        # Prepare to collect cumulative statistics
        cumulative_stats = {}
    
        # Iterate through all HDF5 files in the directory
        for hdf_path in self.h5_files:
            with h5py.File(hdf_path, 'r') as hdf_file:
                if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys():
                    kcals = hdf_file['/agent_data/kcal'][:]
                    sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
    
                    for sex in np.unique(sexes):
                        sex_label = 'Male' if sex == 0 else 'Female'
                        if sex_label not in cumulative_stats:
                            cumulative_stats[sex_label] = []
    
                        sex_mask = sexes == sex
                        kcals_by_sex = kcals[sex_mask]
    
                        # Process each agent's kcal data, ignoring inf and NaN values at each timestep
                        total_kcals_by_sex = []
                        for kcal_values in kcals_by_sex:
                            valid_kcals = kcal_values[np.isfinite(kcal_values)]  # Filter out inf and NaN values
                            total_kcal = np.sum(valid_kcals)  # Sum the remaining valid kcal values
                            
                            # Ignore NaN, inf, or total kcal values of 0
                            if np.isfinite(total_kcal) and total_kcal > 0:
                                total_kcals_by_sex.append(total_kcal)
    
                        cumulative_stats[sex_label].extend(total_kcals_by_sex)
    
        # Compute and print cumulative statistics
        stats_file_path = os.path.join(self.inputWS, "kcal_statistics_directory.txt")
        with open(stats_file_path, 'w') as output_file:
            for sex_label, values in cumulative_stats.items():
                if values:
                    values = np.array(values)
                    
                    # Further ensure no inf or NaN values are in the array
                    values = values[np.isfinite(values)]
                    
                    if len(values) > 0:
                        mean_kcal = np.mean(values)
                        median_kcal = np.median(values)
                        std_dev_kcal = np.std(values, ddof=1)
                        min_kcal = np.min(values)
                        max_kcal = np.max(values)
    
                        output_file.write(f"Cumulative Statistics for {sex_label}:\n")
                        output_file.write(f"  Average (Mean) Kcal: {mean_kcal:.2f}\n")
                        output_file.write(f"  Median Kcal: {median_kcal:.2f}\n")
                        output_file.write(f"  Standard Deviation of Kcal: {std_dev_kcal:.2f}\n")
                        output_file.write(f"  Minimum Kcal: {min_kcal:.2f}\n")
                        output_file.write(f"  Maximum Kcal: {max_kcal:.2f}\n\n")
                    else:
                        output_file.write(f"No valid kcal values found for {sex_label}.\n\n")
                else:
                    output_file.write(f"No valid kcal values found for {sex_label}.\n\n")
           

        
    def Kcal_histogram_by_timestep_intervals_for_all_simulations(self):
        # Generate histograms of total kcal per agent across HDF5 files (male/female separate).
        import os
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Get the folder name from inputWS
        folder_name = os.path.basename(os.path.normpath(self.inputWS))
        
        # Determine if "left" or "right" should be included in the title
        direction = "left" if "left" in folder_name.lower() else "right" if "right" in folder_name.lower() else ""
    
        # Prepare the output paths for male and female histograms
        male_histogram_path = os.path.join(self.inputWS, f"{folder_name}_male_kcal_histogram.jpg")
        female_histogram_path = os.path.join(self.inputWS, f"{folder_name}_female_kcal_histogram.jpg")
        
        # Initialize lists to collect kcal data across all HDF5 files
        all_male_kcals = []
        all_female_kcals = []
        
        # Set common plot parameters
        plt.rcParams.update({
            'font.size': 6,
            'font.family': 'serif',
            'figure.figsize': (3, 2)  # width, height in inches
        })
        
        # Iterate through all HDF5 files in the directory
        for hdf_path in self.h5_files:
            try:
                with h5py.File(hdf_path, 'r') as hdf_file:
                    if 'agent_data' in hdf_file and 'kcal' in hdf_file['agent_data'].keys() and 'sex' in hdf_file['agent_data'].keys():
                        kcals = hdf_file['/agent_data/kcal'][:]
                        sexes = hdf_file['/agent_data/sex'][:]  # Assumes 0 and 1 encoding for sexes
    
                        # Separate male and female data
                        male_kcals = kcals[sexes == 0]
                        female_kcals = kcals[sexes == 1]
    
                        # Calculate total kcal for each agent across all timesteps
                        male_total_kcal = np.nansum(male_kcals, axis=1)
                        female_total_kcal = np.nansum(female_kcals, axis=1)
    
                        # Replace inf and NaN values with zero
                        male_total_kcal = np.where(np.isfinite(male_total_kcal), male_total_kcal, 0)
                        female_total_kcal = np.where(np.isfinite(female_total_kcal), female_total_kcal, 0)
    
                        # Append the total kcal data to the lists
                        all_male_kcals.extend(male_total_kcal)
                        all_female_kcals.extend(female_total_kcal)
    
            except Exception as e:
                print(f"Failed to process {hdf_path}: {e}")
    
        # Convert lists to numpy arrays for histogram plotting
        all_male_kcals = np.array(all_male_kcals)
        all_female_kcals = np.array(all_female_kcals)
    
        # Create and save male histogram as JPEG
        plt.figure()
        plt.hist(all_male_kcals, bins=20, color='blue', alpha=0.7)
        plt.title(f"Male Total Kcal Usage ({direction})", fontsize=6)
        plt.xlabel("Total Kcal", fontsize=6)
        plt.ylabel("Frequency", fontsize=6)
        plt.tight_layout()
        plt.savefig(male_histogram_path, format='jpeg', dpi=300)
        plt.close()
    
        # Create and save female histogram as JPEG
        plt.figure()
        plt.hist(all_female_kcals, bins=20, color='red', alpha=0.7)
        plt.title(f"Female Total Kcal Usage ({direction})", fontsize=6)
        plt.xlabel("Total Kcal", fontsize=6)
        plt.ylabel("Probability Density", fontsize=6)
        plt.tight_layout()
        plt.savefig(female_histogram_path, format='jpeg', dpi=300)
        plt.close()
    
        print(f"Kcal histograms saved to {male_histogram_path} and {female_histogram_path}")
            


                    
    def kaplan_curve(self, shapefile_path, tiffWS):
        h5_files = self.h5_files
        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            output_folder = os.path.dirname(h5_file)
            jpeg_filename = f"{base_name}_Kaplan_Meier_Curve.jpeg"
            jpeg_filepath = os.path.join(output_folder, jpeg_filename)
            txt_filename = f"{base_name}_Kaplan_Meier_Statistics.txt"
            txt_filepath = os.path.join(output_folder, txt_filename)
    
            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty:
                continue
    
            # Load TIFF data for coordinate reference
            with rasterio.open(tiffWS) as tif:
                tif_crs = tif.crs
    
            # Adjust shapefile CRS if needed
            if gdf.crs != tif_crs:
                gdf = gdf.to_crs(tif_crs)
    
            # Filter self.ts for the current HDF5 file
            ts_filtered = self.ts[self.ts['filename'] == h5_file]
    
            # Perform spatial intersection with the shapefile
            intersection = gpd.overlay(ts_filtered, gdf, how='intersection')
    
            # Get the list of all agent IDs in the filtered data
            all_agents = ts_filtered['agent'].unique()
            total_agents = len(all_agents)
    
            if intersection.empty:
                continue
    
            # Get the unique list of agents that are found within the rectangle
            unique_agents_in_rectangle = intersection['agent'].unique()
            num_agents_in_rectangle = len(unique_agents_in_rectangle)
    
            # Prepare the first entry times for each agent
            entry_times = {agent: intersection[intersection['agent'] == agent]['timestep'].min()
                           for agent in unique_agents_in_rectangle}
    
            # Calculate total kcal consumed by each successful agent
            total_kcal = {agent: ts_filtered[ts_filtered['agent'] == agent]['kcal'].sum()
                          for agent in unique_agents_in_rectangle}
    
            # Convert to arrays for Kaplan-Meier analysis
            entry_times_array = np.array(list(entry_times.values()))
    
            # Create the survival data array (True if entered the rectangle, False if not)
            survival_data = np.array([(True, time) for time in entry_times_array], dtype=[('event', bool), ('time', int)])
    
            # Perform Kaplan-Meier estimation
            time, survival_prob = kaplan_meier_estimator(survival_data['event'], survival_data['time'])
    
            # Calculate the standard error and confidence intervals (95% CI)
            n = len(survival_data)
            se = np.sqrt((survival_prob * (1 - survival_prob)) / n)
            lower_ci = survival_prob - 1.96 * se
            upper_ci = survival_prob + 1.96 * se
    
            # Ensure CI bounds are within [0, 1]
            lower_ci = np.maximum(lower_ci, 0)
            upper_ci = np.minimum(upper_ci, 1)
    
            # Determine time points for various completion percentages
            completion_times = {
                10: time[np.where(survival_prob <= 0.9)[0][0]],
                30: time[np.where(survival_prob <= 0.7)[0][0]],
                50: time[np.where(survival_prob <= 0.5)[0][0]],
                70: time[np.where(survival_prob <= 0.3)[0][0]],
                90: time[np.where(survival_prob <= 0.1)[0][0]],
            }
            final_time = time[-1]
            completion_percentage = (num_agents_in_rectangle / total_agents) * 100
    
            # Calculate kcal statistics for successful agents
            kcal_values = np.array(list(total_kcal.values()))
            min_kcal = np.min(kcal_values)
            max_kcal = np.max(kcal_values)
            avg_kcal = np.mean(kcal_values)
    
            # Write the statistics and kcal information to a text file
            with open(txt_filepath, 'w') as txt_file:
                txt_file.write(f"Kaplan-Meier Statistics for {base_name}\n")
                txt_file.write("-" * 33 + "\n")
                for perc, comp_time in completion_times.items():
                    txt_file.write(f"Time at which {perc}% of agents completed passage: {comp_time}\n")
                txt_file.write(f"Last timestep the final agent crosses into the rectangle: {final_time}\n")
                txt_file.write(f"Percentage of agents that complete passage: {completion_percentage:.2f}%\n\n")
                
                # New section: Kilocalories consumed by successful agents
                txt_file.write("Kilocalories Consumed by Successful Agents:\n")
                txt_file.write("-" * 33 + "\n")
                for agent, kcal in total_kcal.items():
                    txt_file.write(f"Agent {agent}: {kcal:.2f} kcal\n")
                
                # New section: Kilocalories statistics
                txt_file.write("\nKilocalories Statistics:\n")
                txt_file.write(f"Minimum kcal: {min_kcal:.2f}\n")
                txt_file.write(f"Maximum kcal: {max_kcal:.2f}\n")
                txt_file.write(f"Average kcal: {avg_kcal:.2f}\n\n")
    
                txt_file.write("Explanation of the Confidence Interval:\n")
                txt_file.write("The confidence interval represents the range within which we expect the true proportion of agents\n")
                txt_file.write("remaining outside the rectangle to fall, with 95% confidence. The interval is calculated based on\n")
                txt_file.write("the observed data, and it provides a measure of uncertainty around the Kaplan-Meier estimate.\n")
                txt_file.write("A narrower confidence interval indicates greater certainty in the estimate, while a wider interval\n")
                txt_file.write("indicates more uncertainty. The shaded area around the Kaplan-Meier curve on the plot represents\n")
                txt_file.write("this confidence interval.\n")
                txt_file.write("\nWhat the Confidence Interval Says About This Data:\n")
                txt_file.write("In this data set, the confidence interval shows how consistent the agents' behavior is in relation to entering\n")
                txt_file.write("the specified rectangle. A narrow interval suggests that most agents have similar entry times, indicating\n")
                txt_file.write("uniform behavior. Conversely, a wide interval suggests that agent behavior is more varied, with significant\n")
                txt_file.write("differences in when agents enter the rectangle. This variability could be due to differences in individual\n")
                txt_file.write("agent characteristics, external influences, or random factors.\n")
    
            # Plot the Kaplan-Meier survival curve with confidence intervals
            plt.figure(figsize=(3.5, 2.5))  # Slightly larger figure to avoid chopping off the y-axis
            plt.step(time, survival_prob, where="post", label="Kaplan Meier Curve")
            plt.fill_between(time, lower_ci, upper_ci, color='gray', alpha=0.3, step="post", label="Confidence Interval")
            plt.xlabel("Time (Timesteps)", fontsize=6)
            plt.ylabel("Proportion of Agents Remaining Outside the Rectangle", fontsize=6)
            plt.title(f"Kaplan-Meier Curve\n{num_agents_in_rectangle} Agents Entered Rectangle out of {total_agents}", fontsize=6)
    
            # Add legend in the top right corner
            plt.legend(loc="upper right",  fontsize=6)
    
            # Adjust layout and save the plot as a JPEG image
            plt.tight_layout()
            plt.savefig(jpeg_filepath, format='jpeg', dpi=300, bbox_inches='tight')
            plt.close()



                    
                    
    def passage_success(self,finish_line):
        '''find the fish that are successful'''
        
        for filename in self.h5_files:
            dat = self.ts[self.ts.filename == filename]
            agents_in_box = dat[dat.within(finish_line)]
            
            # get the unique list of successful agents 
            succesful = agents_in_box.agent.unique()
            
            # get complete list of agents
            n_agents = dat.agent.unique()
            
            # success rate
            success = len(succesful) / n_agents 
            
            # get first detections in finish area
            passing_times = agents_in_box.groupby('agent')['timestep'].min()
            
            # get passage time stats
            pass_time_0 = passing_times.timestep.min()
            pass_time_10 = np.percentile(passing_times.timestep,10)
            pass_time_25 = np.percentile(passing_times.timestep,25)
            pass_time_50 = np.percentile(passing_times.timestep,50)
            pass_time_75 = np.percentile(passing_times.timestep,75)
            pass_time_90 = np.percentile(passing_times.timestep,90)
            pass_time_100 = passing_times.timestep.max()
            
            # make a row and add it to the success_rates dictionary
            self.success_rates[filename] = [success, 
                                            pass_time_0,
                                            pass_time_10,
                                            pass_time_25,
                                            pass_time_50,
                                            pass_time_75,
                                            pass_time_90,
                                            pass_time_100]
            
    def emergence(self,filename,scenario,crs):
        '''Method quantifies emergent spatial properties of the agent based model.
        
        Our problem is 5D, 2 spatial dimensions, 1 temporal dimension, iterations, 
        and finally scenario.  Comparison of each scenario output will typicall 
        occur in a GIS as those are the final surfaces.  
        
        Emergence first calculates the number of agents per cell per timestep.
        Then, emergence produces a 2 band raster for every iteration where the 
        first band is the average number of agents per cell per timestep and the 
        second band is the standard deviaition.  
        
        Then, Emergence statistically compares iterations, develops and index of 
        similarity, identifies a corridor threshold, and produces a final surface
        for comparison in a GIS.

        Returns
        -------
        corridor raster surface.

        '''
        # Agent coordinates and rasterio affine transform
        x_coords = self.ts.X  # X coordinates of agents
        y_coords = self.ts.Y  # Y coordinates of agents
        transform = self.transform  # affine transform from your rasterio dataset
        
        hdf5_filename = 'intermediate_results.h5'
        
        with h5py.File(os.path.join(self.parent_directory,hdf5_filename), 'w') as hdf5_file:
            for filename in self.ts.filename.unique():
                dat = self.ts[self.ts.filename == filename]
                num_timesteps = self.ts.timestep.max() + 1
                num_iterations = len(self.ts.filename.unique())
                
                # Create a dataset for each filename
                data_over_time = hdf5_file.create_dataset(
                    filename,
                    shape=(num_timesteps, self.height, self.width),
                    dtype=np.float32,
                    chunks=(1, self.height, self.width),
                    compression="gzip"
                )
                
                for timestep in range(num_timesteps):
                    t_dat = dat[dat.timestep == timestep]
                    
                    # Convert geographic coordinates to pixel indices using your function
                    rows, cols = geo_to_pixel(t_dat.X, t_dat.Y, transform)
                    
                    # Combine row and column indices to get unique cell identifiers
                    cell_indices = np.stack((cols, rows), axis=1)
                    
                    # Count unique cells
                    unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
                    valid_rows, valid_cols = unique_cells[:, 1], unique_cells[:, 0]  # Unpack the unique cell indices
                    
                    # Initialize a 2D array with zeros
                    agent_counts_grid = np.zeros((self.height, self.width), dtype=int)
                    
                    # Ensure the indices are within the grid bounds and update the agent_counts_grid
                    within_bounds = (valid_rows >= 0) & (valid_rows < self.height) & (valid_cols >= 0) & (valid_cols < self.width)
                    agent_counts_grid[valid_rows[within_bounds], valid_cols[within_bounds]] = counts[within_bounds]            
                    
                    # Insert the 2D array into the pre-allocated 3D array in HDF5
                    data_over_time[timestep, :, :] = agent_counts_grid
                    print(f'file {filename} timestep {timestep} complete')
            
            # Now aggregate results from HDF5
            all_data = []
            for filename in self.ts.filename.unique():
                data = da.from_array(hdf5_file[filename], chunks=(1, self.height, self.width))
                all_data.append(data)
            
            all_data = da.stack(all_data, axis=0)  # Stack along the new iteration axis
            
            # Calculate the average and standard deviation count per cell over all iterations and timesteps
            self.average_per_cell = da.mean(all_data, axis=(0, 1)).astype(np.float32).compute()
            self.sd_per_cell = da.std(all_data, axis=(0, 1)).astype(np.float32).compute()
        
        # Create dual band raster and write to output directory
        output_file = f'{scenario}_dual_band.tif'
        
        with rasterio.open(
            os.path.join(self.parent_directory,output_file),
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=2,  # Two bands
            dtype=self.average_per_cell.dtype,
            crs=crs,
            transform=self.transform
        ) as dst:
            dst.write(self.average_per_cell, 1)  # Write the average to the first band
            dst.write(self.sd_per_cell, 2)       # Write the standard deviation to the second band
        
        print(f'Dual band raster {output_file} created successfully.')
   
