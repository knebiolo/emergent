"""Simulation step helpers: compose kernels to run one agent timestep."""
from typing import Tuple
import numpy as np
from emergent.fish_passage import merged_kernels


def step_agents(sog: np.ndarray, heading: np.ndarray, x_vel: np.ndarray, y_vel: np.ndarray, mask: np.ndarray, density: float, surface_areas: np.ndarray, drag_coeffs: np.ndarray, wave_drag: np.ndarray, swim_behav: np.ndarray, max_s_U: np.ndarray, max_p_U: np.ndarray, battery: np.ndarray, per_rec: np.ndarray, ttf: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single timestep for agents.

    Returns: (drags, new_battery, swim_speeds)
    """
    ss, bl_s, prolonged, sprint, sustained, drags, new_batt = merged_kernels.drag_and_battery(
        sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, per_rec, ttf, dt, True
    )
    return drags, new_batt, ss
