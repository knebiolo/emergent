"""Minimal Agent Runner: wires behavior -> movement -> energy -> PID for smoke tests."""
import numpy as np

from emergent.fish_passage.behavior.manager import BehaviorManager
from emergent.fish_passage.movement import swim_core, drag_and_battery, calc_battery, merged_battery
from emergent.fish_passage.pid import PID_controller


class AgentRunner:
    def __init__(self, n_agents, behavior_weights=None):
        self.n = n_agents
        self.num_agents = n_agents
        self.positions = np.zeros((n_agents, 2), dtype=float)
        self.headings = np.zeros(n_agents, dtype=float)
        self.speeds = np.zeros(n_agents, dtype=float)
        self.battery = np.ones(n_agents, dtype=float)
        self.behavior = BehaviorManager(weights=behavior_weights or {})
        self.pid = PID_controller(n_agents, k_p=1.0, k_i=0.0, k_d=0.0)

    def step(self, dt=1.0):
        # Behavior decides desired heading/speed (returns a dict)
        commands = self.behavior.step(self.positions, self.headings, dt)
        # expected keys: 'desired_heading_vec' (N,2), 'desired_speed' (N,)
        import numpy as _np
        desired_heading_vec = _np.asarray(commands.get('desired_heading_vec'), dtype=float)
        desired_speed = _np.asarray(commands.get('desired_speed'), dtype=float)
        # convert vector to heading angle (radians) for movement kernels
        desired_heading = _np.arctan2(desired_heading_vec[:, 1], desired_heading_vec[:, 0])
        # Apply swim_core to move agents
        # provide a typed env_forces array to satisfy numba-typed functions
        env_forces = _np.zeros_like(self.positions)
        new_pos, new_speeds = swim_core(self.positions, desired_heading, desired_speed, env_forces, dt)
        # Compute drag and battery updates
        forces, battery_updates = drag_and_battery(new_pos, new_speeds, desired_heading, None, dt)
        # Update battery via merged_battery if available
        try:
            self.battery = merged_battery(self.battery, np.ones_like(self.battery), new_speeds, dt)
        except Exception:
            # fallback: subtract battery_updates
            self.battery = self.battery - battery_updates
        # commit positions and speeds
        self.positions = new_pos
        self.speeds = new_speeds
        return {
            'positions': self.positions,
            'speeds': self.speeds,
            'battery': self.battery,
            'desired_heading_vec': desired_heading_vec,
            'desired_speed': desired_speed,
        }
