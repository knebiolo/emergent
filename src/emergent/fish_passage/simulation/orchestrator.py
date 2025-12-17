"""Simulation orchestrator: schedules timesteps, warms up numba, applies PID, and logs events."""
import numpy as np
from typing import Callable, List

from emergent.fish_passage.simulation.agent_runner import AgentRunner
from emergent.fish_passage.movement import _numba_warmup_for_sim as warmup


class Orchestrator:
    def __init__(self, n_agents: int, warmup_numba: bool = True, pid_gains=(1.0, 0.0, 0.0)):
        self.n_agents = n_agents
        self.runner = AgentRunner(n_agents)
        self.hooks: List[Callable] = []
        self.log = []
        self.pid_gains = pid_gains
        if warmup_numba:
            try:
                warmup.warmup_swim_core(self.runner.behavior.step)
            except Exception:
                # warmup best-effort
                pass

    def add_hook(self, fn: Callable):
        self.hooks.append(fn)

    def run(self, steps: int = 10, dt: float = 1.0):
        # attach PID controller properly
        self.runner.pid = self.runner.pid or None
        # per-step logging
        for t in range(steps):
            out = self.runner.step(dt=dt)
            # compute status mask: status==3 indicates dead/fatigued
            battery = out['battery']
            status = np.zeros(self.n_agents, dtype=int)
            status[battery <= 0.0] = 3
            # PID update (simulate desired -> control)
            desired_vec = out['desired_heading_vec']
            desired_speed = out['desired_speed']
            # compute error as difference between desired speed and current speed
            error = np.vstack((desired_vec[:,0] - out['speeds'], desired_vec[:,1] - out['speeds'])).T
            pid_out = self.runner.pid.update(error, dt, status)
            # call hooks
            for h in self.hooks:
                try:
                    h(t, out, pid_out)
                except Exception:
                    pass
            # record log entry
            self.log.append({'t': t, 'positions': out['positions'].copy(), 'speeds': out['speeds'].copy(), 'battery': out['battery'].copy(), 'pid': pid_out.copy()})
        return self.log
