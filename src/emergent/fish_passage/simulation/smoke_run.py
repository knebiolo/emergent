"""Integration smoke-run harness: runs a short scenario and returns final state."""
from emergent.fish_passage.simulation.agent_runner import AgentRunner

def run_short_scenario(n_agents=5, steps=3, dt=1.0):
    runner = AgentRunner(n_agents)
    for _ in range(steps):
        runner.step(dt=dt)
    return runner.positions, runner.speeds, runner.battery
