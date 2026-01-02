"""Test fatigued fish fallback when thrust < drag."""
import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.movement import movement


def _make_sim_with_physics(num_agents=2):
    """Create a minimal simulation with enough physics to test fallback."""
    class Sim:
        pass
    
    sim = Sim()
    sim.num_agents = num_agents
    sim.length = np.array([500.0] * num_agents, dtype=float)
    sim.weight = np.array([2.0] * num_agents, dtype=float)
    sim.X = np.array([100.0] * num_agents, dtype=float)
    sim.Y = np.array([100.0] * num_agents, dtype=float)
    sim.prev_X = np.array([100.0] * num_agents, dtype=float)
    sim.prev_Y = np.array([100.0] * num_agents, dtype=float)
    sim.heading = np.array([0.0] * num_agents, dtype=float)  # facing east
    sim.ideal_sog = np.array([0.5] * num_agents, dtype=float)
    sim.sog = np.array([0.5] * num_agents, dtype=float)
    sim.Hz = np.array([2.0] * num_agents, dtype=float)
    
    # Water velocity - strong current against fish
    sim.x_vel = np.array([-1.0] * num_agents, dtype=float)  # strong westward current
    sim.y_vel = np.array([0.0] * num_agents, dtype=float)
    
    # Swim behavior: 1=normal, 3=fatigued
    sim.swim_behav = np.array([1, 3], dtype=int)
    
    sim.max_s_U = np.array([2.0] * num_agents, dtype=float)
    sim.max_s_U_fatigued = np.array([1.0] * num_agents, dtype=float)
    sim.wave_drag = np.array([1.0] * num_agents, dtype=float)
    sim.water_temp = 10.0
    sim.drag_coeff = lambda re: np.ones_like(re, dtype=float)
    sim.dead = np.zeros(num_agents, dtype=int)
    sim.is_stuck = np.zeros(num_agents, dtype=bool)
    sim.pid_tuning = False
    
    # Thrust and drag (will be set by movement)
    sim.thrust = np.zeros((num_agents, 2), dtype=float)
    sim.drag = np.zeros((num_agents, 2), dtype=float)
    sim.error = np.zeros((num_agents, 2), dtype=float)
    sim.integral = np.zeros((num_agents, 2), dtype=float)
    sim.pid_adjustment = np.zeros((num_agents, 2), dtype=float)
    sim.max_practical_sog = np.zeros((num_agents, 2), dtype=float)
    
    return sim


def test_fatigued_fish_fallback_when_thrust_less_than_drag():
    """Test that fatigued fish with thrust < drag fall backward."""
    sim = _make_sim_with_physics(num_agents=2)
    mv = movement(sim)
    
    # Manually set thrust and drag to simulate insufficient thrust
    # Agent 0 (normal): thrust > drag (can maintain position)
    sim.thrust[0] = [1.0, 0.0]  # strong thrust forward
    sim.drag[0] = [-0.5, 0.0]   # moderate drag backward
    
    # Agent 1 (fatigued): thrust < drag (cannot maintain position)
    sim.thrust[1] = [0.2, 0.0]  # weak thrust forward (only 2 Hz)
    sim.drag[1] = [-1.0, 0.0]   # strong drag backward
    
    # Create PID controller
    from emergent.salmon_abm.pid import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    
    # Call swim method
    mask = np.array([True, True], dtype=bool)
    tired_mask = (sim.swim_behav == 3)
    dxdy = mv.swim(t=0, dt=1.0, pid_controller=pid, mask=mask)
    
    # Check that fatigued fish has backward acceleration
    surge = sim.thrust + sim.drag
    acc = surge / sim.weight[:, np.newaxis]
    
    # Agent 0 (normal): should have positive forward acceleration
    assert acc[0, 0] > 0, f"Normal fish should have forward acceleration, got {acc[0, 0]}"
    
    # Agent 1 (fatigued): should have negative forward acceleration (falling back)
    assert acc[1, 0] < 0, f"Fatigued fish with insufficient thrust should have backward acceleration, got {acc[1, 0]}"
    
    # Check that fallback flag is set for agent 1
    assert hasattr(sim, 'fish_falling_back'), "Simulation should have fish_falling_back attribute"
    assert sim.fish_falling_back[1] == True, "Fatigued fish with negative surge should be marked as falling back"
    assert sim.fish_falling_back[0] == False, "Normal fish should not be marked as falling back"
    
    print(f"✓ Agent 0 (normal): acc_x = {acc[0, 0]:.3f} (forward)")
    print(f"✓ Agent 1 (fatigued): acc_x = {acc[1, 0]:.3f} (backward - falling back)")


def test_fatigued_fish_no_fallback_when_thrust_sufficient():
    """Test that fatigued fish with thrust >= drag do NOT fall back."""
    sim = _make_sim_with_physics(num_agents=2)  # Need 2 agents to match swim_behav setup
    sim.swim_behav[:] = 3  # both fatigued
    mv = movement(sim)
    
    # Set thrust >= drag (both fish can hold position)
    sim.thrust[0] = [1.0, 0.0]
    sim.drag[0] = [-0.5, 0.0]
    sim.thrust[1] = [0.8, 0.0]
    sim.drag[1] = [-0.6, 0.0]
    
    from emergent.salmon_abm.pid import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    
    mask = np.array([True, True], dtype=bool)
    dxdy = mv.swim(t=0, dt=1.0, pid_controller=pid, mask=mask)
    
    surge = sim.thrust + sim.drag
    acc = surge / sim.weight[:, np.newaxis]
    
    # Both should have positive or zero forward acceleration
    assert acc[0, 0] >= 0, f"Fatigued fish with sufficient thrust should not fall back, got {acc[0, 0]}"
    assert acc[1, 0] >= 0, f"Fatigued fish with sufficient thrust should not fall back, got {acc[1, 0]}"
    
    # Should NOT be marked as falling back
    if hasattr(sim, 'fish_falling_back'):
        assert sim.fish_falling_back[0] == False, "Fatigued fish with sufficient thrust should not be marked as falling back"
        assert sim.fish_falling_back[1] == False, "Fatigued fish with sufficient thrust should not be marked as falling back"
    
    print(f"✓ Fatigued fish 0 with sufficient thrust: acc_x = {acc[0, 0]:.3f} (holding)")
    print(f"✓ Fatigued fish 1 with sufficient thrust: acc_x = {acc[1, 0]:.3f} (holding)")


if __name__ == '__main__':
    print("Testing fatigued fish fallback physics...\n")
    test_fatigued_fish_fallback_when_thrust_less_than_drag()
    print()
    test_fatigued_fish_no_fallback_when_thrust_sufficient()
    print("\n✓ All fallback physics tests passed!")
