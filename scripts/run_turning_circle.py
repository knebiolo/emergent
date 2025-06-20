# -*- coding: utf-8 -*-

# run_turncircle_test.py

from emergent.ship_abm.simulation_core import simulation, compute_turncircle_metrics
from emergent.ship_abm.config import SIMULATION_BOUNDS, xml_url
import matplotlib.pyplot as plt
import numpy as np

def run_turncircle_test():
    sim = simulation(
        port_name="Baltimore",
        dt=0.2,
        T=600,
        n_agents=1,
        use_ais=False,
        test_mode="turncircle"
    )

    sim.spawn()

    for _ in range(sim.steps):
        sim.ship.step(sim.state, sim.ship.commanded_rpm, sim.goals, wind=np.zeros((2, 1)), current=np.zeros((2, 1)), dt=sim.dt)
        sim.t += sim.dt
        sim.state = sim.ship.state
        sim.pos = sim.ship.pos
        sim.psi = sim.ship.psi

        sim.psi_history.append(sim.psi[0])
        sim.t_history.append(sim.t)
        sim.history[0].append(sim.pos[:, 0].copy())

    metrics = compute_turncircle_metrics(sim)
    print("\n🚢 TURNING CIRCLE METRICS")
    for k, v in metrics.items():
        print(f"{k:>30}: {v:.2f}")

    # Optional: plot turn circle
    xy = np.array(sim.history[0])
    plt.figure(figsize=(8, 8))
    plt.plot(xy[:, 0], xy[:, 1], label="Ship Path")
    plt.scatter(xy[0, 0], xy[0, 1], c='g', label='Start')
    plt.axis("equal")
    plt.title("Turning Circle")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_turncircle_test()
