import csv
from emergent.ship_abm.simulation_core import simulation
import numpy as np

OUT = 'new_york_one_agent.csv'

# short headless sim for quick sanity check
sim = simulation(port_name='New York', dt=0.5, T=60.0, n_agents=1, load_enc=False)

# override wind/current to zero for a baseline
sim.wind_fn = lambda lon, lat, now: np.zeros((2, sim.n))
sim.current_fn = lambda lon, lat, now: np.zeros((2, sim.n))

# initialize positions and states if spawn not invoked
if sim.n == 1 and sim.pos.shape[1] != 1:
    sim.pos = np.zeros((2,1))

# run loop manually to capture traces
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'x', 'y', 'psi'])

    for step in range(sim.steps):
        t = step * sim.dt
        # get current nu from state [u,v,p,r] stored in sim.state
        nu = sim.state[[0,1,3], :]
        # compute controls
        hd, sp, rud = sim._compute_controls_and_update(nu, t)
        # compute dynamics step: call ship.step for vectorized ships
        new_state, rpms = sim.ship.step(sim.state, sim.ship.commanded_rpm, sim.goals, sim.wind_fn(0,0,0), sim.current_fn(0,0,0), sim.dt)
        sim.state = new_state
        sim.pos[0] = sim.state[0]
        sim.pos[1] = sim.state[1]
        sim.psi = sim.state[7:8].flatten()
        w.writerow([t, float(sim.pos[0,0]), float(sim.pos[1,0]), float(sim.psi[0])])

print('Wrote', OUT)
