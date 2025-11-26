import time
import numpy as np
from emergent.ship_abm.simulation_core import simulation

# small headless run to detect commanded_rpm==0 events
sim = simulation(port_name='Seattle', n_agents=2, verbose=True, load_enc=False)
# spawn agents (use spawn() if available)
try:
    sim.spawn()
except Exception:
    # some setups pre-spawn in __init__; ignore if not present
    pass

for step in range(200):
    t = step * sim.dt
    # run one step of headless sim loop (reuse internal run pieces)
    # call controls + step dynamics manually to observe commanded_rpm
    nu = np.vstack([sim.state[0], sim.state[1], sim.state[3]])
    hd_cmds, sp_cmds, rud_cmds = sim._compute_controls_and_update(nu, t)
    # print commanded RPMs
    print(f"t={t:.2f}s commanded_rpm={sim.ship.commanded_rpm}")
    # check for zero throttle
    zeros = [i for i, r in enumerate(sim.ship.commanded_rpm) if r == 0.0]
    if zeros:
        print(f"Zero-throttle detected at t={t:.2f} for vessels: {zeros}")
        break
    sim._step_dynamics(hd_cmds, sp_cmds, rud_cmds)
    sim.t += sim.dt
print('done')
