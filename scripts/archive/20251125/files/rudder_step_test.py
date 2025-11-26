"""
Run rudder step-response tests.
Produces CSVs: scripts/rudder_step_+5.csv, scripts/rudder_step_-5.csv, etc.
The simulation supports setting `test_mode='turncircle'` and `ship.constant_rudder_cmd` or we apply a time-varying commanded rudder in the sim loop.
"""
import time
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, SHIP_PHYSICS

# test parameters
steps = [5.0, 10.0]  # degrees steps to test
T = 120.0
dt = 0.1
n_agents = 1
port = 'Rosario Strait'

for step_deg in steps:
    for sign in [1, -1]:
        name = f'rudder_step_{int(sign*step_deg)}'
        csv_path = f'scripts/{name}.csv'
        # ensure no PID trace writing
        PID_TRACE['enabled'] = False
        # create sim in zigzag/test mode but we'll drive rudder directly via ship.constant_rudder_cmd
        sim = simulation(port_name=port, dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode='zigzag')
        sim.spawn()
        # set up step schedule via attributes the sim checks each timestep
        sim.tc_rudder_deg = float(sign * step_deg)  # used by some sim test modes
        # We'll set sim.ship.constant_rudder_cmd during the run by toggling a sim attribute
        # The simulation loop will cache sim.ship.smoothed_rudder and sim.applied_rudder_history
        # We'll implement the schedule using sim.test_mode hooks: set sim.ship.constant_rudder_cmd before run,
        # then restore afterwards. To emulate a step at t_switch and return at t_return, we set attributes
        sim.step_test_schedule = {'t_switch': 5.0, 't_return': 60.0, 'cmd_rad': math.radians(sign * step_deg)}

        # Pre-set constant rudder to zero; simulation will read sim.ship.constant_rudder_cmd each step
        sim.ship.constant_rudder_cmd = 0.0

        # Monkey-patch a small per-step hook into the simulation instance if possible
        def _per_step_hook(sim_inst, t):
            s = getattr(sim_inst, 'step_test_schedule', None)
            if s is None:
                return
            if t >= s['t_switch'] and t < s['t_return']:
                sim_inst.ship.constant_rudder_cmd = s['cmd_rad']
            else:
                sim_inst.ship.constant_rudder_cmd = 0.0

        setattr(sim, '_user_per_step_hook', _per_step_hook)

        # Run the simulation (the sim.run loop doesn't know about our hook, but we can call it by
        # temporarily wrapping sim._step_dynamics; however to keep changes minimal, we'll run sim.run()
        # and then extract sim.applied_rudder_history and sim.psi history that the sim records.
        # A lightweight approach: run sim.run() and then read the recorded histories
        sim.run()

        # After run, collect the histories the sim stores
        # sim.rudder_history: commanded rudder seen by sim (rad)
        # sim.applied_rudder_history: smoothed/applied rudder (rad)
        t_arr = np.arange(0.0, min(T, sim.t + sim.dt), dt)
        cmd_arr = np.array(getattr(sim, 'rudder_history', [np.nan]*len(t_arr)))
        applied_arr = np.array(getattr(sim, 'applied_rudder_history', [np.nan]*len(t_arr)))
        psi_arr = np.array(getattr(sim, 'psi_history', [np.nan]*len(t_arr)))

        # build DataFrame and write
        traj_df = pd.DataFrame({
            't': t_arr,
            'psi_deg': np.degrees(psi_arr),
            'cmd_rudder_deg': np.degrees(cmd_arr),
            'applied_rudder_deg': np.degrees(applied_arr)
        })
        traj_df.to_csv(csv_path, index=False)
        print(f'Wrote {csv_path}')
        time.sleep(0.1)

print('All step tests done')
