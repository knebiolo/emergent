"""Inspect intermediate variables used by movement.frequency() for debugging.

Run after creating a small simulation to access sim.* attributes.
"""
import numpy as np
from emergent.salmon_abm.movement import movement
from emergent.salmon_abm.simulation import simulation
import os


def main():
    base = os.path.join('data', 'salmon_abm')
    sim = simulation(
        model_dir='outputs/debug',
        model_name='debug_freq',
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=os.path.join(base, 'start_loc_river_right.shp'),
        env_files=[],
        longitudinal_profile=os.path.join(base, 'longitudinal.shp'),
        num_timesteps=10,
        num_agents=5,
        db_path=os.path.join('outputs','debug','debug_freq.h5')
    )

    mv = movement(sim)
    mask = np.where(sim.dead == 0, True, False)
    t = 0
    dt = 1.0

    # Force some values to non-zero to see calculations
    sim.ideal_sog = np.repeat(0.475, sim.num_agents)
    sim.heading = np.repeat(0.0, sim.num_agents)
    sim.x_vel = np.zeros(sim.num_agents)
    sim.y_vel = np.zeros(sim.num_agents)
    # initialize attributes used by movement functions
    if not hasattr(sim, 'wave_drag'):
        sim.wave_drag = np.ones(sim.num_agents, dtype=float)
    if not hasattr(sim, 'Hz'):
        sim.Hz = np.zeros(sim.num_agents, dtype=float)
    # attach helper functions expected by movement to the sim instance
    if not hasattr(sim, 'drag_coeff'):
        sim.drag_coeff = lambda reynolds: np.interp(reynolds, [2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5], [0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])

    mv.thrust_fun(mask, t, dt)
    mv.drag_fun(mask, t, dt)
    mv.frequency(mask, t, dt)

    print('thrust:', sim.thrust[:5])
    print('drag:', sim.drag[:5])
    print('Hz:', sim.Hz[:5])

    sim.close()

if __name__ == '__main__':
    main()
