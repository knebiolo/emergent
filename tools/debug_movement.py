"""Debug movement: run a tiny simulation and print per-step internal values.

This script calls `simulation.timestep()` directly so we can inspect `thrust`,
`drag`, `error`, and position deltas after each step.
"""
import os
import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io


def discover_env_files(base_dir):
    keys = ['vel_x.tif','vel_y.tif','vel_dir.tif','vel_mag.tif','depth.tif','elev.tif','wsel.tif','wetted_perimeter.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def main():
    base = os.path.join('data', 'salmon_abm')
    env_files = discover_env_files(base)
    longitudinal = os.path.join(base, 'longitudinal.shp')
    start_poly = os.path.join(base, 'start_loc_river_right.shp')

    os.makedirs('outputs/debug', exist_ok=True)

    sim = simulation(
        model_dir='outputs/debug',
        model_name='debug_movement',
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=longitudinal,
        num_timesteps=20,
        num_agents=5,
        db_path=os.path.join('outputs','debug','debug_movement.h5')
    )

    os.makedirs('outputs/debug', exist_ok=True)
    sim.debug = True

    n_steps = 5
    dt = 1.0
    for t in range(n_steps):
        print('\n=== STEP', t, '===')
        sim.prev_X = sim.X.copy()
        sim.prev_Y = sim.Y.copy()
        # call functions in explicit order and print intermediate results
        mv = sim._movement
        mask = np.where(sim.dead == 0, True, False)
        try:
            mv.frequency(mask, t, dt)
        except Exception as e:
            print('frequency raised:', e)
        print('Hz after frequency:', sim.Hz[:5])

        try:
            mv.thrust_fun(mask, t, dt)
        except Exception as e:
            print('thrust_fun raised:', e)
        print('thrust after thrust_fun:', sim.thrust[:5])

        try:
            mv.drag_fun(mask, t, dt)
        except Exception as e:
            print('drag_fun raised:', e)
        print('drag after drag_fun:', sim.drag[:5])

        try:
            dxdy = mv.swim(t, dt, sim.pid_controller, mask)
        except Exception as e:
            print('swim raised:', e)
            dxdy = None
        print('dxdy returned by swim:', dxdy[:5] if dxdy is not None else None)

        # apply movement and compute velocities
        if dxdy is not None:
            try:
                sim.X = sim.X + dxdy[:, 0]
                sim.Y = sim.Y + dxdy[:, 1]
            except Exception:
                pass

        dx = sim.X - sim.prev_X
        dy = sim.Y - sim.prev_Y
        print('X[:5]:', sim.X[:5])
        print('Y[:5]:', sim.Y[:5])
        print('dx[:5]:', dx[:5])
        print('dy[:5]:', dy[:5])
        print('ideal_sog[:5]:', sim.ideal_sog[:5])
        print('x_vel[:5]:', sim.x_vel[:5])
        print('y_vel[:5]:', sim.y_vel[:5])
        print('Hz[:5]:', sim.Hz[:5])
        print('dead[:5]:', sim.dead[:5])
        print('error[:5]:', getattr(sim, 'error', None)[:5] if hasattr(sim, 'error') else None)

    sim.close()

if __name__ == '__main__':
    main()
