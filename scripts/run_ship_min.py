import numpy as np
from emergent.ship_abm.ship_model import ship

def main():
    # simple single-ship initial state: x,y,u,v,p,r,phi,psi
    state0 = np.zeros((8, 1), dtype=float)
    pos0 = np.zeros((2, 1), dtype=float)
    psi0 = np.array([0.0])
    goals = np.array([[1000.0], [0.0]])  # goal 1000 m east

    s = ship(state0, pos0, psi0, goals)
    print("Ship instance created: n=", s.n)

    # test speed_to_rpm and thrust/drag/dynamics
    try:
        rpm = s.speed_to_rpm(5.0)
        print("speed_to_rpm(5.0) ->", rpm)
    except Exception as e:
        print("speed_to_rpm failed:", e)

    # run small dynamics step
    state_uvpr = np.zeros((4, 1))
    prop_thrust = np.array([0.0])
    drag_force = np.array([0.0])
    wind_force = np.zeros((2, 1))
    current_force = np.zeros((2, 1))
    rudder = np.array([0.0])

    try:
        ud, vd, pd, rd = s.dynamics(state_uvpr, prop_thrust, drag_force, wind_force, current_force, rudder)
        print('dynamics ->', ud, vd, pd, rd)
    except Exception as e:
        print('dynamics failed:', e)

if __name__ == '__main__':
    main()
