import numpy as np
from emergent.ship_abm.ship_model import ship

# Minimal smoke test for colregs lock set/clear behavior

def run_headon_sequence():
    # two ships facing each other on x-axis
    n = 2
    # state dims: [x;y;u;v;p;r;phi;psi] per-ship
    # but colregs expects positions (2,n) and nu (2,n) and psi (n,)
    positions = np.array([[0.0, 1000.0], [0.0, 0.0]]).T  # will transpose later
    # positions shape should be (2,n)
    positions = positions.T
    # velocities in body frame u,v: both moving toward each other at 5 m/s
    nu = np.zeros((2, n))
    nu[0, 0] = 5.0
    nu[0, 1] = -5.0
    # headings: ship0 heading east (0 rad), ship1 heading west (pi rad)
    psi = np.array([0.0, np.pi])

    # create a ship instance with dummy state arrays
    state0 = np.zeros((8, n))
    pos0 = positions
    psi0 = psi
    goals = np.zeros((2, n))
    sh = ship(state0, pos0, psi0, goals)

    # run colregs for a few steps and print roles
    for t in range(5):
        head, speed_des, rpm, roles = sh.colregs(1.0, positions, nu, psi, np.array([0.0,0.0]))
        print(f"t={t} roles={roles} crossing_lock={sh.crossing_lock.tolist()} crossing_heading={sh.crossing_heading.tolist()} post_avoid={sh.post_avoid_timer.tolist()}")
        # advance positions slightly toward each other
        positions[0,0] += nu[0,0]
        positions[0,1] += nu[0,1]

if __name__ == '__main__':
    run_headon_sequence()
