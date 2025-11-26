import numpy as np
from emergent.ship_abm.simulation_core import simulation

# Headless test: instantiate simulation core without GUI/ENC (fast)
def run_headless(port='Seattle', agents=2, steps=10, dt=1.0, load_enc=False, verbose=True):
    sim = simulation(port_name=port, dt=dt, T=steps*dt, n_agents=agents, load_enc=load_enc, verbose=verbose)
    # Ensure waypoints are present so spawn() will create ship model(s)
    if not hasattr(sim, 'waypoints') or len(getattr(sim, 'waypoints', [])) == 0:
        # Use projected coordinates (meters) with large magnitudes so spawn()
        # treats them as already-projected. Place agents ~1500 m apart.
        base_x = 1_000_000.0
        sim.waypoints = [ [(base_x, 0.0), (base_x + 1500.0, 0.0)], [(base_x + 1500.0, 0.0), (base_x, 0.0)] ]
    # spawn the simulation entities (states, pos, ship model)
    try:
        sim.spawn()
    except Exception as e:
        print('spawn() failed:', e)
        # fallback: create minimal ship model data
        if not hasattr(sim, 'pos'):
            sim.pos = np.array([[0.0, 1000.0], [0.0, 0.0]]).T
        if not hasattr(sim, 'state'):
            sim.state = np.zeros((8, sim.n))
        if not hasattr(sim, 'psi'):
            sim.psi = np.array([0.0, np.pi])
    print('Starting headless colregs test: n=', sim.n)
    for step in range(steps):
        # compute nu: body-frame u,v from state if available else fallback
        try:
            u = sim.state[2] if sim.state.shape[0] > 2 else np.zeros(sim.n)
            v = sim.state[3] if sim.state.shape[0] > 3 else np.zeros(sim.n)
            nu = np.vstack([u, v])
        except Exception:
            nu = np.zeros((2, sim.n))
        # call colregs
        col_hd, col_sp, col_rpm, roles = sim.ship.colregs(sim.dt, sim.pos, nu, sim.psi, sim.ship.commanded_rpm)
        # Diagnostic: print positions and pairwise distances
        print('positions:', sim.pos.tolist())
        for i in range(sim.n):
            for j in range(sim.n):
                if i==j: continue
                delta = sim.pos[:, j] - sim.pos[:, i]
                dist = np.linalg.norm(delta)
                print(f'  dist {i}->{j} = {dist:.1f}')
        from emergent.ship_abm.config import COLLISION_AVOIDANCE
        print(f"safe_dist={COLLISION_AVOIDANCE['safe_dist']}")
        print(f"step={step} roles={roles} crossing_lock={sim.ship.crossing_lock.tolist()} heading_cmds_deg={np.degrees(col_hd).tolist()} speed_cmds={col_sp}")
        # advance simple positions toward each other to provoke CPA
        # move each agent along its psi vector by 5 m
        for i in range(sim.n):
            sim.pos[0,i] += 5.0 * np.cos(sim.psi[i])
            sim.pos[1,i] += 5.0 * np.sin(sim.psi[i])
    print('done')

if __name__ == '__main__':
    run_headless()
